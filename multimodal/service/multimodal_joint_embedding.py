"""Optional local Qwen joint image/metadata embedding relevance scorer."""
from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
import time

import numpy as np

_MODULE_LOCK = threading.Lock()


class LocalJointEmbeddingReranker:
    """Text queries and single-image documents; unsupported content raises."""

    def __init__(self, path, device='cuda:0', timeout=30, capture_metadata=None):
        self.path = Path(path).resolve()
        self.script = self.path / 'scripts' / 'qwen3_vl_embedding.py'
        if not self.path.is_dir() or not self.script.is_file():
            raise ValueError('local embedding model and upstream script are required')
        if not np.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('invalid joint embedding timeout')
        self.device, self.timeout = device, timeout
        self.capture_metadata = capture_metadata
        self.lock = threading.Lock()
        self.model = self.processor = self.helper = None
        self.cache = OrderedDict()
        self.dimensions = None

    def _load(self):
        import torch
        name = '_structmem_joint_' + hashlib.sha256(str(self.script).encode()).hexdigest()
        with _MODULE_LOCK:
            module = sys.modules.get(name)
            if module is None:
                spec = importlib.util.spec_from_file_location(name, self.script)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(name, None)
                    raise
        model, info = module.Qwen3VLForEmbedding.from_pretrained(
            self.path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('incomplete local embedding weights')
        model = model.to(self.device).eval()
        processor = module.Qwen3VLProcessor.from_pretrained(
            self.path, local_files_only=True, padding_side='right')
        helper = object.__new__(module.Qwen3VLEmbedder)
        helper.max_length, helper.min_pixels, helper.max_pixels = 8192, 4096, 1843200
        helper.total_pixels, helper.fps = 7864320, 1
        helper.num_frames = helper.max_frames = 64
        helper.default_instruction = "Represent the user's input."
        helper.processor = processor
        self.model, self.processor, self.helper = model, processor, helper

    def _prepare(self, parts, image_decoder, *, query=False):
        if query:
            if len(parts) != 1 or parts[0].get('type') != 'text' or not parts[0].get('text', '').strip():
                raise ValueError('joint embedding requires one text query')
            return {'text': parts[0]['text']}
        if len(parts) != 1 or parts[0].get('type') != 'image_url':
            raise ValueError('joint embedding requires one original image document')
        value = parts[0]['image_url']
        uri = value['url'] if isinstance(value, dict) else value
        prefix, separator, encoded = uri.partition(',')
        if (not separator or prefix not in {'data:image/jpeg;base64', 'data:image/png;base64', 'data:image/webp;base64'}
                or len(encoded) > ((10 * 1024 * 1024 + 2) // 3) * 4):
            raise ValueError('invalid joint embedding image')
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError('joint embedding image exceeds limit')
        item = {'image': image_decoder(value)}
        if self.capture_metadata is not None:
            text = self.capture_metadata.text_from_bytes(raw)
            if text is not None:
                item['text'] = text
        return item

    def _infer(self, item):
        import torch
        from qwen_vl_utils import process_vision_info
        if self.model is None:
            self._load()
        conversation = self.helper.format_model_input(**item)
        rendered = self.processor.apply_chat_template(
            [conversation], add_generation_prompt=True, tokenize=False)
        images, videos, kwargs = process_vision_info(
            [conversation], image_patch_size=16, return_video_metadata=True,
            return_video_kwargs=True)
        if videos is not None:
            raise ValueError('joint embedding video is unsupported')
        batch = self.processor(text=rendered, images=images, padding=True,
                               do_resize=False, return_tensors='pt', **kwargs)
        if batch['input_ids'].shape[1] > 8192:
            raise ValueError('joint embedding input exceeds context budget')
        image_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        present = bool((batch['input_ids'] == image_id).any())
        expected = 'image' in item
        if present != expected or (expected and (
                'pixel_values' not in batch or len(batch['image_grid_thw']) != 1)):
            raise ValueError('joint embedding image input mismatch')
        with torch.inference_mode():
            result = self.model(**batch.to(self.device), use_cache=False)
            hidden = self.helper._pooling_last(result.last_hidden_state, batch['attention_mask'])
            return torch.nn.functional.normalize(hidden.float(), p=2, dim=-1).cpu().numpy()[0]

    def _vector(self, item):
        image = item.get('image')
        key = ((hashlib.sha256(image.tobytes()).hexdigest(), image.size, item.get('text'))
               if image is not None else (None, item['text']))
        if key not in self.cache:
            vector = np.asarray(self._infer(item), dtype=np.float32)
            if (vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all()
                    or abs(float(np.linalg.norm(vector)) - 1) > 2e-6
                    or (self.dimensions is not None and len(vector) != self.dimensions)):
                raise ValueError('invalid joint embedding vector')
            self.dimensions = len(vector)
            if len(self.cache) >= 4096:
                self.cache.popitem(last=False)
            self.cache[key] = vector
        self.cache.move_to_end(key)
        return self.cache[key]

    def score(self, query, documents, image_decoder):
        if not 1 <= len(documents) <= 20:
            raise ValueError('joint embedding requires one to twenty candidates')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('joint embedding is busy')
        try:
            deadline = time.monotonic() + self.timeout
            q = self._prepare(query, image_decoder, query=True)
            docs = [self._prepare(d, image_decoder) for d in documents]
            if time.monotonic() >= deadline:
                raise TimeoutError('joint embedding budget exhausted')
            qv = self._vector(q).astype(np.float64)
            scores = []
            for document in docs:
                if time.monotonic() >= deadline:
                    raise TimeoutError('joint embedding budget exhausted')
                scores.append(float(qv @ self._vector(document).astype(np.float64)))
            if time.monotonic() >= deadline:
                raise TimeoutError('joint embedding budget exhausted')
            return scores
        finally:
            self.lock.release()
