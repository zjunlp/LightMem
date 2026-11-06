"""Optional local Qwen3-VL relevance scoring. No downloads or answer generation."""
from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np


class LocalReranker:
    """Lazy local-only loading, bounded work, and no queue behind GPU inference."""

    instruction = 'Given a search query, retrieve relevant candidates that answer the query.'

    def __init__(self, path, device='cuda:0', timeout=30):
        if not Path(path).is_dir():
            raise ValueError('reranker model must be an existing local directory')
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('reranker timeout must be positive and finite')
        self.path, self.device, self.timeout = str(path), device, timeout
        self.lock = threading.Lock()
        self.model = self.processor = self.head = None

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(self.path, local_files_only=True,
                                                 padding_side='left')
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            self.path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').to(self.device).eval()
        vocab = processor.tokenizer.get_vocab()
        head = torch.nn.Linear(lm.lm_head.weight.shape[1], 1, bias=False)
        head.weight.data.copy_((lm.lm_head.weight[vocab['yes']] -
                               lm.lm_head.weight[vocab['no']]).unsqueeze(0))
        head = head.to(self.device, dtype=lm.dtype).eval()
        self.processor, self.model, self.head = processor, lm.model, head

    @staticmethod
    def _content(parts, prefix, image_decoder):
        content = [{'type': 'text', 'text': prefix}]
        for part in parts:
            if part['type'] == 'image_url':
                content.append({'type': 'image', 'image': image_decoder(part['image_url']),
                                'min_pixels': 4096, 'max_pixels': 1310720})
            elif part['type'] != 'text':
                raise ValueError('unsupported reranker content part')
        text = ' '.join(p['text'] for p in parts if p['type'] == 'text')
        if text:
            content.append({'type': 'text', 'text': text})
        if len(content) == 1:
            raise ValueError('empty reranker content')
        return content

    def score(self, query, documents, image_decoder):
        if not documents or len(documents) > 20:
            raise ValueError('reranker requires one to twenty candidates')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('reranker is busy')
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            deadline = time.monotonic() + self.timeout
            with torch.inference_mode():
                if self.model is None:
                    self._load()
                query_content = self._content(query, '<Query>:', image_decoder)
                scores = []
                for document in documents:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('reranker budget exhausted')
                    content = [{'type': 'text', 'text': '<Instruct>: ' + self.instruction}]
                    content += query_content + self._content(document, '\n<Document>:', image_decoder)
                    messages = [dict(role='system', content=[dict(type='text', text=(
                        'Judge whether the Document meets the requirements based on the Query '
                        'and the Instruct provided. Note that the answer can only be "yes" or "no".'))]),
                                dict(role='user', content=content)]
                    # Vision errors propagate; never replace malformed images by NULL.
                    images, videos, video_kwargs = process_vision_info(
                        [messages], image_patch_size=16, return_video_kwargs=True,
                        return_video_metadata=True)
                    if videos is not None:
                        raise ValueError('video is unsupported')
                    rendered = self.processor.apply_chat_template(
                        [messages], tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=rendered, images=images, padding=True,
                                            return_tensors='pt', do_resize=False, **video_kwargs)
                    if inputs['input_ids'].shape[1] > 8192:
                        raise ValueError('reranker input exceeds context budget')
                    image_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
                    expected = sum(c['type'] == 'image' for c in content)
                    present = int((inputs['input_ids'] == image_id).sum())
                    if bool(expected) != bool(present) or (expected and 'pixel_values' not in inputs):
                        raise ValueError('reranker image input missing')
                    if expected and int(inputs['image_grid_thw'].shape[0]) != expected:
                        raise ValueError('reranker image count mismatch')
                    inputs = inputs.to(self.device)
                    hidden = self.model(**inputs, use_cache=False).last_hidden_state[:, -1]
                    value = torch.sigmoid(self.head(hidden)).item()
                    if not np.isfinite(value):
                        raise ValueError('nonfinite reranker score')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('reranker budget exhausted')
                    scores.append(value)
                return scores
        finally:
            self.lock.release()
