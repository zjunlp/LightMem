"""Bounded remote per-part image/text embedding adapter."""
from __future__ import annotations

import base64
import io
import warnings
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
from PIL import Image, ImageOps



def embedding_image_url(url: str, threshold: int = 0, maxedge: int = 2048, quality: int = 95) -> str:
    """Optionally transform embedding input; never mutate original evidence."""
    if type(threshold) is not int or not 0 <= threshold <= 10 * 1024 * 1024:
        raise ValueError('invalid image preprocessing threshold')
    if type(maxedge) is not int or not 1 <= maxedge <= 8192 or type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError('invalid image preprocessing parameters')
    if threshold == 0:
        return url
    prefix, separator, encoded = url.partition(',')
    formats = {'data:image/png;base64': 'PNG', 'data:image/jpeg;base64': 'JPEG', 'data:image/webp;base64': 'WEBP'}
    if not separator or prefix not in formats or len(encoded) > ((10 * 1024 * 1024 + 2) // 3) * 4:
        raise ValueError('invalid inline image')
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError('image exceeds decoded limit')
    if len(raw) <= threshold:
        return url
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as original:
            if original.format != formats[prefix]:
                raise ValueError('image MIME mismatch')
            image = ImageOps.exif_transpose(original).convert('RGB')
            image.thumbnail((maxedge, maxedge))
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode('ascii')


class RemoteEncoder:
    def __init__(self, base_url: str, model: str, api_key: str, *,
                 concurrency: int = 4, timeout: float = 60,
                 client: httpx.Client | None = None, batch_size: int = 32, image_batch_size: int = 8, response_format: str = 'singlevector', image_threshold_bytes: int = 0,
                 image_max_edge: int = 2048, image_jpeg_quality: int = 95,
                 query_prompt_name: str | None = None, passage_prompt_name: str | None = None,
                 input_mode: str = 'multimodal_parts', dimensions: int | None = None):
        if not api_key or not model or not base_url.startswith(('https://', 'http://')):
            raise ValueError('remote embedding configuration is incomplete')
        if not 1 <= concurrency <= 64 or not 0 < timeout <= 300:
            raise ValueError('invalid remote embedding resource limits')
        if not 1 <= batch_size <= 128 or not 1 <= image_batch_size <= batch_size:
            raise ValueError('invalid remote embedding batch limits')
        if response_format not in {'singlevector', 'multivector'}:
            raise ValueError('invalid embedding response format')
        if input_mode not in {'multimodal_parts', 'text_only'}:
            raise ValueError('invalid embedding input mode')
        if dimensions is not None and (type(dimensions) is not int or not 1 <= dimensions <= 8192):
            raise ValueError('embedding dimensions must be an integer between 1 and 8192')
        for prompt in (query_prompt_name, passage_prompt_name):
            if prompt is not None and (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 128):
                raise ValueError('invalid embedding prompt name')
        self.prompt_names = {'query': query_prompt_name, 'passage': passage_prompt_name}
        self.input_mode, self.configured_dimensions = input_mode, dimensions
        if type(image_threshold_bytes) is not int or not 0 <= image_threshold_bytes <= 10 * 1024 * 1024:
            raise ValueError('invalid image preprocessing threshold')
        if type(image_max_edge) is not int or not 1 <= image_max_edge <= 8192 or type(image_jpeg_quality) is not int or not 1 <= image_jpeg_quality <= 100:
            raise ValueError('invalid image preprocessing parameters')
        self.image_threshold_bytes, self.image_max_edge, self.image_jpeg_quality = image_threshold_bytes, image_max_edge, image_jpeg_quality
        self.response_format = response_format
        self.batch_size, self.image_batch_size = batch_size, image_batch_size
        self.base_url = base_url.rstrip('/')
        self.embedding_model = model
        self._api_key = api_key
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self.slots = threading.BoundedSemaphore(concurrency)
        self.concurrency = concurrency
        self.pending = threading.BoundedSemaphore(concurrency * 2)
        self.executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='embedding')
        self.closed = False
        self.state_lock = threading.Lock()
        self.dimensions = None
        self.loaded = False
        identity_parts = [self.base_url, model, 'per-part-text-image-v2', response_format]
        if input_mode != 'multimodal_parts' or dimensions is not None:
            identity_parts.append(['input-mode-v1', input_mode, dimensions])
        if image_threshold_bytes:
            identity_parts.append(['exif-RGB-thumbnail-JPEG-v1', image_threshold_bytes, image_max_edge, image_jpeg_quality])
        if query_prompt_name is not None or passage_prompt_name is not None:
            identity_parts.append(['prompt-names-v1', query_prompt_name, passage_prompt_name])
        identity = json.dumps(identity_parts, separators=(',', ':'))
        self.signature = 'remote-parts-v2:' + hashlib.sha256(identity.encode()).hexdigest()

    def encode(self, parts: list[dict]) -> np.ndarray:
        vectors = [value for value in self.encode_parts(parts) if value is not None and len(value)]
        if not vectors:
            raise ValueError('content has no encodable parts')
        return np.concatenate(vectors, axis=0)

    def encode_parts(self, parts: list[dict], *, purpose: str = 'passage') -> list[np.ndarray | None]:
        if purpose not in self.prompt_names:
            raise ValueError('invalid embedding purpose')
        batches = []
        current = []
        indices = []
        images = 0
        for index, part in enumerate(parts):
            if self.input_mode == 'text_only' and (
                    part.get('type') != 'text' or not str(part.get('text') or '').strip()):
                continue
            is_image = part['type'] == 'image_url'
            if current and (len(current) >= self.batch_size or (is_image and images >= self.image_batch_size)):
                batches.append((indices, current))
                current, indices, images = [], [], 0
            current.append(part)
            indices.append(index)
            images += int(is_image)
        if current:
            batches.append((indices, current))
        if not batches:
            if self.input_mode == 'text_only':
                return [None] * len(parts)
            raise ValueError('no embedding inputs')
        if self.closed:
            raise RuntimeError('embedding encoder is closed')
        result = [None] * len(parts)
        if len(batches) == 1:
            indices, batch = batches[0]
            encoded = self._encode_batch(batch, self.prompt_names[purpose])
            for index, value in zip(indices, encoded, strict=True):
                result[index] = value
            return result
        futures = []
        try:
            for indices, batch in batches:
                for _, previous in futures:
                    if previous.done():
                        previous.result()
                # Bound both running work and queued tasks across callers.
                if not self.pending.acquire(timeout=self.timeout):
                    raise RuntimeError('remote embedding queue limit reached')
                try:
                    future = self.executor.submit(self._encode_batch, batch, self.prompt_names[purpose])
                except BaseException:
                    self.pending.release()
                    raise
                future.add_done_callback(lambda _: self.pending.release())
                futures.append((indices, future))
            for indices, future in futures:
                for index, value in zip(indices, future.result(), strict=True):
                    result[index] = value
            return result
        except BaseException:
            for _, future in futures:
                future.cancel()
            raise

    def close(self):
        with self.state_lock:
            if self.closed:
                return
            self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        # Direct single-batch calls also hold slots; drain before client close.
        for _ in range(self.concurrency):
            self.slots.acquire()
        self.client.close()

    def _encode_batch(self, parts: list[dict], prompt_name: str | None = None) -> list[np.ndarray]:
        inputs = []
        for part in parts:
            if part['type'] == 'text':
                inputs.append(str(part['text']) if self.input_mode == 'text_only' else {'text': part['text']})
            elif part['type'] == 'image_url':
                if self.input_mode == 'text_only':
                    raise ValueError('image parts cannot be sent in text-only embedding mode')
                image = part['image_url']
                url = image['url'] if isinstance(image, dict) else image
                inputs.append({'image': embedding_image_url(url, self.image_threshold_bytes, self.image_max_edge, self.image_jpeg_quality)})
            else:
                raise ValueError('unsupported embedding content part')
        if not inputs:
            raise ValueError('no embedding inputs')
        payload = {'model': self.embedding_model, 'input': inputs, 'encoding_format': 'float'}
        if self.configured_dimensions is not None:
            payload['dimensions'] = self.configured_dimensions
        if prompt_name is not None:
            payload['prompt_name'] = prompt_name
        if self.closed:
            raise RuntimeError('embedding encoder is closed')
        if not self.slots.acquire(timeout=self.timeout):
            raise RuntimeError('remote embedding concurrency limit reached')
        try:
            if self.closed:
                raise RuntimeError('embedding encoder is closed')
            response = None
            for attempt in range(3):
                try:
                    response = self.client.post(self.base_url + '/embeddings',
                        headers={'Authorization': 'Bearer ' + self._api_key},
                        json=payload,
                        timeout=self.timeout)
                except httpx.TransportError:
                    if attempt == 2:
                        raise RuntimeError('remote embedding transport failure') from None
                else:
                    if response.status_code == 200:
                        break
                    if response.status_code != 429 and response.status_code < 500:
                        raise RuntimeError(f'remote embedding request rejected (HTTP {response.status_code})')
                    if attempt == 2:
                        raise RuntimeError(f'remote embedding service unavailable (HTTP {response.status_code})')
                time.sleep(0.25 * (2 ** attempt))
            try:
                rows = response.json()['data']
                if not isinstance(rows, list) or len(rows) != len(inputs):
                    raise ValueError('embedding count mismatch')
                ordered = [None] * len(inputs)
                for row in rows:
                    index = row['index']
                    if type(index) is not int or not 0 <= index < len(inputs) or ordered[index] is not None:
                        raise ValueError('invalid embedding index')
                    values = row['embeddings' if self.response_format == 'multivector' else 'embedding']
                    if self.response_format == 'singlevector':
                        values = [values]
                    if not isinstance(values, list) or not values or any(
                        not isinstance(v, list) or not v or any(type(x) not in (int, float) for x in v) for v in values
                    ):
                        raise ValueError('invalid embedding values')
                    vectors = np.asarray(values, dtype=np.float32)
                    if vectors.ndim != 2 or not np.isfinite(vectors).all():
                        raise ValueError('invalid embedding matrix')
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    if not np.isfinite(norms).all() or np.any(norms <= 0):
                        raise ValueError('invalid embedding norm')
                    ordered[index] = vectors / norms
                dimensions = {vectors.shape[1] for vectors in ordered}
                if len(dimensions) != 1:
                    raise ValueError('inconsistent embedding dimensions')
                dimension = dimensions.pop()
                with self.state_lock:
                    expected_dimension = self.configured_dimensions if self.configured_dimensions is not None else self.dimensions
                    if expected_dimension is not None and expected_dimension != dimension:
                        raise ValueError('embedding dimensions changed')
                    self.dimensions = dimension
                    self.loaded = True
                return ordered
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise ValueError('malformed remote embedding response') from exc
        finally:
            self.slots.release()
