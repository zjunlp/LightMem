import json

import httpx
import numpy as np
import pytest

from service.multimodal_embedding import RemoteEncoder


PARTS = [{'type': 'text', 'text': 'hello'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]


def encoder(handler, **kwargs):
    return RemoteEncoder('https://embedding.example/v1', 'model', 'private',
                         client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_ordered_per_part_vectors_and_identity():
    def handler(request):
        body = json.loads(request.content)
        assert body['input'] == [{'text': 'hello'}, {'image': 'data:image/png;base64,AA=='}]
        assert body['encoding_format'] == 'float'
        return httpx.Response(200, json={'data': [{'index': 1, 'embedding': [0, 4]}, {'index': 0, 'embedding': [3, 0]}]})
    model = encoder(handler)
    np.testing.assert_equal(model.encode(PARTS), [[1, 0], [0, 1]])
    assert model.loaded
    assert 'private' not in model.signature
    assert model.signature == RemoteEncoder('https://embedding.example/v1', 'model', 'different').signature


def test_text_only_embedding_sends_plain_text_and_keeps_image_slot_unembedded():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [3] + [4] + [0] * 1022} for i, _ in enumerate(body['input'])]})

    model = encoder(handler, input_mode='text_only', dimensions=1024)
    vectors = model.encode_parts(PARTS)
    assert calls[0]['input'] == ['hello']
    assert calls[0]['dimensions'] == 1024
    assert vectors[0].shape == (1, 1024)
    assert vectors[1] is None
    np.testing.assert_allclose(model.encode(PARTS)[0, :2], [.6, .8])
    assert model.signature != encoder(handler).signature


def test_text_only_embedding_rejects_image_only_encoding():
    model = encoder(lambda _: httpx.Response(200, json={'data': []}), input_mode='text_only')
    assert model.encode_parts([PARTS[1]]) == [None]
    with pytest.raises(ValueError, match='no encodable parts'):
        model.encode([PARTS[1]])


def test_query_and_passage_prompts_are_explicit_and_batched():
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [1, 0]} for i in range(len(body['input']))]})
    model = encoder(handler, query_prompt_name='query', passage_prompt_name='passage', batch_size=2,
                    image_batch_size=1)
    model.encode_parts(PARTS * 3, purpose='query')
    assert len(calls) == 3 and all(b['prompt_name'] == 'query' for b in calls)
    calls.clear()
    model.encode_parts(PARTS)
    assert calls[0]['prompt_name'] == 'passage'
    calls.clear()
    default = encoder(handler)
    default.encode(PARTS)
    assert 'prompt_name' not in calls[0]
    assert model.signature != default.signature
    assert model.signature != encoder(handler, query_prompt_name='other', passage_prompt_name='passage').signature
    assert model.signature != encoder(handler, query_prompt_name='query', passage_prompt_name='other').signature
    with pytest.raises(ValueError):
        model.encode_parts(PARTS, purpose='invalid')


@pytest.mark.parametrize('prompt', ['', ' ', 1, 'x' * 129])
def test_invalid_prompt_configuration(prompt):
    with pytest.raises(ValueError):
        encoder(lambda _: httpx.Response(200), query_prompt_name=prompt)


def test_default_encoder_identity_preserves_existing_vector_compatibility():
    import hashlib
    identity = json.dumps(['https://embedding.example/v1', 'model',
                           'per-part-text-image-v2', 'singlevector'], separators=(',', ':'))
    assert encoder(lambda _: httpx.Response(200)).signature == 'remote-parts-v2:' + hashlib.sha256(identity.encode()).hexdigest()


@pytest.mark.parametrize('rows', [[], [{'index': 0, 'embedding': [1]}],
    [{'index': 0, 'embedding': [1, 0]}, {'index': 0, 'embedding': [1, 0]}],
    [{'index': 0, 'embedding': [0, 0]}, {'index': 1, 'embedding': [1, 0]}],
    [{'index': 0, 'embedding': [float('nan'), 0]}, {'index': 1, 'embedding': [1, 0]}],
    [{'index': 0, 'embedding': [1, 0]}, {'index': 1, 'embedding': [1]}],
    [{'index': True, 'embedding': [1, 0]}, {'index': 0, 'embedding': [1, 0]}]])
def test_malformed_remote_vectors_rejected(rows):
    def handler(request):
        # stdlib permits NaN so the adapter must explicitly reject it.
        return httpx.Response(200, content=json.dumps({'data': rows}))
    model = encoder(handler)
    with pytest.raises(ValueError, match='malformed'):
        model.encode(PARTS)
    assert not model.loaded


def test_retries_bounded_and_rejection_not_retried(monkeypatch):
    monkeypatch.setattr('service.multimodal_embedding.time.sleep', lambda _: None)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429)
    with pytest.raises(RuntimeError):
        encoder(handler).encode(PARTS)
    assert len(calls) == 3
    calls.clear()
    def reject(request):
        calls.append(request)
        return httpx.Response(401)
    with pytest.raises(RuntimeError):
        encoder(reject).encode(PARTS)
    assert len(calls) == 1


def test_batch_limits_and_dimension_changes():
    batches = []
    def handler(request):
        batch = json.loads(request.content)['input']
        batches.append(batch)
        return httpx.Response(200, json={'data': [{'index': i, 'embedding': [1, 0]} for i in range(len(batch))]})
    model = encoder(handler, batch_size=3, image_batch_size=1)
    vectors = model.encode(PARTS * 4)
    assert vectors.shape == (8, 2)
    assert all(len(batch) <= 3 and sum('image' in p for p in batch) <= 1 for batch in batches)
    model.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        'data': [{'index': 0, 'embedding': [1, 0, 0]}, {'index': 1, 'embedding': [1, 0, 0]}]})))
    with pytest.raises(ValueError, match='malformed'):
        model.encode(PARTS)


def test_multivector_variable_token_counts_reorder_and_format_identity():
    def handler(request):
        return httpx.Response(200, json={'data': [
            {'index': 1, 'embeddings': [[0, 2], [1, 1], [3, 0]]},
            {'index': 0, 'embeddings': [[2, 0], [0, 3]]}]})
    model = encoder(handler, response_format='multivector')
    vectors = model.encode_parts(PARTS)
    assert [v.shape for v in vectors] == [(2, 2), (3, 2)]
    np.testing.assert_allclose(vectors[0], [[1, 0], [0, 1]])
    assert model.signature != encoder(handler).signature
    assert model.encode(PARTS).shape == (5, 2)


@pytest.mark.parametrize('data', [
    [{'index': 0, 'embeddings': []}, {'index': 1, 'embeddings': [[1, 0]]}],
    [{'index': 0, 'embeddings': [[1, 0]]}, {'index': 1, 'embeddings': [[1, 0, 0]]}],
    [{'index': 0, 'embeddings': [[0, 0]]}, {'index': 1, 'embeddings': [[1, 0]]}],
    [{'index': 0, 'embeddings': [[True, 0]]}, {'index': 1, 'embeddings': [[1, 0]]}],
])
def test_multivector_invalid_token_matrices_rejected(data):
    model = encoder(lambda _: httpx.Response(200, json={'data': data}), response_format='multivector')
    with pytest.raises(ValueError):
        model.encode_parts(PARTS)


def test_embedding_image_preprocessing_matches_policy_and_does_not_mutate():
    import base64
    import io
    from PIL import Image, ImageOps
    from service.multimodal_embedding import embedding_image_url
    stream = io.BytesIO()
    image = Image.new('RGBA', (64, 32), (200, 30, 70, 128))
    image.save(stream, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
    assert embedding_image_url(url) == url
    assert embedding_image_url(url, len(stream.getvalue())) == url
    prepared = embedding_image_url(url, 1, 16, 95)
    expected = io.BytesIO()
    transformed = ImageOps.exif_transpose(image).convert('RGB')
    transformed.thumbnail((16, 16))
    transformed.save(expected, format='JPEG', quality=95)
    assert base64.b64decode(prepared.split(',')[1]) == expected.getvalue()
    original = [{'type': 'image_url', 'image_url': {'url': url}}]
    def handler(request):
        outgoing = json.loads(request.content)['input'][0]['image']
        assert outgoing == prepared and outgoing != url
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1, 0]}]})
    model = encoder(handler, image_threshold_bytes=1, image_max_edge=16)
    model.encode(original)
    assert original[0]['image_url']['url'] == url
    assert model.signature != encoder(handler).signature
    assert encoder(handler, image_max_edge=16).signature == encoder(handler).signature


@pytest.mark.parametrize('kwargs', [{'image_threshold_bytes': -1}, {'image_threshold_bytes': 11*1024*1024},
    {'image_max_edge': 0}, {'image_max_edge': 9000}, {'image_jpeg_quality': 101}, {'image_jpeg_quality': True}])
def test_image_preprocessing_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        encoder(lambda _: None, **kwargs)


def test_parallel_batches_overlap_preserving_part_order_and_close():
    import threading
    barrier = threading.Barrier(2, timeout=3)
    def handler(request):
        value = json.loads(request.content)['input'][0]['text']
        barrier.wait()
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [int(value), 1]}]})
    model = encoder(handler, concurrency=2, batch_size=1, image_batch_size=1)
    vectors = model.encode([{'type': 'text', 'text': str(i)} for i in [1, 2]])
    np.testing.assert_allclose(vectors, np.array([[1, 1], [2, 1]]) / np.linalg.norm([[1, 1], [2, 1]], axis=1, keepdims=True))
    model.close()
    model.close()
    assert model.client.is_closed
    with pytest.raises(RuntimeError, match='closed'):
        model.encode(PARTS)


def test_global_concurrency_bound_across_simultaneous_callers():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    active = maximum = 0
    lock = threading.Lock()
    def handler(request):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(.025)
        with lock:
            active -= 1
        values = json.loads(request.content)['input']
        return httpx.Response(200, json={'data': [{'index': i, 'embedding': [1, 1]} for i in range(len(values))]})
    model = encoder(handler, concurrency=2, batch_size=1, image_batch_size=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(model.encode, [PARTS * 3, PARTS * 3]))
    assert [r.shape for r in results] == [(6, 2), (6, 2)]
    assert maximum == 2
    model.close()
