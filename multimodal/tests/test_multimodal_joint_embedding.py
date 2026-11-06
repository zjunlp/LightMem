"""Local optional backend input fidelity, resource limits, and API routing."""
import base64
import io

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

import service.multimodal as mm
from service.multimodal_joint_embedding import LocalJointEmbeddingReranker


def image_part(color='blue'):
    output = io.BytesIO()
    Image.new('RGB', (2, 2), color).save(output, format='PNG')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
            base64.b64encode(output.getvalue()).decode()}}


@pytest.fixture
def model_path(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    (scripts / 'qwen3_vl_embedding.py').write_text('# synthetic test; never imported\n')
    return tmp_path


def test_exact_metadata_joint_input_and_cache(model_path, monkeypatch):
    seen = []
    class Metadata:
        def text_from_bytes(self, raw):
            assert raw.startswith(b'\x89PNG')
            return 'capture metadata'
    model = LocalJointEmbeddingReranker(model_path, capture_metadata=Metadata())
    def infer(item):
        seen.append(item)
        return np.array([1., 0.], dtype=np.float32)
    monkeypatch.setattr(model, '_infer', infer)
    query = [{'type': 'text', 'text': 'original query'}]
    docs = [[image_part()]]
    assert model.score(query, docs, mm.decode_image) == [1.]
    assert seen[0] == {'text': 'original query'}
    assert seen[1]['text'] == 'capture metadata'
    assert seen[1]['image'].getpixel((0, 0)) == (0, 0, 255)
    assert model.score(query, docs, mm.decode_image) == [1.]
    assert len(seen) == 2


@pytest.mark.parametrize('document', [[], [{'type': 'text', 'text': 'original'}],
    [image_part(), image_part()], [image_part(), {'type': 'text', 'text': 'extra original text'}]])
def test_unsupported_content_rejected_before_inference(model_path, monkeypatch, document):
    model = LocalJointEmbeddingReranker(model_path)
    monkeypatch.setattr(model, '_infer', lambda _: pytest.fail('unsupported input reached model'))
    with pytest.raises(ValueError):
        model.score([{'type': 'text', 'text': 'q'}], [document], mm.decode_image)


def test_busy_and_bad_vectors_release_lock(model_path, monkeypatch):
    model = LocalJointEmbeddingReranker(model_path)
    with model.lock:
        with pytest.raises(RuntimeError, match='busy'):
            model.score([], [[image_part()]], mm.decode_image)
    monkeypatch.setattr(model, '_infer', lambda _: np.array([np.nan, 0.]))
    with pytest.raises(ValueError, match='vector'):
        model.score([{'type': 'text', 'text': 'q'}], [[image_part()]], mm.decode_image)
    assert not model.lock.locked()


def test_cache_is_bounded_and_model_dimensions_cannot_change(model_path, monkeypatch):
    model = LocalJointEmbeddingReranker(model_path)
    monkeypatch.setattr(model, '_infer', lambda _: np.array([1., 0.]))
    for i in range(4097):
        model._vector({'text': str(i)})
    assert len(model.cache) == 4096 and (None, '0') not in model.cache
    monkeypatch.setattr(model, '_infer', lambda _: np.array([1., 0., 0.]))
    with pytest.raises(ValueError, match='vector'):
        model._vector({'text': 'different dimensions'})


def test_backend_selected_through_actual_api(model_path, tmp_path, monkeypatch):
    monkeypatch.setenv('MULTIMODAL_RERANKER_MODEL', str(model_path))
    monkeypatch.setenv('MULTIMODAL_RERANKER_BACKEND', 'joint_embedding')
    monkeypatch.setenv('MULTIMODAL_RERANKER_MODE', 'consensus')
    monkeypatch.setattr(mm, 'bm25_scores', lambda docs, text: np.array([3., 2., 1.]))
    def infer(self, item):
        if 'image' not in item:
            return np.array([1., 0.])
        r, g, b = item['image'].getpixel((0, 0))
        return np.array([1., 0.]) if r else np.array([0., 1.]) if b else np.array([-1., 0.])
    monkeypatch.setattr(LocalJointEmbeddingReranker, '_infer', infer)
    parts = [image_part(c) for c in ['blue', 'green', 'red']]
    headers = {'Authorization': 'Bearer synthetic'}
    with TestClient(mm.create_app(str(tmp_path / 'db'), api_key='synthetic')) as client:
        response = client.post('/v1/memories/add', headers=headers, json={
            'user_id': 'u', 'session_id': 's', 'request_id': 'r', 'messages': [
                {'role': 'user', 'content': [part]} for part in parts]})
        assert response.status_code == 200
        response = client.post('/v1/memories/search', headers=headers,
                               json={'user_id': 'u', 'query': 'q', 'top_k': 3})
        assert response.status_code == 200
        assert [item['content'] for item in response.json()['data']] == [[parts[i]] for i in [0, 2, 1]]
        assert client.get('/health').json()['reranker_error'] is None


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), 301])
def test_invalid_configuration(model_path, timeout):
    with pytest.raises(ValueError):
        LocalJointEmbeddingReranker(model_path, timeout=timeout)


def test_budget_exhaustion_never_returns_partial_scores(model_path, monkeypatch):
    import service.multimodal_joint_embedding as module
    now = [0.]
    model = LocalJointEmbeddingReranker(model_path, timeout=1)
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    def infer(item):
        now[0] += 2
        return np.array([1., 0.])
    monkeypatch.setattr(model, '_infer', infer)
    with pytest.raises(TimeoutError, match='budget'):
        model.score([{'type': 'text', 'text': 'q'}], [[image_part()]], mm.decode_image)
    assert not model.lock.locked()
