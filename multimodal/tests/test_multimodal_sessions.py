import numpy as np
import pytest

from service.multimodal_sessions import dense_session_tail
from service.multimodal import MemoryService


def test_expansion_protects_prefix_and_selects_distinct_uncovered_sessions():
    sessions = ['covered'] * 100 + ['new', 'new', 'other', '', None, 'third']
    scores = np.arange(len(sessions), dtype=float)
    result = dense_session_tail(list(range(len(sessions))), scores, sessions)
    assert result[:20] == list(range(20))
    assert result[20:23] == [105, 102, 101]
    assert len(result) == len(set(result)) == len(sessions)
    assert result[23:100] == list(range(20, 97))


def test_short_gallery_and_empty_sessions_do_not_invent_evidence():
    assert dense_session_tail([1, 0], [1, 1], ['', None]) == [1, 0]
    assert dense_session_tail([], [], []) == []


@pytest.mark.parametrize('scores,order', [([float('nan')], [0]), ([float('inf')], [0]),
                                         ([1, 2], [0]), ([1], [1]), ([1, 2], [0, 0])])
def test_invalid_inputs_rejected(scores, order):
    with pytest.raises(ValueError):
        dense_session_tail(order, scores, ['s'] * len(scores))


@pytest.mark.parametrize('options', [{}, {'remote_config': {}, 'lexical_weight': 0},
                                    {'clip_model': 'dummy', 'lexical_weight': 1},
                                    {'clip_model': 'dummy', 'lexical_weight': 0, 'context_weight': .1},
                                    {'clip_model': 'dummy', 'lexical_weight': 0, 'rerank_mode': 'direct'}])
def test_expansion_requires_explicit_compatible_configuration(tmp_path, options):
    with pytest.raises(ValueError, match='dense session expansion'):
        MemoryService(str(tmp_path / 'db'), dense_session_expansion=True, **options)


def test_http_scope_disabled_parity_and_stop_first_overflow(tmp_path, monkeypatch):
    import base64
    import io
    import json
    import httpx
    from PIL import Image
    from fastapi.testclient import TestClient
    import service.multimodal as mm

    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={'data': [dict(index=i, embedding=[1, 0])
                                                 for i in range(len(body['input']))]})
    cfg = dict(base_url='https://synthetic.invalid/v1', model='synthetic', api_key='synthetic',
               client=httpx.Client(transport=httpx.MockTransport(handler)))
    service = mm.MemoryService(str(tmp_path/'db'), remote_config=cfg, lexical_weight=0,
                               rerank_mode='session_diverse', dense_session_expansion=True)
    monkeypatch.setattr(mm, 'score_vectors', lambda query, vectors, *args, **kwargs:
                        np.arange(len(vectors), 0, -1, dtype=float))
    monkeypatch.setattr(mm, 'MemoryService', lambda *a, **kw: service)
    stream = io.BytesIO()
    Image.new('RGB', (8, 8)).save(stream, format='PNG')
    part = dict(type='image_url', image_url=dict(url='data:image/png;base64,' +
                                                base64.b64encode(stream.getvalue()).decode()))
    headers = {'Authorization': 'Bearer secret'}
    with TestClient(mm.create_app(str(tmp_path/'db'), api_key='secret')) as client:
        for user in ('u', 'other'):
            for i in range(105):
                message = dict(role='user', content=[part if i in (0, 100) else
                                                     dict(type='text', text=f'item {i}')])
                response = client.post('/add', headers=headers, json=dict(
                    user_id=user, request_id=str(i), session_id='common' if i < 100 else f's{i}',
                    messages=[message]))
                assert response.status_code == 200
        with service.connect() as db:
            ids = [r[0] for r in db.execute('SELECT id FROM memories WHERE user_id=? ORDER BY sequence', ('u',))]
        query = dict(user_id='u', query='query', top_k=100)
        result = client.post('/search', headers=headers, json=query)
        assert result.status_code == 200
        assert [r['id'] for r in result.json()['data']] == ids[:20] + ids[100:] + ids[20:95]
        with monkeypatch.context() as limited:
            limited.setattr(mm, 'MAX_MEDIA_BYTES', mm.image_bytes(part))
            result = client.post('/search', headers=headers, json=query)
            assert [r['id'] for r in result.json()['data']] == ids[:20]
        service.dense_session_expansion = False
        baseline = client.post('/search', headers=headers, json=query)
        assert [r['id'] for r in baseline.json()['data']] == ids[:100]
