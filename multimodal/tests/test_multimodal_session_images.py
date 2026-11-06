import numpy as np
import pytest
from service.multimodal_sessions import dense_session_image_tail
from service.multimodal import MemoryService


def test_support_head_session_images_ties_full_continuation():
    sessions = ['a'] * 10 + ['b'] * 10 + ['tail'] * 80 + ['a', 'a', 'b', 'tail', '', None]
    images = [False] * 100 + [True] * 6
    result = dense_session_image_tail(list(range(106)), [1.] * 106, sessions, images)
    assert result[:22] == list(range(20)) + [100, 102]
    assert result[22:] == list(range(20, 100)) + [101, 103, 104, 105]
    assert len(result) == len(set(result)) == 106
    images[0] = True
    result = dense_session_image_tail(list(range(106)), np.arange(106), sessions, images)
    assert result[:21] == list(range(20)) + [102]


def test_short_empty_and_blank_session():
    assert dense_session_image_tail([], [], [], []) == []
    assert dense_session_image_tail([1, 0], [1, 2], ['a', 'a'], [False, True]) == [1, 0]
    assert dense_session_image_tail(list(range(101)), [1.] * 101, [''] * 101,
                                    [False] * 100 + [True]) == list(range(101))


@pytest.mark.parametrize('scores,mask', [([float('nan')], [True]), ([1], []), ([1], [1])])
def test_malformed_support_rejected(scores, mask):
    with pytest.raises(ValueError):
        dense_session_image_tail([0], scores, ['a'], mask)


def test_support_requires_expansion(tmp_path):
    with pytest.raises(ValueError, match='session image support'):
        MemoryService(str(tmp_path/'db'), session_image_support=True)


def test_http_image_scope_budget_disabled_and_failure_fallback(tmp_path, monkeypatch):
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
                               rerank_mode='session_diverse', dense_session_expansion=True, session_image_support=True)
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
            for i in range(102):
                message = dict(role='user', content=[part if i >= 100 else
                                                     dict(type='text', text=f'item {i}')])
                response = client.post('/add', headers=headers, json=dict(
                    user_id=user, request_id=str(i), session_id='a' if i % 2 == 0 else 'b',
                    messages=[message]))
                assert response.status_code == 200
        with service.connect() as db:
            ids = [r[0] for r in db.execute('SELECT id FROM memories WHERE user_id=? ORDER BY sequence', ('u',))]
        query = dict(user_id='u', query='query', top_k=100)
        result = client.post('/search', headers=headers, json=query)
        assert result.status_code == 200
        assert [r['id'] for r in result.json()['data']] == ids[:20] + ids[100:] + ids[20:98]
        with monkeypatch.context() as limited:
            limited.setattr(mm, 'MAX_MEDIA_BYTES', mm.image_bytes(part))
            result = client.post('/search', headers=headers, json=query)
            assert [r['id'] for r in result.json()['data']] == ids[:20] + ids[100:101]
        with monkeypatch.context() as broken:
            def fail(*args):
                raise ValueError('synthetic failure')
            broken.setattr(mm, 'dense_session_image_tail', fail)
            result = client.post('/search', headers=headers, json=query)
            assert [r['id'] for r in result.json()['data']] == ids[:100]
        service.session_image_support = False
        baseline = client.post('/search', headers=headers, json=query)
        assert [r['id'] for r in baseline.json()['data']] == ids[:100]
