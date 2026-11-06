import base64
import io
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service.multimodal import create_app, MemoryService, AddRequest


def image_uri(color='red'):
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8), color).save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def payload(request='r1', user='alice'):
    return dict(request_id=request, user_id=user, session_id='s1', messages=[{
        'role': 'user', 'content': [
            {'type': 'text', 'text': 'A red bicycle at the lake'},
            {'type': 'image_url', 'image_url': {'url': image_uri()}},
            {'type': 'text', 'text': 'Purchased in spring'},
        ]}])


HEADERS = {'Authorization': 'Bearer secret'}


def test_text_embedding_v4_mode_skips_images_but_preserves_original_evidence(tmp_path):
    import httpx
    import json
    from fastapi import HTTPException
    from service.multimodal import SearchRequest
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert all(isinstance(value, str) for value in body['input'])
        assert body['dimensions'] == 1024
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [1, 0] + [0] * 1022} for i in range(len(body['input']))]})

    config = dict(base_url='https://embedding.example/v1', model='text-embedding-v4',
        api_key='synthetic', input_mode='text_only', dimensions=1024,
        client=httpx.Client(transport=httpx.MockTransport(handler)))
    service = MemoryService(str(tmp_path / 'text-embedding-v4.db'), remote_config=config,
                            lexical_weight=0)
    original = payload()
    service.add(AddRequest.model_validate(original))
    assert calls[0]['input'] == ['A red bicycle at the lake', 'Purchased in spring']
    result = service.search(SearchRequest(user_id='alice', query='bicycle lake', top_k=1))
    assert len(result['data']) == 1
    assert any(part['type'] == 'image_url' for part in result['data'][0]['content'])
    with pytest.raises(HTTPException, match='image-only queries') as error:
        service.search(SearchRequest(user_id='alice', query=[{
            'type': 'image_url', 'image_url': {'url': image_uri()}}], top_k=1))
    assert error.value.status_code == 503
    service.close()


def test_text_only_embedding_does_not_return_unscorable_image_only_records(tmp_path):
    import httpx
    import json
    from service.multimodal import SearchRequest

    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [1, 0]} for i in range(len(body['input']))]})

    config = dict(base_url='https://embedding.example/v1', model='text-embedding-v4',
        api_key='synthetic', input_mode='text_only', client=httpx.Client(
            transport=httpx.MockTransport(handler)))
    service = MemoryService(str(tmp_path / 'text-only-gallery.db'), remote_config=config,
                            lexical_weight=0)
    service.add(AddRequest.model_validate(dict(request_id='image-only', user_id='alice',
        session_id='gallery', messages=[{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': image_uri()}}]}])))
    service.add(AddRequest.model_validate(dict(request_id='text', user_id='alice',
        session_id='gallery', messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': 'A note about a bicycle'}]}])))
    result = service.search(SearchRequest(user_id='alice', query='bicycle', top_k=10))
    assert len(result['data']) == 1
    assert result['data'][0]['content'] == [{'type': 'text', 'text': 'A note about a bicycle'}]
    service.close()


def test_prompt_roles_cover_add_query_cache_and_lazy_reencoding(tmp_path):
    import httpx
    import json
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={'data': [
            {'index': i, 'embeddings': [[1, 0]]} for i in range(len(body['input']))]})
    def config(passage):
        return dict(base_url='https://embedding.example/v1', model='model', api_key='synthetic',
                    query_prompt_name='query', passage_prompt_name=passage, response_format='multivector',
                    client=httpx.Client(transport=httpx.MockTransport(handler)))
    path = str(tmp_path / 'db')
    service = MemoryService(path, remote_config=config('passage'), lexical_weight=0)
    service.add(AddRequest.model_validate(dict(user_id='u', session_id='s', request_id='r',
        messages=[dict(role='user', content=[dict(type='text', text='same text')])])))
    assert [b['prompt_name'] for b in calls] == ['passage']
    from service.multimodal import SearchRequest
    query = SearchRequest(user_id='u', query='same text', top_k=1)
    assert len(service.search(query)['data']) == 1
    service.search(query)
    assert [b['prompt_name'] for b in calls] == ['passage', 'query']
    service.close()
    # A changed document profile must not reuse vectors from the prior profile.
    service = MemoryService(path, remote_config=config('new-passage'), lexical_weight=0)
    assert len(service.search(query)['data']) == 1
    assert [b['prompt_name'] for b in calls][-2:] == ['query', 'new-passage']
    service.close()


def test_ordered_original_evidence_restart_isolation_and_auth(tmp_path):
    path = str(tmp_path / 'memory.db')
    data = payload()
    client = TestClient(create_app(path, api_key='secret'))
    assert client.get('/health').json()['image_queries'] is False
    assert client.post('/add', json=data).status_code == 401
    assert client.post('/v1/memories/add', json=data, headers=HEADERS).status_code == 200
    assert client.post('/add', json=data, headers=HEADERS).status_code == 200
    restarted = TestClient(create_app(path, api_key='secret'))
    query = dict(user_id='alice', query=[{'type': 'text', 'text': 'bicycle lake'}], top_k=100)
    result = restarted.post('/v1/memories/search', json=query, headers=HEADERS)
    assert result.status_code == 200
    assert len(result.json()['data']) == 1
    assert result.json()['data'][0]['content'] == data['messages'][0]['content']
    query['user_id'] = 'bob'
    assert restarted.post('/search', json=query, headers=HEADERS).json() == {'data': []}
    assert restarted.post('/add', json=payload(user='bob'), headers=HEADERS).status_code == 200
    data['session_id'] = 'different'
    assert restarted.post('/add', json=data, headers=HEADERS).status_code == 409


def test_images_validation_and_truthful_unavailable_query(tmp_path):
    client = TestClient(create_app(str(tmp_path / 'db'), api_key='secret'))
    for value in ['https://example.org/image.png', 'data:image/png;base64,bm90LWltYWdl',
                  image_uri().replace('image/png', 'image/jpeg')]:
        data = payload()
        data['messages'][0]['content'][1]['image_url'] = value
        assert client.post('/add', json=data, headers=HEADERS).status_code == 422
    result = client.post('/search', json=dict(user_id='alice', query=[{
        'type': 'image_url', 'image_url': {'url': image_uri()}}], top_k=10), headers=HEADERS)
    assert result.status_code == 503
    data = payload()
    data['messages'][0]['content'][1]['image_url'] = {'url': image_uri()}
    assert client.post('/add', json=data, headers=HEADERS).status_code == 200


def test_atomic_retries_and_collision(tmp_path):
    path = str(tmp_path / 'db')
    service = MemoryService(path)
    request = AddRequest.model_validate(payload())
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(r['success'] for r in pool.map(service.add, [request] * 16))
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM memories').fetchone()[0] == 1


def test_lexical_rank_and_body_limit(tmp_path, monkeypatch):
    client = TestClient(create_app(str(tmp_path / 'db'), api_key='secret'))
    assert client.post('/add', json=payload(), headers=HEADERS).status_code == 200
    other = payload(request='r2')
    other['messages'][0]['content'] = [{'type': 'text', 'text': 'Blue ocean waves'}]
    assert client.post('/add', json=other, headers=HEADERS).status_code == 200
    result = client.post('/search', json=dict(user_id='alice', query='ocean waves', top_k=1), headers=HEADERS)
    assert result.json()['data'][0]['content'] == other['messages'][0]['content']
    monkeypatch.setattr('service.multimodal.MAX_BODY_BYTES', 20)
    assert client.post('/add', json=payload(), headers=HEADERS).status_code == 413


def test_answer_evidence_preserves_source_and_binds_adjacent_turns(tmp_path):
    from service.multimodal import SearchRequest, text_of
    service = MemoryService(str(tmp_path / 'db'), answer_evidence=True)
    data = dict(request_id='context', user_id='alice', session_id='trip', messages=[
        {'role': 'user', 'timestamp': 1710000000000,
         'content': 'Where did I leave the red bicycle?'},
        {'role': 'assistant', 'timestamp': 1710000001000,
         'content': 'You left it at the lakeside station.'},
        {'role': 'user', 'timestamp': 1710000002000,
         'content': 'Please remind me of that location tomorrow.'},
    ])
    service.add(AddRequest.model_validate(data))
    result = service.search(SearchRequest(user_id='alice', query='red bicycle location', top_k=1))
    item = result['data'][0]
    assert item['id']
    text = text_of(item['content'])
    assert 'role=user; session=trip; time=1710000000000' in text
    assert 'role=assistant; session=trip; time=1710000001000' in text
    assert 'lakeside station' in text


def test_answer_evidence_keeps_image_parts_for_evaluator_qa(tmp_path):
    from service.multimodal import SearchRequest
    service = MemoryService(str(tmp_path / 'db'), answer_evidence=True)
    data = payload()
    service.add(AddRequest.model_validate(data))
    result = service.search(SearchRequest(user_id='alice', query='red bicycle', top_k=1))
    content = result['data'][0]['content']
    assert isinstance(content, list)
    assert any(part['type'] == 'image_url' for part in content)
    assert any(part['type'] == 'text' and 'role=user; session=s1' in part['text'] for part in content)


def test_answer_evidence_counts_adjacent_images_in_response_budget(tmp_path, monkeypatch):
    import pytest
    from fastapi import HTTPException
    from service.multimodal import SearchRequest
    service = MemoryService(str(tmp_path / 'db'), answer_evidence=True)
    data = payload()
    data['messages'].append(dict(role='assistant', content='The location is Lakeside Station.'))
    service.add(AddRequest.model_validate(data))
    monkeypatch.setattr('service.multimodal.MAX_MEDIA_BYTES', 1)
    monkeypatch.setattr('service.multimodal.image_bytes',
                        lambda part: 2 if part['type'] == 'image_url' else 0)
    with pytest.raises(HTTPException) as error:
        service.search(SearchRequest(user_id='alice', query='Lakeside Station location', top_k=1))
    assert error.value.status_code == 413


def test_answer_evidence_does_not_expand_across_sessions_or_users(tmp_path):
    from service.multimodal import SearchRequest, text_of
    service = MemoryService(str(tmp_path / 'db'), answer_evidence=True)
    for user, session, request, content in [
        ('alice', 'private', 'a', 'Private project detail.'),
        ('bob', 'shared', 'b', 'Another user secret.'),
        ('alice', 'shared', 'c', 'The blue bicycle is at the lake.'),
    ]:
        service.add(AddRequest.model_validate(dict(user_id=user, session_id=session,
            request_id=request, messages=[dict(role='user', content=content)])))
    result = service.search(SearchRequest(user_id='alice', query='blue bicycle lake', top_k=1))
    text = text_of(result['data'][0]['content'])
    assert 'blue bicycle' in text
    assert 'Private project' not in text
    assert 'Another user' not in text


def test_answer_context_top_k_expands_and_deduplicates_only_head_hits(tmp_path, monkeypatch):
    from service.multimodal import SearchRequest, text_of
    service = MemoryService(str(tmp_path / 'db'), answer_context_top_k=1)
    service.add(AddRequest.model_validate(dict(
        request_id='bounded-context', user_id='alice', session_id='same-session', messages=[
            {'role': 'user', 'timestamp': 1000, 'content': 'First source detail about the trip.'},
            {'role': 'assistant', 'timestamp': 2000, 'content': 'Second source detail gives the station name.'},
            {'role': 'user', 'timestamp': 3000, 'content': 'Third source detail gives the departure time.'},
        ],
    )))
    monkeypatch.setattr('service.multimodal.bm25_scores',
                        lambda documents, query: np.array([10, 9, 8], dtype=np.float32))
    result = service.search(SearchRequest(user_id='alice', query='trip station time', top_k=3))
    assert len(result['data']) == 3
    first, second, third = (text_of(item['content']) for item in result['data'])
    assert 'First source detail' in first
    assert 'Second source detail' in first
    assert 'Third source detail' not in first
    assert 'Source content already included with Memory 1' in second
    assert 'Third source detail' in third
    assert 'First source detail' not in third
    assert all(item['content'] for item in result['data'])


def test_answer_context_top_k_rejects_invalid_limits(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        MemoryService(str(tmp_path / 'bad-float'), answer_context_top_k=1.0)
    with pytest.raises(ValueError):
        MemoryService(str(tmp_path / 'bad-negative'), answer_context_top_k=-1)


def test_update_evidence_bridges_only_a_short_selected_session_gap(tmp_path, monkeypatch):
    from service.multimodal import SearchRequest, text_of
    service = MemoryService(str(tmp_path / 'db'), answer_update_bridge=True)
    service.add(AddRequest.model_validate(dict(
        request_id='update', user_id='alice', session_id='plan', messages=[
            {'role': 'user', 'content': 'We discussed a weekly study plan.'},
            {'role': 'assistant', 'content': 'The first proposal used two weeknights.'},
            {'role': 'user', 'content': 'This is an intermediate scheduling detail.'},
            {'role': 'assistant', 'content': 'The final commitment keeps weekends for family.'},
            {'role': 'user', 'content': 'Please remember the final decision.'},
        ],
    )))
    monkeypatch.setattr('service.multimodal.bm25_scores',
                        lambda documents, query: np.array([10, 9, 2, 1, 8], dtype=np.float32))
    result = service.search(SearchRequest(user_id='alice', query='What was the final commitment?', top_k=3))
    assert len(result['data']) == 3
    evidence = text_of(result['data'][-1]['content'])
    assert 'supplemental same-session evidence' in evidence
    assert 'intermediate scheduling detail' in evidence
    assert 'final commitment keeps weekends for family' in evidence
    plain = service.search(SearchRequest(user_id='alice', query='What did we discuss?', top_k=3))
    assert 'supplemental same-session evidence' not in '\n'.join(text_of(item['content']) for item in plain['data'])


def test_next_turn_evidence_keeps_cardinality_and_never_crosses_a_session(tmp_path, monkeypatch):
    from service.multimodal import SearchRequest, text_of
    service = MemoryService(str(tmp_path / 'db'), answer_next_turn=True)
    service.add(AddRequest.model_validate(dict(
        request_id='first', user_id='alice', session_id='one', messages=[
            {'role': 'assistant', 'content': 'When did you begin the plan?'},
            {'role': 'user', 'content': 'I began at the desk after turning on the lamp.'},
        ],
    )))
    service.add(AddRequest.model_validate(dict(
        request_id='second', user_id='alice', session_id='two', messages=[
            {'role': 'user', 'content': 'Other session secret.'},
        ],
    )))
    monkeypatch.setattr('service.multimodal.bm25_scores',
                        lambda documents, query: np.array([10, 1, 9], dtype=np.float32))
    result = service.search(SearchRequest(user_id='alice', query='When did I begin?', top_k=2))
    assert len(result['data']) == 2
    first, second = (text_of(item['content']) for item in result['data'])
    assert 'next same-session turn' in first
    assert 'desk after turning on the lamp' in first
    assert 'Other session secret' not in first
    assert 'next same-session turn' not in second


def test_image_encoder_rank_and_lexical_record_upgrade(tmp_path):
    from service.multimodal import SearchRequest
    service = MemoryService(str(tmp_path / 'db'))
    first = payload()
    first['messages'][0]['content'] = [{'type': 'image_url', 'image_url': {'url': image_uri('red')}}]
    second = payload(request='r2')
    second['messages'][0]['content'] = [{'type': 'image_url', 'image_url': {'url': image_uri('blue')}}]
    service.add(AddRequest.model_validate(first))
    service.add(AddRequest.model_validate(second))

    class FakeEncoder:
        def __init__(self):
            self.calls = 0

        def encode(self, parts):
            self.calls += 1
            from service.multimodal import decode_image
            pixel = decode_image(parts[0]['image_url']).getpixel((0, 0))
            return np.array([pixel[0] / 255, pixel[2] / 255], dtype=np.float32)

    service.encoder = FakeEncoder()
    service.signature = 'test-encoder'
    query = SearchRequest(user_id='alice', query=second['messages'][0]['content'], top_k=1)
    result = service.search(query)
    assert result['data'][0]['content'] == second['messages'][0]['content']
    assert service.encoder.calls == 3
    assert service.search(query) == result
    assert service.encoder.calls == 3


def test_missing_key_fail_closed(tmp_path):
    client = TestClient(create_app(str(tmp_path / 'db'), api_key=''))
    assert client.post('/add', json=payload(), headers=HEADERS).status_code == 503


def test_aggregate_decoded_media_limits_and_provenance(tmp_path, monkeypatch):
    import service.multimodal as mm
    path = str(tmp_path / 'db')
    service = MemoryService(path)
    data = payload()
    data['messages'][0]['timestamp'] = '2026-10-22T12:00:00Z'
    service.add(AddRequest.model_validate(data))
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT role,session_id,source_timestamp FROM memories').fetchone() == (
            'user', 's1', '"2026-10-22T12:00:00Z"')
    monkeypatch.setattr(mm, 'MAX_MEDIA_BYTES', 1)
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        service.add(AddRequest.model_validate(payload(request='r2')))
    assert error.value.status_code == 413
    with pytest.raises(HTTPException) as error:
        service.search(mm.SearchRequest(user_id='alice', query='red', top_k=10))
    assert error.value.status_code == 413


def test_configured_encoder_failure_has_no_silent_fallback(tmp_path):
    from fastapi import HTTPException
    import pytest
    service = MemoryService(str(tmp_path / 'db'))

    class BrokenEncoder:
        def encode(self, parts):
            raise RuntimeError('failed model')

    service.encoder = BrokenEncoder()
    with pytest.raises(HTTPException) as error:
        service.add(AddRequest.model_validate(payload()))
    assert error.value.status_code == 503
    assert service.last_error == 'RuntimeError'
    with service.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM memories').fetchone()[0] == 0


def test_late_interaction_preserves_exact_image_despite_text_distractors(tmp_path):
    from service.multimodal import SearchRequest, decode_image
    service = MemoryService(str(tmp_path / 'db'))

    class PartsEncoder:
        def encode(self, parts):
            vectors = []
            for part in parts:
                if part['type'] == 'text':
                    vectors.append([0., 1.])
                else:
                    red = decode_image(part['image_url']).getpixel((0, 0))[0] > 128
                    vectors.append([1., 0.] if red else [0.8, 0.6])
            return np.array(vectors, dtype=np.float32)

    service.encoder = PartsEncoder()
    service.signature = 'clip-parts-v2:fake'
    exact = payload()
    # Text vectors deliberately point away from the image vector.
    exact['messages'][0]['content'].append({'type': 'text', 'text': 'another distractor'})
    distractor = payload(request='r2')
    distractor['messages'][0]['content'] = [{'type': 'image_url', 'image_url': {'url': image_uri('blue')}}]
    service.add(AddRequest.model_validate(exact))
    service.add(AddRequest.model_validate(distractor))
    query = SearchRequest(user_id='alice', query=[{'type': 'image_url', 'image_url': {'url': image_uri()}}], top_k=2)
    result = service.search(query)
    assert result['data'][0]['content'] == exact['messages'][0]['content']
    # A second query verifies persisted per-part matrices retain their dimensions.
    assert service.search(query) == result


def test_remote_add_batches_messages_then_preserves_part_boundaries(tmp_path):
    import json
    import httpx
    from service.multimodal import SearchRequest
    service = MemoryService(str(tmp_path / 'db'), remote_config=dict(
        base_url='https://embedding.example/v1', model='test', api_key='private'))
    calls = []
    def handler(request):
        inputs = json.loads(request.content)['input']
        calls.append(inputs)
        return httpx.Response(200, json={'data': [{'index': i, 'embedding': [1, i + 1]} for i in range(len(inputs))]})
    service.encoder.client = httpx.Client(transport=httpx.MockTransport(handler))
    data = payload()
    data['messages'].append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'Second message'}]})
    service.add(AddRequest.model_validate(data))
    assert len(calls) == 1
    assert len(calls[0]) == 4
    with service.connect() as db:
        vectors = [np.load(io.BytesIO(row[0]), allow_pickle=False) for row in db.execute('SELECT vector FROM memories ORDER BY sequence')]
    assert [v.shape for v in vectors] == [(3, 2), (1, 2)]
    result = service.search(SearchRequest(user_id='alice', query='Second message', top_k=2))
    assert len(result['data']) == 2


def test_multivector_persistence_boundaries_order_and_maxsim(tmp_path):
    import json
    import httpx
    from service.multimodal import SearchRequest, load_content
    from service.multimodal_retrieval import score_vectors
    path = str(tmp_path / 'db')
    config = dict(base_url='https://embedding.example/v1', model='variable', api_key='private', response_format='multivector')
    service = MemoryService(path, remote_config=config, lexical_weight=0)
    def handler(request):
        values = json.loads(request.content)['input']
        rows = []
        for i, part in enumerate(values):
            matrix = [[1, 0], [0, 1], [1, 1]] if 'image' in part else [[1, 0], [1, 1]]
            rows.append({'index': i, 'embeddings': matrix})
        return httpx.Response(200, json={'data': rows[::-1]})
    service.encoder.client = httpx.Client(transport=httpx.MockTransport(handler))
    data = payload()
    data['messages'].append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'Second message'}]})
    service.add(AddRequest.model_validate(data))
    with service.connect() as db:
        records = db.execute('SELECT vector,content FROM memories ORDER BY sequence').fetchall()
    stored = [load_content(blob, json.loads(content)) for blob, content in records]
    assert [matrix.shape for matrix, types in stored] == [(7, 2), (2, 2)]
    assert stored[0][1] == ['text', 'text', 'image_url', 'image_url', 'image_url', 'text', 'text']
    queryparts = [{'type': 'text', 'text': 'mixed'}, data['messages'][0]['content'][1]]
    query = SearchRequest(user_id='alice', query=queryparts, top_k=2)
    result = service.search(query)
    querymatrix, querytypes = service.encode_content(queryparts)
    expected = score_vectors(querymatrix, [v[0] for v in stored], [v[1] for v in stored], querytypes)
    assert [item['score'] for item in result['data']] == sorted(expected, reverse=True)
    assert result['data'][0]['content'] == data['messages'][0]['content']
    restarted = MemoryService(path, remote_config=config, lexical_weight=0)
    restarted.encoder.client = httpx.Client(transport=httpx.MockTransport(handler))
    assert restarted.search(query) == result


def test_preprocessed_embedding_keeps_persisted_and_returned_original_image(tmp_path):
    import json
    import httpx
    from service.multimodal import SearchRequest
    data = payload()
    source = data['messages'][0]['content']
    service = MemoryService(str(tmp_path / 'db'), remote_config=dict(
        base_url='https://embedding.example/v1', model='test', api_key='private', image_threshold_bytes=1))
    def handler(request):
        inputs = json.loads(request.content)['input']
        for part in inputs:
            if 'image' in part:
                assert part['image'].startswith('data:image/jpeg;base64,')
                assert part['image'] != source[1]['image_url']['url']
        return httpx.Response(200, json={'data': [{'index': i, 'embedding': [1, 0]} for i in range(len(inputs))]})
    service.encoder.client = httpx.Client(transport=httpx.MockTransport(handler))
    service.add(AddRequest.model_validate(data))
    result = service.search(SearchRequest(user_id='alice', query='bicycle', top_k=1))
    assert result['data'][0]['content'] == source
    assert data['messages'][0]['content'] == source
    with service.connect() as db:
        assert json.loads(db.execute('SELECT content FROM memories').fetchone()[0]) == source


def test_app_lifespan_closes_lazy_encoder(tmp_path, monkeypatch):
    import httpx
    from service.multimodal_embedding import RemoteEncoder
    instances = []
    original = RemoteEncoder.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.client.close()
        self.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            'data': [{'index': i, 'embedding': [1, 0]} for i in range(3)]})))
        instances.append(self)
    monkeypatch.setattr(RemoteEncoder, '__init__', initialize)
    monkeypatch.setenv('MULTIMODAL_EMBEDDING_BASE_URL', 'https://embedding.example/v1')
    monkeypatch.setenv('MULTIMODAL_EMBEDDING_API_KEY', 'private')
    monkeypatch.setenv('MULTIMODAL_EMBEDDING_INPUT_MODE', 'multimodal_parts')
    monkeypatch.setenv('MULTIMODAL_EMBEDDING_DIMENSIONS', '2')
    with TestClient(create_app(str(tmp_path / 'db'), api_key='secret')) as client:
        assert not instances
        assert client.post('/add', json=payload(), headers=HEADERS).status_code == 200
    assert len(instances) == 1 and instances[0].client.is_closed


def test_query_cache_does_not_cache_results_or_hide_new_additions(tmp_path):
    from service.multimodal import SearchRequest
    service = MemoryService(str(tmp_path / 'db'), lexical_weight=0)
    class CountingEncoder:
        def __init__(self):
            self.calls = 0
        def encode(self, parts):
            self.calls += 1
            return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(parts), 1))
    service.encoder = CountingEncoder()
    service.signature = 'counting'
    service.add(AddRequest.model_validate(payload()))
    query = SearchRequest(user_id='alice', query='same query', top_k=10)
    first = service.search(query)
    assert len(first['data']) == 1
    calls = service.encoder.calls
    assert service.search(query) == first
    assert service.encoder.calls == calls
    service.add(AddRequest.model_validate(payload(request='new-request')))
    calls = service.encoder.calls
    assert len(service.search(query)['data']) == 2
    assert service.encoder.calls == calls
    assert service.query_cache.entries
    service.close()
    assert not service.query_cache.entries and service.query_cache.bytes == 0


def test_context_configuration_and_search_original_evidence(tmp_path):
    import pytest
    from service.multimodal import SearchRequest
    for options in [dict(context_weight=-.1), dict(context_weight=1.1),
                    dict(context_weight=.25, clip_model='unused'),
                    dict(context_weight=.25, clip_model='unused', lexical_weight=0, scoring='modality_rrf'),
                    dict(context_weight=.25, lexical_weight=0)]:
        with pytest.raises(ValueError):
            MemoryService(str(tmp_path / 'bad'), **options)
    service = MemoryService(str(tmp_path / 'db'), clip_model='unused', lexical_weight=0, context_weight=.25)
    class Encoder:
        def encode(self, parts):
            return np.array([[1, 0] if p['text'] == 'high' else [0, 1] for p in parts], dtype=np.float32)
    service.encoder = Encoder()
    data = dict(request_id='r', user_id='u', session_id='s', messages=[
        dict(role='user', content=[dict(type='text', text=value)]) for value in ['low','high','low']])
    service.add(AddRequest.model_validate(data))
    result = service.search(SearchRequest(user_id='u', query='high', top_k=3))
    assert [item['score'] for item in result['data']] == [1, .25, .25]
    assert result['data'][0]['content'] == data['messages'][1]['content']
    service.context_weight = 0
    baseline = service.search(SearchRequest(user_id='u', query='high', top_k=3))
    assert [item['score'] for item in baseline['data']] == [1, 0, 0]


def test_search_media_budget_keeps_complete_ranked_prefix(tmp_path, monkeypatch):
    import service.multimodal as mm
    from service.multimodal import SearchRequest
    service = MemoryService(str(tmp_path / 'db'), lexical_weight=0)
    documents = [
        [{'type': 'text', 'text': 'first'}, {'type': 'image_url', 'image_url': {'url': image_uri('red')}}],
        [{'type': 'text', 'text': 'second'}, {'type': 'image_url', 'image_url': {'url': image_uri('blue')}}],
        [{'type': 'text', 'text': 'small text tail'}],
    ]
    service.add(AddRequest.model_validate(dict(user_id='u', session_id='s', request_id='r',
        messages=[dict(role='user', content=parts) for parts in documents])))
    # With no encoder and zero lexical weight insertion order breaks score ties.
    # Each image costs 6 units; the text-only third record must not jump past second.
    monkeypatch.setattr(mm, 'image_bytes', lambda part: 6 if part['type'] == 'image_url' else 0)
    query = SearchRequest(user_id='u', query='query', top_k=100)
    monkeypatch.setattr(mm, 'MAX_MEDIA_BYTES', 6)
    result = service.search(query)
    assert len(result['data']) == 1
    assert result['data'][0]['content'] == documents[0]
    monkeypatch.setattr(mm, 'MAX_MEDIA_BYTES', 12)
    result = service.search(query)
    assert [item['content'] for item in result['data']] == documents
    assert len(result['data']) <= query.top_k
    monkeypatch.setattr(mm, 'MAX_MEDIA_BYTES', 5)
    from fastapi import HTTPException
    import pytest
    with pytest.raises(HTTPException) as error:
        service.search(query)
    assert error.value.status_code == 413
    with service.connect() as db:
        import json
        assert [json.loads(row[0]) for row in db.execute('SELECT content FROM memories ORDER BY sequence')] == documents


def test_npz_search_reads_only_selected_original_content_and_same_snapshot(tmp_path, monkeypatch):
    from service.multimodal import SearchRequest, serialize_content
    path = str(tmp_path / 'db')
    service = MemoryService(path, lexical_weight=0)
    class Encoder:
        def encode(self, parts):
            return np.tile([[1., 0.]], (len(parts), 1))
    service.encoder = Encoder()
    service.signature = 'snapshot-test'
    def request(identifier, value):
        return AddRequest.model_validate(dict(request_id=identifier, user_id='u', session_id='s',
            messages=[dict(role='user', content=[dict(type='text', text=value)])]))
    for i in range(3):
        service.add(request(str(i), str(i)))
    # Current multivector storage carries modality metadata without source reads.
    with service.connect() as db:
        db.execute('UPDATE memories SET vector=?', (serialize_content(np.array([[1., 0.]],dtype=np.float32), ['text'], True),))
    original_connect = service.connect
    seen = []
    class TracedConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            seen.append((sql, parameters))
            return super().execute(sql, parameters)
    monkeypatch.setattr(service, 'connect', lambda: sqlite3.connect(path, timeout=30, factory=TracedConnection))
    result = service.search(SearchRequest(user_id='u', query='query', top_k=1))
    source_reads = [(sql, parameters) for sql, parameters in seen if sql.startswith('SELECT content')]
    assert len(source_reads) == 1
    assert source_reads[0][1] == ('u', 1)
    assert result['data'][0]['content'] == [dict(type='text', text='0')]
    assert all('id,content' not in sql for sql, _ in seen)
    # Add commits after this Search establishes its read snapshot.
    inserted = [False]
    class ConcurrentConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            cursor = super().execute(sql, parameters)
            if sql.startswith('SELECT id,NULL') and not inserted[0]:
                inserted[0] = True
                monkeypatch.setattr(service, 'connect', original_connect)
                service.add(request('new', 'new'))
            return cursor
    monkeypatch.setattr(service, 'connect', lambda: sqlite3.connect(path, timeout=30, factory=ConcurrentConnection))
    first = service.search(SearchRequest(user_id='u', query='query', top_k=100))
    second = service.search(SearchRequest(user_id='u', query='query', top_k=100))
    assert len(first['data']) == 3
    assert len(second['data']) == 4
