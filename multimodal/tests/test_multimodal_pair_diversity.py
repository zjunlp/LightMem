"""Synthetic source-pair ordering and retrieval prefix contracts."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

import service.multimodal as mm
from service.multimodal_retrieval import image_first_order


def record(identifier, role='user', image=False, **overrides):
    result = dict(id=identifier, role=role, has_image=image, session_id='s',
                  user_id='u', sequence=identifier)
    result.update(overrides)
    return result


def test_order_preserves_candidates_and_delays_paired_reply():
    records = [record(1, image=True), record(2, 'assistant'), record(3)]
    assert image_first_order([2, 3, 1], records) == [1, 3, 2]
    assert image_first_order([2, 3], records) == [2, 3]
    assert image_first_order([3, 1], records) == [3, 1]
    assert image_first_order([], records) == []


@pytest.mark.parametrize('first,second', [
    ({'user_id': 'other'}, {}), ({'session_id': 'other'}, {}),
    ({'session_id': ''}, {'session_id': ''}),
    ({'session_id': ' '}, {'session_id': ' '}),
    ({}, {'sequence': 4}), ({'has_image': False}, {}),
    ({'role': 'assistant'}, {}), ({}, {'role': 'user'}),
])
def test_source_boundaries_prevent_pairing(first, second):
    records = [dict(record(1, image=True), **first), dict(record(2, 'assistant'), **second)]
    assert image_first_order([2, 1], records) == [2, 1]


def test_intervening_source_record_is_not_skipped():
    records = [record(1, image=True), record(3), record(2, 'assistant')]
    assert image_first_order([2, 1], records) == [2, 1]


@pytest.mark.parametrize('candidates,records', [
    ([1, 1], [record(1)]), ([1], [record(1), record(1)]), ([2], [record(1)]),
])
def test_invalid_candidate_identity_is_rejected(candidates, records):
    with pytest.raises(ValueError):
        image_first_order(candidates, records)


def messages(count):
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2), 'red').save(buffer, format='PNG')
    uri = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    rows = [dict(role='user', content=[dict(type='text', text=f'item {i}')])
            for i in range(count)]
    rows[0]['content'].append(dict(type='image_url', image_url=uri))
    rows[1]['role'] = 'assistant'
    return rows


def populate(tmp_path, monkeypatch, count=3, enabled=True):
    service = mm.MemoryService(str(tmp_path / 'db'), pair_diversity=enabled)
    service.add(mm.AddRequest.model_validate(dict(user_id='u', session_id='s',
        request_id='r', messages=messages(count))))
    # The reply ranks first, then its image source, then the remaining records.
    scores = np.arange(count, 0, -1, dtype=float)
    scores[0], scores[1] = scores[1], scores[0]
    monkeypatch.setattr(mm, 'bm25_scores', lambda *args: scores)
    return service


def labels(result):
    return [item['content'][0]['text'] for item in result['data']]


def test_fixed_candidate_pool_before_top_k_and_tail_preserved(tmp_path, monkeypatch):
    service = populate(tmp_path, monkeypatch, count=103)
    calls = []
    def observe(candidates, records):
        calls.append((list(candidates), list(records)))
        return image_first_order(candidates, records)
    monkeypatch.setattr(mm, 'image_first_order', observe)
    for k in (1, 2, 100, 103):
        result = service.search(mm.SearchRequest(user_id='u', query='query', top_k=k))
        expected = ['item 0'] + [f'item {i}' for i in range(2, 100)] + ['item 1']
        expected += ['item 100', 'item 101', 'item 102']
        assert labels(result) == expected[:k]
        assert len(calls[-1][0]) == 100
        assert len(calls[-1][1]) == 103
    assert service.search(mm.SearchRequest(user_id='other', query='query', top_k=2)) == {'data': []}


def test_disabled_preserves_original_ranking(tmp_path, monkeypatch):
    service = populate(tmp_path, monkeypatch, enabled=False)
    def unexpected(*args):
        pytest.fail('disabled ordering must not invoke pair helper')
    monkeypatch.setattr(mm, 'image_first_order', unexpected)
    assert labels(service.search(mm.SearchRequest(user_id='u', query='query', top_k=3))) == [
        'item 1', 'item 0', 'item 2']


def test_reordered_media_budget_stops_at_first_overflow(tmp_path, monkeypatch):
    service = populate(tmp_path, monkeypatch, count=4)
    # Mark a second user record as carrying an image while preserving its source.
    import json
    with service.connect() as db:
        source = json.loads(db.execute('SELECT content FROM memories WHERE sequence=1').fetchone()[0])
        second = json.loads(db.execute('SELECT content FROM memories WHERE sequence=3').fetchone()[0])
        second.append(source[1])
        db.execute('UPDATE memories SET content=? WHERE sequence=3', (json.dumps(second),))
    assert mm.MAX_MEDIA_BYTES == 30 * 1024 * 1024
    monkeypatch.setattr(mm, 'image_bytes', lambda part: 20 * 1024 * 1024 if part['type'] == 'image_url' else 0)
    result = service.search(mm.SearchRequest(user_id='u', query='query', top_k=4))
    assert labels(result) == ['item 0']
    # The smaller text records beyond the overflowing image must not jump ahead.
    monkeypatch.setattr(mm, 'image_bytes', lambda part: 31 * 1024 * 1024 if part['type'] == 'image_url' else 0)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        service.search(mm.SearchRequest(user_id='u', query='query', top_k=4))
    assert error.value.status_code == 413
