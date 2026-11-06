"""Synthetic integration contracts; no external models, APIs or datasets."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

import service.multimodal as mm
from service.multimodal_retrieval import rerank_prefix
from service.multimodal_reranker import LocalReranker


def image_part():
    data = io.BytesIO()
    Image.new('RGB', (32, 32), 'blue').save(data, format='PNG')
    return dict(type='image_url', image_url=dict(
        url='data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()))


class Observe:
    def __init__(self, scores=None, error=None):
        self.calls, self.scores, self.error = [], scores, error

    def score(self, query, documents, decoder):
        self.calls.append((query, documents))
        if self.error:
            raise self.error
        return self.scores if self.scores is not None else list(range(len(documents)))


def populate(tmp_path, monkeypatch, scorer, mode='direct', count=25):
    service = mm.MemoryService(str(tmp_path / 'db'), reranker=scorer, rerank_mode=mode)
    messages = [dict(role='user', content=[dict(type='text', text=f'item {i}')])
                for i in range(count)]
    messages[0]['content'].append(image_part())
    messages[1]['role'] = 'assistant'
    service.add(mm.AddRequest.model_validate(dict(request_id='r', user_id='u',
                                                session_id='s', messages=messages)))
    monkeypatch.setattr(mm, 'bm25_scores', lambda docs, text: np.arange(len(docs), 0, -1))
    return service, messages


def query(service, k=25, user='u'):
    return service.search(mm.SearchRequest(user_id=user, query='blue image', top_k=k))


def labels(result):
    return [r['content'][0]['text'] for r in result['data']]


def test_prefix_modes_and_stable_ties():
    ids = list(range(25))
    assert rerank_prefix(ids, range(20), 'direct') == list(range(19, -1, -1)) + ids[20:]
    # Complete reverse ranks cancel under equal rank sum, retaining baseline ties.
    assert rerank_prefix(ids, range(20), 'consensus') == ids
    assert rerank_prefix(ids, [1] * 20, 'direct') == ids
    assert rerank_prefix([], [], 'consensus') == []


@pytest.mark.parametrize('ids,scores,mode', [
    ([1, 1], [1, 2], 'direct'), ([1, 2], [1], 'direct'),
    ([1, 2], [1, float('nan')], 'direct'),
    ([1], [float('inf')], 'consensus'), ([1], [1], 'invalid'),
])
def test_invalid_reranker_output_is_rejected(ids, scores, mode):
    with pytest.raises(ValueError):
        rerank_prefix(ids, scores, mode)


def test_bounded_pool_source_reply_original_evidence_and_user_isolation(tmp_path, monkeypatch):
    scorer = Observe()
    service, messages = populate(tmp_path, monkeypatch, scorer)
    result = query(service)
    assert labels(result) == [f'item {i}' for i in range(19, -1, -1)] + [f'item {i}' for i in range(20, 25)]
    assert len(scorer.calls[-1][1]) == 20
    assert scorer.calls[-1][1][0] == messages[0]['content'] + [dict(type='text', text='item 1')]
    assert next(r for r in result['data'] if r['content'][0]['text'] == 'item 0')['content'] == messages[0]['content']
    assert labels(query(service, 1)) == ['item 19']
    before = len(scorer.calls)
    assert query(service, user='other') == {'data': []}
    assert len(scorer.calls) == before
    assert service.reranker_error is None


@pytest.mark.parametrize('update', [
    "UPDATE memories SET session_id='other' WHERE sequence=2",
    "UPDATE memories SET session_id='' WHERE sequence IN (1,2)",
    "UPDATE memories SET session_id=' ' WHERE sequence IN (1,2)",
    "UPDATE memories SET sequence=50 WHERE sequence=2",
    "UPDATE memories SET role='user' WHERE sequence=2",
])
def test_source_boundaries_prevent_reply_context(tmp_path, monkeypatch, update):
    scorer = Observe()
    service, messages = populate(tmp_path, monkeypatch, scorer)
    with service.connect() as db:
        db.execute(update)
    query(service)
    assert scorer.calls[-1][1][0] == messages[0]['content']


@pytest.mark.parametrize('scorer', [
    Observe(error=RuntimeError('busy')), Observe(error=TimeoutError()),
    Observe(scores=[float('nan')] * 20), Observe(scores=[1]),
])
def test_failed_rerank_preserves_baseline_and_evidence(tmp_path, monkeypatch, scorer):
    service, messages = populate(tmp_path, monkeypatch, scorer)
    result = query(service)
    assert labels(result) == [f'item {i}' for i in range(25)]
    assert result['data'][0]['content'] == messages[0]['content']
    assert service.reranker_error is not None


def test_byte_budget_applies_after_rerank_and_stops_at_overflow(tmp_path, monkeypatch):
    service, _ = populate(tmp_path, monkeypatch, Observe())
    monkeypatch.setattr(mm, 'MAX_MEDIA_BYTES', 3)
    monkeypatch.setattr(mm, 'image_bytes', lambda part: 4 if part.get('text') == 'item 18' else 0)
    assert labels(query(service)) == ['item 19']


def test_runtime_configuration_and_content_contract(tmp_path):
    with pytest.raises(ValueError):
        LocalReranker(tmp_path / 'missing')
    with pytest.raises(ValueError):
        LocalReranker(tmp_path, timeout=float('nan'))
    parts = [dict(type='text', text='first'), image_part(), dict(type='text', text='last')]
    content = LocalReranker._content(parts, 'Document:', mm.decode_image)
    assert content[0]['text'] == 'Document:' and content[1]['image'].size == (32, 32)
    assert content[2]['text'] == 'first last'
    with pytest.raises(ValueError):
        LocalReranker._content([], '', mm.decode_image)


def test_busy_model_returns_without_importing_optional_dependencies(tmp_path):
    model = LocalReranker(tmp_path)
    with model.lock:
        with pytest.raises(RuntimeError, match='busy'):
            model.score([dict(type='text', text='query')],
                        [[dict(type='text', text='document')]], mm.decode_image)
    with pytest.raises(ValueError):
        model.score([], [], mm.decode_image)


def test_image_locked_keeps_image_evidence_and_tail(tmp_path, monkeypatch):
    service, messages = populate(tmp_path, monkeypatch, Observe(), mode='image_locked')
    result = query(service)
    assert labels(result) == ['item 0'] + [f'item {i}' for i in range(19, 0, -1)] + [f'item {i}' for i in range(20, 25)]
    assert result['data'][0]['content'] == messages[0]['content']
    assert labels(query(service, 1)) == ['item 0']
    assert service.reranker_error is None


def test_image_locked_multiple_images_and_invalid_mask():
    ids = list(range(25))
    mask = [i in {0, 3, 9} for i in range(20)]
    result = rerank_prefix(ids, range(20), 'image_locked', image_mask=mask)
    assert [result[i] for i in (0, 3, 9)] == [0, 3, 9]
    assert result[20:] == ids[20:]
    assert set(result[:20]) == set(ids[:20])
    assert rerank_prefix(ids, [1]*20, 'image_locked', image_mask=mask) == ids
    for invalid in (None, [], [0]*20):
        with pytest.raises(ValueError):
            rerank_prefix(ids, range(20), 'image_locked', image_mask=invalid)


def test_session_diversity_protects_slots_and_prefers_new_sources():
    from service.multimodal_retrieval import session_diverse_order
    ids = list(range(105))
    scores = [float(100-i) for i in range(100)]
    scores[0] = scores[4] = None
    sessions = ['a', 'a', 'b', 'b', 'd'] + ['c'] * 95
    result = session_diverse_order(ids, scores, sessions)
    assert result[:5] == [0, 2, 5, 1, 4]
    assert result[100:] == ids[100:]
    assert len(set(result)) == len(ids)
    assert session_diverse_order([], [], []) == []
    assert session_diverse_order([0, 1], [1, 1], ['', '']) == [0, 1]
    with pytest.raises(ValueError):
        session_diverse_order([0], [float('nan')], ['a'])
    with pytest.raises(ValueError):
        session_diverse_order([0, 0], [1, 2], ['a', 'b'])


class SourceScores:
    timeout = 30

    def __init__(self, invalid=None, failure=None):
        self.invalid, self.failure, self.calls = invalid, failure, []

    def score(self, query, documents, decoder):
        assert len(documents) == 1
        assert all(p['type'] == 'text' for p in documents[0])
        label = int(documents[0][0]['text'].split()[-1])
        self.calls.append(label)
        if label == self.invalid:
            raise ValueError('reranker input exceeds context budget')
        if self.failure:
            raise self.failure
        return [float(label)]


def test_session_diverse_service_preserves_image_and_unscorable_source(tmp_path, monkeypatch):
    scorer = SourceScores(invalid=7)
    service, messages = populate(tmp_path, monkeypatch, scorer, mode='session_diverse')
    result = query(service)
    assert result['data'][0]['content'] == messages[0]['content']
    assert labels(result)[7] == 'item 7'
    assert labels(result)[1] == 'item 24'
    assert 0 not in scorer.calls
    assert len(scorer.calls) == 24
    assert set(labels(result)) == {f'item {i}' for i in range(25)}
    assert service.reranker_error is None


@pytest.mark.parametrize('failure', [RuntimeError('busy'), TimeoutError(), ValueError('invalid session relevance')])
def test_session_diverse_fatal_error_keeps_whole_baseline(tmp_path, monkeypatch, failure):
    service, _ = populate(tmp_path, monkeypatch, SourceScores(failure=failure), mode='session_diverse')
    assert labels(query(service)) == [f'item {i}' for i in range(25)]
    assert service.reranker_error == type(failure).__name__


def test_session_diverse_overall_budget_falls_back(tmp_path, monkeypatch):
    scorer = SourceScores()
    service, _ = populate(tmp_path, monkeypatch, scorer, mode='session_diverse')
    clock = iter([0, 31])
    from types import SimpleNamespace
    monkeypatch.setattr(mm, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    assert labels(query(service)) == [f'item {i}' for i in range(25)]
    assert scorer.calls == []
    assert service.reranker_error == 'TimeoutError'


def test_protected40_preserves_direct_head_and_outer_tail():
    ids = list(range(60))
    scores = list(range(40))
    result = rerank_prefix(ids, scores, 'protected40')
    assert result[:10] == list(range(19, 9, -1))
    assert result[10:30] == list(range(39, 19, -1))
    assert result[30:40] == list(range(9, -1, -1))
    assert result[40:] == ids[40:]
    assert rerank_prefix(ids, [1] * 40, 'protected40') == ids
    assert rerank_prefix([], [], 'protected40') == []
    assert rerank_prefix(ids[:7], range(7), 'protected40') == list(range(6, -1, -1))


class WiderObserve(Observe):
    timeout = 30

    def score(self, query, documents, decoder):
        super().score(query, documents, decoder)
        if len(self.calls) == 2 and self.error_on_second:
            raise RuntimeError('second batch failed')
        return [int(d[0]['text'].split()[-1]) for d in documents]

    error_on_second = False


def test_protected40_service_batches_and_preserves_original_content(tmp_path, monkeypatch):
    scorer = WiderObserve()
    service, messages = populate(tmp_path, monkeypatch, scorer, mode='protected40', count=45)
    result = query(service, 45)
    expected = list(range(19, 9, -1)) + list(range(39, 19, -1)) + list(range(9, -1, -1)) + list(range(40, 45))
    assert labels(result) == [f'item {i}' for i in expected]
    assert [len(c[1]) for c in scorer.calls] == [20, 20]
    assert next(r for r in result['data'] if r['content'][0]['text'] == 'item 0')['content'] == messages[0]['content']
    assert service.reranker_error is None


@pytest.mark.parametrize('failure', ['second_batch', 'deadline'])
def test_protected40_atomic_fallback(tmp_path, monkeypatch, failure):
    scorer = WiderObserve()
    scorer.error_on_second = failure == 'second_batch'
    service, _ = populate(tmp_path, monkeypatch, scorer, mode='protected40', count=45)
    if failure == 'deadline':
        ticks = iter([0, 0, 31])
        monkeypatch.setattr(mm.time, 'monotonic', lambda: next(ticks))
    result = query(service, 45)
    assert labels(result) == [f'item {i}' for i in range(45)]
    assert service.reranker_error == ('RuntimeError' if failure == 'second_batch' else 'TimeoutError')


def test_protected_tail_preserves_head_and_unscored_tail():
    ids = list(range(110))
    scores = list(range(100))
    actual = rerank_prefix(ids, scores, 'protected20_tail100')
    head = rerank_prefix(ids, scores[:40], 'protected40')[:20]
    assert actual[:20] == head
    assert actual[20:100] == [i for i in range(99, -1, -1) if i not in set(head)]
    assert actual[100:] == ids[100:]
    assert rerank_prefix(ids, [1] * 100, 'protected20_tail100') == ids
    assert rerank_prefix([], [], 'protected20_tail100') == []


def test_protected_tail_service_batches(tmp_path, monkeypatch):
    scorer = WiderObserve()
    service, _ = populate(tmp_path, monkeypatch, scorer, mode='protected20_tail100', count=105)
    result = query(service, 100)
    head = list(range(19, 9, -1)) + list(range(39, 29, -1))
    expected = head + [i for i in range(99, -1, -1) if i not in set(head)]
    assert labels(result) == [f'item {i}' for i in expected]
    assert [len(c[1]) for c in scorer.calls] == [20] * 5
    assert service.reranker_error is None


@pytest.mark.parametrize('failure', ['last_batch', 'nonfinite', 'short_batch', 'deadline'])
def test_protected_tail_atomic_fallback_after_partial_scoring(tmp_path, monkeypatch, failure):
    class FailLate(WiderObserve):
        def score(self, query, documents, decoder):
            values = super().score(query, documents, decoder)
            if len(self.calls) == 5:
                if failure == 'last_batch':
                    raise RuntimeError('last batch failed')
                if failure == 'nonfinite':
                    values[-1] = float('nan')
                if failure == 'short_batch':
                    return values[:-1]
            return values
    scorer = FailLate()
    service, _ = populate(tmp_path, monkeypatch, scorer, mode='protected20_tail100', count=105)
    if failure == 'deadline':
        ticks = iter([0] * 10 + [31])
        monkeypatch.setattr(mm.time, 'monotonic', lambda: next(ticks))
    result = query(service, 100)
    assert len(scorer.calls) == 5
    assert labels(result) == [f'item {i}' for i in range(100)]
    assert service.reranker_error == ({'last_batch': 'RuntimeError', 'deadline': 'TimeoutError'}.get(failure, 'ValueError'))


@pytest.mark.parametrize('count', [7, 25, 45, 85])
def test_protected_tail_short_pool_matches_prefix_contract(tmp_path, monkeypatch, count):
    scorer = WiderObserve()
    service, _ = populate(tmp_path, monkeypatch, scorer, mode='protected20_tail100', count=count)
    result = query(service, min(count, 100))
    expected = rerank_prefix(list(range(count)), list(range(count)), 'protected20_tail100')
    assert labels(result) == [f'item {i}' for i in expected]
    assert sum(len(c[1]) for c in scorer.calls) == count
    assert service.reranker_error is None
