from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import numpy as np
import pytest

import service.background as background
from service.main import StructMemService
from service.retrieval import bm25_scores, hybrid_search, prepare_corpus
from service.schemas import AddRequest, MediaPart, SearchRequest
from service.store import RawMemoryStore, StoredMemory


def memories():
    rng = np.random.default_rng(7)
    texts = ["alpha cedar cedar", "北京天气晴朗", "alpha 北京 coffee", "the and", ""]
    result = []
    for index in range(40):
        vector = rng.normal(size=8).astype(np.float32)
        vector /= np.linalg.norm(vector)
        result.append(StoredMemory(
            sequence=index + 1, id=f"m{index}", request_id=f"r{index}",
            user_id="u", session_id=f"s{index // 4}", message_index=index,
            role="user", content=texts[index % len(texts)],
            source_timestamp=None, created_at=datetime.now(timezone.utc),
            embedding=vector,
            media=(MediaPart(type="image_url", image_url="https://example.com/test.jpg", caption="coffee 天气"),)
            if index % 7 == 0 else (),
            embedding_chunks=(vector * 0.9,) if index % 3 == 0 else (),
        ))
    return result


@pytest.mark.parametrize("query", ["alpha cedar", "北京天气", "alpha 北京", "the and", ""])
@pytest.mark.parametrize("fusion", ["minmax", "rrf"])
def test_cached_and_incremental_ranking_exactly_match_original(query, fusion):
    rows = memories()
    prefix = prepare_corpus(rows[:17], state=(17, 17))
    corpus = prepare_corpus(rows, state=(40, 40), prefix=prefix)
    assert corpus.lexical.frequencies[0] is prefix.lexical.frequencies[0]
    assert not corpus.matrix.flags.writeable
    documents = ["\n".join([m.content, *(x.caption or "" for x in m.media)]).strip() for m in rows]
    for cjk in [False, True]:
        index = corpus.cjk_lexical if cjk else corpus.lexical
        np.testing.assert_array_equal(
            index.scores(query, cjk_ngrams=cjk), bm25_scores(documents, query, cjk_ngrams=cjk),
        )
        for controls in [{}, {"session_recency_weight": .2, "session_repeat_penalty": .1,
                               "session_anchor_seed_k": 4}, {"session_diversity": True}]:
            kwargs = dict(
                query_vector=rows[3].embedding, query_text=query, top_k=25,
                lexical_weight=.5, rrf_k=60, fusion=fusion, result_window=1,
                result_window_seed_k=5, semantic_bridge_seed_k=3,
                semantic_bridge_min_similarity=.3, cjk_ngrams=cjk, **controls,
            )
            expected = hybrid_search(rows, **kwargs)
            actual = hybrid_search(rows, prepared=corpus, **kwargs)
            assert [(h.memory.id, h.score) for h in actual] == [(h.memory.id, h.score) for h in expected]


class Embedder:
    identity = "synthetic-background-tests"

    def embed_documents(self, texts):
        return [[1.0, float(len(t) % 3)] for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def service(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_BACKGROUND_INDEX", "true")
    monkeypatch.setenv("MEMORY_INDEX_IDLE_SECONDS", "0")
    monkeypatch.setenv("MEMORY_SESSION_FACTS", "false")
    return StructMemService(
        embedder=Embedder(),
        store=RawMemoryStore(tmp_path / "store.db", embedding_identity=Embedder.identity),
    )


def add(svc, index, user="u"):
    return svc.add(AddRequest.model_validate({
        "request_id": f"{user}-{index}", "user_id": user,
        "session_id": f"session-{index}",
        "messages": [{"role": "user", "content": f"Cedar meeting confirmation {index}"}],
    }))


def search(svc, user="u"):
    return svc.search(SearchRequest(query="Cedar meeting", user_id=user, top_k=100))


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("Background task did not finish")


def test_foreground_works_before_ready_after_append_and_after_restart(tmp_path, monkeypatch):
    svc = service(tmp_path, monkeypatch)
    try:
        with svc.index.foreground():
            svc.index.start()
            add(svc, 1)
            assert svc.index.get("u") is None
            first = search(svc).model_dump()
            assert len(first["data"]) == 1
            assert svc.index.builds == 0
        wait_for(lambda: svc.index.get("u"))
        assert search(svc).model_dump() == first
        with svc.index.foreground():
            add(svc, 2)
            assert svc.index.get("u") is None
            second = search(svc).model_dump()
            assert len(second["data"]) == 2
            assert search(svc, "other").data == []
        wait_for(lambda: svc.index.get("u"))
        assert search(svc).model_dump() == second
    finally:
        svc.index.close()
    restarted = service(tmp_path, monkeypatch)
    try:
        assert restarted.index.get("u") is None
        assert search(restarted).model_dump() == second
    finally:
        restarted.index.close()


def test_append_during_build_cannot_publish_stale_results(tmp_path, monkeypatch):
    svc = service(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = background.prepare_corpus

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(background, "prepare_corpus", delayed)
    try:
        svc.index.start()
        add(svc, 1)
        assert entered.wait(5)
        add(svc, 2)
        baseline = search(svc).model_dump()
        assert len(baseline["data"]) == 2
        release.set()
        ready = wait_for(lambda: svc.index.get("u"))
        assert ready.state == (2, 2)
        assert search(svc).model_dump() == baseline
    finally:
        release.set()
        svc.index.close()


def test_failed_build_keeps_retrieval_available(tmp_path, monkeypatch):
    svc = service(tmp_path, monkeypatch)

    def fail(*a, **kw):
        raise RuntimeError("simulated build failure")

    monkeypatch.setattr(background, "prepare_corpus", fail)
    try:
        svc.index.start()
        add(svc, 1)
        wait_for(lambda: svc.index.failures)
        assert len(search(svc).data) == 1
    finally:
        svc.index.close()


def test_checkpoint_yields_to_foreground_and_shutdown_is_bounded(tmp_path):
    store = RawMemoryStore(tmp_path / "store.db", embedding_identity="test")
    index = background.BackgroundIndex(store, idle_seconds=0)
    entered = threading.Event()

    def checkpoint():
        index._checkpoint()
        entered.set()

    with index.foreground():
        worker = threading.Thread(target=checkpoint)
        worker.start()
        assert not entered.wait(.05)
    assert entered.wait(2)
    worker.join(2)
    with index.foreground():
        index.start()
        index.schedule("u")
        index.close()
    assert not index._thread.is_alive()


def test_cache_memory_budget_and_expiry(tmp_path, monkeypatch):
    rows = memories()
    with pytest.raises(BufferError):
        prepare_corpus(rows, state=(40, 40), max_bytes=1)
    svc = service(tmp_path, monkeypatch)
    add(svc, 1)
    svc.index._build("u")
    assert svc.index.get("u") is not None
    # Delete data through the store's SQL connection: count/state invalidation
    # must reject a formerly valid cached snapshot as well.
    with svc.store._connect() as c:
        c.execute("DELETE FROM requests")
    assert svc.index.get("u") is None
    svc.index.close()


def test_cache_expires_and_evicts_users(tmp_path, monkeypatch):
    from types import SimpleNamespace

    clock = [time.monotonic()]
    monkeypatch.setattr(background, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    svc = service(tmp_path, monkeypatch)
    svc.index = background.BackgroundIndex(
        svc.store, idle_seconds=0, max_users=1, ttl_seconds=1,
    )
    try:
        add(svc, 1)
        svc.index._build("u")
        assert svc.index.get("u") is not None
        clock[0] += 2
        assert svc.index.get("u") is None
        svc.index._build("u")
        add(svc, 1, user="v")
        svc.index._build("v")
        assert svc.index.get("v") is not None
        assert svc.index.get("u") is None
    finally:
        svc.index.close()
