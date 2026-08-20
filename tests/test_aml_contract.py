from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

import service.main as main
from service.retrieval import hybrid_search
from service.schemas import AddRequest, AddResponse, SearchRequest
from service.store import ConflictingRequestError, RawMemoryStore, StoredMemory


class FakeService:
    def __init__(self):
        self.requests = []

    def add(self, request):
        self.requests.append(request)
        return {
            "success": True,
            "request_id": request.request_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }

    def search(self, request):
        return {
            "data": [
                {"id": "memory-1", "content": "I moved to Shanghai.", "score": 1.0}
            ]
        }


def test_contract_routes(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(main, "service", fake)
    client = TestClient(main.app)

    health = client.get("/health")
    assert health.status_code == 200
    added = client.post(
        "/add",
        json={
            "request_id": "req-1",
            "messages": [{"role": "user", "content": "I moved to Shanghai."}],
            "user_id": "user-1",
            "session_id": "session-1",
        },
    )
    assert added.status_code == 200
    assert added.json()["success"] is True
    searched = client.post(
        "/search",
        json={"query": "Where did I move?", "user_id": "user-1", "top_k": 100},
    )
    assert searched.status_code == 200
    assert searched.json()["data"][0]["content"]


def test_raw_store_is_idempotent_and_detects_conflicts(tmp_path):
    store = RawMemoryStore(tmp_path / "memory.sqlite3", embedding_identity="fake:2")
    request = AddRequest(
        request_id="req-1",
        messages=[{"role": "user", "content": "Remember this."}],
        user_id="user-1",
        session_id="session-1",
    )
    response = AddResponse(
        success=True,
        request_id="req-1",
        user_id="user-1",
        session_id="session-1",
    )
    store.add(request, response, np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert store.get(request) == response
    store.add(request, response, np.asarray([[1.0, 0.0]], dtype=np.float32))
    assert len(store.list_user("user-1")) == 1

    conflicting = AddRequest(
        request_id="req-1",
        messages=[{"role": "user", "content": "Changed."}],
        user_id="user-1",
        session_id="session-1",
    )
    with pytest.raises(ConflictingRequestError):
        store.get(conflicting)


class FakeEmbedder:
    identity = "fake:2"

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.calls.append(text)
        return [1.0, 0.0]


def _service(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_RESULT_WINDOW", "1")
    monkeypatch.setenv("MEMORY_RESULT_WINDOW_SEED_K", "20")
    monkeypatch.setenv("MEMORY_SEMANTIC_BRIDGE_SEED_K", "10")
    monkeypatch.setenv("MEMORY_SEMANTIC_BRIDGE_MIN_SIMILARITY", "0.72")
    embedder = FakeEmbedder()
    store = RawMemoryStore(
        tmp_path / "memory.sqlite3", embedding_identity=embedder.identity
    )
    return main.StructMemService(embedder=embedder, store=store), embedder


def test_search_preserves_raw_messages_uses_options_and_honors_top_k(
    tmp_path, monkeypatch
):
    service, embedder = _service(tmp_path, monkeypatch)
    service.add(
        AddRequest(
            request_id="req-1",
            user_id="user-1",
            session_id="session-1",
            messages=[
                {
                    "role": "Alice",
                    "content": "The access code is ZX-42.",
                    "timestamp": 1704067200000,
                },
                {"role": "Bob", "content": "I wrote it down."},
                {"role": "Alice", "content": "Keep it somewhere safe."},
            ],
        )
    )

    response = service.search(
        SearchRequest(
            query="Which access code was mentioned?",
            options=["AB-10", "ZX-42"],
            user_id="user-1",
            top_k=2,
        )
    )

    assert len(response.data) == 2
    assert (
        response.data[0].content
        == "[2024-01-01T00:00:00Z] Alice: The access code is ZX-42."
    )
    assert response.data[1].content == "Bob: I wrote it down."
    assert "Candidate answers" in embedder.calls[-1]
    assert "ZX-42" in embedder.calls[-1]


def test_search_is_user_scoped_and_skips_embedding_for_empty_user(
    tmp_path, monkeypatch
):
    service, embedder = _service(tmp_path, monkeypatch)
    service.add(
        AddRequest(
            request_id="req-1",
            user_id="user-1",
            session_id="session-1",
            messages=[{"role": "user", "content": "Private memory."}],
        )
    )
    call_count = len(embedder.calls)

    response = service.search(
        SearchRequest(query="private", user_id="other-user", top_k=100)
    )

    assert response.data == []
    assert len(embedder.calls) == call_count


def _memory(index, session_id, content, embedding):
    vector = np.asarray(embedding, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return StoredMemory(
        sequence=index + 1,
        id=f"memory-{index}",
        request_id=f"request-{session_id}",
        user_id="user-1",
        session_id=session_id,
        message_index=index,
        role="user",
        content=content,
        source_timestamp=None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        embedding=vector,
    )


def test_semantic_bridge_expands_a_cross_session_candidate_and_honors_cap():
    memories = [
        _memory(0, "session-a", "rare answer", [0.8, 0.6]),
        _memory(1, "session-a", "same-session paraphrase", [0.81, 0.59]),
        _memory(2, "session-b", "semantic continuation", [0.95, 0.31]),
        _memory(3, "session-c", "other", [-1.0, 0.0]),
    ]
    common = {
        "query_vector": np.asarray([0.0, 1.0], dtype=np.float32),
        "query_text": "rare",
        "top_k": 2,
        "lexical_weight": 0.5,
        "rrf_k": 60,
        "result_window": 0,
        "result_window_seed_k": 20,
    }

    baseline = hybrid_search(
        memories,
        **common,
        semantic_bridge_seed_k=0,
        semantic_bridge_min_similarity=0.72,
    )
    expanded = hybrid_search(
        memories,
        **common,
        semantic_bridge_seed_k=1,
        semantic_bridge_min_similarity=0.72,
    )

    assert [hit.memory.id for hit in baseline] == ["memory-0", "memory-1"]
    assert [hit.memory.id for hit in expanded] == ["memory-0", "memory-2"]
    assert len({hit.memory.id for hit in expanded}) == 2


def test_semantic_bridge_preserves_ranking_when_no_candidate_passes_threshold():
    memories = [
        _memory(0, "session-a", "rare answer", [0.8, 0.6]),
        _memory(1, "session-a", "same-session paraphrase", [0.81, 0.59]),
        _memory(2, "session-b", "weak relation", [0.95, 0.31]),
    ]
    common = {
        "query_vector": np.asarray([0.0, 1.0], dtype=np.float32),
        "query_text": "rare",
        "top_k": 2,
        "lexical_weight": 0.5,
        "rrf_k": 60,
        "result_window": 0,
        "result_window_seed_k": 20,
        "semantic_bridge_min_similarity": 0.99,
    }

    baseline = hybrid_search(memories, **common, semantic_bridge_seed_k=0)
    expanded = hybrid_search(memories, **common, semantic_bridge_seed_k=1)

    assert [hit.memory.id for hit in expanded] == [hit.memory.id for hit in baseline]
