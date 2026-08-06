from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import service.main as main
from service.schemas import AddRequest, AddResponse, SearchRequest


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
        return {"data": [{"id": "memory-1", "content": "I moved to Shanghai.", "score": 1.0}]}


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


def test_request_registry_is_idempotent_and_detects_conflicts(tmp_path):
    registry = main.RequestRegistry(tmp_path / "requests.sqlite3")
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
    registry.save(request, response)
    assert registry.get_or_conflict(request) == response

    conflicting = AddRequest(
        request_id="req-1",
        messages=[{"role": "user", "content": "Changed."}],
        user_id="user-1",
        session_id="session-1",
    )
    with pytest.raises(HTTPException) as error:
        registry.get_or_conflict(conflicting)
    assert error.value.status_code == 409


def test_search_returns_item_top_k_plus_summary_fraction():
    class FakeEmbedder:
        def embed(self, query):
            return [1.0]

    class FakeRetriever:
        def __init__(self, payload_key, count, score):
            self.payload_key = payload_key
            self.count = count
            self.score = score
            self.limits = []

        def search(self, query_vector, limit, filters, return_full):
            self.limits.append(limit)
            return [
                {
                    "id": f"{self.payload_key}-{index}",
                    "score": self.score - index / 100,
                    "payload": {self.payload_key: f"result-{index}"},
                }
                for index in range(min(limit, self.count))
            ]

    entries = FakeRetriever("memory", count=5, score=1.0)
    summaries = FakeRetriever("summary", count=1, score=0.5)
    service = object.__new__(main.StructMemService)
    service.lock = threading.RLock()
    service.memory = SimpleNamespace(
        text_embedder=FakeEmbedder(),
        embedding_retriever=entries,
        summary_retriever=summaries,
    )

    response = service.search(
        SearchRequest(query="query", user_id="user-1", top_k=5)
    )

    assert entries.limits == [5]
    assert summaries.limits == [1]
    assert len(response.data) == 6
    assert sum(item.id.startswith("memory-") for item in response.data) == 5
    assert sum(item.id.startswith("summary-") for item in response.data) == 1
