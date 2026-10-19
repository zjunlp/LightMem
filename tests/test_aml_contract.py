from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
from dataclasses import replace
import pytest
from fastapi.testclient import TestClient

import service.main as main
from service.retrieval import hybrid_search
from service.retrieval import lexical_tokens
from service.schemas import AddRequest, AddResponse, MediaPart, SearchRequest
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-01-01T00:00:00+08:00", 1704038400000),
        ("2024-01-01T00:00:00Z", 1704067200000),
        ("2024-01-01T00:00:00", 1704067200000),
        ("2024/01/01 00:00", 1704067200000),
    ],
)
def test_timestamp_millis_respects_timezone_offsets(value, expected):
    assert main._timestamp_millis(value) == expected


def test_lexical_tokens_add_cjk_bigrams_without_changing_english_words():
    tokens = lexical_tokens("我喜欢绿茶，也喜欢。green tea", cjk_ngrams=True)
    assert "喜欢" in tokens
    assert "绿茶" in tokens
    assert "green" in tokens
    assert "tea" in tokens


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


def test_store_rejects_existing_vectors_without_embedding_identity(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = RawMemoryStore(path, embedding_identity="fake:2")
    request = AddRequest(
        request_id="req-1", user_id="user-1", session_id="session-1",
        messages=[{"role": "user", "content": "Remember this."}],
    )
    response = AddResponse(
        success=True, request_id="req-1", user_id="user-1", session_id="session-1",
    )
    store.add(request, response, np.asarray([[1.0, 0.0]], dtype=np.float32))
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM store_metadata WHERE key = 'embedding_identity'")
    with pytest.raises(ValueError, match="without an embedding identity"):
        RawMemoryStore(path, embedding_identity="different:2")


def test_raw_store_keeps_derived_facts_with_source_provenance(tmp_path):
    store = RawMemoryStore(tmp_path / "memory.sqlite3", embedding_identity="fake:2")
    request = AddRequest(
        request_id="req-facts",
        messages=[{"role": "user", "content": "I prefer tea.", "timestamp": 1704067200000}],
        user_id="user-1",
        session_id="session-1",
    )
    response = AddResponse(success=True, request_id="req-facts", user_id="user-1", session_id="session-1")
    store.add(request, response, np.asarray([[1.0, 0.0]], dtype=np.float32))
    store.add_derived(request, ["The user prefers tea."], np.asarray([[0.0, 1.0]], dtype=np.float32))
    store.add_derived(request, ["The user prefers tea."], np.asarray([[0.0, 1.0]], dtype=np.float32))
    memories = store.list_user("user-1")
    assert [memory.derived for memory in memories] == [False, True]
    assert memories[1].request_id == request.request_id
    assert memories[1].session_id == request.session_id
    assert memories[1].source_timestamp == 1704067200000


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


@pytest.mark.parametrize(
    "model,revision,expected_model,expected_revision",
    [
        (None, None, "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
        ("BAAI/bge-small-en-v1.5", None, "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
        ("custom/model", None, "custom/model", None),
        ("custom/model", "custom-revision", "custom/model", "custom-revision"),
    ],
)
def test_local_embedder_passes_revision_for_selected_model(
    monkeypatch, model, revision, expected_model, expected_revision
):
    for name in (
        "MEMORY_LOCAL_EMBEDDING_MODEL", "MEMORY_EMBEDDING_MODEL",
        "MEMORY_LOCAL_EMBEDDING_REVISION", "MEMORY_EMBEDDING_QUERY_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)
    if model:
        monkeypatch.setenv("MEMORY_LOCAL_EMBEDDING_MODEL", model)
    if revision:
        monkeypatch.setenv("MEMORY_LOCAL_EMBEDDING_REVISION", revision)
    loaded = {}

    def load_model(model_name, **kwargs):
        loaded.update(model=model_name, **kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=load_model),
    )
    embedder = main.SentenceTransformerEmbedder()
    assert loaded["model"] == expected_model
    assert loaded["revision"] == expected_revision
    if expected_model == "BAAI/bge-m3":
        assert embedder.query_prefix == ""


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


@pytest.mark.parametrize(
    ("messages", "query", "expected_weight"),
    [
        (["I moved to Paris.", "My favorite season is spring."], "Where did I move?", 0.05),
        (["我搬到了上海。", "我喜欢绿茶。"], "我搬到了哪里？", 0.5),
        (["我搬到了上海。", "I like green tea."], "Where did I move?", 0.05),
    ],
)
def test_search_selects_lexical_weight_from_memory_corpus_language(
    tmp_path, monkeypatch, messages, query, expected_weight
):
    monkeypatch.setenv("MEMORY_LEXICAL_WEIGHT", "0.05")
    monkeypatch.setenv("MEMORY_CJK_LEXICAL_WEIGHT", "0.5")
    monkeypatch.setenv("MEMORY_CJK_CORPUS_RATIO_THRESHOLD", "0.05")
    service, _ = _service(tmp_path, monkeypatch)
    service.add(AddRequest(
        request_id="req-language", user_id="user-language", session_id="session-1",
        messages=[{"role": "user", "content": content} for content in messages],
    ))
    captured = {}

    def capture(_memories, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main, "hybrid_search", capture)
    service.search(SearchRequest(
        query=query, user_id="user-language", top_k=10,
    ))

    assert captured["lexical_weight"] == expected_weight


def test_invalid_timestamp_does_not_break_add_or_embedding(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    response = service.add(AddRequest(
        request_id="req-invalid-time", user_id="user-time", session_id="session-1",
        messages=[{"role": "user", "content": "Keep this.", "timestamp": 10**30}],
    ))
    assert response.success is True
    assert service.store.list_user("user-time")[0].source_timestamp is None


def test_timestamp_document_mode_is_part_of_model_identity(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: object()),
    )
    monkeypatch.setenv("MEMORY_EMBED_TIMESTAMPS", "true")
    with_dates = main.SentenceTransformerEmbedder().identity
    monkeypatch.setenv("MEMORY_EMBED_TIMESTAMPS", "false")
    without_dates = main.SentenceTransformerEmbedder().identity
    assert with_dates != without_dates


def test_speaker_document_mode_indexes_role_once_and_versions_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MEMORY_EMBED_SPEAKERS", "true")
    enabled, embedder = _service(tmp_path / "enabled", monkeypatch)
    assert enabled.add(AddRequest(
        request_id="speaker-enabled", user_id="user-1", session_id="session-1",
        messages=[
            {"role": "Alice", "content": "I prefer green tea."},
            {"role": "user", "content": "User: already labeled."},
        ],
    )).success
    assert embedder.calls[0] == [
        "Alice: I prefer green tea.", "User: already labeled."
    ]
    enabled_identity = main._embedding_document_identity()

    monkeypatch.setenv("MEMORY_EMBED_SPEAKERS", "false")
    disabled, embedder = _service(tmp_path / "disabled", monkeypatch)
    assert disabled.add(AddRequest(
        request_id="speaker-disabled", user_id="user-1", session_id="session-1",
        messages=[{"role": "Alice", "content": "I prefer green tea."}],
    )).success
    assert embedder.calls[0] == ["I prefer green tea."]
    assert enabled_identity != main._embedding_document_identity()


def test_multi_window_uses_best_chunk_and_persists_raw_turn(tmp_path, monkeypatch):
    class WindowEmbedder(FakeEmbedder):
        def window_documents(self, text):
            return text.split("|")

        def embed_documents(self, texts):
            self.calls.append(texts)
            return [
                [1.0, 0.0] if "target-second" in text else
                [-1.0, 0.0] if "target-first" in text else
                [0.8, 0.6]
                for text in texts
            ]

    monkeypatch.setenv("MEMORY_MULTI_WINDOW", "true")
    monkeypatch.setenv("MEMORY_RESULT_WINDOW", "0")
    monkeypatch.setenv("MEMORY_SEMANTIC_BRIDGE_SEED_K", "0")
    monkeypatch.setenv("MEMORY_LEXICAL_WEIGHT", "0")
    monkeypatch.delenv("MEMORY_FUSION", raising=False)
    embedder = WindowEmbedder()
    path = tmp_path / "window.sqlite3"
    store = RawMemoryStore(path, embedding_identity=embedder.identity)
    service = main.StructMemService(embedder=embedder, store=store)
    assert service.fusion == "rrf"
    service.add(AddRequest(
        request_id="window-request", user_id="user-1", session_id="session-1",
        messages=[
            {"role": "user", "content": "other evidence"},
            {"role": "user", "content": "target-first|target-second"},
        ],
    ))
    reopened = RawMemoryStore(path, embedding_identity=embedder.identity)
    memories = reopened.list_user("user-1")
    assert len(memories) == 2
    assert len(memories[1].embedding_chunks) == 1
    reopened_service = main.StructMemService(embedder=embedder, store=reopened)
    result = reopened_service.search(SearchRequest(
        query="target", user_id="user-1", top_k=1,
    ))
    assert result.data[0].id == memories[1].id
    assert result.data[0].content == "user: target-first|target-second"


def test_multi_window_mode_versions_embedding_identity(monkeypatch):
    monkeypatch.setenv("MEMORY_MULTI_WINDOW", "false")
    single = main._embedding_document_identity()
    monkeypatch.setenv("MEMORY_MULTI_WINDOW", "true")
    assert main._embedding_document_identity() != single


def test_session_facts_are_written_but_excluded_for_temporal_queries(
    tmp_path, monkeypatch
):
    service, _ = _service(tmp_path, monkeypatch)
    service.session_facts = True
    service._session_facts = lambda _request: ["The user moved to Paris."]
    service.add(
        AddRequest(
            request_id="req-facts", user_id="user-1", session_id="session-1",
            messages=[{"role": "user", "content": "I moved to Paris.", "timestamp": 1704067200000}],
        )
    )
    assert len(service.store.list_user("user-1")) == 2
    response = service.search(
        SearchRequest(query="When did I move?", user_id="user-1", top_k=10)
    )
    assert len(response.data) == 1
    assert "memory_fact" not in response.data[0].content


def test_session_fact_failure_does_not_fail_or_erase_raw_add(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    service.session_facts = True

    def unavailable(_request):
        raise RuntimeError("fact model unavailable")

    service._session_facts = unavailable
    response = service.add(
        AddRequest(
            request_id="req-facts-failure", user_id="user-1", session_id="session-1",
            messages=[{"role": "user", "content": "Raw evidence survives."}],
        )
    )
    assert response.success is True
    memories = service.store.list_user("user-1")
    assert len(memories) == 1
    assert memories[0].derived is False


def test_media_is_preserved_and_caption_is_indexed(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    request = AddRequest(
        request_id="req-media", user_id="user-1", session_id="session-1",
        messages=[
            {
                "role": "user", "content": "I saved this for later.",
                "media": [{
                    "type": "image_url", "image_url": "data:image/png;base64,AAAA",
                    "caption": "A green tea box on a kitchen table.",
                }],
            },
            {
                "role": "user", "media": [{
                    "type": "image_url", "image_url": "data:image/png;base64,BBBB",
                }],
            },
        ],
    )
    assert service.add(request).success is True
    indexed = service.store.list_user("user-1")
    assert len(indexed) == 1
    assert indexed[0].media[0].caption == "A green tea box on a kitchen table."
    all_memories = service.store.list_user("user-1", include_unindexed=True)
    assert len(all_memories) == 2
    response = service.search(
        SearchRequest(query="Where is the green tea box?", user_id="user-1", top_k=10)
    )
    assert response.data[0].media[0].image_url == "data:image/png;base64,AAAA"


def test_openai_image_url_shape_is_normalized_before_storage(tmp_path, monkeypatch):
    service, _ = _service(tmp_path, monkeypatch)
    request = AddRequest(
        request_id="req-object-image", user_id="user-image", session_id="session-1",
        messages=[{
            "role": "user", "content": "I saved this photo.",
            "media": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,DDDD"},
                "caption": "A red umbrella by the door.",
            }],
        }],
    )
    assert service.add(request).success is True
    result = service.search(SearchRequest(
        query="Where was the red umbrella?", user_id="user-image", top_k=1,
    ))
    assert result.data[0].media[0].image_url == "data:image/png;base64,DDDD"


@pytest.mark.parametrize("content", ["I saved this.", ""])
def test_caption_terms_break_dense_ties_without_changing_evidence(
    tmp_path, monkeypatch, content
):
    service, _ = _service(tmp_path, monkeypatch)
    service.add(
        AddRequest(
            request_id="req-caption-lexical",
            user_id="user-1",
            session_id="session-1",
            messages=[
                {"role": "user", "content": "I saved an unrelated notebook."},
                {
                    "role": "user", "content": content,
                    "media": [{
                        "type": "image_url", "image_url": "data:image/png;base64,CCCC",
                        "caption": "A cobalt thermos beside a window.",
                    }],
                },
            ],
        )
    )
    response = service.search(
        SearchRequest(query="Where is the cobalt thermos?", user_id="user-1", top_k=1)
    )
    assert response.data[0].media[0].caption == "A cobalt thermos beside a window."
    assert response.data[0].content == f"user: {content}"


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


def test_session_recency_uses_first_add_order_and_is_opt_in():
    memories = [
        _memory(0, "older", "old best match", [1.0, 0.0]),
        _memory(1, "older", "old unrelated", [0.0, 1.0]),
        _memory(2, "newer", "recent close match", [0.99, 0.14]),
    ]
    common = {
        "memories": memories,
        "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
        "query_text": "unmatched",
        "top_k": 1,
        "lexical_weight": 0.0,
        "rrf_k": 60,
        "result_window": 0,
        "result_window_seed_k": 20,
        "semantic_bridge_seed_k": 0,
        "semantic_bridge_min_similarity": 0.72,
    }
    assert hybrid_search(**common)[0].memory.id == "memory-0"
    assert hybrid_search(**common, session_recency_weight=0.2)[0].memory.id == "memory-2"


def test_service_passes_session_recency_setting_to_search(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_SESSION_RECENCY_WEIGHT", "0.2")
    monkeypatch.setenv("MEMORY_RESULT_WINDOW", "0")
    monkeypatch.setenv("MEMORY_SEMANTIC_BRIDGE_SEED_K", "0")
    embedder = FakeEmbedder()
    store = RawMemoryStore(tmp_path / "memory.sqlite3", embedding_identity=embedder.identity)
    service = main.StructMemService(embedder=embedder, store=store)
    for session_id in ("older", "newer"):
        service.add(AddRequest(
            request_id=session_id, user_id="user-1", session_id=session_id,
            messages=[{"role": "user", "content": f"{session_id} note"}],
        ))
    result = service.search(SearchRequest(
        query="unmatched", user_id="user-1", top_k=1,
    ))
    assert result.data[0].content == "user: newer note"


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


def test_session_diversity_prefers_distinct_sessions_before_fill():
    memories = [
        _memory(0, "session-a", "first", [1.0, 0.0]),
        _memory(1, "session-a", "second", [0.99, 0.1]),
        _memory(2, "session-b", "third", [0.8, 0.6]),
    ]
    common = {
        "memories": memories,
        "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
        "query_text": "unmatched",
        "top_k": 2,
        "lexical_weight": 0.0,
        "rrf_k": 60,
        "result_window": 0,
        "result_window_seed_k": 20,
        "semantic_bridge_seed_k": 0,
        "semantic_bridge_min_similarity": 0.72,
    }
    ordinary = hybrid_search(**common)
    diverse = hybrid_search(**common, session_diversity=True)
    assert [hit.memory.id for hit in ordinary] == ["memory-0", "memory-1"]
    assert [hit.memory.id for hit in diverse] == ["memory-0", "memory-2"]


def test_session_diversity_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_SESSION_DIVERSITY", raising=False)
    monkeypatch.setenv("MEMORY_RESULT_WINDOW", "0")
    monkeypatch.setenv("MEMORY_SEMANTIC_BRIDGE_SEED_K", "0")
    embedder = FakeEmbedder()
    store = RawMemoryStore(
        tmp_path / "memory.sqlite3", embedding_identity=embedder.identity
    )
    service = main.StructMemService(embedder=embedder, store=store)
    service.add(AddRequest(
        request_id="req-a", user_id="user-1", session_id="session-a",
        messages=[
            {"role": "user", "content": "First evidence."},
            {"role": "user", "content": "Second evidence."},
        ],
    ))
    service.add(AddRequest(
        request_id="req-b", user_id="user-1", session_id="session-b",
        messages=[{"role": "user", "content": "Third evidence."}],
    ))
    request = SearchRequest(query="unmatched", user_id="user-1", top_k=2)
    memories = store.list_user("user-1")
    by_content = {memory.content: memory.id for memory in memories}
    assert [item.id for item in service.search(request).data] == [
        by_content["First evidence."], by_content["Second evidence."]
    ]

    monkeypatch.setenv("MEMORY_SESSION_DIVERSITY", "true")
    enabled = main.StructMemService(embedder=embedder, store=store)
    assert [item.id for item in enabled.search(request).data] == [
        by_content["First evidence."], by_content["Third evidence."]
    ]


def test_session_rank_decay_prefers_coherent_session_evidence():
    memories = [
        _memory(0, "session-a", "first", [1.0, 0.0]),
        _memory(1, "session-a", "second", [0.99, 0.1]),
        _memory(2, "session-b", "third", [0.8, 0.6]),
        _memory(3, "session-c", "fourth", [0.7, 0.7]),
    ]
    common = {
        "memories": memories,
        "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
        "query_text": "unmatched",
        "top_k": 3,
        "lexical_weight": 0.0,
        "rrf_k": 60,
        "result_window": 0,
        "result_window_seed_k": 20,
        "semantic_bridge_seed_k": 0,
        "semantic_bridge_min_similarity": 0.72,
        "session_diversity": True,
    }
    hits = hybrid_search(**common, session_rank_decay=0.5)
    assert [hit.memory.id for hit in hits] == ["memory-0", "memory-2", "memory-3"]


def test_session_anchor_replaces_only_a_repeated_slot_with_earliest_user():
    memories = [
        replace(_memory(0, "session-a", "assistant match", [1.0, 0.0]), role="assistant"),
        replace(_memory(1, "session-a", "assistant repeat", [0.99, 0.1]), role="assistant"),
        _memory(2, "session-a", "user anchor", [0.0, 1.0]),
        _memory(3, "session-b", "other", [0.8, 0.6]),
    ]
    common = {
        "memories": memories,
        "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
        "query_text": "unmatched",
        "top_k": 2,
        "lexical_weight": 0.0,
        "rrf_k": 60,
        "result_window": 0,
        "result_window_seed_k": 20,
        "semantic_bridge_seed_k": 0,
        "semantic_bridge_min_similarity": 0.72,
        "fusion": "minmax",
    }
    baseline = hybrid_search(**common, session_anchor_seed_k=0)
    anchored = hybrid_search(**common, session_anchor_seed_k=2)
    assert [hit.memory.id for hit in baseline] == ["memory-0", "memory-1"]
    assert [hit.memory.id for hit in anchored] == ["memory-0", "memory-2"]
    assert {hit.memory.session_id for hit in anchored} == {"session-a"}
    assert len(anchored) == 2


def test_session_anchor_skips_already_returned_user_and_preserves_disabled_order():
    memories = [
        _memory(0, "session-a", "user first", [1.0, 0.0]),
        replace(_memory(1, "session-a", "assistant match", [0.99, 0.1]), role="assistant"),
        replace(_memory(2, "session-a", "assistant repeat", [0.98, 0.2]), role="assistant"),
        _memory(3, "session-a", "later user", [0.0, 1.0]),
    ]
    common = {
        "memories": memories,
        "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
        "query_text": "unmatched",
        "top_k": 2,
        "lexical_weight": 0.0,
        "rrf_k": 60,
        "result_window": 0,
        "result_window_seed_k": 20,
        "semantic_bridge_seed_k": 0,
        "semantic_bridge_min_similarity": 0.72,
        "fusion": "minmax",
    }
    disabled = hybrid_search(**common, session_anchor_seed_k=0)
    enabled = hybrid_search(**common, session_anchor_seed_k=2)
    assert [hit.memory.id for hit in disabled] == ["memory-0", "memory-1"]
    assert [hit.memory.id for hit in enabled] == ["memory-0", "memory-1"]


def test_session_repeat_penalty_is_optional_soft_reranking():
    memories = [
        _memory(0, "session-a", "a0", [1.0, 0.0]),
        _memory(1, "session-a", "a1", [1.0, 0.0]),
        _memory(2, "session-a", "a2", [1.0, 0.0]),
        _memory(3, "session-b", "b0", [1.0, 0.0]),
        _memory(4, "session-c", "c0", [1.0, 0.0]),
    ]
    common = {
        "memories": memories,
        "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
        "query_text": "unmatched",
        "top_k": 4,
        "lexical_weight": 0.0,
        "rrf_k": 60,
        "fusion": "minmax",
        "result_window": 0,
        "result_window_seed_k": 20,
        "semantic_bridge_seed_k": 0,
        "semantic_bridge_min_similarity": 0.72,
    }
    omitted = hybrid_search(**common)
    disabled = hybrid_search(**common, session_repeat_penalty=0.0)
    penalized = hybrid_search(**common, session_repeat_penalty=0.1)
    assert [hit.memory.id for hit in omitted] == [hit.memory.id for hit in disabled]
    assert [hit.score for hit in omitted] == [hit.score for hit in disabled]
    assert [hit.memory.id for hit in penalized] == [
        "memory-0", "memory-3", "memory-4", "memory-1"
    ]
    assert "memory-2" not in [hit.memory.id for hit in penalized]
    assert len({hit.memory.session_id for hit in penalized}) == 3
    assert [hit.score for hit in penalized] == sorted(
        (hit.score for hit in penalized), reverse=True
    )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_session_repeat_penalty_rejects_invalid_values(value):
    memory = _memory(0, "session-a", "only", [1.0, 0.0])
    with pytest.raises(ValueError, match="session_repeat_penalty"):
        hybrid_search(
            [memory],
            query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
            query_text="only",
            top_k=1,
            lexical_weight=0.0,
            rrf_k=60,
            result_window=0,
            result_window_seed_k=20,
            semantic_bridge_seed_k=0,
            semantic_bridge_min_similarity=0.72,
            session_repeat_penalty=value,
        )


def test_session_repeat_penalty_is_passed_from_service_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_SESSION_REPEAT_PENALTY", "0.1")
    monkeypatch.setenv("MEMORY_RESULT_WINDOW", "0")
    monkeypatch.setenv("MEMORY_SEMANTIC_BRIDGE_SEED_K", "0")
    embedder = FakeEmbedder()
    store = RawMemoryStore(tmp_path / "memory.sqlite3", embedding_identity=embedder.identity)
    service = main.StructMemService(embedder=embedder, store=store)
    assert service.session_repeat_penalty == pytest.approx(0.1)
    service.add(AddRequest(
        request_id="req-repeat", user_id="user-1", session_id="session-a",
        messages=[{"role": "user", "content": "evidence"}],
    ))
    captured = {}

    def capture(_memories, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(main, "hybrid_search", capture)
    service.search(SearchRequest(query="evidence", user_id="user-1", top_k=1))
    assert captured["session_repeat_penalty"] == pytest.approx(0.1)


def test_session_anchor_runs_after_window_and_semantic_expansion():
    anchor = replace(
        _memory(0, "session-a", "first user evidence", [0.0, 1.0]),
        media=(MediaPart(
            type="image_url", image_url="data:image/png;base64,AAAA",
            caption="Original anchor caption",
        ),),
    )
    memories = [
        anchor,
        replace(_memory(1, "session-a", "assistant one", [0.8, 0.6]), role="assistant"),
        replace(_memory(2, "session-a", "assistant two", [0.98, 0.2]), role="assistant"),
        replace(_memory(3, "session-a", "direct match", [1.0, 0.0]), role="assistant"),
        _memory(4, "session-b", "semantic bridge", [0.99, 0.1]),
        _memory(5, "session-c", "unrelated", [-1.0, 0.0]),
    ]
    common = {
        "memories": memories,
        "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
        "query_text": "unmatched",
        "top_k": 3,
        "lexical_weight": 0.0,
        "rrf_k": 60,
        "fusion": "minmax",
        "result_window": 1,
        "result_window_seed_k": 1,
        "semantic_bridge_seed_k": 1,
        "semantic_bridge_min_similarity": 0.9,
    }
    expanded = hybrid_search(**common, session_anchor_seed_k=0)
    anchored = hybrid_search(**common, session_anchor_seed_k=1)
    assert [hit.memory.id for hit in expanded] == [
        "memory-3", "memory-2", "memory-4"
    ]
    assert [hit.memory.id for hit in anchored] == [
        "memory-3", "memory-0", "memory-4"
    ]
    assert len(anchored) == 3
    assert len({hit.memory.id for hit in anchored}) == 3
    assert {hit.memory.session_id for hit in anchored} == {"session-a", "session-b"}
    assert anchored[1].memory.content == "first user evidence"
    assert anchored[1].memory.media[0].caption == "Original anchor caption"
    assert [hit.score for hit in anchored] == pytest.approx(
        [1 / 61, 1 / 62, 1 / 63]
    )
