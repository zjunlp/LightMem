from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .schemas import AddRequest, AddResponse, SearchItem, SearchRequest, SearchResponse


_default_data_dir = "/data" if os.access("/data", os.W_OK) else "/tmp/structmem-data"
DATA_DIR = Path(os.getenv("MEMORY_DATA_DIR", _default_data_dir))
DATA_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_DB = DATA_DIR / "request_registry.sqlite3"


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value else fallback


def _build_config() -> dict:
    llm_key = _env("MEMORY_LLM_API_KEY", _env("OPENAI_API_KEY"))
    embedding_key = _env("MEMORY_EMBEDDING_API_KEY", llm_key)
    default_base_url = _env("MEMORY_BASE_URL", _env("OPENAI_API_BASE", "https://api.openai.com/v1"))
    llm_base_url = _env("MEMORY_LLM_BASE_URL", default_base_url)
    embedding_base_url = _env("MEMORY_EMBEDDING_BASE_URL", default_base_url)
    llm_model = _env("MEMORY_LLM_MODEL", "gpt-5.4-mini")
    embedding_model = _env("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dims = int(_env("MEMORY_EMBEDDING_DIMS", "1536"))

    return {
        "pre_compress": False,
        "topic_segment": False,
        "messages_use": "hybrid",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": llm_model,
                "api_key": llm_key,
                "openai_base_url": llm_base_url,
                "max_tokens": 2000,
            },
        },
        "extract_threshold": 0.0,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "openai",
            "configs": {
                "model": embedding_model,
                "api_key": embedding_key,
                "openai_base_url": embedding_base_url,
                "embedding_dims": embedding_dims,
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": "memory_entries",
                "embedding_model_dims": embedding_dims,
                "path": str(DATA_DIR / "qdrant-entries"),
                "on_disk": True,
            },
        },
        "summary_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": "memory_summaries",
                "embedding_model_dims": embedding_dims,
                "path": str(DATA_DIR / "qdrant-summaries"),
                "on_disk": True,
            },
        },
        "update": "offline",
        "extraction_mode": "event",
    }


class RequestRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS requests ("
                "request_id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, "
                "response_json TEXT NOT NULL)"
            )

    def get_or_conflict(self, request: AddRequest) -> AddResponse | None:
        payload_hash = hashlib.sha256(
            json.dumps(request.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT payload_hash, response_json FROM requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
        if row is None:
            return None
        if not hmac.compare_digest(row[0], payload_hash):
            raise HTTPException(status_code=409, detail={"reason": "request_id payload conflict"})
        return AddResponse.model_validate_json(row[1])

    def save(self, request: AddRequest, response: AddResponse) -> None:
        payload_hash = hashlib.sha256(
            json.dumps(request.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO requests(request_id, payload_hash, response_json) VALUES (?, ?, ?)",
                (request.request_id, payload_hash, response.model_dump_json()),
            )


def _to_lightmem_messages(request: AddRequest) -> list[dict]:
    now_ms = int(time.time() * 1000)
    messages = []
    for index, message in enumerate(request.messages):
        timestamp_ms = message.timestamp or now_ms + index
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
        role = message.role if message.role in {"user", "assistant"} else "user"
        messages.append(
            {
                "role": role,
                "content": message.content,
                "speaker_id": role,
                "speaker_name": role,
                "time_stamp": timestamp,
            }
        )
    return messages


class StructMemService:
    def __init__(self):
        import lightmem.memory.lightmem as lightmem_module
        from lightmem.memory.lightmem import LightMemory

        self.lock = threading.RLock()
        self.registry = RequestRegistry(REQUEST_DB)
        self.lightmem_module = lightmem_module
        self.memory = LightMemory.from_config(_build_config())
        self.summary_batch_entries = max(1, int(_env("MEMORY_SUMMARY_BATCH_ENTRIES", "100")))

    def _pending_entries(self, user_id: str) -> int:
        pending, _ = self.memory.embedding_retriever.scroll(
            scroll_filter={"user_id": user_id, "consolidated": False},
            limit=self.summary_batch_entries,
            with_payload=False,
            with_vectors=False,
        )
        return len(pending)

    def _summarize_user(self, user_id: str) -> None:
        # The upstream pointer is process-global. The service lock makes the
        # reset and consolidation scan atomic, while user_id scopes the data.
        self.lightmem_module.GLOBAL_LAST_SUMMARY_TIME = None
        self.memory.summarize(
            user_id=user_id,
            retrieval_scope="global",
            time_window=int(_env("MEMORY_SUMMARY_WINDOW", "3600")),
            top_k_seeds=int(_env("MEMORY_SUMMARY_TOP_K", "15")),
            process_all=True,
        )

    def add(self, request: AddRequest) -> AddResponse:
        with self.lock, self.registry.lock:
            existing = self.registry.get_or_conflict(request)
            if existing is not None:
                return existing

            add_result = self.memory.add_memory(
                _to_lightmem_messages(request),
                force_segment=True,
                force_extract=True,
                user_id=request.user_id,
            )
            if int(add_result.get("memory_entries", 0)) == 0:
                raise HTTPException(
                    status_code=502,
                    detail={"reason": "StructMem extraction produced no memory entries"},
                )

            if self._pending_entries(request.user_id) >= self.summary_batch_entries:
                self._summarize_user(request.user_id)
            response = AddResponse(
                success=True,
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            self.registry.save(request, response)
            return response

    def search(self, request: SearchRequest) -> SearchResponse:
        with self.lock:
            query_vector = self.memory.text_embedder.embed(request.query)
            user_filter = {"user_id": request.user_id}
            entries = self.memory.embedding_retriever.search(
                query_vector=query_vector,
                limit=request.top_k,
                filters=user_filter,
                return_full=True,
            )
            summaries = self.memory.summary_retriever.search(
                query_vector=query_vector,
                limit=request.top_k,
                filters=user_filter,
                return_full=True,
            )

            results = []
            for item in entries:
                payload = item.get("payload", {})
                content = str(payload.get("memory", "")).strip()
                if content:
                    results.append((item.get("score"), str(item["id"]), content, payload.get("time_stamp")))
            for item in summaries:
                payload = item.get("payload", {})
                content = str(payload.get("summary", "")).strip()
                if content:
                    results.append((item.get("score"), str(item["id"]), content, payload.get("created_at")))

            results.sort(key=lambda item: item[0] if item[0] is not None else -1.0, reverse=True)
            return SearchResponse(
                data=[
                    SearchItem(
                        id=item[1],
                        content=item[2],
                        score=item[0],
                        created_at=_parse_datetime(item[3]),
                    )
                    for item in results[: request.top_k]
                ]
            )


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


settings_api_key = _env("MEMORY_API_KEY")
service: StructMemService | None = None


def verify_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    if not settings_api_key:
        return
    candidates = [x_api_key]
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() in {"bearer", "token"}:
            candidates.append(value)
    if not any(candidate and hmac.compare_digest(candidate, settings_api_key) for candidate in candidates):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"reason": "invalid API key"})


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = StructMemService()
    yield


app = FastAPI(title="StructMem AML Adapter", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/add", response_model=AddResponse, dependencies=[Depends(verify_api_key)])
def add_memory(request: AddRequest) -> AddResponse:
    if service is None:
        raise HTTPException(status_code=503, detail={"reason": "service is starting"})
    return service.add(request)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
def search_memory(request: SearchRequest) -> SearchResponse:
    if service is None:
        raise HTTPException(status_code=503, detail={"reason": "service is starting"})
    return service.search(request)
