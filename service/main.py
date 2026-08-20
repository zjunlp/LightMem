from __future__ import annotations

import hmac
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, status
from openai import OpenAI

from .retrieval import hybrid_search, retrieval_query
from .schemas import AddRequest, AddResponse, SearchItem, SearchRequest, SearchResponse
from .store import ConflictingRequestError, RawMemoryStore, embedding_text


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value else fallback


_default_data_dir = "/data" if os.access("/data", os.W_OK) else "/tmp/structmem-data"
DATA_DIR = Path(_env("MEMORY_DATA_DIR", _default_data_dir) or _default_data_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Embedder(Protocol):
    identity: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self) -> None:
        api_key = _env("MEMORY_EMBEDDING_API_KEY", _env("OPENAI_API_KEY"))
        base_url = _env(
            "MEMORY_EMBEDDING_BASE_URL",
            _env(
                "MEMORY_BASE_URL", _env("OPENAI_API_BASE", "https://api.openai.com/v1")
            ),
        )
        self.model = (
            _env(
                "MEMORY_OPENAI_EMBEDDING_MODEL",
                _env("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
            or "text-embedding-3-small"
        )
        self.dimensions = int(_env("MEMORY_EMBEDDING_DIMS", "1536") or "1536")
        self.identity = f"openai:{self.model}:{self.dimensions}:content-v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=[str(text).replace("\n", " ") for text in inputs],
            model=self.model,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = (
            _env(
                "MEMORY_LOCAL_EMBEDDING_MODEL",
                _env("MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            )
            or "BAAI/bge-small-en-v1.5"
        )
        self.revision = (
            _env(
                "MEMORY_LOCAL_EMBEDDING_REVISION",
                "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            )
            or "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
        )
        configured_prefix = os.getenv("MEMORY_EMBEDDING_QUERY_PREFIX")
        if configured_prefix is None:
            configured_prefix = (
                "Represent this sentence for searching relevant passages: "
                if "bge" in self.model_name.casefold()
                and "-en-" in self.model_name.casefold()
                else ""
            )
        self.query_prefix = configured_prefix
        self.identity = (
            f"sentence-transformers:{self.model_name}@{self.revision}:"
            f"query={self.query_prefix}:content-v1"
        )
        device = _env("MEMORY_EMBEDDING_DEVICE")
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            revision=self.revision,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]


def _build_embedder() -> Embedder:
    provider = (
        _env("MEMORY_EMBEDDING_PROVIDER", "sentence_transformers") or ""
    ).casefold()
    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        return SentenceTransformerEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    raise ValueError(
        "MEMORY_EMBEDDING_PROVIDER must be 'sentence_transformers' or 'openai'"
    )


def _normalized_matrix(vectors: object, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1 and expected_rows == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
        raise ValueError("Embedder returned an invalid vector matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embedder returned non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedder returned a zero-length vector")
    return matrix / norms


class StructMemService:
    """AML adapter using lossless raw evidence and hybrid retrieval."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        store: RawMemoryStore | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self.embedder = embedder or _build_embedder()
        self.store = store or RawMemoryStore(
            DATA_DIR / "raw_memory.sqlite3", embedding_identity=self.embedder.identity
        )
        self.lexical_weight = float(_env("MEMORY_LEXICAL_WEIGHT", "0.5") or "0.5")
        self.rrf_k = int(_env("MEMORY_RRF_K", "60") or "60")
        self.result_window = int(_env("MEMORY_RESULT_WINDOW", "1") or "1")
        self.result_window_seed_k = int(
            _env("MEMORY_RESULT_WINDOW_SEED_K", "20") or "20"
        )
        self.semantic_bridge_seed_k = int(
            _env("MEMORY_SEMANTIC_BRIDGE_SEED_K", "10") or "10"
        )
        self.semantic_bridge_min_similarity = float(
            _env("MEMORY_SEMANTIC_BRIDGE_MIN_SIMILARITY", "0.72") or "0.72"
        )
        if self.lexical_weight < 0:
            raise ValueError("MEMORY_LEXICAL_WEIGHT cannot be negative")
        if self.rrf_k < 1:
            raise ValueError("MEMORY_RRF_K must be positive")
        if self.result_window < 0 or self.result_window_seed_k < 1:
            raise ValueError("Invalid result-window configuration")
        if self.semantic_bridge_seed_k < 0:
            raise ValueError("MEMORY_SEMANTIC_BRIDGE_SEED_K cannot be negative")
        if not 0 <= self.semantic_bridge_min_similarity <= 1:
            raise ValueError(
                "MEMORY_SEMANTIC_BRIDGE_MIN_SIMILARITY must be between 0 and 1"
            )

    def add(self, request: AddRequest) -> AddResponse:
        with self.lock:
            try:
                existing = self.store.get(request)
            except ConflictingRequestError as error:
                raise HTTPException(
                    status_code=409, detail={"reason": str(error)}
                ) from error
            if existing is not None:
                return existing

            documents = [embedding_text(message) for message in request.messages]
            vectors = _normalized_matrix(
                self.embedder.embed_documents(documents), len(documents)
            )
            response = AddResponse(
                success=True,
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            try:
                self.store.add(request, response, vectors)
            except ConflictingRequestError as error:
                raise HTTPException(
                    status_code=409, detail={"reason": str(error)}
                ) from error
            return response

    def search(self, request: SearchRequest) -> SearchResponse:
        with self.lock:
            memories = self.store.list_user(request.user_id)
            if not memories:
                return SearchResponse(data=[])

            query_text = retrieval_query(request.query, request.options)
            query_vector = _normalized_matrix(self.embedder.embed_query(query_text), 1)[
                0
            ]
            hits = hybrid_search(
                memories,
                query_vector=query_vector,
                query_text=query_text,
                top_k=request.top_k,
                lexical_weight=self.lexical_weight,
                rrf_k=self.rrf_k,
                result_window=self.result_window,
                result_window_seed_k=self.result_window_seed_k,
                semantic_bridge_seed_k=self.semantic_bridge_seed_k,
                semantic_bridge_min_similarity=self.semantic_bridge_min_similarity,
            )
            return SearchResponse(
                data=[
                    SearchItem(
                        id=hit.memory.id,
                        content=hit.memory.render(),
                        score=hit.score,
                        created_at=hit.memory.event_time,
                    )
                    for hit in hits
                ]
            )


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
    if not any(
        candidate and hmac.compare_digest(candidate, settings_api_key)
        for candidate in candidates
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "invalid API key"},
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = StructMemService()
    yield


app = FastAPI(
    title="StructMem Raw Hybrid AML Adapter", version="0.3.0", lifespan=lifespan
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.3.0"}


@app.post("/add", response_model=AddResponse, dependencies=[Depends(verify_api_key)])
def add_memory(request: AddRequest) -> AddResponse:
    if service is None:
        raise HTTPException(status_code=503, detail={"reason": "service is starting"})
    return service.add(request)


@app.post(
    "/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)]
)
def search_memory(request: SearchRequest) -> SearchResponse:
    if service is None:
        raise HTTPException(status_code=503, detail={"reason": "service is starting"})
    return service.search(request)
