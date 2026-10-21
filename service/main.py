from __future__ import annotations

import hmac
import math
import os
import re
import threading
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, status
from openai import OpenAI

from .retrieval import CJK_PATTERN, hybrid_search, retrieval_query
from .schemas import AddRequest, AddResponse, SearchItem, SearchRequest, SearchResponse
from .store import ConflictingRequestError, RawMemoryStore, embedding_text
from .background import BackgroundIndex


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value else fallback


def _embedding_timestamps_enabled() -> bool:
    return (_env("MEMORY_EMBED_TIMESTAMPS", "false") or "false").casefold() in {
        "1", "true", "yes"
    }


def _embedding_speakers_enabled() -> bool:
    return (_env("MEMORY_EMBED_SPEAKERS", "true") or "true").casefold() in {
        "1", "true", "yes"
    }


def _multi_window_enabled() -> bool:
    return (_env("MEMORY_MULTI_WINDOW", "false") or "false").casefold() in {
        "1", "true", "yes"
    }


def _token_windows(tokenizer, text: str) -> list[str]:
    token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if not token_ids:
        return [text]
    chunks = []
    for start in range(0, len(token_ids), 320):
        ids = token_ids[start:start + 384]
        chunks.append(tokenizer.decode(ids, skip_special_tokens=True))
        if start + 384 >= len(token_ids):
            break
    return chunks


def _embedding_document_identity() -> str:
    """Version the persisted vector identity with the document text shape."""
    # v2 records the current body-plus-caption document shape. The version
    # prevents an older body-only database from being reused silently.
    timestamp_mode = "date-prefix-v2" if _embedding_timestamps_enabled() else "raw-v2"
    speaker_mode = "speaker-v1" if _embedding_speakers_enabled() else "raw-v1"
    window_mode = "window-384x64-v1" if _multi_window_enabled() else "single-v1"
    return f"document={speaker_mode}+{timestamp_mode}+{window_mode}"


def _timestamp_millis(value: int | str | None) -> int | None:
    if isinstance(value, int):
        if not -(2**63) <= value <= 2**63 - 1:
            return None
        try:
            datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = re.sub(r"\s*\([A-Za-z]{3}\)", "", value).strip()
    for pattern in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(normalized, pattern).replace(
                tzinfo=timezone.utc
            ).timestamp() * 1000)
        except (OSError, OverflowError, ValueError):
            pass
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return int(parsed.timestamp() * 1000)
    except (OSError, OverflowError, ValueError):
        return None


_default_data_dir = "/data" if os.access("/data", os.W_OK) else "/tmp/structmem-data"
DATA_DIR = Path(_env("MEMORY_DATA_DIR", _default_data_dir) or _default_data_dir)
DATA_DIR.mkdir(parents=True, exist_ok=True)
_TEMPORAL_QUERY = re.compile(
    r"\b(?:when|how long|how many (?:days|weeks|months|years)|before|after|ago|"
    r"earliest|latest|first|last|today|yesterday|tomorrow)\b",
    re.IGNORECASE,
)
_FACT_PROMPT = """Extract concise retrieval facts from this conversation chunk.
Only state explicit facts from the source. Preserve names, dates, quantities,
preferences, changes, and outcomes. Do not infer or answer unseen questions.
Return at most 12 bullet points.

Conversation:
{conversation}"""


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
        self.identity = (
            f"openai:{self.model}:{self.dimensions}:content-v1:"
            f"{_embedding_document_identity()}"
        )
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

        self.batch_size = int(_env("MEMORY_EMBEDDING_BATCH_SIZE", "64"))
        max_concurrency = int(_env("MEMORY_EMBEDDING_MAX_CONCURRENCY", "2"))
        if self.batch_size < 1 or max_concurrency < 1:
            raise ValueError("Embedding batch size and max concurrency must be positive")
        self._inference_slots = threading.BoundedSemaphore(max_concurrency)
        self.model_name = (
            _env(
                "MEMORY_LOCAL_EMBEDDING_MODEL",
                _env("MEMORY_EMBEDDING_MODEL", "BAAI/bge-m3"),
            )
            or "BAAI/bge-m3"
        )
        default_revision = {
            "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
            "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        }.get(self.model_name)
        self.revision = _env(
            "MEMORY_LOCAL_EMBEDDING_REVISION", default_revision
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
            f"query={self.query_prefix}:content-v1:{_embedding_document_identity()}"
        )
        device = _env("MEMORY_EMBEDDING_DEVICE")
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            revision=self.revision,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        with self._inference_slots:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def window_documents(self, text: str) -> list[str]:
        return _token_windows(self.model.tokenizer, text)


class EnsembleSentenceTransformerEmbedder:
    """Average cosine similarity by concatenating normalized model embeddings."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        raw_models = _env("MEMORY_ENSEMBLE_EMBEDDING_MODELS", "") or ""
        self.model_names = [name.strip() for name in raw_models.split(",") if name.strip()]
        if len(self.model_names) < 2:
            raise ValueError("MEMORY_ENSEMBLE_EMBEDDING_MODELS needs at least two models")
        self.revision = _env("MEMORY_LOCAL_EMBEDDING_REVISION", "") or None
        self.device = _env("MEMORY_EMBEDDING_DEVICE")
        self.models = [
            SentenceTransformer(name, device=self.device, revision=self.revision)
            for name in self.model_names
        ]
        configured_prefix = os.getenv("MEMORY_EMBEDDING_QUERY_PREFIX")
        self.query_prefix = configured_prefix if configured_prefix is not None else (
            "Represent this sentence for searching relevant passages: "
        )
        self.identity = (
            "ensemble:" + "|".join(self.model_names)
            + f":revision={self.revision}:query={self.query_prefix}:"
            f"content-v1:{_embedding_document_identity()}"
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        matrices = [
            np.asarray(model.encode(
                texts, batch_size=64, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ), dtype=np.float32)
            for model in self.models
        ]
        combined = np.concatenate(matrices, axis=1) / math.sqrt(len(matrices))
        return combined.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def window_documents(self, text: str) -> list[str]:
        return _token_windows(self.models[0].tokenizer, text)


def _build_embedder() -> Embedder:
    provider = (
        _env("MEMORY_EMBEDDING_PROVIDER", "sentence_transformers") or ""
    ).casefold()
    if provider in {"sentence_transformers", "sentence-transformers", "local"}:
        return SentenceTransformerEmbedder()
    if provider in {"ensemble", "sentence_transformers_ensemble"}:
        return EnsembleSentenceTransformerEmbedder()
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
        self.multi_window = _multi_window_enabled()
        self.embedder = embedder or _build_embedder()
        if self.multi_window and not callable(getattr(self.embedder, "window_documents", None)):
            raise ValueError("MEMORY_MULTI_WINDOW requires a tokenizer-backed local embedder")
        self.store = store or RawMemoryStore(
            DATA_DIR / "raw_memory.sqlite3", embedding_identity=self.embedder.identity
        )
        self.index = (
            BackgroundIndex(
                self.store,
                idle_seconds=float(_env("MEMORY_INDEX_IDLE_SECONDS", "0.25")),
                max_bytes=int(_env("MEMORY_INDEX_CACHE_MB", "512")) * 1024 * 1024,
                max_users=int(_env("MEMORY_INDEX_MAX_USERS", "32")),
                ttl_seconds=float(_env("MEMORY_INDEX_TTL_SECONDS", "300")),
            )
            if (_env("MEMORY_BACKGROUND_INDEX", "true") or "true").casefold()
            in {"1", "true", "yes"} else None
        )
        self.lexical_weight = float(_env("MEMORY_LEXICAL_WEIGHT", "0.1") or "0.1")
        self.cjk_lexical_weight = float(
            _env("MEMORY_CJK_LEXICAL_WEIGHT", "0.5") or "0.5"
        )
        self.cjk_corpus_ratio_threshold = float(
            _env("MEMORY_CJK_CORPUS_RATIO_THRESHOLD", "0.05") or "0.05"
        )
        self.rrf_k = int(_env("MEMORY_RRF_K", "60") or "60")
        self.fusion = _env("MEMORY_FUSION", "rrf" if self.multi_window else "minmax")
        self.session_diversity = (
            _env("MEMORY_SESSION_DIVERSITY", "false") or "false"
        ).casefold() in {"1", "true", "yes"}
        self.session_rank_decay = float(_env("MEMORY_SESSION_RANK_DECAY", "0.5") or "0.5")
        self.session_recency_weight = float(
            _env("MEMORY_SESSION_RECENCY_WEIGHT", "0") or "0"
        )
        self.session_anchor_seed_k = int(
            _env("MEMORY_SESSION_ANCHOR_SEED_K", "0") or "0"
        )
        self.session_repeat_penalty = float(
            _env("MEMORY_SESSION_REPEAT_PENALTY", "0") or "0"
        )
        self.embed_timestamps = _embedding_timestamps_enabled()
        self.session_facts = (
            _env("MEMORY_SESSION_FACTS", "false") or "false"
        ).casefold() in {"1", "true", "yes"}
        self.fact_client: OpenAI | None = None
        self.fact_model = _env("MEMORY_FACTS_MODEL", "deepseek-chat") or "deepseek-chat"
        if self.session_facts:
            fact_key = _env("MEMORY_FACTS_API_KEY", _env("OPENAI_API_KEY"))
            if not fact_key:
                raise ValueError("MEMORY_FACTS_API_KEY is required when MEMORY_SESSION_FACTS=true")
            self.fact_client = OpenAI(
                api_key=fact_key,
                base_url=_env("MEMORY_FACTS_BASE_URL", _env("MEMORY_BASE_URL", "https://api.deepseek.com")),
            )
        # Window expansion duplicates locally related evidence and displaces direct
        # matches when the platform requests a small fixed top-k.
        self.result_window = int(_env("MEMORY_RESULT_WINDOW", "0") or "0")
        self.result_window_seed_k = int(
            _env("MEMORY_RESULT_WINDOW_SEED_K", "20") or "20"
        )
        self.semantic_bridge_seed_k = int(
            _env("MEMORY_SEMANTIC_BRIDGE_SEED_K", "0") or "0"
        )
        self.semantic_bridge_min_similarity = float(
            _env("MEMORY_SEMANTIC_BRIDGE_MIN_SIMILARITY", "0.72") or "0.72"
        )
        if self.lexical_weight < 0 or self.cjk_lexical_weight < 0:
            raise ValueError("Lexical weights cannot be negative")
        if not 0 <= self.cjk_corpus_ratio_threshold <= 1:
            raise ValueError("MEMORY_CJK_CORPUS_RATIO_THRESHOLD must be between 0 and 1")
        if self.rrf_k < 1:
            raise ValueError("MEMORY_RRF_K must be positive")
        if not 0 <= self.session_rank_decay <= 1:
            raise ValueError("MEMORY_SESSION_RANK_DECAY must be between 0 and 1")
        if not math.isfinite(self.session_recency_weight) or self.session_recency_weight < 0:
            raise ValueError("MEMORY_SESSION_RECENCY_WEIGHT must be finite and non-negative")
        if self.session_anchor_seed_k < 0:
            raise ValueError("MEMORY_SESSION_ANCHOR_SEED_K cannot be negative")
        if not math.isfinite(self.session_repeat_penalty) or self.session_repeat_penalty < 0:
            raise ValueError(
                "MEMORY_SESSION_REPEAT_PENALTY must be finite and non-negative"
            )
        if self.fusion not in {"rrf", "minmax"}:
            raise ValueError("MEMORY_FUSION must be 'rrf' or 'minmax'")
        if self.result_window < 0 or self.result_window_seed_k < 1:
            raise ValueError("Invalid result-window configuration")
        if self.semantic_bridge_seed_k < 0:
            raise ValueError("MEMORY_SEMANTIC_BRIDGE_SEED_K cannot be negative")
        if not 0 <= self.semantic_bridge_min_similarity <= 1:
            raise ValueError(
                "MEMORY_SEMANTIC_BRIDGE_MIN_SIMILARITY must be between 0 and 1"
            )

    def add(self, request: AddRequest) -> AddResponse:
        with self.index.foreground() if self.index else nullcontext():
            response = self._add(request)
            if self.index:
                self.index.schedule(request.user_id)
            return response

    def _add(self, request: AddRequest) -> AddResponse:
        # The benchmark may send human-readable timestamps; the store uses epoch ms.
        normalized_messages = [
            message.model_copy(update={"timestamp": _timestamp_millis(message.timestamp)})
            for message in request.messages
        ]
        request = request.model_copy(update={"messages": normalized_messages})
        try:
            existing = self.store.get(request)
        except ConflictingRequestError as error:
            raise HTTPException(status_code=409, detail={"reason": str(error)}) from error
        if existing is not None:
            return existing
        documents = [self._embedding_document(message) for message in request.messages]
        indexed_positions = [index for index, document in enumerate(documents) if document]
        vectors: list[np.ndarray | None] = [None] * len(request.messages)
        chunk_vectors: list[list[np.ndarray] | None] | None = (
            [None] * len(request.messages) if self.multi_window else None
        )
        if indexed_positions:
            if self.multi_window:
                windows_by_position = {
                    index: self.embedder.window_documents(documents[index])
                    for index in indexed_positions
                }
            else:
                windows_by_position = {
                    index: [documents[index]] for index in indexed_positions
                }
            flattened = [
                window for index in indexed_positions
                for window in windows_by_position[index]
            ]
            indexed_vectors = _normalized_matrix(
                self.embedder.embed_documents(flattened), len(flattened),
            )
            offset = 0
            for index in indexed_positions:
                count = len(windows_by_position[index])
                vectors[index] = indexed_vectors[offset]
                if chunk_vectors is not None:
                    chunk_vectors[index] = list(indexed_vectors[offset + 1:offset + count])
                offset += count
        response = AddResponse(success=True, request_id=request.request_id, user_id=request.user_id, session_id=request.session_id)
        try:
            self.store.add(request, response, vectors, chunk_vectors=chunk_vectors)
        except ConflictingRequestError as error:
            raise HTTPException(status_code=409, detail={"reason": str(error)}) from error
        if self.session_facts:
            # Facts improve recall but raw source evidence is the durable API contract.
            # A transient model failure must not turn a successful Add into a failure.
            try:
                facts = self._session_facts(request)
                if facts:
                    fact_vectors = _normalized_matrix(
                        self.embedder.embed_documents(facts), len(facts)
                    )
                    self.store.add_derived(request, facts, fact_vectors)
            except Exception:
                pass
        return response

    def _embedding_document(self, message) -> str:
        content = embedding_text(message)
        already_labeled = bool(re.match(
            rf"^\s*{re.escape(message.role)}\s*:", content, re.IGNORECASE
        ))
        if _embedding_speakers_enabled() and content and message.role and not already_labeled:
            content = (
                f"[{message.role}] {content}" if self.multi_window
                else f"{message.role}: {content}"
            )
        if not self.embed_timestamps or message.timestamp is None:
            return content
        timestamp = _timestamp_millis(message.timestamp)
        if timestamp is None:
            return content
        try:
            rendered = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return content
        return f"[Session date: {rendered:%Y-%m-%d}] {content}" if content else ""

    def _session_facts(self, request: AddRequest) -> list[str]:
        if self.fact_client is None:
            return []
        conversation = "\n".join(
            f"{message.role}: {message.content}" for message in request.messages
        )
        completion = self.fact_client.chat.completions.create(
            model=self.fact_model,
            temperature=0,
            max_tokens=360,
            messages=[{"role": "user", "content": _FACT_PROMPT.format(conversation=conversation)}],
        )
        raw = str(completion.choices[0].message.content or "")
        return [line.lstrip("-* ").strip() for line in raw.splitlines() if line.strip()]

    def search(self, request: SearchRequest) -> SearchResponse:
        with self.index.foreground() if self.index else nullcontext():
            return self._search(request)

    def _search(self, request: SearchRequest) -> SearchResponse:
        prepared = self.index.get(request.user_id) if self.index else None
        memories = (
            list(prepared.memories) if prepared is not None
            else self.store.list_user(request.user_id)
        )
        if _TEMPORAL_QUERY.search(retrieval_query(request.query, request.options)):
            if any(memory.derived for memory in memories):
                prepared = None
            memories = [memory for memory in memories if not memory.derived]
        if not memories:
            return SearchResponse(data=[])
        query_text = retrieval_query(request.query, request.options)
        query_vector = _normalized_matrix(self.embedder.embed_query(query_text), 1)[0]
        cjk_documents = prepared.cjk_count if prepared is not None else sum(
            bool(CJK_PATTERN.search("\n".join(
                [memory.content, *(part.caption or "" for part in memory.media)]
            )))
            for memory in memories
        )
        corpus_cjk_ratio = cjk_documents / len(memories)
        lexical_weight = self.lexical_weight
        if (
            corpus_cjk_ratio >= self.cjk_corpus_ratio_threshold
            and CJK_PATTERN.search(query_text)
        ):
            lexical_weight = self.cjk_lexical_weight
        hits = hybrid_search(memories, query_vector=query_vector, query_text=query_text,
            top_k=request.top_k, lexical_weight=lexical_weight, rrf_k=self.rrf_k,
            fusion=self.fusion, result_window=self.result_window,
            result_window_seed_k=self.result_window_seed_k,
            semantic_bridge_seed_k=self.semantic_bridge_seed_k,
            semantic_bridge_min_similarity=self.semantic_bridge_min_similarity,
            session_diversity=self.session_diversity,
            session_rank_decay=self.session_rank_decay,
            session_recency_weight=self.session_recency_weight,
            session_anchor_seed_k=self.session_anchor_seed_k,
            session_repeat_penalty=self.session_repeat_penalty,
            prepared=prepared)
        return SearchResponse(data=[
            SearchItem(
                id=hit.memory.id, content=hit.memory.render(), media=list(hit.memory.media),
                score=hit.score, created_at=hit.memory.event_time,
            )
            for hit in hits
        ])


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
    if service.index:
        service.index.start()
    try:
        yield
    finally:
        if service.index:
            service.index.close()


app = FastAPI(
    title="StructMem Raw Hybrid AML Adapter", version="0.3.2", lifespan=lifespan
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.3.2"}


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


# Official leaderboard paths; retain /add and /search for backwards compatibility.
app.add_api_route(
    "/v1/memories/add",
    add_memory,
    methods=["POST"],
    response_model=AddResponse,
    dependencies=[Depends(verify_api_key)],
)
app.add_api_route(
    "/v1/memories/search",
    search_memory,
    methods=["POST"],
    response_model=SearchResponse,
    dependencies=[Depends(verify_api_key)],
)
