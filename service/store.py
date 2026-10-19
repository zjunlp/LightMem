from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .schemas import AddRequest, AddResponse, MediaPart, Message


class ConflictingRequestError(ValueError):
    pass


def _payload_hash(request: AddRequest) -> str:
    serialized = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _memory_id(request: AddRequest, message_index: int) -> str:
    identity = f"{request.user_id}\0{request.request_id}\0{message_index}"
    return "mem_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _fact_id(request: AddRequest, fact_index: int) -> str:
    identity = f"{request.user_id}\0{request.request_id}\0fact\0{fact_index}"
    return "fact_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _timestamp(value: int | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def embedding_text(message: Message) -> str:
    captions = [part.caption for part in message.media if part.caption]
    return "\n".join([message.content.strip(), *captions]).strip()


@dataclass(frozen=True)
class StoredMemory:
    sequence: int
    id: str
    request_id: str
    user_id: str
    session_id: str
    message_index: int
    role: str
    content: str
    source_timestamp: int | None
    created_at: datetime
    embedding: np.ndarray
    derived: bool = False
    media: tuple[MediaPart, ...] = ()
    embedding_chunks: tuple[np.ndarray, ...] = ()

    @property
    def event_time(self) -> datetime:
        return _timestamp(self.source_timestamp) or self.created_at

    def render(self) -> str:
        timestamp = _timestamp(self.source_timestamp)
        prefix = (
            f"[{timestamp.isoformat().replace('+00:00', 'Z')}] " if timestamp else ""
        )
        return f"{prefix}{self.role}: {self.content}"


class RawMemoryStore:
    def __init__(self, path: Path, *, embedding_identity: str) -> None:
        self.path = path
        self.embedding_identity = embedding_identity
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_timestamp INTEGER,
                    created_at TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    derived INTEGER NOT NULL DEFAULT 0,
                    media_json TEXT NOT NULL DEFAULT '[]',
                    indexed INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY (request_id) REFERENCES requests(request_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS message_chunks (
                    message_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    PRIMARY KEY (message_id, chunk_index),
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(user_id, session_id, sequence);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(messages)")
            }
            if "derived" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN derived INTEGER NOT NULL DEFAULT 0"
                )
            if "media_json" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN media_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "indexed" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN indexed INTEGER NOT NULL DEFAULT 1"
                )
            row = connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'embedding_identity'"
            ).fetchone()
            if row is None:
                has_messages = connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM messages)"
                ).fetchone()[0]
                if has_messages:
                    raise ValueError(
                        "Memory database has vectors without an embedding identity; "
                        "rebuild it before changing the embedding configuration"
                    )
                connection.execute(
                    "INSERT INTO store_metadata(key, value) VALUES ('embedding_identity', ?)",
                    (self.embedding_identity,),
                )
            elif row["value"] != self.embedding_identity:
                raise ValueError(
                    "Configured embedding model does not match the model used by this memory database"
                )

    def get(self, request: AddRequest) -> AddResponse | None:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload_hash, response_json FROM requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
        if row is None:
            return None
        if row["payload_hash"] != _payload_hash(request):
            raise ConflictingRequestError(
                "request_id already exists with a different payload"
            )
        return AddResponse.model_validate_json(row["response_json"])

    def add(
        self, request: AddRequest, response: AddResponse,
        vectors: np.ndarray | list[np.ndarray | None],
        chunk_vectors: list[list[np.ndarray] | None] | None = None,
    ) -> None:
        if isinstance(vectors, np.ndarray):
            prepared: list[np.ndarray | None] = [vectors[index] for index in range(len(vectors))]
        else:
            prepared = vectors
        if len(prepared) != len(request.messages):
            raise ValueError("Message and embedding counts do not match")
        if chunk_vectors is None:
            prepared_chunks: list[list[np.ndarray] | None] = [None] * len(prepared)
        else:
            prepared_chunks = chunk_vectors
        if len(prepared_chunks) != len(prepared):
            raise ValueError("Message and chunk embedding counts do not match")
        for vector in prepared:
            if vector is not None and np.asarray(vector).ndim != 1:
                raise ValueError("Each embedding must be one-dimensional")
        for base, chunks in zip(prepared, prepared_chunks, strict=True):
            if chunks and base is None:
                raise ValueError("Chunk embeddings require a base embedding")
            for vector in chunks or ():
                if np.asarray(vector).shape != np.asarray(base).shape:
                    raise ValueError("Chunk embedding dimension does not match base embedding")
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_hash FROM requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != _payload_hash(request):
                    raise ConflictingRequestError(
                        "request_id already exists with a different payload"
                    )
                return
            connection.execute(
                "INSERT INTO requests(request_id, payload_hash, response_json) VALUES (?, ?, ?)",
                (
                    request.request_id,
                    _payload_hash(request),
                    response.model_dump_json(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO messages(
                    id, request_id, user_id, session_id, message_index, role, content,
                    source_timestamp, created_at, embedding, dimension, derived, media_json, indexed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                [
                    (
                        _memory_id(request, index),
                        request.request_id,
                        request.user_id,
                        request.session_id,
                        index,
                        message.role,
                        message.content.strip(),
                        message.timestamp,
                        now,
                        (np.asarray(prepared[index], dtype=np.float32).tobytes()
                         if prepared[index] is not None else b""),
                        (int(np.asarray(prepared[index]).shape[0])
                         if prepared[index] is not None else 0),
                        json.dumps([part.model_dump() for part in message.media]),
                        int(prepared[index] is not None),
                    )
                    for index, message in enumerate(request.messages)
                ],
            )
            chunk_rows = [
                (
                    _memory_id(request, index), chunk_index,
                    np.asarray(vector, dtype=np.float32).tobytes(),
                    int(np.asarray(vector).shape[0]),
                )
                for index, chunks in enumerate(prepared_chunks)
                if chunks
                for chunk_index, vector in enumerate(chunks, 1)
            ]
            if chunk_rows:
                connection.executemany(
                    """
                    INSERT INTO message_chunks(message_id, chunk_index, embedding, dimension)
                    VALUES (?, ?, ?, ?)
                    """,
                    chunk_rows,
                )

    def add_derived(
        self, request: AddRequest, facts: list[str], vectors: np.ndarray
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(facts):
            raise ValueError("Fact and embedding counts do not match")
        if not facts:
            return
        now = datetime.now(timezone.utc).isoformat()
        timestamp = next(
            (message.timestamp for message in request.messages if message.timestamp is not None),
            None,
        )
        with self.lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO messages(
                    id, request_id, user_id, session_id, message_index, role, content,
                    source_timestamp, created_at, embedding, dimension, derived
                ) VALUES (?, ?, ?, ?, ?, 'memory_fact', ?, ?, ?, ?, ?, 1)
                """,
                [
                    (
                        _fact_id(request, index), request.request_id, request.user_id,
                        request.session_id, 1_000_000 + index, fact, timestamp, now,
                        np.asarray(vectors[index], dtype=np.float32).tobytes(),
                        int(vectors.shape[1]),
                    )
                    for index, fact in enumerate(facts)
                ],
            )

    def list_user(self, user_id: str, *, include_unindexed: bool = False) -> list[StoredMemory]:
        with self.lock, self._connect() as connection:
            filter_clause = "" if include_unindexed else "AND indexed = 1"
            rows = connection.execute(
                f"""
                SELECT sequence, id, request_id, user_id, session_id, message_index,
                       role, content, source_timestamp, created_at, embedding, dimension, derived, media_json
                FROM messages WHERE user_id = ? {filter_clause} ORDER BY sequence, message_index
                """,
                (user_id,),
            ).fetchall()
            chunk_rows = connection.execute(
                """
                SELECT c.message_id, c.chunk_index, c.embedding, c.dimension
                FROM message_chunks AS c
                JOIN messages AS m ON m.id = c.message_id
                WHERE m.user_id = ? AND (m.indexed = 1 OR ?)
                ORDER BY c.message_id, c.chunk_index
                """,
                (user_id, int(include_unindexed)),
            ).fetchall()
        chunks_by_id: dict[str, list[np.ndarray]] = {}
        for row in chunk_rows:
            chunks_by_id.setdefault(str(row["message_id"]), []).append(
                np.frombuffer(
                    row["embedding"], dtype=np.float32, count=int(row["dimension"])
                ).copy()
            )
        return [
            StoredMemory(
                sequence=int(row["sequence"]),
                id=str(row["id"]),
                request_id=str(row["request_id"]),
                user_id=str(row["user_id"]),
                session_id=str(row["session_id"]),
                message_index=int(row["message_index"]),
                role=str(row["role"]),
                content=str(row["content"]),
                source_timestamp=(
                    int(row["source_timestamp"])
                    if row["source_timestamp"] is not None
                    else None
                ),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                embedding=np.frombuffer(
                    row["embedding"], dtype=np.float32, count=int(row["dimension"])
                ).copy(),
                derived=bool(row["derived"]),
                media=tuple(
                    MediaPart.model_validate(part)
                    for part in json.loads(str(row["media_json"]))
                ),
                embedding_chunks=tuple(chunks_by_id.get(str(row["id"]), ())),
            )
            for row in rows
        ]
