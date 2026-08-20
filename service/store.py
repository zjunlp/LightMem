from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .schemas import AddRequest, AddResponse, Message


class ConflictingRequestError(ValueError):
    pass


def _payload_hash(request: AddRequest) -> str:
    serialized = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _memory_id(request: AddRequest, message_index: int) -> str:
    identity = f"{request.user_id}\0{request.request_id}\0{message_index}"
    return "mem_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _timestamp(value: int | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def embedding_text(message: Message) -> str:
    return message.content.strip()


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
                    FOREIGN KEY (request_id) REFERENCES requests(request_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(user_id, session_id, sequence);
                """
            )
            row = connection.execute(
                "SELECT value FROM store_metadata WHERE key = 'embedding_identity'"
            ).fetchone()
            if row is None:
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
        self, request: AddRequest, response: AddResponse, vectors: np.ndarray
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(request.messages):
            raise ValueError("Message and embedding counts do not match")
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
                    source_timestamp, created_at, embedding, dimension
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        np.asarray(vectors[index], dtype=np.float32).tobytes(),
                        int(vectors.shape[1]),
                    )
                    for index, message in enumerate(request.messages)
                ],
            )

    def list_user(self, user_id: str) -> list[StoredMemory]:
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, id, request_id, user_id, session_id, message_index,
                       role, content, source_timestamp, created_at, embedding, dimension
                FROM messages WHERE user_id = ? ORDER BY sequence, message_index
                """,
                (user_id,),
            ).fetchall()
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
            )
            for row in rows
        ]
