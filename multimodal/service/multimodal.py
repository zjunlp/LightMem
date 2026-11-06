"""StructMem multimodal API: uvicorn service.multimodal:app.

All original content parts are persisted and returned in order. CLIP is opt-in
via MULTIMODAL_CLIP_MODEL (a local directory); the default is lexical retrieval.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, closing
import base64
import hashlib
import hmac
import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
import warnings
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .retrieval import bm25_scores
from .multimodal_embedding import RemoteEncoder
from .multimodal_sessions import dense_session_tail, dense_session_image_tail
from .multimodal_retrieval import score_vectors, propagate_context, image_first_order, rerank_prefix, session_diverse_order
from .multimodal_cache import QueryCache
from .multimodal_events import event_neighbors, complete_event_ranking, merge_event_tail
from .multimodal_metadata import CaptureMetadata

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BODY_BYTES = 42 * 1024 * 1024
MAX_MEDIA_BYTES = 30 * 1024 * 1024
VERSION = '0.2.1'


def decode_image(url: str | dict) -> Image.Image:
    if isinstance(url, dict):
        url = url['url']
    prefix, sep, encoded = url.partition(',')
    formats = {'data:image/png;base64': 'PNG', 'data:image/jpeg;base64': 'JPEG',
               'data:image/webp;base64': 'WEBP'}
    if not sep or prefix not in formats:
        raise ValueError('image_url must be an inline PNG, JPEG or WebP base64 data URI')
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError('image exceeds 10 MiB')
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError('image exceeds 10 MiB')
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(raw))
            if image.format != formats[prefix]:
                raise ValueError('image MIME type does not match encoded image')
            image.load()
            return image.convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('invalid or unsafe image') from exc


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class TextPart(StrictModel):
    type: Literal['text']
    text: str = Field(min_length=1)

    @field_validator('text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('text cannot be blank')
        return value


class ImageURL(StrictModel):
    url: str


class ImagePart(StrictModel):
    type: Literal['image_url']
    image_url: ImageURL

    @field_validator('image_url', mode='before')
    @classmethod
    def normalize(cls, value):
        return {'url': value} if isinstance(value, str) else value

    @field_validator('image_url')
    @classmethod
    def validate_image(cls, value):
        decode_image(value.url)
        return value


Part = Annotated[TextPart | ImagePart, Field(discriminator='type')]
Content = Annotated[list[Part], Field(min_length=1)]


class Message(StrictModel):
    role: str = Field(min_length=1, max_length=64)
    content: Content
    timestamp: int | str | None = None

    @field_validator('content', mode='before')
    @classmethod
    def normalize_text_content(cls, value):
        if isinstance(value, str):
            return [{'type': 'text', 'text': value}]
        return value


class AddRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=512)
    user_id: str = Field(min_length=1, max_length=512)
    session_id: str = Field(min_length=1, max_length=512)
    messages: list[Message] = Field(min_length=1)


class SearchRequest(StrictModel):
    user_id: str = Field(min_length=1, max_length=512)
    query: Content | str
    top_k: int = Field(ge=1, le=1000)
    options: list[str] | None = None

    @field_validator('query')
    @classmethod
    def nonblank(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError('query cannot be blank')
        return value


def image_bytes(part: dict) -> int:
    if part['type'] != 'image_url':
        return 0
    url = part['image_url']
    url = url['url'] if isinstance(url, dict) else url
    return len(base64.b64decode(url.partition(',')[2]))


def text_of(parts: list[dict]) -> str:
    return '\n'.join(p['text'] for p in parts if p['type'] == 'text')


class ClipEncoder:
    """Lazy model load; local files only, one inference at a time."""
    def __init__(self, path: str, device: str):
        self.path, self.device = path, device
        self.lock = threading.Lock()
        self.model = self.processor = None

    def encode(self, parts: list[dict]) -> np.ndarray:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        with self.lock, torch.inference_mode():
            if self.model is None:
                self.processor = CLIPProcessor.from_pretrained(self.path, local_files_only=True)
                self.model = CLIPModel.from_pretrained(self.path, local_files_only=True).to(self.device).eval()
            vectors = []
            for part in parts:
                if part['type'] == 'text' and part['text'].strip():
                    inputs = self.processor(text=[part['text']], padding=True, truncation=True, return_tensors='pt').to(self.device)
                    vector = self.model.get_text_features(**inputs)
                elif part['type'] == 'image_url':
                    inputs = self.processor(images=decode_image(part['image_url']), return_tensors='pt').to(self.device)
                    vector = self.model.get_image_features(**inputs)
                else:
                    continue
                vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                vectors.append(vector.cpu().numpy()[0])
            if not vectors:
                raise ValueError('content has no encodable parts')
            return np.stack(vectors).astype(np.float32)


def serialize_vectors(vectors: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    return buffer.getvalue()


def pack_content(partvectors, parts):
    present = [(part, vectors) for part, vectors in zip(parts, partvectors, strict=True)
               if vectors is not None and len(vectors)]
    if not present:
        return np.empty((0, 0), dtype=np.float32), []
    matrix = np.concatenate([vectors for _, vectors in present], axis=0)
    types = [part['type'] for part, vectors in present for _ in range(len(vectors))]
    return matrix, types


def serialize_content(matrix, types, multivector=False):
    if not multivector:
        return serialize_vectors(matrix)
    buffer = io.BytesIO()
    np.savez(buffer, matrix=matrix, types=np.asarray(types, dtype='U16'))
    return buffer.getvalue()


def load_content(blob, parts):
    stored = np.load(io.BytesIO(blob), allow_pickle=False)
    if isinstance(stored, np.lib.npyio.NpzFile):
        with stored:
            matrix, types = stored['matrix'], stored['types'].tolist()
        return matrix, types
    if callable(parts):
        parts = parts()
    return stored, [part['type'] for part in parts]


class MemoryService:
    def __init__(self, database: str, clip_model: str | None = None, device: str = 'cpu', remote_config: dict | None = None, scoring: str = 'late', lexical_weight: float = 1, query_cache_mb: float = 64,
                 query_cache_ttl: float = 300, query_cache_entries: int = 128, context_weight: float = 0, pair_diversity: bool = False,
                 reranker=None, rerank_mode: str = 'consensus', capture_metadata=None, event_completion: bool = False, dense_session_expansion: bool = False, session_image_support: bool = False, answer_evidence: bool = False, answer_update_bridge: bool = False, answer_next_turn: bool = False, answer_context_top_k: int = 0):
        if scoring not in {'late', 'modality_zmax', 'modality_rrf'}:
            raise ValueError('invalid multimodal scoring mode')
        if not np.isfinite(lexical_weight) or lexical_weight < 0:
            raise ValueError('lexical weight must be finite and nonnegative')
        if not np.isfinite(query_cache_mb) or query_cache_mb < 0:
            raise ValueError('invalid query cache memory limit')
        if not np.isfinite(context_weight) or not 0 <= context_weight <= 1:
            raise ValueError('context weight must be in [0, 1]')
        if context_weight and (scoring != 'late' or lexical_weight != 0 or not (clip_model or remote_config)):
            raise ValueError('context propagation requires an encoder, late scoring and zero lexical weight')
        if rerank_mode not in {'direct', 'consensus', 'image_locked', 'session_diverse', 'session_reserved', 'protected40', 'protected20_tail100'}:
            raise ValueError('invalid reranker mode')
        if capture_metadata is not None and not (clip_model or remote_config):
            raise ValueError('capture metadata requires a configured encoder')
        if event_completion and (capture_metadata is None or scoring != 'late' or lexical_weight != 0
                                 or context_weight or pair_diversity or rerank_mode != 'consensus'):
            raise ValueError('event completion requires capture metadata, late dense scoring and consensus mode')
        if dense_session_expansion and (not (clip_model or remote_config) or lexical_weight != 0
                                        or context_weight or event_completion or rerank_mode != 'session_diverse'):
            raise ValueError('dense session expansion requires an encoder, zero lexical/context weights and session_diverse mode')
        if session_image_support and not dense_session_expansion:
            raise ValueError('session image support requires dense session expansion')
        if type(answer_evidence) is not bool:
            raise ValueError('answer evidence mode must be boolean')
        if type(answer_update_bridge) is not bool:
            raise ValueError('answer update bridge mode must be boolean')
        if type(answer_next_turn) is not bool:
            raise ValueError('answer next-turn mode must be boolean')
        if type(answer_context_top_k) is not int or not 0 <= answer_context_top_k <= 100:
            raise ValueError('answer context top-k must be an integer between 0 and 100')
        if answer_evidence and answer_context_top_k:
            raise ValueError('full answer evidence and bounded answer context are mutually exclusive')
        self.answer_evidence = answer_evidence
        self.answer_update_bridge = answer_update_bridge
        self.answer_next_turn = answer_next_turn
        self.answer_context_top_k = answer_context_top_k
        self.session_image_support = session_image_support
        self.dense_session_expansion = dense_session_expansion
        self.event_completion = event_completion
        self.capture_metadata = capture_metadata
        self.reranker, self.rerank_mode = reranker, rerank_mode
        self.reranker_error = None
        self.pair_diversity = pair_diversity
        self.context_weight = context_weight
        self.query_cache = QueryCache(int(query_cache_mb * 1024 * 1024), query_cache_ttl, query_cache_entries)
        self.scoring, self.lexical_weight = scoring, lexical_weight
        self.database = database
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.encoder = RemoteEncoder(**remote_config) if remote_config else (ClipEncoder(clip_model, device) if clip_model else None)
        self.signature = self.encoder.signature if remote_config else (f'clip-parts-v2:{clip_model}' if clip_model else 'lexical-only')
        if capture_metadata is not None:
            self.signature += ':' + capture_metadata.signature
        self.last_error = None
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS requests (
                    user_id TEXT, request_id TEXT, digest TEXT,
                    PRIMARY KEY(user_id,request_id));
                CREATE TABLE IF NOT EXISTS memories (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT, id TEXT, content TEXT, text TEXT,
                    vector BLOB, encoder TEXT, created_at TEXT,
                    role TEXT, session_id TEXT, source_timestamp TEXT);
                CREATE INDEX IF NOT EXISTS memories_user ON memories(user_id,sequence);
                CREATE TABLE IF NOT EXISTS capture_events (
                    user_id TEXT, sequence INTEGER, metadata TEXT,
                    PRIMARY KEY(user_id,sequence));
            ''')
            columns = {row[1] for row in db.execute('PRAGMA table_info(memories)')}
            for column in ('role', 'session_id', 'source_timestamp'):
                if column not in columns:
                    db.execute(f'ALTER TABLE memories ADD COLUMN {column} TEXT')

    def close(self):
        self.query_cache.close()
        if self.encoder and hasattr(self.encoder, 'close'):
            self.encoder.close()

    def connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def encode(self, parts):
        try:
            result = np.atleast_2d(self.encoder.encode(parts)).astype(np.float32)
            result = result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
            self.last_error = None
            return result
        except HTTPException:
            raise
        except Exception as exc:
            self.last_error = type(exc).__name__
            raise HTTPException(503, 'configured embedding encoder unavailable') from exc

    def encode_parts(self, parts, *, purpose='passage'):
        try:
            if isinstance(self.encoder, RemoteEncoder):
                result = self.encoder.encode_parts(parts, purpose=purpose)
            else:
                result = [row[None, :] for row in self.encode(parts)]
            self.last_error = None
            return result
        except HTTPException:
            raise
        except Exception as exc:
            self.last_error = type(exc).__name__
            raise HTTPException(503, 'configured embedding encoder unavailable') from exc

    def encode_content(self, parts):
        if self.capture_metadata is not None:
            parts = self.capture_metadata.augment(parts)
        return pack_content(self.encode_parts(parts), parts)

    def encode_query_content(self, parts):
        if isinstance(self.encoder, RemoteEncoder):
            return pack_content(self.encode_parts(parts, purpose='query'), parts)
        if self.capture_metadata is None:
            return self.encode_content(parts)
        return pack_content(self.encode_parts(parts), parts)

    def serialize_content(self, matrix, types):
        multi = (self.capture_metadata is not None or
                 isinstance(self.encoder, RemoteEncoder) and (
                     self.encoder.response_format == 'multivector' or self.encoder.input_mode == 'text_only'))
        return serialize_content(matrix, types, multi)

    def add(self, request: AddRequest):
        payload = request.model_dump(mode='json')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.connect() as db:
            previous = db.execute('SELECT digest FROM requests WHERE user_id=? AND request_id=?',
                                  (request.user_id, request.request_id)).fetchone()
        if previous:
            if previous[0] != digest:
                raise HTTPException(409, 'request_id was already used with different content')
            return self.add_response(request)
        media_size = sum(image_bytes(p.model_dump(mode='json')) for message in request.messages for p in message.content)
        if media_size > MAX_MEDIA_BYTES:
            raise HTTPException(413, 'decoded images exceed 30 MiB')
        records = []
        remote_vectors = None
        prepared_messages = None
        offset = 0
        if isinstance(self.encoder, RemoteEncoder):
            prepared_messages = [[part.model_dump(mode='json') for part in message.content]
                                 for message in request.messages]
            if self.capture_metadata is not None:
                prepared_messages = [self.capture_metadata.augment(parts) for parts in prepared_messages]
            remote_vectors = self.encode_parts([part for parts in prepared_messages for part in parts])
        for index, message in enumerate(request.messages):
            parts = message.model_dump(mode='json')['content']
            if remote_vectors is not None:
                prepared = prepared_messages[index]
                matrix, parttypes = pack_content(remote_vectors[offset:offset + len(prepared)], prepared)
                vector = self.serialize_content(matrix, parttypes)
                offset += len(prepared)
            else:
                vector = self.serialize_content(*self.encode_content(parts)) if self.encoder else None
            identifier = hashlib.sha256(json.dumps([request.user_id, request.request_id, index]).encode()).hexdigest()
            records.append((request.user_id, identifier, json.dumps(parts, ensure_ascii=False),
                            text_of(parts), vector, self.signature, message.role, request.session_id,
                            json.dumps(message.timestamp)))
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT digest FROM requests WHERE user_id=? AND request_id=?',
                                  (request.user_id, request.request_id)).fetchone()
            if previous:
                if previous[0] != digest:
                    raise HTTPException(409, 'request_id conflict')
            else:
                db.execute('INSERT INTO requests VALUES(?,?,?)', (request.user_id, request.request_id, digest))
                db.executemany("INSERT INTO memories(user_id,id,content,text,vector,encoder,role,session_id,source_timestamp,created_at) VALUES(?,?,?,?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))", records)
        return self.add_response(request)

    @staticmethod
    def add_response(request):
        return dict(success=True, request_id=request.request_id, user_id=request.user_id, session_id=request.session_id)

    def search(self, request: SearchRequest):
        parts = ([{'type': 'text', 'text': request.query}] if isinstance(request.query, str)
                 else [p.model_dump(mode='json') for p in request.query])
        if (isinstance(self.encoder, RemoteEncoder) and self.encoder.input_mode == 'text_only'
                and not any(p.get('type') == 'text' and str(p.get('text') or '').strip() for p in parts)):
            raise HTTPException(503, 'image-only queries require an image embedding encoder')
        if not self.encoder and any(p['type'] == 'image_url' for p in parts):
            raise HTTPException(503, 'image query retrieval requires configured image encoder')
        # One WAL read snapshot keeps metadata and original evidence consistent.
        with closing(self.connect()) as db:
            db.execute('BEGIN')
            return self._search_snapshot(request, parts, db)

    def _session_diverse_rerank(self, ranking, rows, parts, original):
        count = min(100, len(ranking))
        deadline = time.monotonic() + getattr(self.reranker, 'timeout', 30)
        relevance, sessions = [], []
        for index in ranking[:count]:
            index = int(index)
            document = original(index)
            sessions.append(rows[index][6])
            if any(p['type'] == 'image_url' for p in document):
                relevance.append(None)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError('session reranker budget exhausted')
            try:
                values = self.reranker.score(parts, [document], decode_image)
                if len(values) != 1 or not np.isfinite(values[0]):
                    raise ValueError('invalid session relevance')
                relevance.append(values[0])
            except ValueError as exc:
                if str(exc) not in {'reranker input exceeds context budget', 'empty reranker content'}:
                    raise
                relevance.append(None)
            if time.monotonic() >= deadline:
                raise TimeoutError('session reranker budget exhausted')
        return session_diverse_order(
            ranking, relevance, sessions,
            protected_head=10 if self.rerank_mode == 'session_reserved' else 0)

    def _search_snapshot(self, request, parts, snapshot):
        rows = snapshot.execute('SELECT id,NULL,text,vector,encoder,created_at,session_id,sequence,role,source_timestamp FROM memories WHERE user_id=? ORDER BY sequence', (request.user_id,)).fetchall()
        loaded = {}

        def original(index):
            if index not in loaded:
                row = snapshot.execute('SELECT content FROM memories WHERE user_id=? AND sequence=?',
                                       (request.user_id, rows[index][7])).fetchone()
                if row is None:
                    raise RuntimeError('source evidence missing from read snapshot')
                loaded[index] = json.loads(row[0])
            return loaded[index]
        if not rows:
            return {'data': []}
        scores = np.zeros(len(rows), dtype=np.float64)
        lexical = bm25_scores([r[2] for r in rows], text_of(parts)) if self.lexical_weight else None
        if lexical is not None and np.any(lexical > 0):
            for rank, idx in enumerate(np.argsort(-lexical, kind='stable')):
                if lexical[idx] > 0:
                    scores[idx] += self.lexical_weight / (60 + rank + 1)
        if self.encoder:
            query, querytypes = self.query_cache.get_or_encode(self.signature, request.user_id, parts, self.encode_query_content)
            vectors = []
            types = []
            for idx, row in enumerate(rows):
                # Lexical-only records added before enabling CLIP are encoded lazily.
                if row[3] is not None and row[4] == self.signature:
                    vector, parttypes = load_content(row[3], lambda: original(idx))
                else:
                    vector, parttypes = self.encode_content(original(idx))
                    with self.connect() as db:
                        db.execute('UPDATE memories SET vector=?,encoder=? WHERE user_id=? AND id=?',
                                   (self.serialize_content(vector, parttypes), self.signature, request.user_id, row[0]))
                vectors.append(vector)
                types.append(parttypes)
            dense = score_vectors(query, vectors, types, querytypes, mode=self.scoring)
            scorable = np.asarray([
                not isinstance(self.encoder, RemoteEncoder) or self.encoder.input_mode != 'text_only'
                or 'text' in parttypes for parttypes in types
            ], dtype=bool)
            if self.context_weight:
                dense = propagate_context(dense, [request.user_id] * len(rows),
                                          [row[6] for row in rows], [row[7] for row in rows],
                                          self.context_weight, valid_vectors=scorable)
            if self.lexical_weight == 0:
                scores = dense
            else:
                for rank, idx in enumerate(np.argsort(-dense, kind='stable')):
                    scores[idx] += 1 / (60 + rank + 1)
            if isinstance(self.encoder, RemoteEncoder) and self.encoder.input_mode == 'text_only':
                scores[~scorable] = -np.inf
        ranking = np.argsort(-scores, kind='stable')
        metadata_order = ranking.tolist() if self.event_completion else None
        if self.pair_diversity:
            # Reorder an explicit bounded candidate set before truncating to K.
            # Source adjacency is checked against all rows in this read snapshot.
            candidates = ranking[:100].tolist()
            candidate_set = set(candidates)
            records = [dict(id=i, role=row[8], session_id=row[6],
                            user_id=request.user_id, sequence=row[7],
                            has_image=(i in candidate_set and row[8] == 'user'
                                       and any(p['type'] == 'image_url' for p in original(i))))
                       for i, row in enumerate(rows)]
            ranking = image_first_order(candidates, records) + ranking[100:].tolist()
        if self.reranker is not None:
            try:
                if self.rerank_mode in {'session_diverse', 'session_reserved'}:
                    ranking = self._session_diverse_rerank(ranking, rows, parts, original)
                else:
                    documents = []
                    window = 100 if self.rerank_mode == 'protected20_tail100' else (40 if self.rerank_mode == 'protected40' else 20)
                    for index in ranking[:window]:
                        index = int(index)
                        document = list(original(index))
                        row = rows[index]
                        if (row[8] == 'user' and any(p['type'] == 'image_url' for p in document)
                                and row[6] and str(row[6]).strip() and index + 1 < len(rows)):
                            reply = rows[index + 1]
                            if (reply[8] == 'assistant' and reply[6] == row[6]
                                    and reply[7] == row[7] + 1):
                                text = text_of(original(index + 1))
                                if text:
                                    document.append(dict(type='text', text=text))
                        documents.append(document)
                    if self.rerank_mode in {'protected40', 'protected20_tail100'}:
                        deadline = time.monotonic() + getattr(self.reranker, 'timeout', 30)
                        relevance = []
                        for offset in range(0, len(documents), 20):
                            if time.monotonic() >= deadline:
                                raise TimeoutError('reranker budget exhausted')
                            batch = documents[offset:offset + 20]
                            values = self.reranker.score(parts, batch, decode_image)
                            if len(values) != len(batch):
                                raise ValueError('reranker candidate/score mismatch')
                            relevance.extend(values)
                            if time.monotonic() >= deadline:
                                raise TimeoutError('reranker budget exhausted')
                    else:
                        relevance = self.reranker.score(parts, documents, decode_image)
                    ranking = rerank_prefix(
                        ranking, relevance, self.rerank_mode,
                        image_mask=[any(p['type'] == 'image_url' for p in original(int(i)))
                                    for i in ranking[:20]])
                self.reranker_error = None
            except Exception as exc:
                self.reranker_error = type(exc).__name__
                logging.getLogger(__name__).warning('Local reranking skipped: %s', type(exc).__name__)
        if self.event_completion:
            records = self._event_records(request.user_id, rows, original, snapshot)
            count = min(100, len(rows))
            completed = complete_event_ranking(metadata_order, scores,
                event_neighbors(records, user_id=request.user_id), limit=count, head=min(20, count))
            ranking = merge_event_tail(list(ranking[:count]), completed, head=min(20, count))
        if self.dense_session_expansion:
            ranking = dense_session_tail(ranking, dense, [row[6] for row in rows])
        if self.session_image_support:
            try:
                image_mask = [any(part['type'] == 'image_url' for part in original(i))
                              for i in range(len(rows))]
                ranking = dense_session_image_tail(ranking, dense, [row[6] for row in rows], image_mask)
            except Exception as exc:
                logging.getLogger(__name__).warning('Session image support skipped: %s', type(exc).__name__)
        selected = [int(index) for index in ranking if np.isfinite(scores[int(index)])][:request.top_k]
        bridge = (self._answer_update_bridge(selected, rows, text_of(parts))
                  if self.answer_update_bridge else {})
        next_turn = (self._answer_next_turn(selected, rows, bridge, text_of(parts))
                     if self.answer_next_turn else {})
        result = {'data': []}
        media_bytes = 0
        seen_context_records = {}
        for rank, index in enumerate(selected, 1):
            content = original(int(index))
            if self.answer_evidence:
                evidence = self._answer_evidence(int(index), rows, original)
            elif self.answer_context_top_k:
                evidence = self._answer_evidence(
                    int(index), rows, original, seen_source_records=seen_context_records,
                    source_rank=rank, expand_neighbors=rank <= self.answer_context_top_k,
                )
            else:
                evidence = content
            if bridge.get(int(index)):
                # Keep the response cardinality within top_k. Supplemental
                # source turns are attached to their selected session anchor.
                evidence = list(evidence) + [dict(type='text', text='[supplemental same-session evidence]')]
                for candidate in bridge[int(index)]:
                    evidence.extend(original(candidate))
            if next_turn.get(int(index)) is not None:
                evidence = list(evidence) + [dict(type='text', text='[next same-session turn]')]
                evidence.extend(original(next_turn[int(index)]))
            # Context expansion can include adjacent images.  Enforce the same
            # response budget on the representation actually returned to QA.
            item_bytes = sum(image_bytes(part) for part in evidence)
            if media_bytes + item_bytes > MAX_MEDIA_BYTES:
                if not result['data']:
                    raise HTTPException(413, 'highest-ranked evidence exceeds 30 MiB image budget')
                break
            result['data'].append(dict(id=rows[index][0], content=evidence,
                                       score=float(scores[index]), created_at=rows[index][5]))
            media_bytes += item_bytes
        return result

    @staticmethod
    def _answer_update_bridge(selected, rows, query):
        """Fill a small selected same-session gap for explicit update queries.

        The trigger is query text only. No labels, answer text or stored-content
        classification affects the selection, and the retrieval ranking remains
        unchanged.
        """
        markers = ('最后', '后来', '当前', '现在', '改成', '调整', '承诺', '不再', '更新', '最新',
                   'final', 'later', 'current', 'updated', 'update', 'commitment')
        query_text = str(query).casefold()
        if not any(marker in query_text for marker in markers):
            return {}
        selected = [int(index) for index in selected]
        selected_set = set(selected)
        sessions = {}
        for index in selected:
            session = rows[index][6]
            if isinstance(session, str) and session.strip():
                sessions.setdefault(session, []).append(index)
        additions, remaining = {}, 12
        for session, indexes in sessions.items():
            sequences = [rows[index][7] for index in indexes]
            lower, upper = min(sequences), max(sequences)
            if len(indexes) < 2 or upper - lower > 24:
                continue
            candidates = [index for index, row in enumerate(rows)
                          if row[6] == session and lower <= row[7] <= upper
                          and index not in selected_set]
            if not candidates:
                continue
            anchor = max(indexes, key=lambda index: rows[index][7])
            additions[anchor] = candidates[:remaining]
            remaining -= len(additions[anchor])
            if remaining == 0:
                break
        return additions

    @staticmethod
    def _answer_next_turn(selected, rows, bridge, query=''):
        """Attach an immediate same-session successor to each selected record.

        It does not change ranking or result cardinality.  The extra turn is
        useful when the question asks for a state transition, temporal
        relation, or visual detail; for ordinary factual questions it is
        omitted so unrelated neighbouring text cannot distract the answer
        model.  A successor already included by the update bridge is omitted,
        so source content appears at most once per selected record.
        """
        query_text = str(query).casefold()
        markers = (
            '最后', '后来', '当前', '现在', '改成', '调整', '承诺', '不再', '更新', '最新',
            '何时', '多久', '之前', '之后', '后来', '第一', '然后', '图片', '照片', '图中',
            'when', 'how long', 'before', 'after', 'later', 'current', 'changed',
            'update', 'updated', 'first', 'then', 'image', 'picture', 'photo', 'scene',
        )
        if not any(marker in query_text for marker in markers):
            return {}
        bridge_ids = {candidate for candidates in bridge.values() for candidate in candidates}
        successors = {}
        for index in map(int, selected):
            row = rows[index]
            candidate = index + 1
            if (candidate < len(rows) and candidate not in bridge_ids
                    and row[6] and rows[candidate][6] == row[6]
                    and rows[candidate][7] == row[7] + 1):
                successors[index] = candidate
        return successors

    @staticmethod
    def _answer_evidence(index, rows, original, *, seen_source_records=None, source_rank=None,
                         expand_neighbors=True):
        """Render ranked source records as bounded, source-faithful answer evidence.

        This response view makes role, source session and timestamp explicit and
        includes immediate same-session context.  It keeps the original image
        parts, because the evaluator forwards image_url parts separately to its
        QA model.
        """
        if not 0 <= index < len(rows):
            raise ValueError('invalid evidence index')
        row = rows[index]
        session, sequence = row[6], row[7]
        included = []
        candidates = (index - 1, index, index + 1) if expand_neighbors else (index,)
        for candidate in candidates:
            if not 0 <= candidate < len(rows):
                continue
            nearby = rows[candidate]
            if candidate == index or (session and str(session).strip()
                                     and nearby[6] == session
                                     and nearby[7] in {sequence - 1, sequence + 1}):
                included.append(candidate)
        evidence = []
        for candidate in dict.fromkeys(included):
            if seen_source_records is not None and candidate in seen_source_records:
                continue
            if seen_source_records is not None:
                seen_source_records[candidate] = source_rank
            nearby = rows[candidate]
            try:
                timestamp = json.loads(nearby[9]) if nearby[9] is not None else nearby[5]
            except (TypeError, json.JSONDecodeError):
                timestamp = nearby[5]
            labels = [f'role={nearby[8]}']
            if nearby[6]:
                labels.append(f'session={nearby[6]}')
            if timestamp is not None:
                labels.append(f'time={timestamp}')
            if source_rank is None:
                label = '[' + '; '.join(labels) + ']'
            else:
                relation = 'target' if candidate == index else ('previous' if candidate < index else 'next')
                label = f'[Source {relation}; role={nearby[8]}; session={nearby[6]}; time={timestamp}]'
            evidence.append(dict(type='text', text=label))
            evidence.extend(original(candidate))
        if not evidence and seen_source_records is not None:
            prior_rank = next((seen_source_records[candidate] for candidate in included
                               if candidate in seen_source_records), None)
            if prior_rank is not None:
                evidence.append(dict(type='text', text=f'[Source content already included with Memory {prior_rank}]'))
        return evidence

    def _event_records(self, user_id, rows, original, snapshot):
        # Cache even missing metadata: original evidence is immutable. Scope both
        # reads and writes to the authenticated gallery, never a global image ID.
        cached = dict(snapshot.execute('SELECT sequence,metadata FROM capture_events WHERE user_id=?',
                                       (user_id,)).fetchall())
        records, pending = [], []
        for index, row in enumerate(rows):
            sequence = row[7]
            if sequence in cached:
                metadata = json.loads(cached[sequence])
            else:
                metadata = None
                content = original(index)
                if len(content) == 1 and content[0]['type'] == 'image_url':
                    value = content[0]['image_url']
                    uri = value['url'] if isinstance(value, dict) else value
                    # Content was validated at Add; parse original bytes before
                    # any encoder resize or RGB conversion discards EXIF.
                    raw = base64.b64decode(uri.split(',', 1)[1], validate=True)
                    metadata = CaptureMetadata.event_from_bytes(raw)
                pending.append((user_id, sequence, json.dumps(metadata)))
            records.append(dict(user_id=user_id, **(metadata or {})))
        if pending:
            with self.connect() as db:
                db.executemany('INSERT OR IGNORE INTO capture_events VALUES(?,?,?)', pending)
        return records


def create_app(database: str | None = None, api_key: str | None = None,
               clip_model: str | None = None, device: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        try:
            yield
        finally:
            if holder:
                await asyncio.to_thread(holder[0].close)

    application = FastAPI(title='StructMem Multimodal', version=VERSION, lifespan=lifespan)
    key = os.environ.get('MEMORY_API_KEY', '') if api_key is None else api_key
    model = clip_model if clip_model is not None else os.environ.get('MULTIMODAL_CLIP_MODEL')
    remote_base = os.environ.get('MULTIMODAL_EMBEDDING_BASE_URL')
    remote_config = dict(base_url=remote_base, model=os.environ.get('MULTIMODAL_EMBEDDING_MODEL', 'text-embedding-v4'),
                         api_key=os.environ.get('MULTIMODAL_EMBEDDING_API_KEY', ''),
                         concurrency=int(os.environ.get('MULTIMODAL_EMBEDDING_CONCURRENCY', '4')),
                         timeout=float(os.environ.get('MULTIMODAL_EMBEDDING_TIMEOUT', '60')),
                         batch_size=int(os.environ.get('MULTIMODAL_EMBEDDING_BATCH_SIZE', '32')),
                         image_batch_size=int(os.environ.get('MULTIMODAL_EMBEDDING_IMAGE_BATCH_SIZE', '8')),
                         response_format=os.environ.get('MULTIMODAL_EMBEDDING_FORMAT', 'singlevector'),
                         image_threshold_bytes=int(os.environ.get('MULTIMODAL_EMBEDDING_IMAGE_THRESHOLD_BYTES', '0')),
                         image_max_edge=int(os.environ.get('MULTIMODAL_EMBEDDING_IMAGE_MAX_EDGE', '2048')),
                         image_jpeg_quality=int(os.environ.get('MULTIMODAL_EMBEDDING_IMAGE_JPEG_QUALITY', '95')),
                         query_prompt_name=os.environ.get('MULTIMODAL_EMBEDDING_QUERY_PROMPT_NAME'),
                         passage_prompt_name=os.environ.get('MULTIMODAL_EMBEDDING_PASSAGE_PROMPT_NAME'),
                         input_mode=os.environ.get('MULTIMODAL_EMBEDDING_INPUT_MODE', 'text_only'),
                         dimensions=int(os.environ.get('MULTIMODAL_EMBEDDING_DIMENSIONS', '1024'))) if remote_base else None
    reranker_path = os.environ.get('MULTIMODAL_RERANKER_MODEL')
    # Construct storage on first authenticated request, not on module import.
    guard = threading.Lock()
    holder = []

    def service():
        with guard:
            if not holder:
                reranker = None
                capture_metadata = None
                if os.environ.get('MULTIMODAL_CAPTURE_GAZETTEER'):
                    from .multimodal_metadata import CaptureMetadata
                    capture_metadata = CaptureMetadata(os.environ['MULTIMODAL_CAPTURE_GAZETTEER'])
                if reranker_path:
                    backend = os.environ.get('MULTIMODAL_RERANKER_BACKEND', 'relevance')
                    options = dict(device=os.environ.get('MULTIMODAL_RERANKER_DEVICE', 'cuda:0'),
                                   timeout=float(os.environ.get('MULTIMODAL_RERANKER_TIMEOUT', '30')))
                    if backend == 'joint_embedding':
                        from .multimodal_joint_embedding import LocalJointEmbeddingReranker
                        if os.environ.get('MULTIMODAL_RERANKER_MODE', 'consensus') != 'consensus':
                            raise ValueError('joint embedding requires consensus mode')
                        reranker = LocalJointEmbeddingReranker(
                            reranker_path, capture_metadata=capture_metadata, **options)
                    elif backend == 'relevance':
                        from .multimodal_reranker import LocalReranker
                        reranker = LocalReranker(reranker_path, **options)
                    else:
                        raise ValueError('invalid local reranker backend')
                holder.append(MemoryService(database or os.environ.get('MULTIMODAL_DB', './data/multimodal.sqlite3'),
                                            model, device or os.environ.get('MULTIMODAL_DEVICE', 'cpu'), remote_config,
                                            os.environ.get('MULTIMODAL_SCORING', 'late'),
                                            float(os.environ.get('MULTIMODAL_LEXICAL_WEIGHT', '1')),
                                            float(os.environ.get('MULTIMODAL_QUERY_CACHE_MB', '64')),
                                            float(os.environ.get('MULTIMODAL_QUERY_CACHE_TTL_SECONDS', '300')),
                                            int(os.environ.get('MULTIMODAL_QUERY_CACHE_MAX_ENTRIES', '128')),
                                            float(os.environ.get('MULTIMODAL_CONTEXT_WEIGHT', '0')),
                                            os.environ.get('MULTIMODAL_PAIR_DIVERSITY', '0') == '1',
                                            reranker, os.environ.get('MULTIMODAL_RERANKER_MODE', 'consensus'),
                                            capture_metadata,
                                            os.environ.get('MULTIMODAL_EVENT_COMPLETION', '0') == '1',
                                            os.environ.get('MULTIMODAL_DENSE_SESSION_EXPANSION', '0') == '1',
                                            os.environ.get('MULTIMODAL_SESSION_IMAGE_SUPPORT', '0') == '1',
                                            os.environ.get('MULTIMODAL_ANSWER_EVIDENCE', '0') == '1',
                                            os.environ.get('MULTIMODAL_UPDATE_EVIDENCE', '0') == '1',
                                            os.environ.get('MULTIMODAL_NEXT_TURN_EVIDENCE', '0') == '1',
                                            int(os.environ.get('MULTIMODAL_ANSWER_CONTEXT_TOP_K', '0'))))
            return holder[0]

    def auth(request: Request):
        authorization = request.headers.get('authorization', '')
        supplied = authorization.split(' ', 1)[1] if authorization.lower().startswith(('bearer ', 'token ')) else request.headers.get('x-api-key', '')
        if not key:
            raise HTTPException(503, 'API key is not configured')
        if not hmac.compare_digest(supplied.encode(), key.encode()):
            raise HTTPException(401, 'invalid API key')

    @application.middleware('http')
    async def body_limit(request: Request, call_next):
        size = 0
        chunks = []
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                return JSONResponse({'detail': 'request wire size exceeds 42 MiB'}, status_code=413)
            chunks.append(chunk)
        request._body = b''.join(chunks)
        return await call_next(request)

    @application.get('/health')
    def health():
        encoder = holder[0].encoder if holder else None
        loaded = bool(getattr(encoder, 'loaded', False) if remote_config else getattr(encoder, 'model', None))
        configured = bool(remote_config or model)
        lexical_weight = holder[0].lexical_weight if holder else float(os.environ.get('MULTIMODAL_LEXICAL_WEIGHT', '1'))
        retrieval = 'remote' if remote_config else ('clip' if model else 'bm25')
        if configured and lexical_weight:
            retrieval += '+bm25'
        return dict(status='ok', version=VERSION, track='multimodal',
                    retrieval=retrieval,
                    scoring=holder[0].scoring if holder else os.environ.get('MULTIMODAL_SCORING', 'late'),
                    lexical_weight=lexical_weight,
                    image_queries=bool(configured and not (
                        isinstance(encoder, RemoteEncoder) and encoder.input_mode == 'text_only')),
                    model_loaded=loaded,
                    ready=bool(not configured or (loaded and not holder[0].last_error)),
                    encoder_error=holder[0].last_error if holder else None,
                    reranker_configured=bool(reranker_path),
                    reranker_loaded=bool(holder and getattr(holder[0].reranker, 'model', None) is not None),
                    reranker_error=holder[0].reranker_error if holder else None)

    @application.post('/add', dependencies=[Depends(auth)])
    @application.post('/v1/memories/add', dependencies=[Depends(auth)])
    def add(request: AddRequest):
        return service().add(request)

    @application.post('/search', dependencies=[Depends(auth)])
    @application.post('/v1/memories/search', dependencies=[Depends(auth)])
    def search(request: SearchRequest):
        return service().search(request)

    return application


app = create_app()
