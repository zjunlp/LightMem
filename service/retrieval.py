from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .store import StoredMemory


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class RetrievalHit:
    memory: StoredMemory
    score: float


def retrieval_query(query: str, options: list[str] | None) -> str:
    if not options:
        return query.strip()
    return (
        query.strip()
        + "\nCandidate answers:\n"
        + "\n".join(option.strip() for option in options)
    )


def lexical_tokens(text: str) -> list[str]:
    return [
        token
        for token in (
            match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)
        )
        if token not in STOP_WORDS
    ]


def bm25_scores(
    documents: list[str], query: str, *, k1: float = 1.2, b: float = 0.75
) -> np.ndarray:
    scores = np.zeros(len(documents), dtype=np.float32)
    query_terms = set(lexical_tokens(query))
    if not documents or not query_terms:
        return scores
    tokenized = [lexical_tokens(document) for document in documents]
    average_length = sum(map(len, tokenized)) / len(tokenized)
    if average_length == 0:
        return scores
    frequencies = [Counter(tokens) for tokens in tokenized]
    document_frequency = Counter(
        term for counts in frequencies for term in query_terms.intersection(counts)
    )
    count = len(documents)
    for index, counts in enumerate(frequencies):
        length = len(tokenized[index])
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            inverse = math.log(
                (count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
                + 1.0
            )
            scores[index] += (
                inverse
                * frequency
                * (k1 + 1.0)
                / (frequency + k1 * (1.0 - b + b * length / average_length))
            )
    return scores


def _expanded_positions(
    ranked: np.ndarray,
    memories: list[StoredMemory],
    *,
    limit: int,
    window: int,
    seed_k: int,
    semantic_bridges: dict[int, int],
    semantic_seed_k: int,
) -> list[int]:
    if window == 0 and not semantic_bridges:
        return [int(position) for position in ranked[:limit]]

    sessions: dict[str, list[int]] = defaultdict(list)
    for position, memory in enumerate(memories):
        sessions[memory.session_id].append(position)
    for positions in sessions.values():
        if all(
            memories[position].source_timestamp is not None for position in positions
        ):
            positions.sort(
                key=lambda position: (
                    memories[position].source_timestamp,
                    memories[position].sequence,
                    memories[position].message_index,
                )
            )
        else:
            positions.sort(
                key=lambda position: (
                    memories[position].sequence,
                    memories[position].message_index,
                )
            )
    session_offsets = {
        position: (positions, offset)
        for positions in sessions.values()
        for offset, position in enumerate(positions)
    }

    expanded: list[int] = []
    seen: set[int] = set()

    def add(position: int | None) -> None:
        if position is not None and position not in seen and len(expanded) < limit:
            seen.add(position)
            expanded.append(position)

    expansion_seed_k = max(seed_k if window else 0, semantic_seed_k)
    for seed_rank, raw_position in enumerate(ranked[:expansion_seed_k], 1):
        position = int(raw_position)
        add(position)
        if seed_rank <= seed_k:
            session_positions, offset = session_offsets[position]
            for distance in range(1, window + 1):
                for neighbor_offset in (offset - distance, offset + distance):
                    if 0 <= neighbor_offset < len(session_positions):
                        add(session_positions[neighbor_offset])
        if seed_rank <= semantic_seed_k:
            add(semantic_bridges.get(position))
    for raw_position in ranked:
        add(int(raw_position))
    return expanded


def _semantic_bridges(
    ranked: np.ndarray,
    memories: list[StoredMemory],
    matrix: np.ndarray,
    *,
    seed_k: int,
    minimum_similarity: float,
) -> dict[int, int]:
    if seed_k == 0 or len(memories) < 2:
        return {}

    seed_positions = np.asarray(ranked[:seed_k], dtype=np.int64)
    similarities = matrix @ matrix[seed_positions].T
    session_ids = np.asarray([memory.session_id for memory in memories])
    bridges: dict[int, int] = {}
    for column, raw_seed in enumerate(seed_positions):
        seed = int(raw_seed)
        column_scores = similarities[:, column].copy()
        column_scores[session_ids == memories[seed].session_id] = -np.inf
        candidate = int(np.argmax(column_scores))
        if column_scores[candidate] >= minimum_similarity:
            bridges[seed] = candidate
    return bridges


def hybrid_search(
    memories: list[StoredMemory],
    *,
    query_vector: np.ndarray,
    query_text: str,
    top_k: int,
    lexical_weight: float,
    rrf_k: int,
    result_window: int,
    result_window_seed_k: int,
    semantic_bridge_seed_k: int,
    semantic_bridge_min_similarity: float,
) -> list[RetrievalHit]:
    if not memories:
        return []
    dimension = memories[0].embedding.shape[0]
    if query_vector.ndim != 1 or query_vector.shape[0] != dimension:
        raise ValueError("Query embedding dimension does not match stored memories")
    if any(memory.embedding.shape != (dimension,) for memory in memories):
        raise ValueError("Stored memories contain inconsistent embedding dimensions")

    matrix = np.vstack([memory.embedding for memory in memories])
    dense_scores = matrix @ query_vector
    dense_order = np.argsort(-dense_scores, kind="stable")
    fused = np.zeros(len(memories), dtype=np.float32)
    for rank, position in enumerate(dense_order, 1):
        fused[int(position)] += 1.0 / (rrf_k + rank)

    if lexical_weight > 0:
        lexical = bm25_scores([memory.content for memory in memories], query_text)
        lexical_positions = np.flatnonzero(lexical > 0)
        lexical_order = lexical_positions[
            np.argsort(-lexical[lexical_positions], kind="stable")
        ]
        for rank, position in enumerate(lexical_order, 1):
            fused[int(position)] += lexical_weight / (rrf_k + rank)

    ranked = np.lexsort((-dense_scores, -fused))
    semantic_bridges = _semantic_bridges(
        ranked,
        memories,
        matrix,
        seed_k=min(semantic_bridge_seed_k, len(memories)),
        minimum_similarity=semantic_bridge_min_similarity,
    )
    positions = _expanded_positions(
        ranked,
        memories,
        limit=min(top_k, len(memories)),
        window=result_window,
        seed_k=min(result_window_seed_k, len(memories)),
        semantic_bridges=semantic_bridges,
        semantic_seed_k=min(semantic_bridge_seed_k, len(memories)),
    )
    return [
        RetrievalHit(memory=memories[position], score=float(fused[position]))
        for position in positions
    ]
