from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .store import StoredMemory


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", re.UNICODE)
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
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


def diversify_by_session(
    ranked: np.ndarray, memories: list[StoredMemory], *, rank_decay: float = 0.0,
    rrf_k: int = 60,
) -> np.ndarray:
    """Put one representative from each session first, then preserve score order.

    A positive decay scores a session by several of its ranked hits. This lets a
    session with coherent evidence outrank one represented by a single accidental
    match while retaining the top document as its representative.
    """
    if rank_decay < 0 or rank_decay > 1:
        raise ValueError("rank_decay must be between 0 and 1")
    if rank_decay:
        session_positions: dict[str, list[int]] = defaultdict(list)
        for rank, raw_position in enumerate(ranked):
            session_positions[str(memories[int(raw_position)].session_id)].append(rank)
        session_order = sorted(
            session_positions,
            key=lambda session: sum(
                rank_decay ** index / (rrf_k + rank + 1)
                for index, rank in enumerate(session_positions[session])
            ),
            reverse=True,
        )
        representatives = [
            int(ranked[session_positions[session][0]]) for session in session_order
        ]
        representative_set = set(representatives)
        return np.asarray(
            representatives + [
                int(raw_position) for raw_position in ranked
                if int(raw_position) not in representative_set
            ],
            dtype=np.int64,
        )
    result: list[int] = []
    seen_sessions: set[str] = set()
    seen_positions: set[int] = set()
    for raw_position in ranked:
        position = int(raw_position)
        session_id = str(memories[position].session_id)
        if session_id not in seen_sessions:
            result.append(position)
            seen_sessions.add(session_id)
            seen_positions.add(position)
    result.extend(
        int(raw_position) for raw_position in ranked
        if int(raw_position) not in seen_positions
    )
    return np.asarray(result, dtype=np.int64)


def retrieval_query(query: str | list[dict], options: list[str] | None) -> str:
    if isinstance(query, list):
        query = "\n".join(
            str(part.get("text") or "")
            for part in query
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if not options:
        return query.strip()
    return (
        query.strip()
        + "\nCandidate answers:\n"
        + "\n".join(option.strip() for option in options)
    )


def lexical_tokens(text: str, *, cjk_ngrams: bool = False) -> list[str]:
    tokens = [
        token
        for token in (
            match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)
        )
        if token not in STOP_WORDS
    ]
    # Whitespace-free CJK text otherwise becomes one sentence-long token, which
    # prevents BM25 from matching a question to an overlapping caption phrase.
    if cjk_ngrams:
        for span in CJK_PATTERN.findall(text):
            tokens.extend(span[index:index + 2] for index in range(len(span) - 1))
    return tokens


def bm25_scores(
    documents: list[str], query: str, *, k1: float = 1.2, b: float = 0.75,
    cjk_ngrams: bool | None = None,
) -> np.ndarray:
    scores = np.zeros(len(documents), dtype=np.float32)
    use_cjk_ngrams = bool(CJK_PATTERN.search(query)) if cjk_ngrams is None else cjk_ngrams
    query_terms = set(lexical_tokens(query, cjk_ngrams=use_cjk_ngrams))
    if not documents or not query_terms:
        return scores
    tokenized = [
        lexical_tokens(document, cjk_ngrams=use_cjk_ngrams) for document in documents
    ]
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


def _session_anchor_positions(
    positions: list[int],
    memories: list[StoredMemory],
    *,
    seed_k: int,
) -> list[int]:
    """Use a repeated session slot for its first raw user message.

    Preserve each session's first hit, the session set, and the result budget.
    """
    selected = list(positions)
    if seed_k <= 0 or len(selected) < 2:
        return selected

    selected_set = set(selected)
    seed_sessions = {
        memories[position].session_id for position in selected[:seed_k]
    }
    slots: dict[str, list[int]] = defaultdict(list)
    for rank, position in enumerate(selected):
        slots[memories[position].session_id].append(rank)
    anchors: dict[str, int] = {}
    for position, memory in enumerate(memories):
        session_id = memory.session_id
        if (
            session_id not in seed_sessions or len(slots[session_id]) < 2
            or memory.derived or memory.role.strip().casefold() != "user"
        ):
            continue
        prior = anchors.get(session_id)
        if prior is None or (memory.sequence, memory.message_index, position) < (
            memories[prior].sequence, memories[prior].message_index, prior
        ):
            anchors[session_id] = position
    for session_id, anchor in anchors.items():
        if anchor not in selected_set:
            selected[slots[session_id][-1]] = anchor
    return selected


def hybrid_search(
    memories: list[StoredMemory],
    *,
    query_vector: np.ndarray,
    query_text: str,
    top_k: int,
    lexical_weight: float,
    rrf_k: int,
    fusion: str = "rrf",
    result_window: int,
    result_window_seed_k: int,
    semantic_bridge_seed_k: int,
    semantic_bridge_min_similarity: float,
    cjk_ngrams: bool | None = None,
    session_diversity: bool = False,
    session_rank_decay: float = 0.0,
    session_recency_weight: float = 0.0,
    session_anchor_seed_k: int = 0,
    session_repeat_penalty: float = 0.0,
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
    for index, memory in enumerate(memories):
        if memory.embedding_chunks:
            chunks = np.vstack(memory.embedding_chunks)
            if chunks.shape[1] != dimension:
                raise ValueError("Stored chunk embeddings contain inconsistent dimensions")
            dense_scores[index] = max(dense_scores[index], float(np.max(chunks @ query_vector)))
    if session_recency_weight < 0:
        raise ValueError("session_recency_weight cannot be negative")
    if session_anchor_seed_k < 0:
        raise ValueError("session_anchor_seed_k cannot be negative")
    if not math.isfinite(session_repeat_penalty) or session_repeat_penalty < 0:
        raise ValueError("session_repeat_penalty must be finite and non-negative")
    if session_recency_weight and len(memories) > 1:
        first_sequence: dict[str, int] = {}
        for memory in memories:
            first_sequence[memory.session_id] = min(
                first_sequence.get(memory.session_id, memory.sequence), memory.sequence
            )
        session_order = {
            session_id: ordinal
            for ordinal, (session_id, _) in enumerate(
                sorted(first_sequence.items(), key=lambda item: item[1])
            )
        }
        age = np.asarray(
            [len(session_order) - session_order[memory.session_id] for memory in memories],
            dtype=np.float32,
        )
        recency = -np.log1p(age)

        def zscore(values: np.ndarray) -> np.ndarray:
            deviation = float(np.std(values))
            if deviation == 0:
                return np.zeros_like(values)
            return (values - float(np.mean(values))) / deviation

        dense_scores = zscore(dense_scores) + session_recency_weight * zscore(recency)
    # Captions are part of the indexed embedding text and must also participate
    # in lexical fusion; otherwise caption-only questions lose exact terms.
    lexical_documents = [
        "\n".join(
            [memory.content, *(part.caption or "" for part in memory.media)]
        ).strip()
        for memory in memories
    ]
    lexical = bm25_scores(lexical_documents, query_text, cjk_ngrams=cjk_ngrams)
    if fusion == "minmax":
        def normalize(scores: np.ndarray) -> np.ndarray:
            minimum, maximum = float(scores.min()), float(scores.max())
            if maximum == minimum:
                return np.zeros_like(scores)
            return (scores - minimum) / (maximum - minimum)

        fused = normalize(dense_scores) + lexical_weight * normalize(lexical)
    else:
        dense_order = np.argsort(-dense_scores, kind="stable")
        fused = np.zeros(len(memories), dtype=np.float32)
        for rank, position in enumerate(dense_order, 1):
            fused[int(position)] += 1.0 / (rrf_k + rank)

    if fusion == "rrf" and lexical_weight > 0:
        lexical_positions = np.flatnonzero(lexical > 0)
        lexical_order = lexical_positions[
            np.argsort(-lexical[lexical_positions], kind="stable")
        ]
        for rank, position in enumerate(lexical_order, 1):
            fused[int(position)] += lexical_weight / (rrf_k + rank)

    if fusion not in {"rrf", "minmax"}:
        raise ValueError("fusion must be 'rrf' or 'minmax'")
    ranked = np.lexsort((-dense_scores, -fused))
    ranking_scores = fused
    if session_repeat_penalty:
        # Penalize repeated evidence softly after the two retrieval signals are
        # fused. The first hit from each session keeps its standardized score;
        # later baseline-ranked hits lose one penalty unit per prior hit.
        deviation = float(np.std(fused))
        standardized = (
            np.zeros_like(fused)
            if deviation == 0
            else (fused - float(np.mean(fused))) / deviation
        )
        ordinal_by_position = np.zeros(len(memories), dtype=np.float32)
        session_counts: dict[str, int] = defaultdict(int)
        for raw_position in ranked:
            position = int(raw_position)
            session_id = str(memories[position].session_id)
            ordinal_by_position[position] = session_counts[session_id]
            session_counts[session_id] += 1
        ranking_scores = standardized - session_repeat_penalty * ordinal_by_position
        ranked = np.lexsort((-dense_scores, -ranking_scores))
    if session_diversity:
        ranked = diversify_by_session(
            ranked, memories, rank_decay=session_rank_decay, rrf_k=rrf_k
        )
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
    positions = _session_anchor_positions(
        positions, memories, seed_k=session_anchor_seed_k,
    )
    # Context selection is a final reranking step. Report its reciprocal rank
    # score instead of attributing another message's similarity to an anchor.
    return [
        RetrievalHit(
            memory=memories[position],
            score=(
                1.0 / (rrf_k + rank)
                if session_anchor_seed_k
                else float(ranking_scores[position])
            ),
        )
        for rank, position in enumerate(positions, 1)
    ]
