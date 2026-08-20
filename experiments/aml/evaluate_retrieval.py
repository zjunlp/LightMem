from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.retrieval import bm25_scores, lexical_tokens  # noqa: E402


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
VARIANTS = ("baseline", "semantic10", "anchor10", "combined10")


def stratified_sample(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["question_type"]].append(row)

    selected: list[dict] = []
    for question_type in sorted(groups):
        group = groups[question_type]
        count = max(1, round(len(group) * fraction))
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['question_id']}".encode("utf-8")
            ).digest(),
        )
        selected.extend(ordered[:count])
    return sorted(selected, key=lambda row: row["question_id"])


def flatten_messages(row: dict) -> tuple[list[str], list[str], list[int]]:
    documents: list[str] = []
    session_ids: list[str] = []
    message_offsets: list[int] = []
    for session_id, session in zip(
        row["haystack_session_ids"], row["haystack_sessions"], strict=True
    ):
        for offset, message in enumerate(session):
            documents.append(str(message["content"]).strip())
            session_ids.append(str(session_id))
            message_offsets.append(offset)
    return documents, session_ids, message_offsets


def fused_ranking(
    document_vectors: np.ndarray,
    query_vector: np.ndarray,
    documents: list[str],
    query: str,
    *,
    lexical_weight: float = 0.5,
    rrf_k: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    dense_scores = document_vectors @ query_vector
    dense_order = np.argsort(-dense_scores, kind="stable")
    fused = np.zeros(len(documents), dtype=np.float32)
    for rank, position in enumerate(dense_order, 1):
        fused[int(position)] += 1.0 / (rrf_k + rank)

    lexical = bm25_scores(documents, query)
    lexical_positions = np.flatnonzero(lexical > 0)
    lexical_order = lexical_positions[
        np.argsort(-lexical[lexical_positions], kind="stable")
    ]
    for rank, position in enumerate(lexical_order, 1):
        fused[int(position)] += lexical_weight / (rrf_k + rank)
    return np.lexsort((-dense_scores, -fused)), dense_scores


def session_neighbors(
    session_ids: list[str], message_offsets: list[int]
) -> dict[int, tuple[int | None, int | None]]:
    sessions: dict[str, list[int]] = defaultdict(list)
    for position, session_id in enumerate(session_ids):
        sessions[session_id].append(position)
    result: dict[int, tuple[int | None, int | None]] = {}
    for positions in sessions.values():
        positions.sort(key=message_offsets.__getitem__)
        for offset, position in enumerate(positions):
            result[position] = (
                positions[offset - 1] if offset else None,
                positions[offset + 1] if offset + 1 < len(positions) else None,
            )
    return result


def semantic_bridges(
    ranked: np.ndarray,
    document_vectors: np.ndarray,
    session_ids: list[str],
    *,
    seed_k: int,
    minimum_similarity: float,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for raw_seed in ranked[:seed_k]:
        seed = int(raw_seed)
        similarities = document_vectors @ document_vectors[seed]
        similarities[np.asarray(session_ids) == session_ids[seed]] = -np.inf
        candidate = int(np.argmax(similarities))
        if similarities[candidate] >= minimum_similarity:
            result[seed] = candidate
    return result


def anchor_bridges(
    ranked: np.ndarray,
    documents: list[str],
    session_ids: list[str],
    dense_scores: np.ndarray,
    *,
    seed_k: int,
) -> dict[int, int]:
    token_sets = [set(lexical_tokens(document)) for document in documents]
    document_frequency = Counter(token for tokens in token_sets for token in tokens)
    maximum_frequency = max(3, math.ceil(len(documents) * 0.01))
    postings: dict[str, list[int]] = defaultdict(list)
    for position, tokens in enumerate(token_sets):
        for token in tokens:
            if len(token) >= 3 and document_frequency[token] <= maximum_frequency:
                postings[token].append(position)

    result: dict[int, int] = {}
    for raw_seed in ranked[:seed_k]:
        seed = int(raw_seed)
        candidates: Counter[int] = Counter()
        for token in token_sets[seed]:
            frequency = document_frequency[token]
            if len(token) < 3 or frequency > maximum_frequency:
                continue
            weight = math.log((len(documents) + 1) / (frequency + 1)) + 1.0
            for candidate in postings[token]:
                if session_ids[candidate] != session_ids[seed]:
                    candidates[candidate] += weight
        if candidates:
            result[seed] = max(
                candidates,
                key=lambda position: (candidates[position], dense_scores[position]),
            )
    return result


def expand(
    ranked: np.ndarray,
    neighbors: dict[int, tuple[int | None, int | None]],
    *,
    limit: int,
    local_seed_k: int,
    structure_seed_k: int,
    semantic: dict[int, int] | None = None,
    anchors: dict[int, int] | None = None,
) -> list[int]:
    expanded: list[int] = []
    seen: set[int] = set()

    def add(position: int | None) -> None:
        if position is not None and position not in seen and len(expanded) < limit:
            seen.add(position)
            expanded.append(position)

    for seed_rank, raw_seed in enumerate(ranked[:local_seed_k], 1):
        seed = int(raw_seed)
        add(seed)
        before, after = neighbors[seed]
        add(before)
        add(after)
        if len(expanded) >= limit:
            break
        if (
            seed_rank <= structure_seed_k
            and semantic is not None
            and seed in semantic
            and len(expanded) < limit
        ):
            add(semantic[seed])
        if (
            seed_rank <= structure_seed_k
            and anchors is not None
            and seed in anchors
            and len(expanded) < limit
        ):
            add(anchors[seed])
    for raw_position in ranked:
        add(int(raw_position))
    return expanded


def cached_embeddings(
    model,
    row: dict,
    documents: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / f"{row['question_id']}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return cached["documents"], cached["query"]

    document_vectors = np.asarray(
        model.encode(
            documents,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    query_vector = np.asarray(
        model.encode(
            [QUERY_PREFIX + row["question"]],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0],
        dtype=np.float32,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, documents=document_vectors, query=query_vector)
    return document_vectors, query_vector


def summarize(records: list[dict], top_k: int) -> dict:
    output: dict[str, dict] = {}
    recall_key = f"recall@{top_k}"
    hit_key = f"hit@{top_k}"
    for variant in VARIANTS:
        variants = [record[variant] for record in records]
        output[variant] = {
            recall_key: sum(item["recall"] for item in variants) / len(variants),
            hit_key: sum(item["hit"] for item in variants) / len(variants),
            "by_type": {},
        }
        for question_type in sorted({record["question_type"] for record in records}):
            subset = [
                record[variant]
                for record in records
                if record["question_type"] == question_type
            ]
            output[variant]["by_type"][question_type] = {
                "count": len(subset),
                recall_key: sum(item["recall"] for item in subset) / len(subset),
                hit_key: sum(item["hit"] for item in subset) / len(subset),
            }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--revision", default=BGE_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--semantic-threshold", type=float, default=0.72)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/structmem-longmemeval-cache"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")

    from sentence_transformers import SentenceTransformer

    rows = json.loads(args.dataset.read_text())
    sample = stratified_sample(rows, args.fraction, args.seed)
    model = SentenceTransformer(
        args.model,
        revision=args.revision,
        device=args.device,
        local_files_only=True,
    )
    records: list[dict] = []
    print(f"sample={len(sample)}/{len(rows)} seed={args.seed}", flush=True)

    for index, row in enumerate(sample, 1):
        documents, session_ids, offsets = flatten_messages(row)
        document_vectors, query_vector = cached_embeddings(
            model, row, documents, args.cache_dir
        )
        ranked, dense_scores = fused_ranking(
            document_vectors, query_vector, documents, row["question"]
        )
        neighbors = session_neighbors(session_ids, offsets)
        semantic = semantic_bridges(
            ranked,
            document_vectors,
            session_ids,
            seed_k=10,
            minimum_similarity=args.semantic_threshold,
        )
        anchors = anchor_bridges(
            ranked,
            documents,
            session_ids,
            dense_scores,
            seed_k=10,
        )
        positions = {
            "baseline": expand(
                ranked,
                neighbors,
                limit=args.top_k,
                local_seed_k=20,
                structure_seed_k=0,
            ),
            "semantic10": expand(
                ranked,
                neighbors,
                limit=args.top_k,
                local_seed_k=20,
                structure_seed_k=10,
                semantic=semantic,
            ),
            "anchor10": expand(
                ranked,
                neighbors,
                limit=args.top_k,
                local_seed_k=20,
                structure_seed_k=10,
                anchors=anchors,
            ),
            "combined10": expand(
                ranked,
                neighbors,
                limit=args.top_k,
                local_seed_k=20,
                structure_seed_k=10,
                semantic=semantic,
                anchors=anchors,
            ),
        }
        answer_sessions = set(map(str, row["answer_session_ids"]))
        record = {
            "question_id": row["question_id"],
            "question_type": row["question_type"],
        }
        for variant, selected_positions in positions.items():
            retrieved_sessions = {
                session_ids[position] for position in selected_positions
            }
            overlap = answer_sessions.intersection(retrieved_sessions)
            record[variant] = {
                "recall": len(overlap) / len(answer_sessions),
                "hit": float(bool(overlap)),
            }
        records.append(record)
        print(
            f"[{index:02d}/{len(sample)}] {row['question_id']} {row['question_type']}",
            flush=True,
        )

    result = {
        "dataset": str(args.dataset),
        "sample_fraction": args.fraction,
        "sample_seed": args.seed,
        "sample_size": len(sample),
        "model": args.model,
        "model_revision": args.revision,
        "top_k": args.top_k,
        "semantic_threshold": args.semantic_threshold,
        "summary": summarize(records, args.top_k),
        "records": records,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
