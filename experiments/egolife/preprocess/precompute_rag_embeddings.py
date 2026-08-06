#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import argparse
import json
import logging
import os
from typing import Dict, Tuple

from em2mem.embedding import EmbeddingModel
from em2mem.memory.multimodal_memory_cell import MemoryCell
from em2mem.memory.semantic_graph import SemanticMemory


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _first_existing(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def build_episodic_caption_file_map(memory_cell_dir: str, person: str) -> Dict[str, str]:
    file_map: Dict[str, str] = {}
    patterns = {
        "30sec": [os.path.join(memory_cell_dir, f"{person}_record.json")],
        "3min": [os.path.join(memory_cell_dir, "temporal_context_views", "temporal_context_views_3min.json")],
        "10min": [os.path.join(memory_cell_dir, "temporal_context_views", "temporal_context_views_10min.json")],
        "1h": [os.path.join(memory_cell_dir, "temporal_context_views", "temporal_context_views_1h.json")],
    }
    for granularity, candidates in patterns.items():
        path = _first_existing(candidates)
        if path:
            file_map[granularity] = path
        else:
            logger.warning("Missing %s caption file under %s", granularity, memory_cell_dir)
    return file_map


def resolve_device(gpu: str) -> Tuple[str, str]:
    if gpu.lower() == "cpu":
        return "cpu", ""
    return "cuda:0", gpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute EgoLife RAG text embedding caches.")
    parser.add_argument("--person", default="A1_JAKE", help="Subject ID")
    parser.add_argument("--memory-cell-dir", required=True, help="Directory containing EgoLife multimodal memory cell files")
    parser.add_argument("--semantic-file", default=None, help="Semantic graph JSON file for semantic RAG cache")
    parser.add_argument("--text-embedding-model", default="Qwen/Qwen3-Embedding-4B", help="Text embedding model id or local path")
    parser.add_argument("--cache-tag", default=None, help="Dense cache tag. Defaults to basename(memory-cell-dir).")
    parser.add_argument("--gpu", default="0", help='GPU id for precompute, or "cpu"')
    parser.add_argument("--batch-size", type=int, default=None, help="Override EPISODIC_DENSE_BATCH_SIZE")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild even when dense cache metadata matches")
    args = parser.parse_args()

    if args.batch_size is not None:
        os.environ["EPISODIC_DENSE_BATCH_SIZE"] = str(args.batch_size)

    device, visible_devices = resolve_device(args.gpu)
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    cache_tag = args.cache_tag or os.path.basename(os.path.normpath(args.memory_cell_dir))
    caption_files = build_episodic_caption_file_map(args.memory_cell_dir, args.person)
    if "30sec" not in caption_files:
        raise FileNotFoundError(
            f"30sec memory cell file is required: {os.path.join(args.memory_cell_dir, f'{args.person}_record.json')}"
        )

    logger.info("Loading text embedding model: %s", args.text_embedding_model)
    embedding_model = EmbeddingModel(
        text_model_name=args.text_embedding_model,
        device=device,
    )
    embedding_model.load_model("text")

    memory_cell = MemoryCell(
        embedding_model=embedding_model,
        llm_model=None,
        prompt_template_manager=None,
        cache_tag=cache_tag,
    )
    memory_cell.load_captions_from_files(caption_files)
    memory_cell.build_dense_index(force_rebuild=args.force_rebuild)

    if args.semantic_file:
        if not os.path.exists(args.semantic_file):
            logger.warning("Semantic graph file not found; semantic dense cache skipped: %s", args.semantic_file)
        else:
            logger.info("Loading semantic graph: %s", args.semantic_file)
            with open(args.semantic_file, "r", encoding="utf-8") as f:
                semantic_data = json.load(f)
            semantic_memory = SemanticMemory(
                embedding_model=embedding_model,
                cache_tag=cache_tag,
            )
            semantic_memory.load_triples_from_data(semantic_data)
            semantic_memory.build_dense_cache(force_rebuild=args.force_rebuild)

    logger.info("EgoLife episodic embedding cache ready: .cache/episodic_dense/%s", cache_tag)
    logger.info("EgoLife semantic embedding cache ready: .cache/semantic_dense/%s", cache_tag)


if __name__ == "__main__":
    main()
