#!/usr/bin/env python3
# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors
import os
import json
import re
import glob
import queue as pyqueue
import argparse
import logging
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Tuple, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from em2mem.embedding import EmbeddingModel
from em2mem.llm import LLMModel, PromptTemplateManager
from em2mem.memory import EM2Memory, QAResult, transform_timestamp


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")


# -----------------------------------------------------
# helpers
# -----------------------------------------------------

def load_json(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    match = re.match(r"\(?([A-Za-z])[\.\)]?\s*", text.strip())
    return match.group(1).upper() if match else None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])

    if pred_norm == gold_candidate:
        return True

    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True

    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}"),
    ]
    if pred_norm in full_patterns:
        return True

    return False


def find_30sec_segment(target_timestamp: int, segments_30sec: List[Dict[str, Any]]) -> Tuple[int, int]:
    for segment in segments_30sec:
        date = segment.get("date", "")
        start_time_raw = segment.get("start_time", 0)
        end_time_raw = segment.get("end_time", 0)

        day = date.replace("DAY", "").replace("Day", "") if isinstance(date, str) else str(date)

        if isinstance(start_time_raw, str):
            start_time = int(day + start_time_raw.zfill(8))
        elif isinstance(start_time_raw, int):
            start_time = int(day + str(start_time_raw).zfill(8))
        else:
            continue

        if isinstance(end_time_raw, str):
            end_time = int(day + end_time_raw.zfill(8))
        elif isinstance(end_time_raw, int):
            end_time = int(day + str(end_time_raw).zfill(8))
        else:
            continue

        if start_time <= target_timestamp <= end_time:
            return (start_time, end_time)

    return (0, 0)


def parse_target_time(row: Dict[str, Any], segments_30sec: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    target_time_list = []

    if "time" in row["target_time"] and row["target_time"]["time"]:
        time_str = row["target_time"]["time"]
        time_str_upper = time_str.upper()

        if "DAY" in time_str_upper:
            parts = re.split(r"DAY|Day", time_str, maxsplit=1)
            if len(parts) == 2:
                start_time_str = parts[0]
                day_and_end = parts[1].split("_")
                if len(day_and_end) == 2:
                    end_day = day_and_end[0]
                    end_time_str = day_and_end[1]
                    start_day = row["target_time"]["date"].replace("DAY", "").replace("Day", "")

                    start_time = int(start_day + start_time_str.zfill(8))
                    end_time = int(end_day + end_time_str.zfill(8))
                    target_time_list.append((start_time, end_time))
        else:
            day = row["target_time"]["date"].replace("DAY", "").replace("Day", "")
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30sec_segment(target_timestamp, segments_30sec)
            if segment != (0, 0):
                target_time_list.append(segment)

    elif "time_list" in row["target_time"] and row["target_time"]["time_list"]:
        day = row["target_time"]["date"].replace("DAY", "").replace("Day", "")
        for time_str in row["target_time"]["time_list"]:
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30sec_segment(target_timestamp, segments_30sec)
            if segment != (0, 0):
                target_time_list.append(segment)

    return target_time_list


def _first_existing(candidates: List[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _glob_first(candidates: List[str]) -> Optional[str]:
    for pattern in candidates:
        matched = sorted(glob.glob(pattern))
        if matched:
            return matched[0]
    return None


def build_episodic_caption_file_map(base_dir: str, subject: str) -> Dict[str, str]:
    file_map: Dict[str, str] = {}
    scale_to_patterns = {
        "30sec": [
            os.path.join(base_dir, f"{subject}_record.json")
        ],
        "3min": [
            os.path.join(base_dir, "temporal_context_views", "temporal_context_views_3min.json")
        ],
        "10min": [
            os.path.join(base_dir, "temporal_context_views", "temporal_context_views_10min.json")
        ],
        "1h": [
            os.path.join(base_dir, "temporal_context_views", "temporal_context_views_1h.json")
        ],
    }
    scale_to_globs = {
        "30sec": [os.path.join(base_dir, "*30sec.json"), os.path.join(base_dir, "*30s.json")],
        "3min": [os.path.join(base_dir, "*3min.json")],
        "10min": [os.path.join(base_dir, "*10min.json")],
        "1h": [os.path.join(base_dir, "*1h.json")],
    }
    for scale in ["30sec", "3min", "10min", "1h"]:
        path = _first_existing(scale_to_patterns[scale]) or _glob_first(scale_to_globs[scale])
        if path:
            file_map[scale] = path
    return file_map


def build_episodic_sidecar_file_maps(base_dir: str, model_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    triplet_files = {}
    graph_files = {}
    scales = ["30sec", "3min", "10min", "1h"]
    for scale in scales:
        triplet_files[scale] = os.path.join(base_dir, scale, f"episodic_triplets_{scale}_{model_name}.json")
        graph_files[scale] = os.path.join(base_dir, scale, f"episodic_graph_{scale}_{model_name}.json")
    return triplet_files, graph_files


def filter_existing_files(file_map: Dict[str, str]) -> Dict[str, str]:
    existing = {}
    for k, v in file_map.items():
        if os.path.exists(v):
            existing[k] = v
        else:
            logger.warning(f"Missing file for {k}: {v}")
    return existing


def extract_query_time(row: Dict[str, Any]) -> int:
    return int(row["query_time"]["date"][-1] + row["query_time"]["time"].zfill(8))


def shard_rows_by_time(
    indexed_rows: List[Tuple[int, Dict[str, Any]]],
    num_workers: int,
) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """
    Sort by query_time ascending, then round-robin into shards.
    Each shard remains monotonic in query_time.
    """
    sorted_rows = sorted(indexed_rows, key=lambda x: extract_query_time(x[1]))
    shards: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(num_workers)]
    for i, item in enumerate(sorted_rows):
        shards[i % num_workers].append(item)
    return [shard for shard in shards if shard]


def build_worker_em2mem(config: Dict[str, Any], worker_id: int) -> Tuple[EM2Memory, List[Dict[str, Any]]]:
    gpu_ids = config.get("gpu_ids", [0])
    physical_gpu_id = gpu_ids[worker_id % len(gpu_ids)]

    logger.info(f"[worker-{worker_id}] Binding to physical GPU {physical_gpu_id}")

    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(physical_gpu_id)
        device_str = f"cuda:{physical_gpu_id}"
    else:
        device_str = "cpu"

    logger.info(f"[worker-{worker_id}] Initializing models on {device_str}...")
    embedding_model = EmbeddingModel(
        text_model_name=config["text_embedding_model"],
        device=device_str,
    )
    embedding_model.load_model("text")
    retriever_llm_model = LLMModel(model_name=config["retriever_model"])
    respond_llm_model = LLMModel(model_name=config["respond_model"], fps=1)
    prompt_template_manager = PromptTemplateManager()

    em2mem = EM2Memory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        max_rounds=config["max_rounds"],
        max_errors=config["max_errors"],
        episodic_cache_tag=config["episodic_cache_tag"],
    )
    em2mem.set_retrieval_top_k(
        episodic=config["episodic_top_k"],
        semantic=config["semantic_top_k"],
        visual=config["visual_top_k"],
    )

    logger.info(f"[worker-{worker_id}] Loading data into EM2Memory...")
    em2mem.load_episodic_captions(caption_files=config["episodic_caption_files"])

    if config["episodic_triplet_files"] or config["episodic_graph_files"]:
        logger.info(f"[worker-{worker_id}] Loading episodic triplet/graph sidecar...")
        em2mem.load_episodic_sidecar(
            triplet_files=config["episodic_triplet_files"] or None,
            graph_files=config["episodic_graph_files"] or None,
        )
    else:
        logger.warning(f"[worker-{worker_id}] No episodic sidecar files found.")

    semantic_results = load_json(config["semantic_path"])
    em2mem.load_semantic_triples(data=semantic_results)

    visual_evidence_data = load_json(config["visual_evidence_file"])
    em2mem.load_visual_clips(
        embeddings_path=config["visual_embeddings_path"],
        clips_data=visual_evidence_data,
    )

    logger.info(f"[worker-{worker_id}] Warming up episodic dense cache...")
    if hasattr(em2mem, "prepare_episodic_dense_index"):
        em2mem.prepare_episodic_dense_index(force_rebuild=False)
    if hasattr(em2mem, "prepare_semantic_dense_index"):
        logger.info(f"[worker-{worker_id}] Warming up semantic dense cache...")
        em2mem.prepare_semantic_dense_index(force_rebuild=False)

    episodic_captions_30sec = load_json(config["episodic_caption_files"]["30sec"])
    return em2mem, episodic_captions_30sec

def process_shard(
    worker_id: int,
    shard: List[Tuple[int, Dict[str, Any]]],
    config: Dict[str, Any],
    progress_queue,
) -> List[Tuple[int, Dict[str, Any], int]]:
    em2mem, episodic_captions_30sec = build_worker_em2mem(config, worker_id)

    results: List[Tuple[int, Dict[str, Any], int]] = []
    evaluate_true_local = 0

    try:
        for dataset_idx, row in shard:
            ID = row["ID"]
            query_type = row["type"]
            question = row["question"]
            answer = row["answer"]

            logger.info(f"[worker-{worker_id}] Processing ID {ID}: {question[:80]}...")

            choices = {}
            for key, label in [("choice_a", "A"), ("choice_b", "B"), ("choice_c", "C"), ("choice_d", "D")]:
                if key in row and row[key]:
                    choices[label] = row[key]

            query_time = extract_query_time(row)
            target_time_list = parse_target_time(row, episodic_captions_30sec)

            qa_result: Optional[QAResult] = None
            try:
                qa_result = em2mem.answer(
                    query=question,
                    choices=choices,
                    until_time=query_time,
                )
                response = qa_result.answer
            except Exception as e:
                logger.error(f"[worker-{worker_id}] Error processing ID {ID}: {e}")
                response = "Error"

            evaluate = evaluate_prediction(response, answer, choices)
            evaluate_true_local += int(evaluate)

            result_entry = {
                "ID": ID,
                "type": query_type,
                "question": question,
                "choices": choices,
                "answer": answer,
                "response": response,
                "round_history": qa_result.round_history if qa_result else [],
                "num_rounds": qa_result.num_rounds if qa_result else 0,
                "evaluate": evaluate,
                "query_time": query_time,
                "query_time_str": transform_timestamp(str(query_time)),
                "target_time": target_time_list,
                "target_time_str": [
                    (transform_timestamp(str(start)), transform_timestamp(str(end)))
                    for start, end in target_time_list
                ],
            }
            results.append((dataset_idx, result_entry, int(evaluate)))

            logger.info(
                f"[worker-{worker_id}] ID {ID} Answer: {response}, Gold: {answer}, Correct: {evaluate} "
                f"// Local Accuracy: {evaluate_true_local}/{len(results)} = {evaluate_true_local / len(results):.4f}"
            )

            if progress_queue is not None:
                progress_queue.put({"delta": 1, "correct": int(evaluate), "id": ID, "worker_id": worker_id})
    finally:
        try:
            em2mem.cleanup()
        except Exception:
            pass

    return results


# -----------------------------------------------------
# main
# -----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EgoLifeQA Evaluation with event-centric EM2Memory")
    parser.add_argument("--person", type=str, default="A1_JAKE", help="Subject ID")
    parser.add_argument("--retriever-model", type=str, default="gpt-5-mini", help="LLM model for retrieval")
    parser.add_argument("--respond-model", type=str, default="gpt-5", help="LLM model for answer generation")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before aborting")
    parser.add_argument("--episodic-top-k", type=int, default=3, help="Final number of event anchors")
    parser.add_argument("--semantic-top-k", type=int, default=10, help="Number of semantic facts to retrieve")
    parser.add_argument("--visual-top-k", type=int, default=3, help="Max images per final event anchor")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel worker processes")
    parser.add_argument("--gpu-list", type=str, default="0,1", help='Comma-separated physical GPU ids, e.g. "0,1"')
    parser.add_argument("--text-embedding-model", type=str, default="Qwen/Qwen3-Embedding-4B", help="Text embedding model id or local path")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife", help="Data directory")
    parser.add_argument("--memory-cell-dir", type=str, default=None, help="Directory for multimodal memory cells.")
    parser.add_argument("--semantic-graph-dir", type=str, help="Directory for semantic graph files.")
    parser.add_argument("--visual-dir", type=str, help="Directory for visual evidence files.")
    parser.add_argument("--visual-evidence-file", type=str, help="JSON file containing visual evidence data.")

    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_list.split(",") if x.strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]

    subject = args.person
    data_dir = args.data_dir

    memory_cell_dir = args.memory_cell_dir
    semantic_graph_dir = args.semantic_graph_dir
    visual_dir = args.visual_dir
    visual_evidence_file = args.visual_evidence_file

    episodic_cache_tag = os.path.basename(os.path.normpath(memory_cell_dir))

    logger.info("Loading evaluation data...")
    eval_data_path = os.path.join(data_dir, f"EgoLifeQA/EgoLifeQA_{subject}.json")
    eval_data = load_json(eval_data_path)

    episodic_caption_files = build_episodic_caption_file_map(memory_cell_dir, subject)
    episodic_caption_files = filter_existing_files(episodic_caption_files)
    if "30sec" not in episodic_caption_files:
        raise FileNotFoundError("30sec caption file is required for target-time parsing.")

    episodic_triplet_files, episodic_graph_files = build_episodic_sidecar_file_maps(
        memory_cell_dir,
        args.retriever_model,
    )

    semantic_path = os.path.join(semantic_graph_dir, f"semantic_graph_{args.retriever_model}.json")

    visual_embeddings_path = None
    if visual_dir:
        candidate_visual_path = os.path.join(visual_dir, "visual_embeddings.pkl")
        if os.path.exists(candidate_visual_path):
            visual_embeddings_path = candidate_visual_path
            logger.info(f"Using optional visual embeddings: {visual_embeddings_path}")
        else:
            logger.info("Visual embeddings not found; proceeding with keyframe-only visual evidence.")
    else:
        logger.info("No visual_dir provided; proceeding with keyframe-only visual evidence.")

    worker_config = {
        "subject": subject,
        "retriever_model": args.retriever_model,
        "respond_model": args.respond_model,
        "max_rounds": args.max_rounds,
        "max_errors": args.max_errors,
        "episodic_top_k": args.episodic_top_k,
        "semantic_top_k": args.semantic_top_k,
        "visual_top_k": args.visual_top_k,
        "text_embedding_model": args.text_embedding_model,
        "episodic_cache_tag": episodic_cache_tag,
        "episodic_caption_files": episodic_caption_files,
        "episodic_triplet_files": episodic_triplet_files,
        "episodic_graph_files": episodic_graph_files,
        "semantic_path": semantic_path,
        "visual_embeddings_path": visual_embeddings_path,
        "visual_evidence_file": visual_evidence_file,
        "gpu_ids": gpu_ids,
    }

    indexed_rows = list(enumerate(eval_data))
    num_workers = max(1, args.num_workers)
    shards = shard_rows_by_time(indexed_rows, num_workers)

    logger.info(f"Starting parallel evaluation on {len(eval_data)} samples with {len(shards)} worker shards...")

    results_by_idx: List[Optional[Dict[str, Any]]] = [None] * len(eval_data)
    evaluate_true = 0

    if len(shards) == 1:
        progress_queue = pyqueue.Queue()
        result_holder: List[List[Tuple[int, Dict[str, Any], int]]] = []

        def _run_single():
            result_holder.append(process_shard(0, shards[0], worker_config, progress_queue))

        t = threading.Thread(target=_run_single, daemon=True)
        t.start()

        completed = 0
        global_correct = 0
        with tqdm(total=len(eval_data), desc="Eval") as pbar:
            while t.is_alive() or not progress_queue.empty():
                try:
                    msg = progress_queue.get(timeout=0.5)
                    delta = int(msg.get("delta", 0)) if isinstance(msg, dict) else int(msg)
                    correct = int(msg.get("correct", 0)) if isinstance(msg, dict) else 0
                    completed += delta
                    global_correct += correct
                    pbar.update(delta)
                    if completed > 0:
                        global_acc = global_correct / completed
                        pbar.set_postfix_str(f"Global Acc: {global_correct}/{completed} = {global_acc:.4f}")
                        logger.info(
                            f"[global] Progress: {completed}/{len(eval_data)} // "
                            f"Global Accuracy: {global_correct}/{completed} = {global_acc:.4f}"
                        )
                except pyqueue.Empty:
                    continue

        t.join()
        shard_results = result_holder[0]

        for idx, result_entry, correct in shard_results:
            results_by_idx[idx] = result_entry
            evaluate_true += correct

    else:
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        progress_queue = manager.Queue()
        futures = []

        with ProcessPoolExecutor(max_workers=len(shards), mp_context=ctx) as executor:
            for worker_id, shard in enumerate(shards):
                futures.append(executor.submit(process_shard, worker_id, shard, worker_config, progress_queue))

            completed = 0
            global_correct = 0
            with tqdm(total=len(eval_data), desc="Eval") as pbar:
                while completed < len(eval_data):
                    try:
                        msg = progress_queue.get(timeout=0.5)
                        delta = int(msg.get("delta", 0)) if isinstance(msg, dict) else int(msg)
                        correct = int(msg.get("correct", 0)) if isinstance(msg, dict) else 0
                        completed += delta
                        global_correct += correct
                        pbar.update(delta)
                        if completed > 0:
                            global_acc = global_correct / completed
                            pbar.set_postfix_str(f"Global Acc: {global_correct}/{completed} = {global_acc:.4f}")
                            logger.info(
                                f"[global] Progress: {completed}/{len(eval_data)} // "
                                f"Global Accuracy: {global_correct}/{completed} = {global_acc:.4f}"
                            )
                    except pyqueue.Empty:
                        if all(f.done() for f in futures):
                            break

            for fut in futures:
                shard_results = fut.result()
                for idx, result_entry, correct in shard_results:
                    results_by_idx[idx] = result_entry
                    evaluate_true += correct

    results = [x for x in results_by_idx if x is not None]
    if len(results) != len(eval_data):
        raise RuntimeError(f"Missing results: got {len(results)}, expected {len(eval_data)}")

    output_path = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('-', '_')}_{args.respond_model.replace('-', '_')}",
        f"egolife_eval_{subject}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    final_accuracy = evaluate_true / len(results) if results else 0
    logger.info(f"\n{'=' * 50}")
    logger.info("Evaluation Complete")
    logger.info(f"Subject: {subject}")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Correct: {evaluate_true}")
    logger.info(f"Accuracy: {final_accuracy:.4f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
