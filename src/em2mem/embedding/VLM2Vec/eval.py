import datetime
import logging
import json
import random
import time

import numpy as np
import os
import pickle
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, AutoConfig
from datasets import Dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.eval_collator import MultimodalEvalDataCollator
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
from src.eval_utils.metrics import RankingMetrics
from src.model.model import MMEBModel
from src.model.processor import get_backbone_name, load_processor, COLPALI
from src.utils import batch_to_device, print_rank, print_master
import multiprocessing
from multiprocessing import Pool, cpu_count
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger(__name__)


def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size


def encode_embeddings(
    model: MMEBModel,
    loader: DataLoader,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    full_dataset: Dataset,
    encode_side: str,
    description: str = "Encoding"
) -> tuple[np.ndarray, list]:
    """
    Encodes embeddings for a given dataset using the model, handling both standard and
    late-interaction models in a DDP-safe manner.
    """
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Check if the model is a late-interaction type
    is_late_interaction = (model_args.model_backbone == COLPALI)

    local_embeds = []
    local_gt_infos = []
    local_max_len = 0

    model.eval()
    with torch.no_grad():
        for inputs, dataset_info in tqdm(loader, desc=f"{description} (rank {local_rank})", disable=local_rank > 0):
            inputs = batch_to_device(inputs, training_args.device)
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                # Determine if encoding query or target based on available keys
                if encode_side == "qry":
                    output = model(qry=inputs)
                    reps = output["qry_reps"].detach()
                    local_gt_infos.extend(dataset_info)  # to retain all information per query
                else:
                    output = model(tgt=inputs)
                    reps = output["tgt_reps"].detach()
                    local_gt_infos.extend([info["cand_name"] for info in dataset_info])  # to retain ground-truth labels

            if is_late_interaction and reps.dim() == 3:
                local_max_len = max(local_max_len, reps.shape[1])

            local_embeds.append(reps)

    if not local_embeds:
        # Handle cases where a rank gets no data
        return np.array([]), []

    # === DDP Synchronization and Padding for Late-Interaction Models ===
    if is_late_interaction:
        if dist.is_initialized():
            # 1. Find the global maximum sequence length across all ranks
            local_max_len_tensor = torch.tensor(local_max_len, device=training_args.device)
            dist.all_reduce(local_max_len_tensor, op=dist.ReduceOp.MAX)
            global_max_len = local_max_len_tensor.item()
        else:
            global_max_len = local_max_len

        # 2. Pad all local embeddings to the global max length
        padded_embeds = []
        for reps_batch in local_embeds:
            if reps_batch.dim() == 3:
                B, L, H = reps_batch.shape
                padding_size = global_max_len - L
                padded_batch = F.pad(reps_batch, (0, 0, 0, padding_size), "constant", 0)
                padded_embeds.append(padded_batch)
            else: # Should not happen if model is consistently late-interaction
                padded_embeds.append(reps_batch)

        embeds_tensor = torch.cat(padded_embeds, dim=0).contiguous()
    else: # Standard dense models
        embeds_tensor = torch.cat(local_embeds, dim=0).contiguous()


    # === Gather embeddings and keys from all ranks ===
    if dist.is_initialized() and full_dataset.num_rows >= world_size:
        print_master(f"Gathering {encode_side} embeddings across all ranks...")

        # Use the more efficient all_gather_into_tensor for tensors
        output_shape = list(embeds_tensor.shape)
        output_shape[0] = full_dataset.num_rows
        embeds_tensor = embeds_tensor.to(training_args.device)
        gathered_embeds_tensor = torch.empty(output_shape, dtype=embeds_tensor.dtype, device=training_args.device)
        dist.all_gather_into_tensor(gathered_embeds_tensor, embeds_tensor)
        final_embeddings = gathered_embeds_tensor.cpu().float().numpy()
        # Gather metadata, for which all_gather_object is appropriate
        gathered_gt_infos = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_gt_infos, local_gt_infos)
        all_gt_infos = [key for rank_keys in gathered_gt_infos for key in rank_keys]
    else:
        all_gt_infos = local_gt_infos
        final_embeddings = embeds_tensor.cpu().float().numpy()

    return final_embeddings, all_gt_infos


def main():
    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    local_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # DEBUG PRINTS for Distributed Setup
    print_master("Distributed init debug info:")
    print_master(f"RANK: {os.environ.get('RANK')}")
    print_master(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print_master(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print_master(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print_master(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    if dist.is_initialized():
        print_rank(f"dist.get_rank(): {dist.get_rank()}")
        print_rank(f"dist.get_world_size(): {dist.get_world_size()}")

    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    # --- Model Loading ---
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not getattr(model_args, "model_backbone", None):
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_master(f'Model Backbone: {model_args.model_backbone}')
    # --- DDP-Safe Model Loading ---
    # Step 1: Only the master process (rank 0) downloads the model.
    if local_rank == 0:
        processor = load_processor(model_args, data_args)
        model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
        print_master(f"[rank=0] Loading the model from Huggingface: {model_args.model_name}...")
    # Step 2: All processes wait here. The non-master processes will pause
    # until the master process (rank 0) finishes downloading and exits this barrier.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # Step 3: Now that the model is cached, the non-master processes load it from the local cache.
    if local_rank != 0:
        print_rank(f"Loading the model from cache...")
        processor = load_processor(model_args, data_args)
        time.sleep(random.randint(2 * local_rank, 3 * local_rank))
        model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)
    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)


    # --- Main Evaluation Loop ---
    for dataset_idx, (dataset_name, task_config) in enumerate(dataset_configs.items()):
        # 0. load dataset
        if dist.is_initialized():
            dist.barrier()
        print_master(f"--- Evaluating {dataset_name} ---")

        query_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_qry")
        cand_embed_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_tgt")
        dataset_info_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_info.jsonl")

        do_query = not os.path.exists(query_embed_path) or not os.path.exists(dataset_info_path)
        do_cand = not os.path.exists(cand_embed_path)

        if do_query or do_cand:
            if data_args.data_basedir is not None:
                # Construct full paths for data files if --data_basedir is provided
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if data_args.data_basedir and task_config.get(key):
                        task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

            full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(model_args=model_args, data_args=data_args, **task_config)
            full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
            eval_qry_dataset, eval_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset
            # Pad datasets to be divisible by world_size before splitting
            if dist.is_initialized():
                padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                padded_cand_dataset, _ = pad_dataset_to_divisible(full_eval_cand_dataset, world_size)
                eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=local_rank, world_size=world_size)
                eval_cand_dataset = split_dataset_by_node(padded_cand_dataset, rank=local_rank, world_size=world_size)
            else:
                padded_qry_dataset, padded_cand_dataset = full_eval_qry_dataset, full_eval_cand_dataset

        # --- 1. Compute Query Embeddings ---
        if do_query:
            print_master("Encoding queries...")
            eval_qry_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "qry")
            eval_qry_loader = DataLoader(eval_qry_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=eval_qry_collator, num_workers=training_args.dataloader_num_workers)
            query_embeds, gt_infos = encode_embeddings(model, eval_qry_loader, training_args, model_args, padded_qry_dataset, encode_side="qry", description=f"Queries for {dataset_name}")
            query_embeds = query_embeds[:len(full_eval_qry_dataset)]  # world_size>1, trim the padded data points
            gt_infos = gt_infos[:len(full_eval_qry_dataset)]
            if local_rank == 0:
                with open(query_embed_path, 'wb') as f:
                    pickle.dump(query_embeds, f)
                with open(dataset_info_path, 'w') as f:
                    for info in gt_infos:
                        f.write(json.dumps(info) + '\n')
                print_master(f"Saved query embeddings to {query_embed_path}")
            if dist.is_initialized():
                dist.barrier()


        # --- 2. Compute Candidate Embeddings ---
        if do_cand:
            print_master("Encoding candidates...")
            eval_cand_collator = MultimodalEvalDataCollator(processor, model_args, data_args, "cand")
            eval_cand_loader = DataLoader(eval_cand_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=eval_cand_collator, num_workers=training_args.dataloader_num_workers)

            cand_embeds, all_cand_ids = encode_embeddings(model, eval_cand_loader, training_args, model_args, padded_cand_dataset, encode_side="cand", description=f"Candidates for {dataset_name}")
            cand_embeds = cand_embeds[:len(full_eval_cand_dataset)]  # world_size>1, trim the padded data points
            all_cand_ids = all_cand_ids[:len(full_eval_cand_dataset)]

            if local_rank == 0:
                cand_embed_dict = {cand_id: embed for cand_id, embed in zip(all_cand_ids, cand_embeds)}
                with open(cand_embed_path, 'wb') as f: pickle.dump(cand_embed_dict, f)
                print_master(f"Saved candidate embeddings to {cand_embed_path}")

        if dist.is_initialized():
            dist.barrier()

        # --- 3. Compute Scores (on master rank only) ---
        if local_rank == 0:
            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
            if os.path.exists(score_path):
                try:
                    with open(score_path, "r") as f:
                        score_dict = json.load(f)
                    print_master(f"Score of {dataset_name} (loaded from previous run): {score_path}")
                    formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
                    print_master(formatted)
                    continue
                except Exception as e:
                    print_master(f"Failed to load score for {dataset_name}, skipping {dataset_name}")
            with open(query_embed_path, 'rb') as f: qry_embeds = pickle.load(f)
            with open(cand_embed_path, 'rb') as f: cand_embed_dict = pickle.load(f)
            gt_infos = [json.loads(l) for l in open(dataset_info_path)]
            pred_dicts = []

            rank_against_all_candidates = task_config.get("eval_type", "global") == "global"
            if rank_against_all_candidates:
                cand_keys = list(cand_embed_dict.keys())
                cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])
                # Handle late-interaction scoring
                if qry_embeds.ndim == 3: # Query: [N_q, L_q, H] | Candidate: [N_c, L_c, H]
                    qry_embed = torch.from_numpy(qry_embeds)
                    cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]
                    scores = processor.score(qry_embed, cand_embeds, batch_size=64)  # use ColPali score function
                    ranked_candids = torch.argsort(-scores, dim=1).cpu().numpy().tolist()
                else: # Dense
                    cosine_scores = np.dot(qry_embeds, cand_embeds.T)
                    ranked_candids = np.argsort(-cosine_scores, axis=1)
                for qid, (ranked_candid, gt_info) in tqdm(enumerate(zip(ranked_candids, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                    rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                    rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None
                    assert rel_scores is None or len(rel_docids) == len(rel_scores)
                    pred_dicts.append({
                        "prediction": [cand_keys[i] for i in ranked_candid],
                        "label": rel_docids,
                        "rel_scores": rel_scores,
                    })
            else:
                for qid, (qry_embed, gt_info) in tqdm(enumerate(zip(qry_embeds, gt_infos)), desc=f"Calculating scores for {dataset_name}"):
                    cand_embeds = np.stack([cand_embed_dict[key] for key in gt_info["cand_names"]])
                    if qry_embeds.ndim == 3: # Query: [N_q, L_q, H] | Candidate: [N_c, L_c, H]
                        qry_embed = torch.from_numpy(np.array(qry_embed)).unsqueeze(0)
                        cand_embeds = [torch.from_numpy(np.array(t)) for t in cand_embeds]
                        scores = processor.score(qry_embed, cand_embeds, batch_size=1024)  # use ColPali score function
                        ranked_candids = torch.argsort(-scores, dim=1).cpu().numpy().tolist()[0]
                    else:
                        cosine_score = np.dot(qry_embed, cand_embeds.T)
                        ranked_candids = np.argsort(-cosine_score)
                    rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                    rel_scores = gt_info["rel_scores"] if "rel_scores" in gt_info else None

                    assert rel_scores is None or len(rel_docids) == len(rel_scores)
                    pred_dicts.append({
                        "prediction": [gt_info["cand_names"][i] for i in ranked_candids],
                        "label": rel_docids,
                        "rel_scores": rel_scores,
                    })

            score_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_score.json")
            pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")

            metrics_to_report = task_config["metrics"] if task_config.get("metrics", None) is not None else ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
            metrics = RankingMetrics(metrics_to_report)
            score_dict = metrics.evaluate(pred_dicts)
            formatted = {k: f"{v:.4f}" for k, v in score_dict.items()}
            score_dict["num_pred"] = len(pred_dicts)
            score_dict["num_data"] = len(gt_infos)
            print_master(f"Score of {dataset_name}:")
            print_master(formatted)
            print_master(f"Outputting final score to: {score_path}")
            with open(score_path, "w") as f:
                json.dump(score_dict, f, indent=4)
            with open(pred_path, "w") as f:
                for pred in pred_dicts:
                    f.write(json.dumps(pred) + '\n')


if __name__ == "__main__":
    main()
