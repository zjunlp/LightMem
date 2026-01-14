import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
from copy import deepcopy
from memories import (
    CONFIG_MAPPING,
    MEMORY_LAYERS_MAPPING,
    DATASET_MAPPING,
)
from memories.datasets.base import QuestionAnswerPair
from typing import (
    Dict,
    Any,
    Optional,
    List,
)

_LOCK = threading.Lock()

from collections.abc import Mapping
from types import MappingProxyType
from pydantic import BaseModel
import sys

def to_jsonable(obj):
    """
    Convert any Python object into a type acceptable by json.dump.
    - Scalars/None: returned as-is
    - list/tuple/set: processed recursively
    - dict / Mapping / mappingproxy: processed recursively
    - pydantic BaseModel: use model_dump() then process recursively
    - Other complex types: convert to str(obj)
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump())
    
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(i) for i in obj]
    
    if isinstance(obj, (dict, Mapping, MappingProxyType)):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    
    # Convert remaining types to string to avoid json.dump errors
    return str(obj)


def memory_search(
    layer_type: str,
    user_id: str,
    questions: List[QuestionAnswerPair],
    config: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    strict: bool = True, 
    dataset_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search memories for a given user based on questions."""
    config = config or {}
    llm_model = config["llm_model"]
    config["user_id"] = user_id
    config["save_dir"] = f"{layer_type}_{llm_model}/{user_id}"
    
    # Load memory layer configuration and class using lazy mapping
    config_cls = CONFIG_MAPPING[layer_type]
    config = config_cls(**config)
    
    with _LOCK:
        layer_cls = MEMORY_LAYERS_MAPPING[layer_type]
        layer = layer_cls(config)
    
    # Load the pre-built memory
    with _LOCK:
        if not layer.load_memory(user_id):
            msg = f"No memory found for user {user_id}."
            if strict:
                raise ValueError(msg)
            else:
                # For some baselines, there are a few cases 
                # these baselines cannot process without throwing an error.
                # We simply return an empty memory for these cases.
                print(msg)
                return [
                    {
                        "retrieved_memories": [
                            {
                                "used_content": "[NO RETRIEVED MEMORIES]"
                            }
                        ],
                        "qa_pair": qa_pair,
                    }
                    for qa_pair in questions 
                ]
    
    # Perform retrieval for each question
    # Ensure it supports len(), and convenient for logging total count
    if dataset_type == "LoCoMo":
        original_count = len(questions)
        questions = [
            qa for qa in questions 
            if qa.metadata.get("category_id") != 5
        ]
        filtered_count = original_count - len(questions)
        if filtered_count > 0:
            print(f"[INFO] {user_id}: Filtered out {filtered_count} questions with category=5")
            
    questions = list(questions)
    total_q = len(questions)
    print(f"[INFO] {user_id}: {total_q} questions to search.")

    retrievals = []
    pbar = tqdm(
        questions,
        total=total_q,
        desc=f"{user_id}",
        leave=False,       # Avoid too many 100% progress remnants under nohup
    )

    for qa_pair in pbar:
        query = qa_pair.question
        if dataset_type == "LoCoMo":
            speaker_names = qa_pair.metadata.get("speaker_names", [])
            if len(speaker_names) != 2:
                print(f"[WARNING] Expected 2 speaker names, got {len(speaker_names)}, using standard retrieval")
                retrieved_memories = layer.retrieve(query, k=top_k)
            else:
                speaker_1_name, speaker_2_name = speaker_names
                speaker_1_memories = layer.retrieve(
                    query, k=top_k, name_filter=speaker_1_name
                )
                speaker_2_memories = layer.retrieve(
                    query, k=top_k, name_filter=speaker_2_name
                )
                retrieved_memories = {
                    "speaker_1": {
                        "name": speaker_1_name,
                        "memories": speaker_1_memories,
                    },
                    "speaker_2": {
                        "name": speaker_2_name,
                        "memories": speaker_2_memories,
                    }
                }
        else:
            retrieved_memories = layer.retrieve(query, k=top_k)
            
        # When MemZero enables Graph, the return is a dictionary
        if isinstance(retrieved_memories, dict):
            if "memories" in retrieved_memories and "relations" in retrieved_memories:
                retrieval_result = {
                    "retrieved_memories": retrieved_memories["memories"],
                    "graph_relations": retrieved_memories["relations"],
                    "qa_pair": qa_pair,
                    "user_id": user_id,
                }
            elif "speaker_1" in retrieved_memories and "speaker_2" in retrieved_memories:
                retrieval_result = {
                    "retrieved_memories": retrieved_memories,  
                    "qa_pair": qa_pair,
                    "user_id": user_id,
                    "dataset_type": "LoCoMo",  
                }
            else:
                retrieval_result = {
                    "retrieved_memories": retrieved_memories,
                    "qa_pair": qa_pair,
                    "user_id": user_id,
                }
        else:
            retrieval_result = {
                "retrieved_memories": retrieved_memories,
                "qa_pair": qa_pair,
                "user_id": user_id,
            }
        retrievals.append(retrieval_result)

    return retrievals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to search memories for a given user based on questions."
    )
    parser.add_argument(
        "--memory-type",
        choices=list(MEMORY_LAYERS_MAPPING.keys()),
        type=str,
        required=True,
        help="The type of the memory layer to be searched."
    )
    parser.add_argument(
        "--dataset-type",
        choices=list(DATASET_MAPPING.keys()),
        type=str,
        required=True,
        help="The type of the dataset used to search the memory layer."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The path to the dataset."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="The number of threads to use for the search."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to sample the dataset if the user provides the sample size."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Subset size from dataset."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to JSON config for memory method."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of memories to retrieve for each query."
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="The starting index of the trajectories to be processed."
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="The ending index of the trajectories to be processed."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Whether to raise an error if no memory is found for a user."
    )
    args = parser.parse_args()
    
    # Prepare the dataset using lazy mapping
    ds_cls = DATASET_MAPPING[args.dataset_type]
    dataset = ds_cls.read_raw_data(args.dataset_path)
    if args.sample_size is not None:
        dataset = dataset.sample(size=args.sample_size, seed=args.seed)
    print("The dataset is loaded successfully.")
    
    # Load configuration
    config = None
    if args.config_path is not None:
        with open(args.config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)
    
    llm_model = config["llm_model"]
    # Process index range
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(dataset)
    args.start_idx, args.end_idx = max(0, args.start_idx), min(args.end_idx, len(dataset))
    if args.start_idx >= args.end_idx:
        raise ValueError("The starting index must be less than the ending index.")
    
    # Perform memory 
    print("Searching memories for each trajectory...")
    retrievals = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for trajectory, qa_pairs in zip(*dataset[args.start_idx: args.end_idx]):
            user_id = f"user_{dataset.__class__.__name__}_{trajectory.metadata['id']}"
            future = executor.submit(
                memory_search,
                args.memory_type,
                user_id,
                qa_pairs,
                config=deepcopy(config),
                top_k=args.top_k,
                strict=args.strict,
                dataset_type=args.dataset_type,
            )
            futures.append(future)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Searching memories"
        ):
            results = future.result()
            retrievals.extend(results)

    for item in retrievals:
        item["qa_pair"] = item["qa_pair"].model_dump()
    output_path = f"{args.memory_type}_{llm_model}_{args.dataset_type}_{args.top_k}_{args.start_idx}_{args.end_idx}.json"

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(
            # retrievals, 
            to_jsonable(retrievals),
            f, 
            ensure_ascii=False, 
            indent=4,
        )
    print(f"Saved {len(retrievals)} results to {output_path}.")