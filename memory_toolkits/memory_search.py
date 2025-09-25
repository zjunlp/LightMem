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

def memory_search(
    layer_type: str,
    user_id: str,
    questions: List[QuestionAnswerPair],
    config: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    strict: bool = True, 
) -> List[Dict[str, Any]]:
    """Search memories for a given user based on questions."""
    config = config or {}
    config["user_id"] = user_id
    config["save_dir"] = f"{layer_type}/{user_id}"
    
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
    retrievals = []
    for qa_pair in questions:
        query = qa_pair.question
        # Perform retrieval using the unified interface
        retrieved_memories = layer.retrieve(query, k=top_k)
        retrieval_result = {
            "retrieved_memories": retrieved_memories,
            "qa_pair": qa_pair,
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
    print("✅ The dataset is loaded successfully.")
    
    # Load configuration
    config = None
    if args.config_path is not None:
        with open(args.config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)
    
    # Process index range
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(dataset)
    args.start_idx, args.end_idx = max(0, args.start_idx), min(args.end_idx, len(dataset))
    if args.start_idx >= args.end_idx:
        raise ValueError("The starting index must be less than the ending index.")
    
    # Perform memory 
    print("🔍 Searching memories for each trajectory...")
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
            )
            futures.append(future)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="🔍 Searching memories"
        ):
            results = future.result()
            retrievals.extend(results)

    for item in retrievals:
        item["qa_pair"] = item["qa_pair"].model_dump()
    output_path = f"{args.memory_type}_{args.dataset_type}_{args.top_k}_{args.start_idx}_{args.end_idx}.json"

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(
            retrievals, 
            f, 
            ensure_ascii=False, 
            indent=4,
        )
    print(f"✅ Saved {len(retrievals)} results to {output_path}.")