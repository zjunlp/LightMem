from memories.datasets.base import QuestionAnswerPair
from memories import (
    CONFIG_MAPPING,
    MEMORY_LAYERS_MAPPING,
)
from inference_utils.operators import (
    MemoryConstructionErrorChecker,
    RetrievalErrorChecker,
)
import argparse
import json
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from tqdm import tqdm
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union, 
    Tuple,
)

from collections.abc import Mapping
from types import MappingProxyType
from pydantic import BaseModel

_LOCK = threading.Lock()

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

def _build_memory_units_text(retrieved_memories: List[Dict[str, Any]]) -> str:
    """Build text representation of retrieved memory units."""
    contents = []
    for i, mem in enumerate(retrieved_memories):
        content = mem.get("used_content", '')
        if not isinstance(content, str):
            content = str(content)
        if content:
            contents.append(f"### Memory Unit {i + 1}:\n{content}")
    return "\n\n".join(contents) if contents else "[NO MEMORY UNITS RETRIEVED]"


def _build_evidences_text(evidences: List[str]) -> str:
    """Build text representation of source evidences."""
    contents = []
    for i, evidence in enumerate(evidences):
        contents.append(f"### Evidence {i + 1}:\n{evidence}")
    return "\n\n".join(contents) if contents else "[NO EVIDENCES]"


def run_error_attribution_for_user(
    user_id: str,
    failed_instances: List[Dict[str, Any]],
    memory_type: str,
    construction_checker: MemoryConstructionErrorChecker,
    retrieval_checker: RetrievalErrorChecker,
    memory_config: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    batch_size: int = 4,
    pbar: Optional[tqdm] = None,
    pbar_lock: Optional[threading.Lock] = None,
) -> List[Dict[str, Any]]:
    """
    Run the error attribution algorithm for a single user's failed instances.
    
    Parameters
    ----------
    user_id: str
        The user ID.
    failed_instances: List[Dict[str, Any]]
        The failed instances for this user.
    memory_type: str
        The type of the memory layer.
    construction_checker: MemoryConstructionErrorChecker
        The operator for checking memory construction errors.
    retrieval_checker: RetrievalErrorChecker
        The operator for checking retrieval errors.
    memory_config: Dict[str, Any], optional
        Configuration for the memory layer.
    top_k: int
        Number of memory units to retrieve for each evidence.
    batch_size: int
        Batch size for LLM inference.
        
    Returns
    -------
    attribution_results: List[Dict[str, Any]]
        The error attribution results for this user's failed instances.
    """
    # Load memory layer once for this user
    config = memory_config.copy() if memory_config else {}
    llm_model = config["llm_model"]
    config["user_id"] = user_id
    config["save_dir"] = f"{memory_type}_{llm_model}/{user_id}"
    
    config_cls = CONFIG_MAPPING[memory_type]
    config = config_cls(**config)
    
    with _LOCK:
        layer_cls = MEMORY_LAYERS_MAPPING[memory_type]
        layer = layer_cls(config)
    
    with _LOCK:
        memory_loaded = layer.load_memory(user_id)
        if not memory_loaded:
            raise ValueError(f"No memory found for user {user_id}")
    
    attribution_results = []
    
    for item in failed_instances:
        try:
            qa_pair_data = item["qa_pair"]
            if isinstance(qa_pair_data, dict):
                qa_pair = QuestionAnswerPair(**qa_pair_data)
            else:
                qa_pair = qa_pair_data
            
            question = qa_pair.question
            golden_answers = list(qa_pair.answer_list)
            prediction = item["prediction"]
            retrieved_memories = item["retrieved_memories"]
            
            # Extract source evidences from metadata
            source_evidences = qa_pair.metadata.get("source_evidences", [])
            if not source_evidences:
                raise ValueError(f"No source evidences found for question answer pair {qa_pair.id}")
            
            # Step 1: Check memory construction errors
            construction_errors = []
            evidence_memory_searches: List[Tuple[str, List[Dict[str, Union[str, Dict[str, Any]]]]]] = []
            
            # Search memory for each evidence using the pre-loaded memory layer
            for evidence in source_evidences:
                evidence = evidence["content"]
                retrieved_for_evidence = layer.retrieve(evidence, k=top_k)
                evidence_memory_searches.append((evidence, retrieved_for_evidence))
            
            # Prepare batch inputs for construction error checking
            question_list = []
            golden_answers_list = []
            source_evidence_list = []
            retrieved_memory_units_list = []
            evidence_indices = []
            
            for idx, (evidence, retrieved_for_evidence) in enumerate(evidence_memory_searches):
                question_list.append(question)
                golden_answers_list.append(golden_answers)
                source_evidence_list.append(evidence)
                retrieved_memory_units_list.append(
                    _build_memory_units_text(retrieved_for_evidence)
                )
                evidence_indices.append(idx)
            
            if question_list:
                construction_results = construction_checker(
                    question_list,
                    golden_answers_list,
                    source_evidence_list,
                    retrieved_memory_units_list,
                    batch_size=batch_size,
                    aggregate=True,
                    temperature=1.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "memory_construction_error_check",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "explanation": {
                                        "type": "string",
                                        "description": "Brief explanation of the reasoning."
                                    },
                                    "is_present": {
                                        "type": "boolean",
                                        "description": "Whether the essential information is present in memory units."
                                    }
                                },
                                "required": ["explanation", "is_present"],
                                "additionalProperties": False
                            }
                        },
                    },
                )
                
                for idx, result in zip(evidence_indices, construction_results):
                    evidence, _ = evidence_memory_searches[idx]
                    if not result["is_present"]:
                        construction_errors.append(
                            {
                                "evidence": evidence,
                                "explanation": result["explanation"],
                            }
                        )
            
            if construction_errors:
                # Memory construction error detected
                attribution_results.append(
                    {
                        "qa_pair": qa_pair.model_dump(),
                        "prediction": prediction,
                        "error_type": "memory_construction_error",
                        "error_details": construction_errors,
                        "retrieved_memories": retrieved_memories,
                        "user_id": user_id,
                    }
                )
                continue
            
            # Step 2: Check retrieval errors
            retrieval_result = retrieval_checker(
                [question],
                [golden_answers],
                [_build_evidences_text(source_evidences)],
                [_build_memory_units_text(retrieved_memories)],
                batch_size=1,
                aggregate=True,
                temperature=1.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "retrieval_error_check",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "explanation": {
                                    "type": "string",
                                    "description": "Brief explanation of the reasoning."
                                },
                                "is_sufficient": {
                                    "type": "boolean",
                                    "description": "Whether the retrieval results sufficiently cover the source evidences."
                                }
                            },
                            "required": ["explanation", "is_sufficient"],
                            "additionalProperties": False
                        }
                    },
                },
            )
            
            retrieval_check = retrieval_result[0]
            
            if not retrieval_check["is_sufficient"]:
                # Retrieval error detected
                attribution_results.append(
                    {
                        "qa_pair": qa_pair.model_dump(),
                        "prediction": prediction,
                        "error_type": "retrieval_error",
                        "error_details": {
                            "explanation": retrieval_check["explanation"],
                        },
                        "retrieved_memories": retrieved_memories,
                        "user_id": user_id,
                    }
                )
                continue
            
            # Step 3: Response error (retrieval is correct but answer is wrong)
            attribution_results.append(
                {
                    "qa_pair": qa_pair.model_dump(),
                    "prediction": prediction,
                    "error_type": "response_error",
                    "error_details": {
                        "explanation": (
                            "Retrieval results contain all necessary information, " 
                            "but the generated answer is still incorrect."
                        ),
                    },
                    "retrieved_memories": retrieved_memories,
                    "user_id": user_id,
                }
            )
            
        finally:
            if pbar is not None:
                if pbar_lock is None:
                    pbar.update(1)
                else:
                    with pbar_lock:
                        pbar.update(1)
        
    return attribution_results


def run_error_attribution(
    evaluation_results: List[Dict[str, Any]],
    memory_type: str,
    judge_model: str,
    memory_config: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    batch_size: int = 4,
    num_workers: int = 4,
    interface_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the error attribution algorithm on failed instances.
    
    Parameters
    ----------
    evaluation_results: List[Dict[str, Any]]
        The evaluation results containing QA pairs, predictions, and correctness flags.
    memory_type: str
        The type of the memory layer.
    judge_model: str
        The model name for the LLM judge.
    memory_config: Dict[str, Any], optional
        Configuration for the memory layer.
    top_k: int
        Number of memory units to retrieve for each evidence.
    batch_size: int
        Batch size for LLM inference.
    num_workers: int
        Number of threads for parallel processing across users.
    interface_kwargs: Dict[str, Any], optional
        Additional kwargs for the LLM interface.
        
    Returns
    -------
    attribution_results: List[Dict[str, Any]]
        The error attribution results for each failed instance.
    """
    interface_kwargs = interface_kwargs or {}
    
    # Filter failed instances
    failed_instances = [
        item for item in evaluation_results 
        if not item.get("is_correct", True)
    ]
    
    if not failed_instances:
        print("No failed instances found. All answers are correct.")
        return []
    
    print(f"Found {len(failed_instances)} failed instances for error attribution.")
    
    # Group failed instances by user_id
    user_to_instances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in failed_instances:
        user_id = item["user_id"]
        user_to_instances[user_id].append(item)
    
    print(f"Grouped into {len(user_to_instances)} users.")
    
    # Initialize operators
    construction_checker = MemoryConstructionErrorChecker(
        prompt_name="memory-construction-error-check",
        model_name=judge_model,
        **interface_kwargs,
    )
    
    retrieval_checker = RetrievalErrorChecker(
        prompt_name="retrieval-error-check",
        model_name=judge_model,
        **interface_kwargs,
    )
    
    total_failed = len(failed_instances)
    pbar_lock = threading.Lock()
    pbar = tqdm(total=total_failed, desc="Attributing failed instances", unit="inst")

    # Process users in parallel
    attribution_results = []
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for user_id, user_instances in user_to_instances.items():
                future = executor.submit(
                    run_error_attribution_for_user,
                    user_id,
                    user_instances,
                    memory_type,
                    construction_checker,
                    retrieval_checker,
                    memory_config=deepcopy(memory_config),
                    top_k=top_k,
                    batch_size=batch_size,
                    pbar=pbar,
                    pbar_lock=pbar_lock,
                )
                futures.append(future)
            
            for future in tqdm(
                as_completed(futures), 
                total=len(futures), 
                desc="Processing users"
            ):
                results = future.result()
                attribution_results.extend(results)
    finally:
        pbar.close()
    
    return attribution_results


def compute_error_statistics(attribution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics of error types.
    
    Parameters
    ----------
    attribution_results: List[Dict[str, Any]]
        The error attribution results.
        
    Returns
    -------
    statistics: Dict[str, Any]
        The statistics of error types.
    """
    error_counts = {
        "memory_construction_error": 0,
        "retrieval_error": 0,
        "response_error": 0,
    }
    
    for result in attribution_results:
        error_type = result["error_type"]
        error_counts[error_type] += 1
    
    total = len(attribution_results)
    error_ratios = {
        k: v / total if total > 0 else 0.0 
        for k, v in error_counts.items()
    }
    
    return {
        "total_failed": total,
        "error_counts": error_counts,
        "error_ratios": error_ratios,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to perform error attribution on failed QA instances."
    )
    parser.add_argument(
        "--evaluation-results-path",
        type=str,
        required=True,
        help="Path to the evaluation results JSON file."
    )
    parser.add_argument(
        "--memory-type",
        choices=list(MEMORY_LAYERS_MAPPING.keys()),
        type=str,
        required=True,
        help="The type of the memory layer."
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for the LLM judge."
    )
    parser.add_argument(
        "--memory-config-path",
        type=str,
        default=None,
        help="Path to JSON config for memory layer."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of memory units to retrieve for each evidence."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for LLM inference."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of threads for parallel processing across users."
    )
    parser.add_argument(
        "--api-config-path",
        type=str,
        default=None,
        help="Path to the API config file."
    )
    args = parser.parse_args()
    
    # Prepare interface kwargs
    interface_kwargs: Dict[str, Any] = {}
    if args.api_config_path is not None:
        with open(args.api_config_path, 'r') as f:
            api_config = json.load(f)
        interface_kwargs["api_keys"] = api_config["api_keys"]
        interface_kwargs["base_urls"] = api_config["base_urls"]
    elif os.environ.get("OPENAI_API_KEY") is not None:
        interface_kwargs["api_keys"] = [os.environ.get("OPENAI_API_KEY")]
        interface_kwargs["base_urls"] = [os.environ.get("OPENAI_API_BASE")]
    
    # Load evaluation results
    with open(args.evaluation_results_path, 'r', encoding="utf-8") as f:
        evaluation_results = json.load(f)
    print(f"Loaded {len(evaluation_results)} evaluation results from {args.evaluation_results_path}.")
    
    # Load memory configuration
    memory_config = None
    if args.memory_config_path is not None:
        with open(args.memory_config_path, 'r', encoding="utf-8") as f:
            memory_config = json.load(f)
    
    # Run error attribution
    print("Running error attribution algorithm...")
    attribution_results = run_error_attribution(
        evaluation_results,
        memory_type=args.memory_type,
        judge_model=args.judge_model,
        memory_config=memory_config,
        top_k=args.top_k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        interface_kwargs=interface_kwargs,
    )
    
    # Compute and print statistics
    statistics = compute_error_statistics(attribution_results)
    print("\nError Attribution Statistics:")
    print(f"  Total failed instances: {statistics['total_failed']}")
    print(f"  Memory construction errors: {statistics['error_counts']['memory_construction_error']} "
          f"({statistics['error_ratios']['memory_construction_error']:.2%})")
    print(f"  Retrieval errors: {statistics['error_counts']['retrieval_error']} "
          f"({statistics['error_ratios']['retrieval_error']:.2%})")
    print(f"  Response errors: {statistics['error_counts']['response_error']} "
          f"({statistics['error_ratios']['response_error']:.2%})")
    
    # Save results
    output_path = args.evaluation_results_path.rsplit('.', 1)[0] + "_attribution.json"
    final_output = {
        "statistics": statistics,
        "attribution_results": attribution_results,
    }
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(
            to_jsonable(final_output),
            f,
            ensure_ascii=False,
            indent=4,
        )
    print(f"Saved error attribution results to {output_path}.")