from memories.datasets.base import QuestionAnswerPair
from inference_utils.operators import (
    QuestionAnsweringOperator,
    LLMExactMatch,
    LocomoQAOperator,
    LocomoGraphQAOperator,
    _parse_json_response
)
import numpy as np
import argparse
import json 
import os 
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
)
from inference_utils.prompts import PROMPT_COLLECTIONS

from collections.abc import Mapping
from types import MappingProxyType
from pydantic import BaseModel

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


def _build_context_text(retrieved_memories: List[Dict[str, Any]]) -> str:
    contents = []
    for i, mem in enumerate(retrieved_memories):
        content = mem.get("used_content") or mem.get("content", '')
        if not isinstance(content, str):
            raise AssertionError("The used_content is not a string for the current memory unit.")
        if not content:
            raise AssertionError("The used_content is empty for the current memory unit.")
        contents.append(f"### Memory {i + 1}:\n{content}")
    return "\n\n".join(contents)

def _build_locomo_context_text(retrieved_memories: Dict[str, Any]) -> Dict[str, Any]:
    speaker_1_data = retrieved_memories["speaker_1"]
    speaker_2_data = retrieved_memories["speaker_2"]
    
    speaker_1_contents = []
    for i, mem in enumerate(speaker_1_data["memories"]):
        content = mem.get("used_content") or mem.get("content", '')
        if content:
            speaker_1_contents.append(f"Memory {i + 1}: {content}")
    
    speaker_2_contents = []
    for i, mem in enumerate(speaker_2_data["memories"]):
        content = mem.get("used_content") or mem.get("content", '')
        if content:
            speaker_2_contents.append(f"Memory {i + 1}: {content}")
    
    return {
        "speaker_1_name": speaker_1_data["name"],
        "speaker_1_memories": "\n\n".join(speaker_1_contents) if speaker_1_contents else "[No memories]",
        "speaker_2_name": speaker_2_data["name"],
        "speaker_2_memories": "\n\n".join(speaker_2_contents) if speaker_2_contents else "[No memories]",
    }

def _build_graph_text(relations: Any) -> str:
    if not relations:
        return ""
    # Expand according to mem0 return format if needed.
    # The following is a very rough example:
    lines = ["### Graph Relations:"]
    for rel in relations:
        # Inspect the structure of `rel` if necessary; placeholder here
        lines.append(str(rel))
    return "\n".join(lines)

def answer_questions(
    retrievals: List[Dict[str, Any]],
    qa_model: str,
    qa_batch_size: int = 4,
    add_question_timestamp: bool = False, 
    interface_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    interface_kwargs = interface_kwargs or {}
    questions: List[str] = []
    is_locomo = any(item.get("dataset_type") == "LoCoMo" for item in retrievals)
    has_graph = any(item.get("graph_relations") is not None for item in retrievals)
    if is_locomo:
        if has_graph:
            s1_names, s1_mems, s1_graphs = [], [], []
            s2_names, s2_mems, s2_graphs = [], [], []
            prompt_name = "locomo-question-answering-graph-memory-system"
            
            for item in retrievals:
                questions.append(item["qa_pair"].question)
                ctx_dict = _build_locomo_context_text(item["retrieved_memories"])
                graph_data = item.get("graph_relations", {})
                
                s1_names.append(ctx_dict["speaker_1_name"])
                s1_mems.append(ctx_dict["speaker_1_memories"])
                s1_graphs.append(_build_graph_text(graph_data.get("speaker_1", [])))
                
                s2_names.append(ctx_dict["speaker_2_name"])
                s2_mems.append(ctx_dict["speaker_2_memories"])
                s2_graphs.append(_build_graph_text(graph_data.get("speaker_2", [])))

            qa_operator = LocomoGraphQAOperator(prompt_name=prompt_name, model_name=qa_model, **interface_kwargs)
            responses = qa_operator(questions, s1_names, s1_mems, s1_graphs, s2_names, s2_mems, s2_graphs, 
                                    batch_size=qa_batch_size, aggregate=False, temperature=0.0)
        else:    
            s1_names, s1_mems, s2_names, s2_mems = [], [], [], []
            prompt_name = "locomo-question-answering-flat-memory-system"
            
            for item in retrievals:
                qa_pair: QuestionAnswerPair = item["qa_pair"]
                questions.append(qa_pair.question)
                ctx_dict = _build_locomo_context_text(item["retrieved_memories"])
                s1_names.append(ctx_dict["speaker_1_name"])
                s1_mems.append(ctx_dict["speaker_1_memories"])
                s2_names.append(ctx_dict["speaker_2_name"])
                s2_mems.append(ctx_dict["speaker_2_memories"])

            qa_operator = LocomoQAOperator(prompt_name=prompt_name, model_name=qa_model, **interface_kwargs)
            
            responses = qa_operator(
                questions, s1_names, s1_mems, s2_names, s2_mems, 
                batch_size=qa_batch_size, aggregate=False, temperature=0.0
            )
    else:
        contexts: List[str] = []
        for item in retrievals:
            qa_pair: QuestionAnswerPair = item["qa_pair"]
            questions.append(f"{qa_pair.question}\nQuestion Timestamp: {qa_pair.get_string_timestamp()}" if add_question_timestamp else qa_pair.question)
            base_ctx = _build_context_text(item["retrieved_memories"])
            rel_ctx = _build_graph_text(item.get("graph_relations"))
            contexts.append(base_ctx + "\n\n" + rel_ctx if rel_ctx else base_ctx)

        qa_operator = QuestionAnsweringOperator(prompt_name="question-answering", model_name=qa_model, **interface_kwargs)
        responses = qa_operator(questions, contexts, batch_size=qa_batch_size, aggregate=False, temperature=0.0)

    print(f"Using prompt: {prompt_name} for QA generation.")
    
    return responses

def evaluate_answers(
    retrievals: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    judge_model: str,
    judge_batch_size: int = 4,
    interface_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    interface_kwargs = interface_kwargs or {}
    is_locomo = any(item.get("dataset_type") == "LoCoMo" for item in retrievals)
    question_list: List[str] = []
    golden_answers_list: List[List[str]] = []
    prediction_list: List[str] = []
    prompt_name_per_index: List[Tuple[str, str]] = []
    category_per_index: List[Any] = []  
    for i, item in enumerate(retrievals):
        qa_pair: QuestionAnswerPair = item["qa_pair"]
        question_list.append(qa_pair.question)
        golden_answers_list.append([ans for ans in qa_pair.answer_list])
        pred = predictions[i].get("processed_content") 
        # LLMs are not robust to the empty string.
        if pred is None:
            raise ValueError(f"The prediction is None for the question {qa_pair.question}.")
        prediction_list.append(pred)

        category = qa_pair.metadata.get("category", "unknown")
        category_per_index.append(category)

        if is_locomo:
            prompt_name = "locomo-judge"
            qtype = "locomo"
        else:
            qtype = qa_pair.metadata.get("question_type", "normal")
            if qtype == "normal":
                prompt_name = "exact-match"
            elif "_abs" in qa_pair.metadata.get("id", ''):
                prompt_name = "longmemeval-abstention"
            else:
                candidate = f"longmemeval-{qtype}"
                if candidate in PROMPT_COLLECTIONS:
                    prompt_name = candidate
                else:
                    # Fallback to exact-match
                    prompt_name = "exact-match"

        prompt_name_per_index.append((prompt_name, qtype))

    print(f"Using prompt: {prompt_name} for judgment.")
    judge_operator = LLMExactMatch(
        prompt_name=prompt_name,
        model_name=judge_model,
        **interface_kwargs,
    )

    groups: Dict[Tuple[str, str], List[int]] = {}
    for idx, p in enumerate(prompt_name_per_index):
        if p not in groups:
            groups[p] = [] 
        groups[p].append(idx)

    judge_outputs: List[Optional[Dict[str, Any]]] = [None] * len(retrievals)
    correctness_flags: List[Optional[bool]] = [None] * len(retrievals)

    for (prompt_name, qtype), idx_list in groups.items():
        judge_operator.set_prompt(prompt_name)
        batched_questions = [question_list[i] for i in idx_list]
        batched_golden = [golden_answers_list[i] for i in idx_list]
        batched_predictions = [prediction_list[i] for i in idx_list]
        results = judge_operator(
            batched_questions,
            batched_golden,
            batched_predictions,
            batch_size=judge_batch_size,
            aggregate=False,
            temperature=0.0, 
        )
        for local_pos, global_idx in enumerate(idx_list):
            out = results[local_pos]
            judge_outputs[global_idx] = out
            content = out.get("processed_content")
            if content is None:
                raise ValueError(f"The content is None for the question {batched_questions[local_pos]}.")
            if is_locomo:
                try:
                    result_json = _parse_json_response(content)
                    is_correct = result_json.get("label", "").upper() == "CORRECT"
                except:
                    is_correct = "CORRECT" in content.upper() and "WRONG" not in content.upper()
            else:
                is_correct = "yes" in content.lower()

            correctness_flags[global_idx] = is_correct
        
        # Aggregate each group's results and print the average accuracy
        accuracy = np.mean(
            [correctness_flags[global_idx] for global_idx in idx_list]
        ).item()
        print(f"The accuracy for {qtype} (prompt name: {prompt_name}) is {accuracy:.4f}.")
    
    category_stats = {}
    for idx, (category, is_correct) in enumerate(zip(category_per_index, correctness_flags)):
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  Category {category}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")
    
    if is_locomo:
        print("\n" + "="*60)
        print("Question Type Accuracy (LoCoMo):")
        print("="*60)
        
        qtype_stats = {}
        for i, item in enumerate(retrievals):
            qa_pair = item["qa_pair"]
            qtype = qa_pair.metadata.get("question_type", "unknown")
            is_correct = correctness_flags[i]
            
            if qtype not in qtype_stats:
                qtype_stats[qtype] = {"correct": 0, "total": 0}
            qtype_stats[qtype]["total"] += 1
            if is_correct:
                qtype_stats[qtype]["correct"] += 1
        
        for qtype in sorted(qtype_stats.keys()):
            stats = qtype_stats[qtype]
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            print(f"  {qtype}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")
    
    # Print the overall accuracy
    print("\n" + "="*60)
    accuracy = np.mean(correctness_flags).item()
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("="*60 + "\n")

    finalized = []
    for i in range(len(retrievals)):
        finalized.append(
            {
                "judge_response": judge_outputs[i],
                "is_correct": correctness_flags[i],
                "category": category_per_index[i],  
            }
        )
    return finalized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to evaluate the answers of the search results."
    )
    parser.add_argument(
        "--search-results-path",
        type=str,
        required=True,
        help="Path to the search results."
    )
    parser.add_argument(
        "--qa-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name/path for question answering."
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name/path for judgment (exact match)."
    )
    parser.add_argument(
        "--qa-batch-size",
        type=int,
        default=4,
        help="Batch size for QA generation."
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=4,
        help="Batch size for judge model."
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
    
    with open(args.search_results_path, 'r') as f:
        retrievals = json.load(f)
    for item in retrievals:
        item["qa_pair"] = QuestionAnswerPair(**item["qa_pair"])
    print(f"Loaded {len(retrievals)} search results from {args.search_results_path}.")

    # Answer questions
    print("Generating answers with QA model...")
    qa_responses = answer_questions(
        retrievals,
        qa_model=args.qa_model,
        qa_batch_size=args.qa_batch_size,
        interface_kwargs=interface_kwargs,
    )

    # Evaluate answers
    print("Evaluating answers with judge model...")
    judge_results = evaluate_answers(
        retrievals,
        qa_responses,
        judge_model=args.judge_model,
        judge_batch_size=args.judge_batch_size,
        interface_kwargs=interface_kwargs,
    )

    # Assemble final outputs
    final_results: List[Dict[str, Any]] = []
    for i, item in enumerate(retrievals):
        qa_pair: QuestionAnswerPair = item["qa_pair"]
        ans_dict = qa_responses[i]
        judge_dict = judge_results[i]
        final_results.append(
            {
                "qa_pair": qa_pair.model_dump(),
                "prediction": ans_dict["processed_content"],
                "judge_response": judge_dict["judge_response"],
                "is_correct": judge_dict["is_correct"],
                "retrieved_memories": item["retrieved_memories"],
                "user_id": item["user_id"],
            }
        )

    output_path = args.search_results_path.rsplit('.', 1)[0] + "_evaluation.json"
    with open(
        output_path, 
        'w', 
        encoding="utf-8"
    ) as f:
        json.dump(
            # final_results, 
            to_jsonable(final_results),
            f, 
            ensure_ascii=False, 
            indent=4, 
        )
    print(f"Saved {len(final_results)} results to {output_path}.")
