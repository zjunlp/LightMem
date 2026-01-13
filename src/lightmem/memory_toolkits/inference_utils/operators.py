from .base_operator import NonCachedLLMOperator
import numpy as np 
import json
import re
from typing import ( 
    List, 
    Dict, 
    Any, 
    Optional, 
)

class QuestionAnsweringOperator(NonCachedLLMOperator):
    """An operator for question answering."""

    def _preprocess(
        self, 
        question_list: List[str], 
        context_list: Optional[List[str]] = None
    ) -> List[List[Dict[str, str]]]: 
        messages_list = [] 
        for i in range(len(question_list)):
            question = question_list[i]
            context = context_list[i] if context_list is not None else None
            if context is not None:
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant."
                    }, 
                    {
                        "role": "user", 
                        "content": self._prompt.substitute(question=question, context=context)
                    }, 
                ]
            else:
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant."
                    }, 
                    {
                        "role": "user", 
                        "content": self._prompt.substitute(question=question)
                    }, 
                ]
            messages_list.append(messages)
        return messages_list 

class LLMExactMatch(NonCachedLLMOperator):
    """An operator for LLM exact match."""

    def _preprocess(
        self, 
        question_list: List[str], 
        golden_answers_list: List[List[str]], 
        prediction_list: List[str], 
        reasoning_process_list: Optional[List[str]] = None
    ) -> List[List[Dict[str, str]]]: 
        messages_list = [] 
        for i in range(len(question_list)):
            question = question_list[i]
            golden_answer_list = golden_answers_list[i]
            prediction = prediction_list[i]
            reasoning_process = reasoning_process_list[i] if reasoning_process_list is not None else None
            if len(golden_answer_list) == 1:
                golden_answer_list = golden_answer_list[0]
            else:
                golden_answer_list = f"[{', '.join(golden_answer for golden_answer in golden_answer_list)}]"
            if reasoning_process is None:
                messages = [
                    {
                        "role": "user", 
                        "content": self._prompt.substitute(
                            question=question, 
                            golden_answers=golden_answer_list, 
                            prediction=prediction
                        )
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user", 
                        "content": self._prompt.substitute(
                            question=question, 
                            golden_answers=golden_answer_list,
                            reasoning_process=reasoning_process,
                            prediction=prediction
                        )
                    }
                ]
            messages_list.append(messages)
        return messages_list 

    def _aggregate(self, responses: List[Dict[str, Any]]) -> float:
        judge_results = np.array(responses)
        # See https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py#L113. 
        judge_results = np.vectorize(
            lambda item: "yes" in item["processed_content"].lower()
        )(judge_results)
        return judge_results.mean().item()


def _parse_json_response(content: str) -> Dict[str, Any]:
    """Parse JSON response from an LLM output, handling markdown code blocks."""
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = content.strip()
    return json.loads(json_str)


class MemoryConstructionErrorChecker(NonCachedLLMOperator):
    """An operator to check if the essential information from a source evidence 
    is present within retrieved memory units."""

    def _preprocess(
        self, 
        question_list: List[str], 
        golden_answers_list: List[List[str]], 
        source_evidence_list: List[str],
        retrieved_memory_units_list: List[str],
    ) -> List[List[Dict[str, str]]]: 
        messages_list = [] 
        for i in range(len(question_list)):
            question = question_list[i]
            golden_answers = golden_answers_list[i]
            source_evidence = source_evidence_list[i]
            retrieved_memory_units = retrieved_memory_units_list[i]

            if len(golden_answers) == 1:
                golden_answers_str = golden_answers[0]
            else:
                golden_answers_str = f"[{', '.join(golden_answers)}]"

            messages = [
                {
                    "role": "user", 
                    "content": self._prompt.substitute(
                        question=question, 
                        golden_answers=golden_answers_str, 
                        source_evidence=source_evidence,
                        retrieved_memory_units=retrieved_memory_units,
                    )
                }
            ]
            messages_list.append(messages)
        return messages_list 

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_json_response(content)
                results.append(
                    {
                        "explanation": parsed.get("explanation", ""),
                        "is_present": parsed.get("is_present", False),
                        "raw_response": content,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                # Fallback: if JSON parsing fails, mark as not present
                results.append(
                    {
                        "explanation": content,
                        "is_present": False,
                        "raw_response": content,
                    }
                )
        return results


class RetrievalErrorChecker(NonCachedLLMOperator):
    """An operator to check if the retrieval results sufficiently cover 
    the key contents of source evidences."""

    def _preprocess(
        self, 
        question_list: List[str], 
        golden_answers_list: List[List[str]], 
        source_evidences_list: List[str],
        retrieval_results_list: List[str],
    ) -> List[List[Dict[str, str]]]: 
        messages_list = [] 
        for i in range(len(question_list)):
            question = question_list[i]
            golden_answers = golden_answers_list[i]
            source_evidences = source_evidences_list[i]
            retrieval_results = retrieval_results_list[i]

            if len(golden_answers) == 1:
                golden_answers_str = golden_answers[0]
            else:
                golden_answers_str = f"[{', '.join(golden_answers)}]"

            messages = [
                {
                    "role": "user", 
                    "content": self._prompt.substitute(
                        question=question, 
                        golden_answers=golden_answers_str, 
                        source_evidences=source_evidences,
                        retrieval_results=retrieval_results,
                    )
                }
            ]
            messages_list.append(messages)
        return messages_list 

    def _aggregate(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for response in responses:
            content = response.get("processed_content", "")
            try:
                parsed = _parse_json_response(content)
                results.append(
                    {
                        "explanation": parsed.get("explanation", ""),
                        "is_sufficient": parsed.get("is_sufficient", False),
                        "raw_response": content,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                # Fallback: if JSON parsing fails, mark as not sufficient
                results.append(
                    {
                        "explanation": content,
                        "is_sufficient": False,
                        "raw_response": content,
                    }
                )
        return results
