from .base_operator import NonCachedLLMOperator
import numpy as np 
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