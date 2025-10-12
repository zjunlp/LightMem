import string
from typing import Dict

_INCOMPLETE_PROMPT_COLLECTIONS = {
    # See https://arxiv.org/abs/2410.10813 and https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py. 
    "longmemeval-single-session-user": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, " 
        "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\n" 
        "Question: $question\n\nCorrect Answer: $golden_answers\n\nModel Response: $prediction\n\n" 
        "Is the model response correct? Answer yes or no only."
    ), 
    "longmemeval-temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
        "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. " 
        "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., " 
        "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\n" 
        "Question: $question\n\nCorrect Answer: $golden_answers\n\nModel Response: $prediction\n\n" 
        "Is the model response correct? Answer yes or no only."
    ), 
    "longmemeval-knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. " 
        "Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct " 
        "as long as the updated answer is the required answer.\n\n" 
        "Question: $question\n\nCorrect Answer: $golden_answers\n\nModel Response: $prediction\n\n" 
        "Is the model response correct? Answer yes or no only."
    ), 
    "longmemeval-single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. " 
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n" 
        "Question: $question\n\nRubric: $golden_answers\n\nModel Response: $prediction\n\n" 
        "Is the model response correct? Answer yes or no only."
    ), 
    "longmemeval-abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. " 
        "The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\n" 
        "Question: $question\n\nExplanation: $golden_answers\n\nModel Response: $prediction\n\n" 
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    ), 
    "question-answering": (
        "Question: $question\nPlease answer the question based on the following memories:\n$context"
    ), 
    # https://arxiv.org/abs/2305.12421
    "exact-match": (
        "Here is a question, a list of golden answers, an AI-generated answer. "
        "Can you judge whether the AI-generated answer is correct according to the question and golden answers?"
        "\nQuestion: $question\nGolden Answers: $golden_answers\nAI-generated answer: $prediction"
        "\nSimply answer Yes or No." 
    ), 
}

def _prepare_prompt_collections(prompt_collections: Dict[str, str]) -> Dict[str, str]:
    prompt_collections = {**prompt_collections}
    # In LongMemEval, the prompt for single-session-assistant and multi-session are the same as single-session-user. 
    prompt_collections["longmemeval-single-session-assistant"] = prompt_collections["longmemeval-single-session-user"]
    prompt_collections["longmemeval-multi-session"] = prompt_collections["longmemeval-single-session-user"]
    return prompt_collections

PROMPT_COLLECTIONS = _prepare_prompt_collections(_INCOMPLETE_PROMPT_COLLECTIONS)

def get_prompt(name: str) -> string.Template:
    """Get the prompt by name."""
    prompt = PROMPT_COLLECTIONS.get(name, None)
    if isinstance(prompt, str):
        template = string.Template(prompt)
        if not template.is_valid():
            raise ValueError(
                f"The prompt {name} is not valid. "
                f"The content of the prompt is: {prompt}."
            )
        return template
    raise ValueError(f"Unknown prompt: {name}.")