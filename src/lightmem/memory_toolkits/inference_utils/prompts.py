import string
from typing import Dict

_INCOMPLETE_PROMPT_COLLECTIONS = {
    # LongMemEval prompts #
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
    # LoCoMo
    "locomo-question-answering-flat-memory-system": (
        "You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.\n\n"
        "# CONTEXT:\n"
        "You have access to memories from two speakers in a conversation. These memories contain "
        "timestamped information that may be relevant to answering the question.\n\n"
        "# INSTRUCTIONS:\n"
        "1. Carefully analyze all provided memories from both speakers\n"
        "2. Pay special attention to the timestamps to determine the answer\n"
        "3. If the question asks about a specific event or fact, look for direct evidence in the memories\n"
        "4. If the memories contain contradictory information, prioritize the most recent memory\n"
        "5. If there is a question about time references (like 'last year', 'two months ago', etc.), "
        "calculate the actual date based on the memory timestamp. For example, if a memory from "
        "4 May 2022 mentions 'went to India last year,' then the trip occurred in 2021.\n"
        "6. Always convert relative time references to specific dates, months, or years. For example, "
        "convert 'last year' to '2022' or 'two months ago' to 'March 2023' based on the memory "
        "timestamp. Ignore the reference while answering the question.\n"
        "7. Focus only on the content of the memories from both speakers. Do not confuse character "
        "names mentioned in memories with the actual users who created those memories.\n"
        "8. The answer should be less than 5-6 words.\n\n"
        "# APPROACH (Think step by step):\n"
        "1. First, examine all memories that contain information related to the question\n"
        "2. Examine the timestamps and content of these memories carefully\n"
        "3. Look for explicit mentions of dates, times, locations, or events that answer the question\n"
        "4. If the answer requires calculation (e.g., converting relative time references), show your work\n"
        "5. Formulate a precise, concise answer based solely on the evidence in the memories\n"
        "6. Double-check that your answer directly addresses the question asked\n"
        "7. Ensure your final answer is specific and avoids vague time references\n\n"
        "Memories:\n\n"
        "$context\n\n"
        "Question: $question\n\n"
        "Answer:"
    ),
    "locomo-question-answering-graph-memory-system": (
        "You are an intelligent memory assistant tasked with retrieving accurate information from "
        "conversation memories.\n\n"
        "# CONTEXT:\n"
        "You have access to memories from two speakers in a conversation. These memories contain "
        "timestamped information that may be relevant to answering the question. You also have "
        "access to knowledge graph relations for each user, showing connections between entities, "
        "concepts, and events relevant to that user.\n\n"
        "# INSTRUCTIONS:\n"
        "1. Carefully analyze all provided memories from both speakers\n"
        "2. Pay special attention to the timestamps to determine the answer\n"
        "3. If the question asks about a specific event or fact, look for direct evidence in the memories\n"
        "4. If the memories contain contradictory information, prioritize the most recent memory\n"
        "5. If there is a question about time references (like 'last year', 'two months ago', etc.), "
        "calculate the actual date based on the memory timestamp. For example, if a memory from "
        "4 May 2022 mentions 'went to India last year,' then the trip occurred in 2021.\n"
        "6. Always convert relative time references to specific dates, months, or years. For example, "
        "convert 'last year' to '2022' or 'two months ago' to 'March 2023' based on the memory "
        "timestamp. Ignore the reference while answering the question.\n"
        "7. Focus only on the content of the memories from both speakers. Do not confuse character "
        "names mentioned in memories with the actual users who created those memories.\n"
        "8. The answer should be less than 5-6 words.\n"
        "9. Use the knowledge graph relations to understand the user's knowledge network and "
        "identify important relationships between entities in the user's world.\n\n"
        "# APPROACH (Think step by step):\n"
        "1. First, examine all memories that contain information related to the question\n"
        "2. Examine the timestamps and content of these memories carefully\n"
        "3. Look for explicit mentions of dates, times, locations, or events that answer the question\n"
        "4. If the answer requires calculation (e.g., converting relative time references), show your work\n"
        "5. Analyze the knowledge graph relations to understand the user's knowledge context\n"
        "6. Formulate a precise, concise answer based solely on the evidence in the memories\n"
        "7. Double-check that your answer directly addresses the question asked\n"
        "8. Ensure your final answer is specific and avoids vague time references\n\n"
        "Memories:\n\n"
        "$context\n\n"
        "Question: $question\n\n"
        "Answer:"
    ),
    "locomo-judge": (
        "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: "
        "(1) a question (posed by one user to another user), "
        "(2) a 'gold' (ground truth) answer, "
        "(3) a generated answer "
        "which you will score as CORRECT/WRONG.\n\n"
        "The point of the question is to ask about something one user should know about the other user based on their prior conversations. "
        "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
        "Question: Do you remember what I got the last time I went to Hawaii?\n"
        "Gold answer: A shell necklace\n"
        "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
        "For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.\n\n"
        "Now it's time for the real question:\n"
        "Question: $question\n"
        "Gold answer: $golden_answers\n"
        "Generated answer: $prediction\n\n"
        "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. "
        "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\n"
        "Just return the label CORRECT or WRONG in a json format with the key as 'label'."
    )
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
