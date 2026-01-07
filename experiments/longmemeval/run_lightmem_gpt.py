from openai import OpenAI
import json
from tqdm import tqdm
import datetime
import time
from lightmem.memory.lightmem import LightMemory

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response) 
    return prompt

def true_or_false(response):
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace('.', '').replace('!', '').replace(':', '').replace(';', '').split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


class LLMModel:
    def __init__(self, model_name, api_key, base_url):
        self.name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = 2000
        self.temperature = 0.0
        self.top_p = 0.8
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call(self, messages: list, **kwargs):
        max_retries = kwargs.get("max_retries", 3)
    
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False
                )
                response = completion.choices[0].message.content
                print(response)
                return response
            except Exception as e:
                print(f"[Retry {attempt + 1}/{max_retries}]  {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise

def load_lightmem(collection_name):
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": "/models/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                },
            }
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": "gpt-4o-mini",
                "api_key": "",
                "max_tokens": 16000,
                "openai_base_url": ""
            }
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": "all-MiniLM-L6-v2",
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f'/{collection_name}',
            }
        },
        "update": "offline",
    }
    lightmem = LightMemory.from_config(config)
    return lightmem

llm_judge = LLMModel("gpt-4o-mini", "", "")

llm = LLMModel("gpt-4o-mini", "", "")

data = json.load(open("longmemeval/longmemeval_s.json", "r"))

INIT_RESULT = {
    "add_input_prompt": [],
    "add_output_prompt": [],
    "api_call_nums": 0
}

for item in tqdm(data):
    print(item["question_id"])
    lightmem = load_lightmem(collection_name=item["question_id"])
    sessions = item["haystack_sessions"]
    timestamps = item["haystack_dates"]

    results_list = []

    time_start = time.time()
    for session, timestamp in zip(sessions, timestamps):
        while session and session[0]["role"] != "user":
            session.pop(0)
        num_turns = len(session) // 2  
        for turn_idx in range(num_turns):
            turn_messages = session[turn_idx*2 : turn_idx*2 + 2]
            if len(turn_messages) < 2 or turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                continue
            for msg in turn_messages:
                msg["time_stamp"] = timestamp
            is_last_turn = (
                session is sessions[-1] and turn_idx == num_turns - 1
            )
            result = lightmem.add_memory(
                messages=turn_messages,
                force_segment=is_last_turn,
                force_extract=is_last_turn,
            )
            if result != INIT_RESULT:
                results_list.append(result)

    time_end = time.time()
    construction_time = time_end - time_start

    related_memories = lightmem.retrieve(item["question"], limit=20)
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({
        "role": "user",
        "content": f"Question time:{item['question_date']} and question:{item['question']}\nPlease answer the question based on the following memories: {str(related_memories)}"
    })
    generated_answer = llm.call(messages)

    if 'abs' in item["question_id"]:
        prompt = get_anscheck_prompt(
            item["question_type"], item["question"], item["answer"], generated_answer, abstention=True
        )
    else:
        prompt = get_anscheck_prompt(
            item["question_type"], item["question"], item["answer"], generated_answer
        )
    messages = [{"role": "user", "content": prompt}]
    response = llm_judge.call(messages)

    correct = 1 if true_or_false(response) else 0

    save_data = {
        "question_id": item["question_id"],
        "results": results_list,
        "construction_time": construction_time,
        "generated_answer": generated_answer,
        "ground_truth": item["answer"],
        "correct": correct,
    }

    filename = f"lightmem/results/result_{item['question_id']}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)