import json
from openai import OpenAI
from tqdm import tqdm
import datetime
import time
import os
import logging
from lightmem.memory.lightmem import LightMemory

def load_lightmem(collection_name):

    compress_ratio = 0.6  # r参数
    shortmem_th = 256    # th参数

    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cuda",
                    "use_llmlingua2": True,
                    "compress_config": {
                        "rate": compress_ratio,
                        "truncation": True   
                    }
                },
            }
        },
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": "glm-4.6",
                "api_key": GLM_API_KEY,
                "max_tokens": 16000,
                "openai_base_url": GLM_API_BASE_URL
            }
        },
        "shortmem_max_tokens": shortmem_th,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
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
                "on_disk": True,# 重要
                "path": f"./qdrant_data/{collection_name}",
            }
        },
        "update": "offline",
        "logging": {
            "level": "DEBUG",
            "file_enabled": True,
            "log_dir": RUN_LOG_DIR,
        }
    }
    lightmem = LightMemory.from_config(config)
    return lightmem

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
                if attempt == max_retries - 1:
                    raise



# ============ API Configuration ============
LLM_MODEL='glm-4.6'
JUDGE_MODEL='gpt-4o-mini'
GLM_API_KEY = 'your_api_key_here'
GLM_API_BASE_URL=''
JUDGE_API_KEY='your_api_key_here'
JUDGE_API_BASE_URL=''

# ============ Model Paths ============
EMBEDDING_MODEL_PATH='/your/path/to/models/all-MiniLM-L6-v2'
LLMLINGUA_MODEL_PATH='/your/path/to/models/llmlingua-2-bert-base-multilingual-cased-meetingbank'

# ============ Data Configuration ============
base_dir = '/your/path/to/qdrant_data'
DATA_PATH='/your/path/to/dataset/longmemeval/longmemeval_s.json'

# ============ Log Configuration ============
LOGS_ROOT = "./logs_offline"

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_ROOT, RUN_TIMESTAMP)
os.makedirs(RUN_LOG_DIR, exist_ok=True)

llm_judge = LLMModel(JUDGE_MODEL, JUDGE_API_KEY, JUDGE_API_BASE_URL)
llm = LLMModel(LLM_MODEL, GLM_API_KEY, GLM_API_BASE_URL)

data = json.load(open(DATA_PATH, "r"))

INIT_RESULT = {
    "update_input_prompt": [],
    "update_output_prompt": [],
    "api_call_nums": 0
}

for item in tqdm(data):
    print(item["question_id"])
    collection_name=item["question_id"]
    
    collection_path = os.path.join(base_dir, collection_name)

    if not os.path.isdir(collection_path):
        continue  

    print(f"Processing collection: {collection_name}")

    try:
        lightmem = load_lightmem(collection_name)
        results_list = []

        lightmem.manager.update_records = {
            "update_input_prompt": [],
            "update_output_prompt": [],
            "api_call_nums": 0
        }

        time_start = time.time()
        lightmem.construct_update_queue_all_entries()
        lightmem.offline_update_all_entries(score_threshold=0.8)
        print(f"Finished updating {collection_name}")
        time_end = time.time()
        update_time = time_end - time_start
        update_records = lightmem.manager.update_records.copy() 
        results_list.append(update_records)
        print(f"Finished updating {collection_name}")
    except Exception as e:
        print(f"Error processing {collection_name}: {e}") 
        update_time = 0
        results_list = []
  

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
        "update_time": update_time,
        "generated_answer": generated_answer,
        "ground_truth": item["answer"],
        "correct": correct,
    }

    filename = f"../results_glm_offline/result_{item['question_id']}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
