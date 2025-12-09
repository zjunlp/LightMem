import json
import ollama
import os
import time
from tqdm import tqdm
from typing import Dict, List, Optional

from lightmem.memory.lightmem import LightMemory


# =========== Ollama Configuration ============
your_ollama_model_name = "your_Ollama_model_name_01"  # such as "llama3:latest"
your_ollama_JUDGE_model_name = "your_Ollama_model_name_02"  # such as "gemma3:latest"
your_ollama_host = "http://localhost:11434"  # default Ollama host is "http://localhost:11434"
your_ollama_options_stable = {
    "num_ctx": 8192,  # set according to the model's context window
    "seed": 42,
    "top_k": 1,
    "top_p": 1.0,
    "temperature": 0.0,
    "repeat_penalty": 1.0,
}  # a stable settings of options for evaluation

# ============ Small Model Paths ============
LLMLINGUA_MODEL_PATH='/your/path/to/models/llmlingua-2-bert-base-multilingual-cased-meetingbank'
EMBEDDING_MODEL_PATH='/your/path/to/models/all-MiniLM-L6-v2'

# ============ Data Configuration ============
DATA_PATH='/your/path/to/data/longmemeval/longmemeval_s.json'
QDRANT_DATA_DIR='../qdrant_data'


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

def load_lightmem(collection_name):
    config = {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": LLMLINGUA_MODEL_PATH,
                    "device_map": "cuda:0",
                    "use_llmlingua2": True,
                },
                "compress_config": {
                    "rate": 0.6
                }
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
            "model_name": "ollama",
            "configs": {
                "model": your_ollama_model_name,
                "host": your_ollama_host,
                "max_tokens": 16384,  # set according to the model's context window
            }
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": EMBEDDING_MODEL_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda:0"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f'{QDRANT_DATA_DIR}/{collection_name}',
            }
        },
        "update": "offline",
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "log_dir": "logs",
            "log_filename_prefix": "run",
            "console_enabled": True,
            "file_level": "DEBUG",
        }
    }
    lightmem = LightMemory.from_config(config)
    return lightmem


class OllamaModel:
    """
    An example Ollama model class for generating responses using the Ollama API.
    You can customize this class as needed to fit your application.
    """
    def __init__(
            self,
            model_name: str,
            ollama_options: Optional[Dict] = None,
            host: str = 'http://localhost:11434',
            headers: Dict = {'x-some-header': 'some-value'},
        ):
        self.name = model_name
        self.host = host
        self.headers = headers
        self.ollama_options = ollama_options
        self.client = ollama.Client(host=self.host, headers=self.headers)

    def call(self, messages: List[Dict[str, str]], **kwargs):
        max_retries = kwargs.get("max_retries", 3)
    
        for attempt in range(max_retries):
            try:
                completion = self.client.chat(
                    model=self.name,
                    messages=messages,
                    options=self.ollama_options,
                )
                response = completion['message']['content']
                print(response)
                return response

            except ollama.ResponseError as e:
                print(f"[ollama.ResponseError] status={e.status_code}  {e.error}")
                if e.status_code == 404:
                    print("Your model may take time to let Ollama download. Please wait...")
                    ollama.pull(self.name)
                else:
                    raise
            except Exception as e:
                print(f"[Retry {attempt + 1}/{max_retries}]  {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise


def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(proj_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    llm = OllamaModel(
        model_name=your_ollama_model_name,
        ollama_options=your_ollama_options_stable,
    )
    llm_judge = OllamaModel(
        model_name=your_ollama_JUDGE_model_name,
        ollama_options=your_ollama_options_stable,
    )

    data = json.load(open(DATA_PATH, "r"))
    data = data[:10]  # for testing

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

        filename = os.path.join(out_dir, f"result_{item['question_id']}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)


# Ollama test:
if __name__ == "__main__":
    main()
