# coding: utf-8
import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import json
from openai import OpenAI
from mem0 import Memory
import os
import datetime
import logging
import time
import argparse
import yaml
logger = logging.getLogger(__name__)

# Remove proxy settings from the environment
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

class resCounter:
    """A class to count and track various metrics related to the question answering process."""
    def __init__(self):
        self.question_id = ""
        self.correct_num = 0
        self.wrong_num = 0
        self.related_memories = []
        self.question_num = 0
        self.input_extract_system_prompt = 0
        self.input_extract_user_prompt = 0
        self.output_extract_prompt = 0
        self.input_update_prompt = 0
        self.output_update_prompt = 0
        self.input_procedural_memory_prompt = 0
        self.output_procedural_memory_prompt = 0
        self.add_time = 0
        self.search_time = 0
        self.memory_count = 0
        self.generated_answer = ""
        self.ground_truth_answer = ""
        self.judge_response = ""
        self.llm_generate_response_calls = 0
    
    def to_dict(self):
        """Convert object to a serializable dictionary."""
        return {
            "question_id": self.question_id,
            "correct_num": self.correct_num,
            "wrong_num": self.wrong_num,
            "question_num": self.question_num,
            "add_time": self.add_time,
            "search_time": self.search_time,
            "input_extract_system_prompt": self.input_extract_system_prompt,
            "input_extract_user_prompt": self.input_extract_user_prompt,
            "output_extract_prompt": self.output_extract_prompt,
            "input_update_prompt": self.input_update_prompt,
            "output_update_prompt": self.output_update_prompt,
            "input_procedural_memory_prompt": self.input_procedural_memory_prompt,
            "output_procedural_memory_prompt": self.output_procedural_memory_prompt,
            "memory_count": self.memory_count,
            "generated_answer": self.generated_answer,
            "ground_truth_answer": self.ground_truth_answer,
            "judge_response": self.judge_response,
            "llm_generate_response_calls": self.llm_generate_response_calls,
            "related_memories": self.related_memories,
        }

class LLMModel:
    """LLM class to generate responses to questions."""
    def __init__(self, model_name, api_key, base_url):
        self.name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = 2000
        self.temperature = 0.0
        self.top_p = 0.8
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"LLM model initialized: {model_name}, base_url: {base_url}")

    def call(self, messages: list, **kwargs):
        max_retries = kwargs.get("max_retries", 3)
        logger.debug(f"LLM call started, message count: {len(messages)}, max retries: {max_retries}")
    
        for attempt in range(max_retries):
            try:
                logger.debug(f"LLM call attempt {attempt + 1}/{max_retries}")
                completion = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False
                )
                response = completion.choices[0].message.content
                logger.debug(f"LLM call successful, response length: {len(response)}")
                print(response)
                return response
            except Exception as e:
                logger.error(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                logger.info(f"Retrying...")

def load_data(data_path, begin=51, end=60):
    data = json.load(open(data_path, "r"))
    data = data[begin:end]
    return data

def load_mem0(config, collection_name, process_mode):
    mem_config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": config["llm_model"], 
                "api_key": config["api_key"],
                "openai_base_url": config["base_url"],
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f'storage_{process_mode}_rag/{collection_name}',
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": config["embedding_model"],
                "embedding_dims": 384,
                "model_kwargs": {"device": config["device"]},
            },
        }
    }
    print(f"mem_config: {mem_config}")
    memory = Memory.from_config(mem_config)
    return memory

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
    return False

def save_results_to_json(results, filename=None):
    """Save results to a JSON file."""
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    if not isinstance(results, list):
        results = [results]
    
    results_dict = [result.to_dict() for result in results]
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to file: {filename}")
    return filename

def main(args):
    begin = args.begin
    end = args.end
    api_key = args.api_key
    base_url = args.base_url
    model_name = args.model_name
    embedding_model = args.embedding_model
    device = args.device
    process_mode = args.process_mode
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs_{process_mode}_rag/debug_simple_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    config = {
        "api_key": api_key,
        "base_url": base_url,
        "llm_model": model_name,
        "embedding_model": embedding_model,
        "device": device
    }

    judge_config_path = 'config.yaml'  # Updated path to relative one
    with open(judge_config_path, 'r', encoding='utf-8') as f:
        judge_cfg = yaml.safe_load(f)
    judge_api_key = judge_cfg.get('api_key')
    judge_base_url = judge_cfg.get('base_url')
    judge_model_name = judge_cfg.get('llm_model')
    if not (judge_api_key and judge_base_url and judge_model_name):
        raise ValueError("Missing judge.api_key/base_url/model_name in config.yaml")
    llm_judge = LLMModel(judge_model_name, judge_api_key, judge_base_url)
    
    data = load_data("longmemeval_s.json", begin, end)  # Updated path to relative one
    llm = LLMModel(model_name, api_key, base_url)
    
    logger.info(f"Processing data range: {begin}-{end}, Process mode: {process_mode}")
    logger.info(f"API Key: {api_key[:10]}...")
    logger.info(f"Base URL: {base_url}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Embedding Model: {embedding_model}")
    
    for item in data: 
        res_counter = resCounter()
        res_counter.question_id = item["question_id"]
        memory = load_mem0(config, item["question_id"], process_mode)

        calls_counter = [0]
        orig_generate_response = getattr(memory.llm, "generate_response", None)
        if callable(orig_generate_response):
            def _wrapped_generate_response(*w_args, **w_kwargs):
                calls_counter[0] += 1
                return orig_generate_response(*w_args, **w_kwargs)
            memory.llm.generate_response = _wrapped_generate_response
        
        sessions = item["haystack_sessions"]
        timestamps = item["haystack_dates"]
        
        time_start = time.time()
        if process_mode == "session":
            for session, timestamp in zip(sessions, timestamps):
                memory.add(session, user_id=item["question_id"], infer=False, metadata={"timestamp": timestamp})
        else:
            for session, timestamp in zip(sessions, timestamps):
                for message in session:
                    memory.add(message["content"], user_id=item["question_id"], infer=False, metadata={"timestamp": timestamp})

        time_end = time.time()
        res_counter.add_time = time_end - time_start
        
        search_time_start = time.time()
        related_memories = memory.search(item["question"], user_id=item["question_id"], limit=20)
        search_time_end = time.time()
        res_counter.search_time = search_time_end - search_time_start
        res_counter.related_memories = related_memories

        if callable(orig_generate_response):
            res_counter.llm_generate_response_calls = calls_counter[0]
            memory.llm.generate_response = orig_generate_response

        messages = []
        messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": f"Question:{item['question']}\nPlease answer the question based on the following memories: {str(related_memories)}"})

        generated_answer = llm.call(messages)
        
        res_counter.generated_answer = generated_answer
        res_counter.ground_truth_answer = item["answer"]
        
        save_results_to_json([res_counter], f"results_{process_mode}_rag/results_gen_{item['question_id']}.json")  # Updated path to relative one

        if 'abs' in item["question_id"]:
            prompt = get_anscheck_prompt(item["question_type"], item["question"], item["answer"], generated_answer, abstention=True)
        else:
            prompt = get_anscheck_prompt(item["question_type"], item["question"], item["answer"], generated_answer)
        messages = [{"role": "user", "content": prompt}]
        response = llm_judge.call(messages)
        res_counter.judge_response = response
        res_counter.question_num += 1
        if true_or_false(response):
            res_counter.correct_num += 1
        else:
            res_counter.wrong_num += 1
        res_counter.input_extract_system_prompt += memory.input_extract_system_prompt
        res_counter.input_extract_user_prompt += memory.input_extract_user_prompt
        res_counter.output_extract_prompt += memory.output_extract_prompt
        res_counter.input_update_prompt += memory.input_update_prompt
        res_counter.output_update_prompt += memory.output_update_prompt
        res_counter.input_procedural_memory_prompt += memory.input_procedural_memory_prompt
        res_counter.output_procedural_memory_prompt += memory.output_procedural_memory_prompt
        get_all_mem = memory.get_all(user_id=item["question_id"])
        print(f"get_all_mem: {get_all_mem}")
        get_all_mem_count = len(get_all_mem["results"])
        print(f"get_all_mem_count: {get_all_mem_count}")
        res_counter.memory_count = get_all_mem_count
        print(res_counter.to_dict())
        save_results_to_json([res_counter], f"results_{process_mode}_rag/results_{item['question_id']}.json")  # Updated path to relative one

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mem0_qwen")
    parser.add_argument("--begin", type=int, default=51)
    parser.add_argument("--end", type=int, default=60)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--embedding_model", type=str, default="models/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--process_mode", type=str, choices=["session", "turn"], default="turn")
    args = parser.parse_args()
    main(args)
