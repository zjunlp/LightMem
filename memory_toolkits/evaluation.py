import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
import time 
import threading

from memories.layers.amem import AMEMConfig, AMEMLayer
from memories.datasets.longmemeval import LongMemEval
from token_monitor import CostStateManager, token_monitor
from monkey_patch import (
    PatchSpec, 
    MonkeyPatcher, 
    make_attr_patch, 
)
from memories.datasets.base import Trajectory
from memories.layers.memzero import MemZeroConfig, MemZeroLayer
from typing import (
    Dict, 
    Any, 
    Optional, 
)

DATASET_MAPPING = {
    "LongMemEval": LongMemEval,
}

MEMORY_MAPPING = {
    "A-MEM": {
        "layer": AMEMLayer,
        "config": AMEMConfig,
    },
    "MemZero": {
        "layer": MemZeroLayer,
        "config": MemZeroConfig,
    },
}

_LOCK = threading.Lock() 

def memory_construction(
    layer_type: str, 
    user_id: str, 
    trajectory: Trajectory, 
    config: Optional[Dict[str, Any]] = None, 
    rerun: bool = False,
) -> Dict[str, float]: 
    """Given a specific interaction trajectory, this function builds a memory."""
    config = config or {}
    # It overrides the user_id in the config. 
    config["user_id"] = user_id 
    # Each user has a distinct config directory. 
    config["save_dir"] = f"{layer_type}/{user_id}" 
    config = MEMORY_MAPPING[layer_type]["config"](**config)
    with _LOCK:
        layer = MEMORY_MAPPING[layer_type]["layer"](config)

    output = {
        "total_process_time": 0.0,
        "avg_process_time": 0.0,
    }

    with _LOCK:
        # It includes I/O operations. 
        if not rerun and layer.load_memory(user_id):
            print(f"🔄 The memory for user {user_id} is loaded successfully 😄.")
            return output
    
    if layer_type == "A-MEM":
        # In this case, we modify an instance's method. 
        # Other instances are not affected. 
        getter, setter = make_attr_patch(layer.memory_layer.llm_controller.llm, "get_completion")
        spec = PatchSpec(
            name=f"{layer.memory_layer.llm_controller.llm.__class__.__name__}.get_completion",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name=lambda *args, **kwargs: (config.llm_model, {}),
                extract_input_dict=lambda *args, **kwargs: {
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You must respond with a JSON object."
                        }, 
                        {
                            "role": "user",
                            "content": kwargs.get("prompt", args[0])
                        }
                    ],
                    "metadata": {
                        "op_type": (
                            "generation"
                            if kwargs.get("prompt", args[0]).startswith("Generate a structured analysis") 
                            else "update"
                        )
                    }
                },
                extract_output_dict=lambda result: {
                    "messages": result
                },
            ),
        )
        specs = [spec] 
    elif layer_type == "MemZero":
        getter, setter = make_attr_patch(layer.mem0.llms.openai.OpenAILLM, "generate_response")
        spec = PatchSpec(
            name = f"{layer.mem0.llms.openai.OpenAILLM.__class__.__name__}.generate_response",
            getter = getter,
            setter = setter,
            wrapper = token_monitor(
                extract_model_name = lambda *args, **kwargs: (config.llm_model, {}),
                extract_input_dict = lambda *args, **kwargs: {
                    "messages": kwargs.get("messages", args[0]),
                    "metadata": {
                        "op_type": (
                            "generation" if kwargs.get("messages", args[0])[0]["content"].startswith("You are a Personal Information Organizer") else "update"
                        )
                    }
                },
                extract_output_dict = lambda response: {
                    "messages": response.choices[0].message.content
                },
            )
        )
        specs = [spec]
    else:
        raise ValueError(f"Unsupported memory type: {layer_type}.")

    with MonkeyPatcher(specs):
        # Start to construct the memory for a specific trajectory. 
        for session in trajectory:
            # TODO: take the case that the message is a question-answe pair into the consideration 
            for message in session:
                start_time = datetime.now() 
                layer.add_message(
                    {"role": message.role, "content": message.content}, 
                    timestamp=session.get_string_timestamp()
                )
                end_time = datetime.now() 
                output["total_process_time"] += (end_time - start_time).total_seconds()
                time.sleep(0.2)
    
    if layer_type == "A-MEM":
        # It includes I/O operations (loading a sentence embedding model).
        with _LOCK:
            layer.consolidate_memories()
    # It includes I/O operations. 
    with _LOCK:
        layer.save_memory() 

    output["avg_process_time"] = output["total_process_time"] / len(trajectory)

    return output 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script used to evaluate various memory layers on various datasets."
    )
    parser.add_argument(
        "--memory-type", 
        choices=list(MEMORY_MAPPING.keys()), 
        type=str, 
        required=True, 
        help="The type of the memory layer to be evaluated."
    )
    parser.add_argument(
        "--dataset-type", 
        choices=list(DATASET_MAPPING.keys()), 
        type=str, 
        required=True, 
        help="The type of the dataset used to evaluate the memory layer."
    )
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        required=True, 
        help="The path to the dataset."
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4, 
        help="The number of threads to use for the evaluation."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed used to sample the dataset if the user provides the sample size."
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None, 
        help="Subset size from dataset."
    )
    parser.add_argument(
        "--rerun", 
        action="store_true", 
        help="Ignore saved memory; rebuild from scratch."
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default=None,
        help="Path to JSON config for memory method."
    )
    parser.add_argument(
        "--token-cost-save-filename", 
        type=str, 
        default="token_cost", 
        help="Path to save the statistics related to the token consumption."
    )
    args = parser.parse_args()

    # Prepare the dataset
    ds_cls = DATASET_MAPPING[args.dataset_type]
    dataset = ds_cls.read_raw_data(args.dataset_path) 
    if args.sample_size is not None:
        dataset = dataset.sample(size=args.sample_size, seed=args.seed)
    print("The dataset is loaded successfully 😄.")
    # print(dataset) calls the __str__ method defined by Pydantic's BaseModel
    print(repr(dataset))
    print()

    config = None 
    if args.config_path is not None:
        with open(args.config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)

    # Get a dummy configuration to infer the corresponding LLM being used 
    if config is None:
        dummy_user_id = "guest" 
        dummy_config = MEMORY_MAPPING[args.memory_type]["config"](user_id=dummy_user_id)
    else:
        dummy_config = MEMORY_MAPPING[args.memory_type]["config"](**config) 

    # Before run the expriment, we should register the base model being used. 
    # Please ensure all types of config classes have a `llm_model` attribute. 
    # The tokenizer is inferred from the model name automatically. 
    llm_model = dummy_config.llm_model
    CostStateManager.register(llm_model)
    del dummy_config 
    print(f"The LLM model 🤖 being used is {llm_model}. It has been registered in `CostStateManager`.")
    print()

    results = [] 
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i, (trajectory, _) in enumerate(dataset):
            user_id = f"{dataset.__class__.__name__}_{i}"
            future = executor.submit(
                memory_construction, 
                args.memory_type, 
                user_id, 
                trajectory, 
                config=config, 
                rerun=args.rerun 
            )
            futures.append(future)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="📉 Processing trajectories"
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing trajectory: {e}")

    if len(results) == len(dataset):
        print("The evaluation is completed successfully 😀.")

    total_time = 0.0 
    avg_time_per_add_session = 0.0 
    for result in results: 
        total_time += result["total_process_time"]
        avg_time_per_add_session += result["avg_process_time"]
    avg_time = total_time / len(results)
    avg_time_per_add_session = avg_time_per_add_session / len(results)
    print(f"The average time per trajectory is {avg_time:.2f} seconds.")
    print(f"The average time per add session is {avg_time_per_add_session:.2f} seconds.")

    # Save the statistics of token comsumption 
    CostStateManager.save_to_json_file(args.token_cost_save_filename)
    CostStateManager.reset() 