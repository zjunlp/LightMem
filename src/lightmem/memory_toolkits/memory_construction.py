import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
import time 
import threading
from importlib.util import find_spec
from copy import deepcopy 
import os
from contextlib import nullcontext

from token_monitor import (
    CostStateManager, 
    token_monitor, 
    CostState, 
    get_tokenizer_for_model,
)
from monkey_patch import (
    PatchSpec, 
    MonkeyPatcher, 
    make_attr_patch, 
)

from memories import (
    CONFIG_MAPPING,
    MEMORY_LAYERS_MAPPING,
    DATASET_MAPPING,
)
from memories.datasets.base import Trajectory
from typing import (
    Dict, 
    Any, 
    Optional, 
    Tuple, 
    List, 
    Callable,
)
from memories.datasets.base import Trajectory, Message, QuestionAnswerPair, Session
from importlib import import_module

_LOCK = threading.Lock()

def _check_langchain_core_imports() -> None: 
    """Check if `langchain_core` is installed."""
    if find_spec("langchain_core") is None:
        raise ImportError("`langchain_core` is not installed. Please install it to use this function.")

def _normalize_langmem_messages(*args, **kwargs) -> Dict[str, List[Dict[str, str]] | str | float | int]:
    """A helper function to process the messages of LangMem."""
    _check_langchain_core_imports() 
    from langchain_core.messages import (
        HumanMessage, 
        SystemMessage, 
        AIMessage, 
        ToolMessage, 
    )

    messages = kwargs.get("messages", args[0])
    assert len(messages) == 1, "Unconsidered Case."
    messages = messages[0]
    normalized_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            normalized_messages.append(
                {
                    "role": "system", 
                    "content": message.content
                }
            )
        elif isinstance(message, HumanMessage):
            normalized_messages.append(
                {
                    "role": "user", 
                    "content": message.content
                }
            ) 
        elif isinstance(message, AIMessage):
            if message.content is not None and not isinstance(message.content, str):
                raise ValueError(
                    f"The content of the message is not a string: {type(message.content)}."
                )
            normalized_messages.append(
                {
                    "role": "assistant", 
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tool_call["id"], 
                            "type": "function", 
                            "function": {
                                "name": tool_call["name"],
                                "arguments": str(tool_call["args"]),
                            }
                        }
                        for tool_call in message.tool_calls
                    ], 
                }
            )
        elif isinstance(message, ToolMessage):
            # See https://platform.openai.com/docs/guides/function-calling 
            normalized_messages.append(
                {
                    "role": "tool", 
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}.")
    
    return normalized_messages

def _extract_langmem_model(
    llm_model: str, 
    query_model: Optional[str], 
    *args,  
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """A helper function to extract the model name and metadata for LangMem."""
    _check_langchain_core_imports()  
    from langchain_core.messages import HumanMessage

    llm_model = llm_model.split(':', 1)[1]
    query_model = query_model.split(':', 1)[1] if query_model is not None else None 

    messages = kwargs.get("messages", args[0])
    assert len(messages) == 1, "Unconsidered Case."
    messages = messages[0]
    # The following parameters are used in LiteLLM's token counter. 
    metadata = {
        "tools": kwargs.get("tools"), 
        "tool_choice": kwargs.get("tool_choice"), 
    } 
    if isinstance(messages[0], HumanMessage) and messages[0].content.startswith(
        "Use parallel tool calling to search for distinct memories relevant to this conversation."
    ):
        if query_model is None:
            raise ValueError("Query model is not provided.")
        return query_model, metadata 
    return llm_model, metadata 

def _extract_langmem_output(response) -> Dict[str, List[Dict[str, str]] | str | float | int]:
    """A helper function to extract the output for LangMem."""
    assert len(response.generations) == 1, "Unconsidered Case."
    assert len(response.generations[0]) == 1, "Unconsidered Case."
    return {
        "messages": _normalize_langmem_messages(
            [[response.generations[0][0].message]]
        )
    } 
    

def memory_construction(
    layer_type: str, 
    user_id: str, 
    trajectory: Trajectory, 
    config: Optional[Dict[str, Any]] = None, 
    rerun: bool = False,
    message_preprocessor: Optional[
        Callable[[Message | QuestionAnswerPair, Session], Dict[str, Any]]
    ] = None,
    **kwargs
) -> Dict[str, float]: 
    """Given a specific interaction trajectory, this function builds a memory."""
    config = config or {}
    llm_model = config["llm_model"]
    # It overrides the user_id in the config. 
    config["user_id"] = user_id 
    # Each user has a distinct config directory. 
    config["save_dir"] = f"{layer_type}_{llm_model}/{user_id}" 
    # Use lazy mapping to load config and layer classes.
    config_cls = CONFIG_MAPPING[layer_type]
    config = config_cls(**config)
    dataset_type = kwargs["dataset_type"]
    with _LOCK:
        layer_cls = MEMORY_LAYERS_MAPPING[layer_type]
        layer = layer_cls(config)

    output = {
        "total_add_time": 0.0,
        "avg_add_time": 0.0,
    }

    with _LOCK:
        # It includes I/O operations. 
        if not rerun and layer.load_memory(user_id):
            print(f"The memory for user {user_id} is loaded successfully.")
            return output
    
    if layer_type == "A-MEM":
        # In this case, we modify an instance's method. 
        # Other instances are not affected. 
        # Note that there is no need to check `response_format` parameter. 
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
        patch_ctx = MonkeyPatcher(specs) 
    elif layer_type == "LangMem":
        getter, setter = make_attr_patch(layer.llm_model, "generate")
        spec = PatchSpec(
            name=f"{layer.llm_model.__class__.__name__}.generate",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name=lambda *args, **kwargs: _extract_langmem_model(
                    config.llm_model, 
                    config.query_model, 
                    *args, 
                    **kwargs
                ),
                extract_input_dict=lambda *args, **kwargs: {
                    # NOTE: LangMem uses the same prompt to generate and update memories. 
                    # These two types of operations are handled by the same forward pass of LLMs. 
                    "messages": _normalize_langmem_messages(*args, **kwargs),
                    "metadata": {
                        "op_type": "generation, update"
                    }
                },
                extract_output_dict=lambda response: _extract_langmem_output(response)
            )
        )
        specs = [spec]
        patch_ctx = MonkeyPatcher(specs)
    elif layer_type in ("MemZero", "MemZeroGraph"):
        getter, setter = make_attr_patch(layer.memory_layer.llm, "generate_response")
        spec = PatchSpec(
            name=f"{layer.memory_layer.llm.__class__.__name__}.generate_response",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name = lambda *args, **kwargs: (config.llm_model, {}),
                extract_input_dict = lambda *args, **kwargs: {
                    # keyword arguments
                    "messages": kwargs["messages"],
                    "metadata": {
                        "op_type": (
                            "generation" if kwargs["messages"][0]["content"].startswith("You are a Personal Information Organizer") else "update"
                        )
                    }
                },
                # generate_response has parsed the output
                extract_output_dict = lambda response: {"messages": response.get("content", "")} if isinstance(response, dict) else {"messages": response}
            )
        )
        specs = [spec]
        patch_ctx = MonkeyPatcher(specs)
    elif layer_type == "FullContext":
        # No need to patch any method for FullContext layer. 
        patch_ctx = nullcontext()
    elif layer_type == "NaiveRAG":
        # No need to patch any method for NaiveRAG layer. 
        patch_ctx = nullcontext()
    else:
        raise ValueError(f"Unsupported memory type: {layer_type}.")

    
    with patch_ctx:
        total_msgs = sum(len(session) for session in trajectory)

        pbar_desc = f" {layer_type} | {user_id}"
        pbar = tqdm(
            total=total_msgs,
            desc=pbar_desc,
            leave=False,   
        )
        # Start to construct the memory for a specific trajectory. 
        for session in trajectory:
            # TODO: take the case that the message is a question-answe pair into the consideration 
            for message in session:
                start_time = datetime.now() 
                msg_dict = {"role": message.role, "content": message.content}
                if message_preprocessor is not None:
                    msg_dict = message_preprocessor(message, session)
                
                timestamp = session.get_string_timestamp()
                layer.add_message(
                    msg_dict, 
                    timestamp=timestamp
                )

                end_time = datetime.now() 
                output["total_add_time"] += (end_time - start_time).total_seconds()

                pbar.update(1)

                time.sleep(0.2)
        pbar.close()

    
    if layer_type == "A-MEM":
        # It includes I/O operations (loading a sentence embedding model).
        with _LOCK:
            layer.consolidate_memories()
    # It includes I/O operations. 
    with _LOCK:
        layer.save_memory() 

    output["avg_add_time"] = output["total_add_time"] / len(trajectory)

    return output 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script used to evaluate various memory layers on various datasets."
    )
    parser.add_argument(
        "--memory-type", 
        choices=list(MEMORY_LAYERS_MAPPING.keys()), 
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
        "--config-path", 
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
    parser.add_argument(
        "--start-idx", 
        type=int, 
        default=None, 
        help="The starting index of the trajectories to be processed."
    )
    parser.add_argument(
        "--end-idx", 
        type=int, 
        default=None, 
        help="The ending index of the trajectories to be processed."
    )
    parser.add_argument(
        "--tokenizer-path", 
        type=str, 
        default=None, 
        help="The path to the tokenizer (only for backbone model)."
    )
    parser.add_argument(
        "--message-preprocessor",
        type=str,
        default=None,
        help=(
            "Dotted path to a message preprocessor function, in the form "
            "'some_module:some_function'. The function should accept "
            "(message, session) and return a dict with at least 'role' "
            "and 'content' keys."
        )
    )
    parser.add_argument(
        "--use-gpt4o-caption",
        action="store_true",
        default=False,
        help=(
            "If set, use GPT-4o to generate image captions in datasets that support it "
            "(e.g., LoCoMo)."
        )
    )
    args = parser.parse_args()

    message_preprocessor = None
    if args.message_preprocessor is not None:
        module_path, func_name = args.message_preprocessor.split(":", 1)
        module = import_module(module_path)
        message_preprocessor = getattr(module, func_name)
    
    if args.use_gpt4o_caption:
        os.environ["USE_GPT4O_CAPTION"] = "1"
    else:
        os.environ["USE_GPT4O_CAPTION"] = "0"

    # Prepare the dataset using lazy mapping
    ds_cls = DATASET_MAPPING[args.dataset_type]
    dataset = ds_cls.read_raw_data(args.dataset_path) 
    if args.sample_size is not None:
        dataset = dataset.sample(size=args.sample_size, seed=args.seed)
    print("The dataset is loaded successfully.")
    # print(dataset) calls the __str__ method defined by Pydantic's BaseModel
    print(repr(dataset))
    print()

    config = None 
    if args.config_path is not None:
        with open(args.config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)

    # Get a dummy configuration to infer the corresponding LLM being used 
    # Use lazy mapping to load config class
    config_cls = CONFIG_MAPPING[args.memory_type]
    if config is None:
        dummy_user_id = "guest" 
        dummy_config = config_cls(user_id=dummy_user_id)
    else:
        dummy_config = config_cls(**config) 
    
    # If token cost file exists, we load it. 
    if os.path.exists(args.token_cost_save_filename + ".json"):
        with open(args.token_cost_save_filename + ".json", 'r') as f:
            token_cost = json.load(f)
        for model, state in token_cost.items():
            is_dict = all(isinstance(value, dict) for value in state.values())
            if not is_dict:
                token_cost[model] = CostState.from_dict(state)
            else:
                token_cost[model] = {
                    op: CostState.from_dict(cs) for op, cs in state.items()
                }
    else: 
        token_cost = {} 
    
    # Before run the expriment, we should register the base model being used. 
    # Please ensure all types of config classes have a `llm_model` attribute. 
    # The tokenizer is inferred from the model name automatically. 
    llm_model = dummy_config.llm_model
    if args.memory_type == "LangMem":
        llm_model = llm_model.split(':', 1)[1]
        if dummy_config.query_model is not None:
            query_model = dummy_config.query_model.split(':', 1)[1]
            if query_model != llm_model:
                state = token_cost.get(query_model)
                if state is not None: 
                    print(
                        f"There is a saved checkpoint for monitoring the token consumption of {query_model}. "
                        "It will be loaded into `CostStateManager`."
                    )
                CostStateManager.register(query_model, state=state)
    state = token_cost.get(llm_model)
    if state is not None: 
        print(
            f"There is a saved checkpoint for monitoring the token consumption of {llm_model}. "
            "It will be loaded into `CostStateManager`."
        )
    if args.tokenizer_path is not None:
        tokenizer = get_tokenizer_for_model(args.tokenizer_path)
    else:
        tokenizer = None 
    CostStateManager.register(llm_model, state=state, tokenizer=tokenizer)
    del dummy_config 
    print(f"The LLM model being used is {llm_model}. It has been registered in `CostStateManager`.")
    print()

    if args.start_idx is None:
        args.start_idx = 0 
    if args.end_idx is None:
        args.end_idx = len(dataset)
    args.start_idx, args.end_idx = max(0, args.start_idx), min(args.end_idx, len(dataset))
    if args.start_idx >= args.end_idx:
        raise ValueError("The starting index must be less than the ending index.")

    results = [] 
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for trajectory, _ in zip(*dataset[args.start_idx: args.end_idx]):
            # Note that this code is for academic purpose, the embedding model will be loaded multiple times. 
            user_id = f"user_{dataset.__class__.__name__}_{trajectory.metadata['id']}"
            future = executor.submit(
                memory_construction, 
                args.memory_type, 
                user_id, 
                trajectory, 
                config=deepcopy(config), 
                rerun=args.rerun,
                message_preprocessor=message_preprocessor,
                dataset_type=args.dataset_type
            )
            futures.append(future)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing trajectories"
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing trajectory: {e}")

    if len(results) == args.end_idx - args.start_idx:
        print("The memory construction process is completed successfully.")

    total_time = 0.0 
    avg_time_per_add_session = 0.0 
    num_vaild_trajectories = 0
    for result in results: 
        # Statistics on the newly processed trajectories
        if result["total_add_time"] > 0:
            total_time += result["total_add_time"]
            avg_time_per_add_session += result["avg_add_time"]
            num_vaild_trajectories += 1 
    avg_time = total_time / max(num_vaild_trajectories, 1)
    avg_time_per_add_session = avg_time_per_add_session / max(num_vaild_trajectories, 1)
    print(
        f"For {args.memory_type}, the average time per trajectory "
        f"({num_vaild_trajectories} in {len(results)}) is {avg_time:.2f} seconds."
    )
    print(
        f"For {args.memory_type}, the average time per operation of adding new session " 
        f"is {avg_time_per_add_session:.2f} seconds."
    )

    # Save the statistics of token comsumption 
    CostStateManager.save_to_json_file(args.token_cost_save_filename)
    CostStateManager.reset() 
