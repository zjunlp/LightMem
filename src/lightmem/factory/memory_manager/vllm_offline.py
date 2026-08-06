"""
This module implements a memory manager using vLLM for **offline** inference.

Require vLLM version >= 0.9.0, and vLLM version 0.11.0 is recommended.
"""

import concurrent
import json
import warnings
from typing import Dict, List, Optional, Literal, Any, Union

import torch
try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("The 'vllm' library is required. Please install it in a new environment using 'pip install vllm', recommended version >= 0.9.0.")

from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.prompts import EXTRACTION_PROMPTS, METADATA_GENERATE_PROMPT
from lightmem.memory.utils import clean_response

DEFAULT_MAX_MODEL_LEN = 128000


class VllmOfflineManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            raise ValueError("VLLM model is not specified. Refer to https://vllm.ai/models/ for available models.")

        if self.config.num_gpu is None:
            self.config.num_gpu = 1
        elif self.config.num_gpu == -1:  # Use all available GPUs
            if torch.cuda.is_available():
                self.config.num_gpu = torch.cuda.device_count() or 1
            else:
                warnings.warn("CUDA not available, using CPU mode.")
                self.config.num_gpu = 0
        
        self.config.gpu_memory_utilization = getattr(self.config, "gpu_memory_utilization", 0.9)

        self.client = LLM(
            model=self.config.model,
            trust_remote_code=self.config.trust_remote_code,
            tensor_parallel_size=max(1, self.config.num_gpu),
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=DEFAULT_MAX_MODEL_LEN,
        )

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from **vLLM offline deployment**.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.

        reference: https://docs.vllm.ai/en/latest/examples/offline_inference/chat_with_tools
        """
        content = response.outputs[0].text.strip()
        
        if tools:
            processed_response = {
                "content": content,
                "tool_calls": [],
            }
            # Offline vLLM doesn't support tool calls in the same way, so return the content
            return processed_response
        else:
            return content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    ) -> Optional[str]:
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            think (bool or str, optional): Thinking level for the model. Defaults to None.

        Returns:
            str: The generated response.
        """
        if self.client is None:
            raise ValueError("vLLM client is not initialized.")
        
        params = SamplingParams(
            seed=self.config.seed,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens,
            stop=self.config.stop,
        )

        if think is None or think is False:
            think = False  # Set to False to strictly disable thinking
        else:
            think = True

        outputs = self.client.chat(
            [messages], 
            params,
            chat_template_kwargs={"enable_thinking": think},
        )

        response = self._parse_response(outputs[0], tools)
        usage_info = self._extract_usage_info(outputs[0])

        return response, usage_info

    def _extract_usage_info(self, response) -> Dict[str, int]:
        prompt_tokens = 0
        completion_tokens = 0

        prompt_token_ids = getattr(response, "prompt_token_ids", None)
        if isinstance(prompt_token_ids, list):
            prompt_tokens = len(prompt_token_ids)

        outputs = getattr(response, "outputs", None)
        if outputs and len(outputs) > 0:
            token_ids = getattr(outputs[0], "token_ids", None)
            if isinstance(token_ids, list):
                completion_tokens = len(token_ids)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def meta_text_extract(
        self,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only",
        topic_id_mapping: Optional[List[List[int]]] = None,
        extraction_mode: Literal["flat", "event"] = "flat",
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.

        Args:
            extract_list: List of message segments to process
            messages_use: Strategy for which messages to use
            topic_id_mapping: For each API call, the global topic IDs
            extraction_mode: "flat" or "event"
            custom_prompts: Optional custom prompts. If None, use defaults from EXTRACTION_PROMPTS

        Returns:
            List of extracted metadata results, None for failed segments
        """
        if not extract_list:
            return []

        default_prompts = EXTRACTION_PROMPTS.get(extraction_mode, {})
        if custom_prompts is None:
            prompts = default_prompts
        else:
            prompts = {**default_prompts, **custom_prompts}

        if extraction_mode == "flat":
            return self._extract_with_prompt(
                system_prompt=prompts.get("factual", METADATA_GENERATE_PROMPT),
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual",
            )
        if extraction_mode == "event":
            factual_results = self._extract_with_prompt(
                system_prompt=prompts["factual"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual",
            )
            relational_results = self._extract_with_prompt(
                system_prompt=prompts["relational"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="relational",
            )
            return self._merge_dual_perspective_results(factual_results, relational_results)

        raise ValueError(f"Unknown extraction_mode: {extraction_mode}")

    def _merge_dual_perspective_results(
        self,
        factual_results: List[Optional[Dict]],
        relational_results: List[Optional[Dict]],
    ) -> List[Optional[Dict]]:
        merged_results = []
        for factual, relational in zip(factual_results, relational_results):
            if factual is None and relational is None:
                merged_results.append(None)
                continue

            merged = {
                "input_prompt": [],
                "output_prompt": "",
                "cleaned_result": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

            if factual is not None:
                merged["input_prompt"].extend(factual.get("input_prompt", []))
                merged["cleaned_result"].extend(factual.get("cleaned_result", []))
                for key in merged["usage"]:
                    merged["usage"][key] += factual.get("usage", {}).get(key, 0)

            if relational is not None:
                merged["input_prompt"].extend(relational.get("input_prompt", []))
                merged["cleaned_result"].extend(relational.get("cleaned_result", []))
                for key in merged["usage"]:
                    merged["usage"][key] += relational.get("usage", {}).get(key, 0)

            merged["output_prompt"] = (
                f"Factual: {factual.get('output_prompt', 'N/A') if factual else 'N/A'}\n"
                f"Relational: {relational.get('output_prompt', 'N/A') if relational else 'N/A'}"
            )
            merged_results.append(merged)
        return merged_results

    def _extract_with_prompt(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: str,
        topic_id_mapping: Optional[List[List[int]]],
        entry_type: str = "factual",
    ) -> List[Optional[Dict]]:
        def concatenate_messages(segment: List[Dict], messages_use: str) -> str:
            role_filter = {
                "user_only": {"user"},
                "assistant_only": {"assistant"},
                "hybrid": {"user", "assistant"},
            }

            if messages_use not in role_filter:
                raise ValueError(f"Invalid messages_use value: {messages_use}")

            allowed_roles = role_filter[messages_use]
            message_lines = []

            for mes in segment:
                if mes.get("role") in allowed_roles:
                    sequence_id = mes["sequence_number"]
                    role = mes["role"]
                    content = mes.get("content", "")
                    speaker_name = mes.get("speaker_name", "")
                    time_stamp = mes.get("time_stamp", "")
                    weekday = mes.get("weekday", "")
                    time_prefix = f"[{time_stamp}, {weekday}] " if time_stamp and weekday else ""
                    if speaker_name:
                        message_lines.append(f"{time_prefix}{sequence_id // 2}.{speaker_name}: {content}")
                    else:
                        message_lines.append(f"{time_prefix}{sequence_id // 2}.{role}: {content}")

            return "\n".join(message_lines)

        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(args):
            api_call_idx, api_call_segments = args
            try:
                user_prompt_parts: List[str] = []
                global_topic_ids: List[int] = []
                if topic_id_mapping and api_call_idx < len(topic_id_mapping):
                    global_topic_ids = topic_id_mapping[api_call_idx]

                for topic_idx, topic_segment in enumerate(api_call_segments):
                    global_topic_id = (
                        global_topic_ids[topic_idx]
                        if topic_idx < len(global_topic_ids)
                        else topic_idx + 1
                    )
                    topic_text = concatenate_messages(topic_segment, messages_use)
                    user_prompt_parts.append(f"--- Topic {global_topic_id} ---\n{topic_text}")

                user_prompt = "\n".join(user_prompt_parts)

                metadata_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                raw_response, usage_info = self.generate_response(
                    messages=metadata_messages,
                )
                metadata_facts = clean_response(raw_response)
                for entry in metadata_facts:
                    entry["entry_type"] = entry_type

                return {
                    "input_prompt": metadata_messages,
                    "output_prompt": raw_response,
                    "cleaned_result": metadata_facts,
                    "usage": usage_info,
                    "entry_type": entry_type,
                }
            except Exception as e:
                print(f"Error processing API call {api_call_idx}: {e}")
                return {
                    "input_prompt": [],
                    "output_prompt": "",
                    "cleaned_result": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "entry_type": entry_type,
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, enumerate(extract_list)))
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                results = [None] * len(extract_list)

        return results
