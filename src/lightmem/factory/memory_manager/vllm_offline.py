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

        return response

    def meta_text_extract(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only"
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.

        Args:
            system_prompt: The system prompt for metadata generation
            all_segments: List of message segments to process
            messages_use: Strategy for which messages to use

        Returns:
            List of extracted metadata results, None for failed segments
        """
        if not extract_list:
            return []
            
        def concatenate_messages(segment: List[Dict], messages_use: str) -> str:
            """Concatenate messages based on usage strategy"""
            role_filter = {
                "user_only": {"user"},
                "assistant_only": {"assistant"},
                "hybrid": {"user", "assistant"}
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
                    message_lines.append(f"{sequence_id}.{role}: {content}")

            return "\n".join(message_lines)
        
        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(api_call_segments: List[List[Dict]]) -> Dict[str, Any]:
            """Process one API call (multiple topic segments inside)"""
            try:
                user_prompt_parts = []
                for idx, topic_segment in enumerate(api_call_segments, start=1):
                    topic_text = concatenate_messages(topic_segment, messages_use)
                    user_prompt_parts.append(f"--- Topic {idx} ---\n{topic_text}")

                user_prompt = "\n".join(user_prompt_parts)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                raw_response = self.generate_response(
                    messages=messages,
                )
                cleaned_result = clean_response(raw_response)
                return {
                    "input_prompt": messages,
                    "output_prompt": raw_response,
                    "cleaned_result": cleaned_result
                }
            except Exception as e:
                print(f"Error processing API call: {e}")
                # When error occurs, return empty but full structure
                return {
                    "input_prompt": [],
                    "output_prompt": "",
                    "cleaned_result": [],
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, extract_list))
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                results = [None] * len(extract_list)

        return results
