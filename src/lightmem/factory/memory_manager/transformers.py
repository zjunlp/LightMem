import concurrent
import json
from typing import Dict, List, Optional, Literal, Any, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response


class TransformersManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            self.config.model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

        if not torch.cuda.is_available() or self.config.num_gpu == 0:
            self.device = "cpu"
        elif self.config.num_gpu == -1:
            self.device = "auto"
        elif self.config.num_gpu == 1:
            self.device = {"": f"cuda:{self.config.main_gpu}"}
        else: # For multiple GPUs, use 'auto' to let Transformers distribute the model across all available GPUs.
            self.device = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, 
            use_fast=True
        )

        self.client = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from the **HuggingFace Transformers model**.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.

        TODO: reference at https://huggingface.co/docs/transformers/main/chat_extras#tool-use
        """
        content = response.strip()
        
        if tools:
            processed_response = {
                "content": content,
                "tool_calls": [],
            }
            # Transformers doesn't support tool calls in the same way, so return the content
            return processed_response
        else:
            return content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.

        Returns:
            str: The generated response.
        """
        params =  {
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
        }

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.client.device)

        outputs = self.client.generate(
            **inputs,
            do_sample=params["do_sample"],
            temperature=params["temperature"],
            max_new_tokens=params["max_tokens"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]

        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        response = self._parse_response(text, tools)

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
