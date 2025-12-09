"""
Need the deployment of vLLM API server first.

The command to start a vLLM API server:
vllm serve [model_tag] [options]
"""

import concurrent
import json
import os
import re
from openai import OpenAI
from typing import List, Dict, Optional, Literal, Any

from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response


class VllmManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            raise ValueError("vLLM model is not specified. Refer to https://vllm.ai/docs/models/ for available models.")

        if self.config.api_key:
            print("Using your provided vLLM API key, make sure your vLLM server supports API key authentication.")
            self.api_key = self.config.api_key or os.getenv("VLLM_API_KEY")
        else:
            self.api_key = None

        self.base_url = self.config.vllm_base_url or "http://localhost:8000"

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(base_url=self.base_url)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from vLLM API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    args = tool_call.function.arguments.strip()
                    match = re.search(r"```(?:json)?\s*(.*?)\s*```", args, re.DOTALL)
                    if match:
                        args = match.group(1)
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(args),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using vLLM.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional vLLM-specific parameters.

        Returns:
            str: The generated response.
        """
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )

        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        str_response = self._parse_response(response, tools)

        return str_response

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
