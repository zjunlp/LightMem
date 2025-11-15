import concurrent
import json
from typing import Dict, List, Optional, Literal, Any, Union

try:
    import ollama
except ImportError:
    raise ImportError("The 'ollama' library is required. Please install it using 'pip install ollama', recommended version >= 0.6.0.")

from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response


class OllamaManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            raise ValueError("Ollama model is not specified. Refer to https://ollama.com/docs/models for available models.")

        self.client = ollama.Client(host=self.config.host or "http://localhost:11434")

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from **Ollama offline deployment**.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.

        reference: https://ollama.com/blog/tool-support
        """
        if tools:
            processed_response = {
                "content": response["message"]["content"],
                "tool_calls": [],
            }

            if response['message']['tool_calls']:
                for tool_call in response['message']['tool_calls']:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.parameters),
                        }
                    )

            return processed_response
        else:
            return response["message"]["content"]

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
            raise ValueError("Ollama client is not initialized.")

        params =  {
            "model": self.config.model,
            "messages": messages,
            "seed": self.config.seed,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "stop": self.config.stop,
        }
        
        completion = self.client.chat(
            model=self.config.model,
            messages=messages,
            format=response_format,
            tools=tools,
            think=think,
            options={
                "num_gpu": self.config.num_gpu,
                "main_gpu": self.config.main_gpu,
                "num_ctx": params["max_tokens"],
                "seed": params["seed"],
                "temperature": params["temperature"],
                "top_k": params["top_k"],
                "top_p": params["top_p"],
                "stop": params["stop"],
            }
        )

        response = self._parse_response(completion, tools)

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
