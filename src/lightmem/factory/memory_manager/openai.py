import concurrent
from collections import defaultdict
from openai import OpenAI
from typing import List, Dict, Optional, Literal, Any
import json, os, warnings
import httpx
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response

model_name_context_windows = {
    "gpt-4o-mini": 128000,
    "qwen3-30b-a3b-instruct-2507": 128000,
    "DEFAULT": 128000,  # Recommended default context window
}


class OpenaiManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            self.config.model = "gpt-4o-mini"
        
        if self.config.model in model_name_context_windows:
            self.context_windows = model_name_context_windows[self.config.model]
        else:
            self.context_windows = model_name_context_windows["DEFAULT"]

        http_client = httpx.Client(verify=False)

        if os.environ.get("OPENROUTER_API_KEY"):  # Use OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = (
                self.config.openai_base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )

            self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
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
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
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
    ) -> Optional[str]:
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        if os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            
            models = getattr(self.config, 'models', None)    
            route = getattr(self.config, 'route', 'fallback') 
            if models:
                openrouter_params["models"] = models
                openrouter_params["route"] = route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        if response_format:
            params["response_format"] = response_format
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        parsed_response = self._parse_response(response, tools)

        return parsed_response, usage_info

    def meta_text_extract(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only",
        topic_id_mapping: Optional[List[List[int]]] = None
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.

        Args:
            system_prompt: The system prompt for metadata generation
            extract_list: List of message segments to process
            messages_use: Strategy for which messages to use
            topic_id_mapping: For each API call, the global topic IDs

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
                    speaker_name = mes.get("speaker_name", "")
                    time_stamp = mes.get("time_stamp", "")
                    weekday = mes.get("weekday", "")
                    
                    time_prefix = ""
                    if time_stamp and weekday:
                        time_prefix = f"[{time_stamp}, {weekday}] "

                    if speaker_name:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{speaker_name}: {content}")
                    else:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{role}: {content}")
            
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
                    if topic_idx < len(global_topic_ids):
                        global_topic_id = global_topic_ids[topic_idx]
                    else:
                        global_topic_id = topic_idx + 1
                    
                    topic_text = concatenate_messages(topic_segment, messages_use)
                    user_prompt_parts.append(f"--- Topic {global_topic_id} ---\n{topic_text}")

                print(f"User prompt for API call {api_call_idx}:\n" + "\n".join(user_prompt_parts))
                user_prompt = "\n".join(user_prompt_parts)
                
                metadata_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                
                raw_response, usage_info = self.generate_response(
                    messages=metadata_messages,
                    response_format={"type": "json_object"},
                )
                metadata_facts = clean_response(raw_response)

                return {
                    "input_prompt": metadata_messages,
                    "output_prompt": raw_response,
                    "cleaned_result": metadata_facts,
                    "usage": usage_info,
                }
                
            except Exception as e:
                print(f"Error processing API call {api_call_idx}: {e}")
                return {
                    "input_prompt": [],
                    "output_prompt": "",
                    "cleaned_result": [],
                    "usage": None,
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, enumerate(extract_list)))
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                results = [None] * len(extract_list)

        return results
    
    def _call_update_llm(self, system_prompt, target_entry, candidate_sources):
        target_memory = target_entry["payload"]["memory"]
        candidate_memories = [c["payload"]["memory"] for c in candidate_sources]

        user_prompt = (
            f"Target memory:{target_memory}\n"
            f"Candidate memories:\n" + "\n".join([f"- {m}" for m in candidate_memories])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text, usage_info = self.generate_response(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response_text)
            if "action" not in result:
                result = {"action": "ignore"}
            result["usage"] = usage_info  
            return result
        except Exception:
            return {"action": "ignore", "usage": usage_info if 'usage_info' in locals() else None}
