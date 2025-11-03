from openai import OpenAI
from typing import List, Dict, Optional
import json
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig

class DeepseekManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config
        if not self.config.model:
            self.config.model = "deepseek-chat"
        self.api_key = self.config.api_key
        self.base_url = (self.config.deepseek_base_url or "https://api.deepseek.com/v1").rstrip("/")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @classmethod
    def from_config(cls, model_name, config):
        cls.validate_config(config)
        return cls(config)

    @staticmethod
    def validate_config(config):
        required_keys = ['api_key', 'endpoint']
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required config keys for DeepSeek LLM: {missing}")

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
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using DeepSeek.

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
            "top_k": self.config.top_k
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)