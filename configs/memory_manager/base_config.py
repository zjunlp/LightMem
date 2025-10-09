from typing import Dict, Optional, Union


class BaseMemoryManagerConfig:
    """
    Config for LLMs.
    """
    def __init__(
        self,
        model: Optional[Union[str, Dict]] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        # Openai specific
        openai_base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        # Ollama specific
        ollama_base_url: Optional[str] = None,
        # DeepSeek specific
        deepseek_base_url: Optional[str] = None,
    ):

        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.enable_vision = enable_vision
        self.vision_details = vision_details
        # Openai specific
        self.openai_base_url = openai_base_url

        # Ollama specific
        self.ollama_base_url = ollama_base_url

        # DeepSeek specific
        self.deepseek_base_url = deepseek_base_url
