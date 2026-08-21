import os

from openai import OpenAI

from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.factory.memory_manager.deepseek import DeepseekManager


class OrcaRouterManager(DeepseekManager):
    """Memory manager backed by [OrcaRouter](https://www.orcarouter.ai).

    OrcaRouter exposes an OpenAI-compatible endpoint at
    ``https://api.orcarouter.ai/v1`` and uses ``sk-orca-`` prefixed API keys.
    It also runs gateway-level, zero-trust security for AI agents on the same
    endpoint — screening every prompt/response and governing every tool call on
    a default-deny basis, with no application code changes.
    """

    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config
        if not self.config.model:
            self.config.model = "orcarouter/auto"
        self.api_key = self.config.api_key or os.getenv("ORCAROUTER_API_KEY")
        self.base_url = (
            self.config.orcarouter_base_url or "https://api.orcarouter.ai/v1"
        ).rstrip("/")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
