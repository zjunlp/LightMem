from openai import OpenAI
from typing import Optional, List, Union
import os
import httpx
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig


class TextEmbedderOpenAI:
    def __init__(self, config: Optional[BaseTextEmbedderConfig] = None):
        self.config = config
        self.model = getattr(config, "model", None) or "text-embedding-3-small"        
        http_client = httpx.Client(verify=False)
        api_key = self.config.api_key 
        base_url = self.config.openai_base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
        self.total_calls = 0
        self.total_tokens = 0

    @classmethod
    def from_config(cls, config: BaseTextEmbedderConfig):
        return cls(config)

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        def preprocess(t):
            return str(t).replace("\n", " ")

        api_params = {"model": self.config.model}
        api_params["dimensions"] = self.config.embedding_dims

        if isinstance(text, list):
            if len(text) == 0:
                return []
            inputs = [preprocess(x) for x in text]
            resp = self.client.embeddings.create(input=inputs, **api_params)
            self.total_calls += 1
            self.total_tokens += resp.usage.total_tokens
            return [item.embedding for item in resp.data]
        else:
            preprocessed = preprocess(text)
            resp = self.client.embeddings.create(input=[preprocessed], **api_params)
            self.total_calls += 1
            self.total_tokens += resp.usage.total_tokens
            return resp.data[0].embedding
        
    def get_stats(self):
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
        }