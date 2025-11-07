from typing import List, Dict, Optional, Any, Union
from lightmem.memory.utils import resolve_tokenizer

class ShortMemBufferManager:
    def __init__(self, max_tokens: int = 2000, tokenizer: Optional[Any] = None):
        self.max_tokens = max_tokens
        self.tokenizer = resolve_tokenizer(tokenizer)
        self.buffer: List[List[Dict[str, Any]]] = [] 
        self.token_count: int = 0 
        print(f"ShortMemBufferManager initialized with max_tokens={self.max_tokens}")
    def _count_tokens(self, messages: List[Dict[str, Any]], messages_use: str) -> int:
        role_map = {
            "user_only": ["user"],
            "assistant_only": ["assistant"],
            "hybrid": ["user", "assistant"],
        }

        allowed_roles = role_map.get(messages_use, [])
        text_list = [msg["content"] for msg in messages if msg["role"] in allowed_roles]

        text = " ".join(text_list)

        if self.tokenizer is None:
            return len(text)
        elif hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text))
        elif isinstance(self.tokenizer, str):
            raise ValueError(
                f"Tokenizer as model_name '{self.tokenizer}' not supported directly. "
                f"Please resolve to actual tokenizer before using."
            )
        else:
            raise TypeError("Invalid tokenizer type")


    def add_segments(self, all_segments: List[List[Dict[str, Any]]], messages_use: str, force_extract: bool = False):
        triggered: List[List[List[Dict[str, Any]]]] = []
        trigger_num = 0

        for seg in all_segments:
            tokens_needed = self._count_tokens(seg, messages_use)
            if self.token_count + tokens_needed > self.max_tokens:
                if self.buffer:  
                    triggered.append(self.buffer.copy())
                    trigger_num += 1
                    self.buffer.clear()
                    self.token_count = 0
            self.buffer.append(seg)
            self.token_count += tokens_needed

        if force_extract and self.buffer:
            triggered.append(self.buffer.copy())
            trigger_num += 1
            self.buffer.clear()
            self.token_count = 0
            
        return trigger_num, triggered

