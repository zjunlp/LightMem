import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from lightmem.configs.pre_compressor.entropy_compress import EntropyCompressorConfig


class EntropyCompressor:
    def __init__(self, config: Optional[EntropyCompressorConfig] = None):
        self.config = config or EntropyCompressorConfig()
        model_name = self.config.entropy_config["model_name"]
        device = self.config.entropy_config["device"]

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")

        self.strategy = self.config.entropy_config["word_level_strategy"]
        self.compress_rate = self.config.entropy_config["compress_rate"]
        self.device = device
        self.max_length = self.config.entropy_config["max_length"]

    def _compute_token_info(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_probs = probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            info = -torch.log2(next_token_probs + 1e-12).squeeze(0).cpu().tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return tokens[1:], info  

    def _aggregate_word_info(self, text: str, tokens: List[str], infos: List[float]):
        words = []
        word_infos = []
        current_word = ""
        current_infos = []

        for tok, inf in zip(tokens, infos):
            if tok.startswith("##") or tok.startswith("▁") or tok.startswith("Ġ"):
                tok_clean = tok.lstrip("##▁Ġ")
                current_word += tok_clean
                current_infos.append(inf)
            else:
                if current_word:
                    if self.strategy == "average":
                        word_infos.append(sum(current_infos) / len(current_infos))
                    elif self.strategy == "first_token":
                        word_infos.append(current_infos[0])
                    words.append(current_word)
                current_word = tok
                current_infos = [inf]

        if current_word:
            if self.strategy == "average":
                word_infos.append(sum(current_infos) / len(current_infos))
            elif self.strategy == "first_token":
                word_infos.append(current_infos[0])
            words.append(current_word)

        return words, word_infos

    def compress(
        self,
        messages: List[Dict[str, str]],
        tokenizer=None
    ):
        compressed_messages = []
        for mes in messages:
            text = mes["content"].strip()
            tokens, infos = self._compute_token_info(text)
            words, word_infos = self._aggregate_word_info(text, tokens, infos)

            keep_k = max(1, int(len(words) * self.compress_rate))
            top_indices = sorted(
                range(len(word_infos)),
                key=lambda i: word_infos[i],
                reverse=True
            )[:keep_k]
            top_indices = sorted(top_indices)
            compressed_text = " ".join([words[i] for i in top_indices])

            mes["content"] = compressed_text.strip()
            compressed_messages.append(mes)

        return compressed_messages
