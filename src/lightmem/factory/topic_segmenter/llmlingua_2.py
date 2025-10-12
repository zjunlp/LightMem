from typing import Dict, Optional, List, Any
import torch, numpy as np
from transformers import AutoTokenizer, AutoModel

class LlmLingua2Segmenter:
    def __init__(self, config: Optional[Dict] = None, shared: bool = False, compressor=None):
        self.config = config

        if shared is False:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.config["model_name"],
                device_map=self.config.get("device_map", None),
                torch_dtype=self.config.get("torch_dtype", None),
                **self.config.get("model_config", {})
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
            self.buffer_len = self.config.get("buffer_len", 512)
        elif compressor is not None:
            self.model = compressor.inner_compressor.model
            self.tokenizer = compressor.inner_compressor.tokenizer
            self.buffer_len = getattr(self.model.config, "max_position_embeddings", 512)

        self.layers = self.config.get("layers", [8, 9, 10, 11])

    def sentence_level_attention(self, buffer_texts: List[str]):
        model, tokenizer = self.model, self.tokenizer
        device = next(model.parameters()).device

        cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.convert_tokens_to_ids('[CLS]')
        sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.convert_tokens_to_ids('[SEP]')

        per_sent_tokens = [tokenizer.encode(s, add_special_tokens=False) for s in buffer_texts]

        input_ids = [cls_id]
        spans = []
        cur = 1
        for ids in per_sent_tokens:
            start = cur
            input_ids.extend(ids)
            cur += len(ids)
            end = cur
            spans.append((start, end))
        input_ids.append(sep_id)
        seq_len = len(input_ids)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor, device=device)

        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=attention_mask, output_attentions=True, return_dict=True)
            attentions = outputs.attentions

        selected = [attentions[i] for i in self.layers]
        att_stack = torch.stack(selected, dim=0)
        att_mean = att_stack.mean(dim=(0, 2))[0].cpu().numpy()

        k = 3
        valid = np.ones(seq_len, dtype=bool)
        if seq_len >= 2 * k:
            valid[:k] = False
            valid[-k:] = False
        else:
            valid[:] = True
            if seq_len > 0:
                cut = max(0, seq_len // 10)
                valid[:cut] = False
                valid[-cut:] = False

        n = len(buffer_texts)
        M = np.zeros((n, n), dtype=float)

        for i in range(n):
            i_start, i_end = spans[i]
            i_pos = np.arange(i_start, i_end)
            if i_pos.size == 0: 
                continue
            i_pos = i_pos[valid[i_pos]]
            if i_pos.size == 0:
                continue

            row_vals = []
            for j in range(i):
                j_start, j_end = spans[j]
                j_pos = np.arange(j_start, j_end)
                if j_pos.size == 0:
                    row_vals.append(0.0)
                    continue
                j_pos = j_pos[valid[j_pos]]
                if j_pos.size == 0:
                    row_vals.append(0.0)
                    continue

                sub = att_mean[np.ix_(i_pos, j_pos)]
                per_token_sum = sub.sum(axis=1)
                mean_att = float(per_token_sum.mean())
                row_vals.append(mean_att)

            if row_vals:
                row_vals = np.array(row_vals, dtype=float)
                s = row_vals.sum()
                if s > 0:
                    row_vals = row_vals / s
                M[i, :i] = row_vals

        return M

    def propose_cut(self, buffer_texts: List[str]) -> Dict[str, Any]:
        n = len(buffer_texts)
        if n == 0:
            return {"boundaries": [0], "cut_index": 0}

        M = self.sentence_level_attention(buffer_texts)
        outer = [M[i, i-1] for i in range(1, n)]

        boundaries = []
        for k in range(1, len(outer)-1):
            if outer[k-1] < outer[k] > outer[k+1]:
                boundaries.append(k)

        if not boundaries:
            boundaries = sorted(set(b for b in boundaries if 0 <= b <= n))

        return boundaries
