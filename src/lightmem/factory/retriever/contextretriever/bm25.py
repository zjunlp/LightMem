import math
import json
from typing import List, Dict, Optional, Tuple


class BM25:
    """
    A minimal BM25 implementation for context-based retrieval over plaintext memory entries.

    Corpus source:
        - Defaults to reading `memory_entries.json` produced by offline/online updates
          when `index_strategy` includes "context".

    Returned format:
        - Aligns with embedding retriever `search(..., return_full=True)` to simplify integration:
          [{"id": <str>, "score": <float>, "payload": {time_stamp, weekday, memory}}]
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.corpus_path = self.config.get("corpus_path", "memory_entries.json")
        self.k1 = float(self.config.get("k1", 1.5))
        self.b = float(self.config.get("b", 0.75))
        self._docs: List[Dict] = []
        self._doc_tokens: List[List[str]] = []
        self._avgdl: float = 0.0
        self._idf: Dict[str, float] = {}
        self._doc_len: List[int] = []
        self._ensure_index()

    # --- public api ---
    def search(self, query: str, limit: int = 10, filters: Dict = None, return_full: bool = True) -> List[Dict]:
        """
        Search top-k relevant entries using BM25 over plaintext memory entries.

        Args:
            query: natural language query
            limit: number of results
            filters: optional dict (currently ignored in BM25; kept for future use)
            return_full: if True, returns dict with id, score, payload
        """
        if not self._docs:
            return []

        q_tokens = self._tokenize(query)
        scores: List[Tuple[int, float]] = []
        for i, tokens in enumerate(self._doc_tokens):
            score = 0.0
            dl = self._doc_len[i]
            for t in q_tokens:
                tf = tokens.count(t)
                if tf == 0:
                    continue
                idf = self._idf.get(t, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:limit]

        results: List[Dict] = []
        for idx, sc in top:
            doc = self._docs[idx]
            payload = {
                "time_stamp": doc.get("time_stamp", ""),
                "weekday": doc.get("weekday", ""),
                "memory": doc.get("memory", ""),
            }
            results.append({"id": doc.get("id", str(idx)), "score": sc, "payload": payload})
        return results if return_full else [{"id": r["id"], "score": r["score"]} for r in results]

    # --- internal helpers ---
    def _ensure_index(self) -> None:
        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Expect list[MemoryEntry-like dict]
                self._docs = data if isinstance(data, list) else []
        except Exception:
            self._docs = []

        self._doc_tokens = [self._tokenize(d.get("memory", "")) for d in self._docs]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        self._compute_idf()

    def _compute_idf(self) -> None:
        N = len(self._doc_tokens)
        if N == 0:
            self._idf = {}
            return
        df: Dict[str, int] = {}
        for tokens in self._doc_tokens:
            seen = set(tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        # BM25 IDF formula with log
        self._idf = {t: math.log(1 + (N - df_t + 0.5) / (df_t + 0.5)) for t, df_t in df.items()}

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in text.split() if t.strip()]