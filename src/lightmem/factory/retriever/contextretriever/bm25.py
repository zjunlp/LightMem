import logging
import os
import pickle
import re
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from lightmem.configs.retriever.bm25 import BM25Config

logger = logging.getLogger(__name__)


class BM25:
    def __init__(self, config: Optional[BM25Config] = None):
        if config is None:
            self.config = BM25Config()
        elif isinstance(config, dict):
            self.config = BM25Config(**config)
        else:
            self.config = config

        self.documents: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None

        if self.config.on_disk and os.path.exists(self.config.index_path):
            try:
                self._load_index()
            except Exception as exc:
                logger.warning("Failed to load BM25 index from %s: %s", self.config.index_path, exc)
        else:
            self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build_index(self) -> None:
        if not self.documents:
            self.bm25 = None
        else:
            tokenized_corpus = [self._tokenize(doc["text"]) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus, k1=self.config.k1, b=self.config.b)
        if self.config.on_disk:
            self._save_index()

    def _save_index(self) -> None:
        index_dir = os.path.dirname(self.config.index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
        with open(self.config.index_path, "wb") as f:
            pickle.dump({"documents": self.documents, "bm25": self.bm25}, f)

    def _load_index(self) -> None:
        with open(self.config.index_path, "rb") as f:
            data = pickle.load(f)
        self.documents = data.get("documents", [])
        self.bm25 = data.get("bm25")
        if self.documents and self.bm25 is None:
            self._build_index()

    def _filter_match(self, payload: Dict, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
        if payload is None:
            return False
        for key, value in filters.items():
            if payload.get(key) != value:
                return False
        return True

    def insert(self, docs: List[str], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> None:
        if payloads is not None and len(payloads) != len(docs):
            raise ValueError("Length of payloads must match length of docs.")
        if ids is not None and len(ids) != len(docs):
            raise ValueError("Length of ids must match length of docs.")

        for idx, text in enumerate(docs):
            doc_id = ids[idx] if ids is not None else str(len(self.documents) + idx)
            payload = payloads[idx] if payloads is not None else {}

            existing_idx = next((i for i, d in enumerate(self.documents) if d["id"] == doc_id), None)
            new_doc = {"id": doc_id, "text": text, "payload": payload}
            if existing_idx is not None:
                self.documents[existing_idx] = new_doc
            else:
                self.documents.append(new_doc)

        self._build_index()

    def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        if not self.documents or self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = []
        for idx, doc in enumerate(self.documents):
            if not self._filter_match(doc.get("payload", {}), filters):
                continue
            ranked.append((doc, scores[idx]))

        ranked.sort(key=lambda x: x[1], reverse=True)
        top_ranked = ranked[:limit]

        results = []
        for doc, score in top_ranked:
            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": float(score),
                    "payload": doc.get("payload", {}),
                }
            )
        return results

    def delete(self, vector_id: str) -> None:
        before_count = len(self.documents)
        self.documents = [doc for doc in self.documents if doc["id"] != vector_id]
        if len(self.documents) != before_count:
            self._build_index()

    def update(self, vector_id: str, text: Optional[str] = None, payload: Optional[Dict] = None) -> None:
        for doc in self.documents:
            if doc["id"] == vector_id:
                if text is not None:
                    doc["text"] = text
                if payload is not None:
                    doc["payload"] = payload
                self._build_index()
                return

    def list(self, filters: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
        matched_docs = []
        for doc in self.documents:
            if self._filter_match(doc.get("payload", {}), filters):
                matched_docs.append(
                    {"id": doc["id"], "text": doc["text"], "payload": doc.get("payload", {})}
                )
                if len(matched_docs) >= limit:
                    break
        return matched_docs

    def reset(self) -> None:
        self.documents = []
        self.bm25 = None
        if self.config.on_disk and os.path.exists(self.config.index_path):
            os.remove(self.config.index_path)

    def exists(self, vector_id: str) -> bool:
        return any(doc["id"] == vector_id for doc in self.documents)

    def get(self, vector_id: str) -> Optional[Dict]:
        for doc in self.documents:
            if doc["id"] == vector_id:
                return {"id": doc["id"], "text": doc["text"], "payload": doc.get("payload", {})}
        return None

    def get_all(self) -> List[Dict]:
        return [{"id": doc["id"], "text": doc["text"], "payload": doc.get("payload", {})} for doc in self.documents]
