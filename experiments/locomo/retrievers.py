from openai import OpenAI
import logging
import os
import time
import sqlite3
import pickle
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
import spacy
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig

SPACY_AVAILABLE = True
logger = logging.getLogger(__name__)

class QdrantEntryLoader:
    def __init__(self, qdrant_path: str):
        self.qdrant_path = qdrant_path

    def _get_qdrant(self, collection_name: str):
        if Qdrant is None or QdrantConfig is None:
            return None
        cfg = QdrantConfig(collection_name=collection_name, path=self.qdrant_path, embedding_model_dims=384, on_disk=True)
        return Qdrant(cfg)

    def load_entries(self, collection_name: str, with_vectors: bool = False) -> List[Dict[str, Any]]:
        q = self._get_qdrant(collection_name)
        logger.info(f"Loading entries from collection: {collection_name} (with_vectors={with_vectors})")
        points = []
        if q is not None:
            try:
                points = q.get_all(with_vectors=with_vectors, with_payload=True)
            except Exception as e:
                logger.warning(f"Qdrant API get_all failed: {e}")

        if not points:
            logger.info("Trying SQLite fallback for collection: %s" % collection_name)
            points = self._fallback_sqlite_read(collection_name, with_vectors=with_vectors)

        logger.info(f"Loaded {len(points)} entries")
        return points

    def _fallback_sqlite_read(self, collection_name: str, with_vectors: bool = True) -> List[Dict[str, Any]]:
        storage_sqlite = os.path.join(self.qdrant_path, collection_name, 'collection', collection_name, 'storage.sqlite')
        if not os.path.exists(storage_sqlite):
            logger.error(f"SQLite file not found: {storage_sqlite}")
            return []

        points = []
        try:
            conn = sqlite3.connect(storage_sqlite)
            cur = conn.execute("SELECT id, point FROM points")
            for row in cur:
                pid, blob = row
                try:
                    obj = pickle.loads(blob)
                except Exception:
                    try:
                        obj = pickle.loads(blob, fix_imports=True)
                    except Exception:
                        continue

                item = {}
                if isinstance(obj, dict):
                    item = obj
                else:
                    if hasattr(obj, '__dict__'):
                        item.update(getattr(obj, '__dict__', {}))
                    for attr in ('id', 'payload', 'vector'):
                        if hasattr(obj, attr) and attr not in item:
                            item[attr] = getattr(obj, attr)

                if 'id' not in item and 'point' in item:
                    item['id'] = item['point'].get('id')
                if 'payload' not in item and 'point' in item:
                    item['payload'] = item['point'].get('payload', {})

                if not with_vectors and 'vector' in item:
                    item.pop('vector', None)

                points.append(item)
            conn.close()
        except Exception as e:
            logger.error(f"Fallback read failed: {e}")

        logger.info(f"Fallback loaded {len(points)} points from SQLite")
        return points

class VectorRetriever:
    def __init__(self, embedder):
        self.embedder = embedder

    def retrieve(self, entries: List[Dict], query_text: str, limit: int = 20) -> List[Dict]:
        try:
            query_vector = self.embedder.embed(query_text)
        except Exception as e:
            logger.error(f"Embedding query failed: {e}")
            return []

        results = []
        skipped = 0
        for entry in entries:
            vec = entry.get('vector')
            if vec is None:
                skipped += 1
                continue
            score = self._cosine_similarity(query_vector, vec)
            results.append({'id': str(entry.get('id')), 'score': float(score), 'payload': entry.get('payload', {}), 'source': 'vector'})

        if skipped:
            logger.warning(f"Skipped {skipped} entries without vectors")

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

def format_related_memories(related: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for item in related:
        payload = item.get('payload', {}) if isinstance(item, dict) else {}
        if not payload and isinstance(item, str):
            out.append(item)
            continue
            
        time_stamp = payload.get('time_stamp') or item.get('time_stamp') or ''
        weekday = payload.get('weekday') or item.get('weekday') or ''
        memory = payload.get('memory') or payload.get('original_memory') or payload.get('compressed_memory') or item.get('memory') or ''
        dt = datetime.fromisoformat(time_stamp.replace('Z', '+00:00'))
        formatted_date = dt.strftime("%d %B %Y")  
        formatted = f"[Memory recorded on: {formatted_date}, {weekday}]\n{memory}"
    
        out.append(formatted.strip())
    
    return "\n\n".join(out)  