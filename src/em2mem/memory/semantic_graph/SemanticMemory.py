import io
import json
import logging
import os
import re
import hashlib
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import igraph as ig

from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


def _md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _safe_cache_tag(x: Optional[str]) -> str:
    if not x:
        return "default"
    x = str(x).strip()
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x or "default"


@dataclass
class SemanticTripleEntry:
    id: str
    subject: str
    predicate: str
    object: str
    timestamp: int

    subject_type: str = ""
    object_type: str = ""
    semantic_summary: str = ""
    support_count: int = 1
    support_days: List[str] = field(default_factory=list)
    support_scales: List[str] = field(default_factory=list)
    confidence: float = 0.5
    habit_strength: str = "low"
    raw_support_count: int = 1
    evidence_event_ids: List[str] = field(default_factory=list)
    provenance_root_ids: List[str] = field(default_factory=list)
    source_doc_ids: List[str] = field(default_factory=list)

    @property
    def triple(self) -> List[str]:
        return [self.subject, self.predicate, self.object]

    @property
    def text(self) -> str:
        if self.semantic_summary:
            return f"{self.subject} {self.predicate} {self.object}. {self.semantic_summary}"
        return " ".join(self.triple)

    def to_display_str(self) -> str:
        base = f"({self.subject}, {self.predicate}, {self.object})"
        extra = f"[support={self.support_count}, confidence={self.confidence:.2f}, habit={self.habit_strength}]"
        if self.semantic_summary:
            return f"{self.semantic_summary} {base} {extra}"
        return f"{base} {extra}"


def _transform_timestamp(ts_str: str) -> str:
    ts_str = str(ts_str)
    if len(ts_str) < 7:
        return ts_str
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


class SemanticMemory:
    def __init__(self, embedding_model: EmbeddingModel, cache_tag: Optional[str] = None):
        self.embedding_model = embedding_model
        self.cache_tag = _safe_cache_tag(cache_tag)
        self.dense_cache_dir = os.path.join(".cache", "semantic_dense", self.cache_tag)

        self.triple_id_to_entry: Dict[str, SemanticTripleEntry] = {}
        self.timestamp_to_triples: Dict[int, List[SemanticTripleEntry]] = {}
        self.available_timestamps: List[int] = []
        self.triple_id_to_embedding: Dict[str, np.ndarray] = {}
        self.dense_cache_built: bool = False

        # "snapshot": use closest timestamp bucket
        # "flat_facts": cumulative all buckets <= until_time
        self.index_mode: str = "snapshot"

        self.indexed_entries: List[SemanticTripleEntry] = []
        self.indexed_time: int = 0
        self.indexed_timestamp: int = 0

        self.graph: Optional[ig.Graph] = None
        self.embeddings: Optional[torch.Tensor] = None
        self.triple_to_entities: Dict[str, Tuple[str, str]] = {}
        self.entity_to_vertex: Dict[str, int] = {}

    # -----------------------------------------------------
    # loading
    # -----------------------------------------------------

    def load_triples_from_file(self, file_path: str) -> None:
        if str(file_path).lower().endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zf:
                json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
                if not json_names:
                    raise ValueError(f"No JSON file found inside semantic zip: {file_path}")
                target_name = json_names[0]
                with zf.open(target_name) as f:
                    data = json.load(io.TextIOWrapper(f, encoding="utf-8"))
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        self.load_triples_from_data(data)

    def _safe_int(self, x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def _safe_float(self, x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _append_entry(self, timestamp: int, entry: SemanticTripleEntry, bucket: List[SemanticTripleEntry]) -> None:
        self.triple_id_to_entry[entry.id] = entry
        bucket.append(entry)

    def _normalize_top_level_facts(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Convert a flat semantic memory file like:
            {"facts": [...], "timeline": ...}
        into:
            {timestamp_str: {"facts": [...]}, ...}
        """
        self.index_mode = "flat_facts"
        normalized_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"facts": []})

        facts_obj = data.get("facts", [])
        if isinstance(facts_obj, list):
            facts_iter = facts_obj
        elif isinstance(facts_obj, dict):
            facts_iter = list(facts_obj.values())
        else:
            facts_iter = []

        logger.info(
            "Normalizing top-level semantic facts: type=%s count=%s",
            type(facts_obj).__name__,
            len(facts_iter),
        )

        for fact in facts_iter:
            if not isinstance(fact, dict):
                continue

            triple = fact.get("triple", []) or []
            if not isinstance(triple, list) or len(triple) < 3:
                continue

            last_seen = fact.get("last_seen") or fact.get("first_seen") or {}
            if not isinstance(last_seen, dict):
                last_seen = {}

            date = str(last_seen.get("date", "DAY9"))
            end_time = str(last_seen.get("end_time", last_seen.get("start_time", "00000000"))).zfill(8)
            m = re.search(r"(\d+)", date)
            day = m.group(1) if m else "9"
            ts_key = f"{int(day)}{end_time}"

            normalized_data[ts_key]["facts"].append({
                "fact_id": fact.get("fact_id"),
                "head": triple[0],
                "relation": triple[1],
                "tail": triple[2],
                "head_type": fact.get("head_type", ""),
                "tail_type": fact.get("tail_type", ""),
                "semantic_summary": fact.get("semantic_summary", ""),
                "support_count": fact.get("support_count", 1),
                "support_days": fact.get("support_days", []),
                "support_scales": fact.get("support_scales", []),
                "confidence": fact.get(
                    "confidence",
                    min(0.95, 0.45 + 0.04 * min(self._safe_int(fact.get("support_count", 1), 1), 8))
                ),
                "habit_strength": fact.get("habit_strength", "low"),
                "raw_support_count": fact.get("raw_support_count", fact.get("support_count", 1)),
                "evidence_event_ids": fact.get("support_docs", []) or fact.get("evidence_event_ids", []),
                "provenance_root_ids": fact.get("provenance_root_ids", []) or fact.get("support_docs", []),
                "source_doc_ids": fact.get("support_docs", []) or fact.get("source_doc_ids", []),
            })

        return dict(normalized_data)

    def _normalize_old_semantic_extraction(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Convert old extraction results:
            {"semantic_triples": {...}, "episodic_evidence": {...}}
        into timestamp buckets.
        """
        semantic_triples = data.get("semantic_triples", {}) or {}
        episodic_evidence = data.get("episodic_evidence", {}) or {}
        normalized_data: Dict[str, Dict[str, Any]] = {}

        for timestamp_str, triples in semantic_triples.items():
            normalized_data[str(timestamp_str)] = {
                "consolidated_semantic_triples": triples or [],
                "consolidated_episodic_evidence": episodic_evidence.get(str(timestamp_str), [])
                if isinstance(episodic_evidence, dict) else [],
            }
        return normalized_data

    def load_triples_from_data(self, data: Dict[str, Any]) -> None:
        self.triple_id_to_entry.clear()
        self.timestamp_to_triples.clear()
        self.available_timestamps = []
        self.triple_id_to_embedding = {}
        self.dense_cache_built = False
        self.index_mode = "snapshot"

        if not isinstance(data, dict):
            logger.warning("Semantic data is not a dict; skipping load")
            return

        logger.info("Semantic load top-level keys: %s", list(data.keys())[:10])

        # Case 1: top-level semantic memory file with facts + timeline
        if "facts" in data:
            data = self._normalize_top_level_facts(data)

        # Case 2: old extraction-style format
        elif "semantic_triples" in data and isinstance(data.get("semantic_triples"), dict):
            data = self._normalize_old_semantic_extraction(data)

        for timestamp_str, content in data.items():
            try:
                timestamp = int(timestamp_str)
            except Exception:
                logger.warning(f"Skipping invalid semantic timestamp key: {timestamp_str}")
                continue

            timestamp_entries: List[SemanticTripleEntry] = []

            if isinstance(content, dict) and "facts" in content:
                facts = content.get("facts", []) or []
                for idx, fact in enumerate(facts):
                    if not isinstance(fact, dict):
                        continue

                    subject = str(fact.get("head", "")).strip()
                    predicate = str(fact.get("relation", "")).strip()
                    obj = str(fact.get("tail", "")).strip()
                    if not subject or not predicate or not obj:
                        continue

                    fact_id = str(fact.get("fact_id", "")).strip() or f"semantic_{timestamp}_{idx}"
                    evidence_event_ids = list(fact.get("evidence_event_ids", []) or [])
                    source_doc_ids = list(fact.get("source_doc_ids", []) or [])
                    provenance_root_ids = list(fact.get("provenance_root_ids", []) or [])

                    if not evidence_event_ids and source_doc_ids:
                        evidence_event_ids = list(source_doc_ids)
                    if not provenance_root_ids and source_doc_ids:
                        provenance_root_ids = list(source_doc_ids)

                    entry = SemanticTripleEntry(
                        id=fact_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        timestamp=timestamp,
                        subject_type=str(fact.get("head_type", "")).strip(),
                        object_type=str(fact.get("tail_type", "")).strip(),
                        semantic_summary=str(fact.get("semantic_summary", "")).strip(),
                        support_count=self._safe_int(fact.get("support_count", 1), 1),
                        support_days=list(fact.get("support_days", []) or []),
                        support_scales=list(fact.get("support_scales", []) or []),
                        confidence=self._safe_float(fact.get("confidence", 0.5), 0.5),
                        habit_strength=str(fact.get("habit_strength", "low")).strip(),
                        raw_support_count=self._safe_int(
                            fact.get("raw_support_count", fact.get("support_count", 1)), 1
                        ),
                        evidence_event_ids=evidence_event_ids,
                        provenance_root_ids=provenance_root_ids,
                        source_doc_ids=source_doc_ids,
                    )
                    self._append_entry(timestamp, entry, timestamp_entries)

            elif isinstance(content, dict) and "consolidated_semantic_triples" in content:
                triples = content.get("consolidated_semantic_triples", []) or []
                raw_support = content.get("consolidated_episodic_evidence", []) or []

                flattened_support: List[str] = []
                for item in raw_support:
                    if isinstance(item, list):
                        flattened_support.extend([str(x) for x in item if str(x).strip()])
                    elif str(item).strip():
                        flattened_support.append(str(item))

                for idx, triple in enumerate(triples):
                    if not isinstance(triple, list) or len(triple) < 3:
                        continue
                    entry_id = f"semantic_{timestamp}_{idx}"
                    entry = SemanticTripleEntry(
                        id=entry_id,
                        subject=str(triple[0]).strip(),
                        predicate=str(triple[1]).strip(),
                        object=str(triple[2]).strip(),
                        timestamp=timestamp,
                        semantic_summary="",
                        support_count=max(1, len(flattened_support)) if flattened_support else 1,
                        confidence=0.5,
                        habit_strength="low",
                        raw_support_count=max(1, len(flattened_support)) if flattened_support else 1,
                        evidence_event_ids=[],
                        provenance_root_ids=flattened_support,
                        source_doc_ids=[],
                    )
                    if entry.subject and entry.predicate and entry.object:
                        self._append_entry(timestamp, entry, timestamp_entries)

            elif isinstance(content, list):
                triples = content
                for idx, triple in enumerate(triples):
                    if not isinstance(triple, list) or len(triple) < 3:
                        continue
                    entry_id = f"semantic_{timestamp}_{idx}"
                    entry = SemanticTripleEntry(
                        id=entry_id,
                        subject=str(triple[0]).strip(),
                        predicate=str(triple[1]).strip(),
                        object=str(triple[2]).strip(),
                        timestamp=timestamp,
                        semantic_summary="",
                        support_count=1,
                        confidence=0.5,
                        habit_strength="low",
                        raw_support_count=1,
                        evidence_event_ids=[],
                        provenance_root_ids=[],
                        source_doc_ids=[],
                    )
                    if entry.subject and entry.predicate and entry.object:
                        self._append_entry(timestamp, entry, timestamp_entries)

            if timestamp_entries:
                self.timestamp_to_triples[timestamp] = timestamp_entries

        self.available_timestamps = sorted(self.timestamp_to_triples.keys())
        logger.info(
            "Loaded semantic facts across %d timestamps (mode=%s)",
            len(self.available_timestamps),
            self.index_mode,
        )

    # -----------------------------------------------------
    # dense cache
    # -----------------------------------------------------

    def _all_entries(self) -> List[SemanticTripleEntry]:
        entries: List[SemanticTripleEntry] = []
        for ts in self.available_timestamps:
            entries.extend(self.timestamp_to_triples.get(ts, []))
        return entries

    def _build_dense_cache_meta(
        self,
        entries: List[SemanticTripleEntry],
        texts: List[str],
    ) -> Dict[str, Any]:
        return {
            "entry_ids": [entry.id for entry in entries],
            "timestamps": [entry.timestamp for entry in entries],
            "text_hashes": [_md5_text(text) for text in texts],
        }

    def _try_load_dense_cache(
        self,
        entries: List[SemanticTripleEntry],
        texts: List[str],
    ) -> bool:
        meta_path = os.path.join(self.dense_cache_dir, "meta.json")
        emb_path = os.path.join(self.dense_cache_dir, "fact_embeddings.npy")

        if not (os.path.exists(meta_path) and os.path.exists(emb_path)):
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected = self._build_dense_cache_meta(entries, texts)
            if meta != expected:
                return False

            embeddings = np.load(emb_path).astype(np.float32)
            if embeddings.shape[0] != len(entries):
                return False

            self.triple_id_to_embedding = {
                entry.id: embeddings[idx]
                for idx, entry in enumerate(entries)
            }
            self.dense_cache_built = True
            logger.info("Loaded semantic dense cache: %d facts", embeddings.shape[0])
            return True
        except Exception as e:
            logger.warning("Failed to load semantic dense cache: %s", e)
            return False

    def _save_dense_cache(
        self,
        entries: List[SemanticTripleEntry],
        texts: List[str],
        embeddings: np.ndarray,
    ) -> None:
        os.makedirs(self.dense_cache_dir, exist_ok=True)
        meta = self._build_dense_cache_meta(entries, texts)
        with open(os.path.join(self.dense_cache_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        np.save(os.path.join(self.dense_cache_dir, "fact_embeddings.npy"), embeddings.astype(np.float32))

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        batch_size = max(1, int(os.environ.get("SEMANTIC_DENSE_BATCH_SIZE", 256)))
        chunks: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.embedding_model.encode_text(batch_texts, batch_size=len(batch_texts))
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings[None, :]
            chunks.append(embeddings)

        return np.concatenate(chunks, axis=0).astype(np.float32)

    def build_dense_cache(self, force_rebuild: bool = False) -> None:
        if self.dense_cache_built and not force_rebuild:
            return

        entries = self._all_entries()
        if not entries:
            logger.warning("No semantic facts loaded; semantic dense cache skipped")
            self.triple_id_to_embedding = {}
            self.dense_cache_built = True
            return

        texts = [entry.text for entry in entries]
        if not force_rebuild and self._try_load_dense_cache(entries, texts):
            return

        logger.info("Building semantic dense cache: %d facts", len(entries))
        embeddings = self._encode_texts(texts)
        self.triple_id_to_embedding = {
            entry.id: embeddings[idx]
            for idx, entry in enumerate(entries)
        }
        self._save_dense_cache(entries, texts, embeddings)
        self.dense_cache_built = True
        logger.info("Built semantic dense cache: %d facts", embeddings.shape[0])

    def _get_entry_embeddings(self, entries: List[SemanticTripleEntry]) -> np.ndarray:
        self.build_dense_cache(force_rebuild=False)

        missing = [entry for entry in entries if entry.id not in self.triple_id_to_embedding]
        if missing:
            logger.warning("Semantic dense cache missing %d facts; encoding fallback batch", len(missing))
            embeddings = self._encode_texts([entry.text for entry in missing])
            for idx, entry in enumerate(missing):
                self.triple_id_to_embedding[entry.id] = embeddings[idx]

        return np.asarray(
            [self.triple_id_to_embedding[entry.id] for entry in entries],
            dtype=np.float32,
        )

    # -----------------------------------------------------
    # indexing
    # -----------------------------------------------------

    def index(self, until_time: int) -> None:
        closest_timestamp = None
        for ts in reversed(self.available_timestamps):
            if ts <= until_time:
                closest_timestamp = ts
                break

        if closest_timestamp is None:
            logger.debug(f"No semantic timestamp found up to {until_time}")
            self.indexed_entries = []
            self.indexed_time = until_time
            self.indexed_timestamp = 0
            return

        if self.index_mode == "flat_facts":
            if self.indexed_time == until_time and self.indexed_entries:
                logger.debug(f"Already indexed cumulative semantic facts up to {until_time}, skipping")
                return

            entries_to_index: List[SemanticTripleEntry] = []
            for ts in self.available_timestamps:
                if ts > until_time:
                    break
                entries_to_index.extend(self.timestamp_to_triples.get(ts, []))
        else:
            if self.indexed_timestamp == closest_timestamp:
                logger.debug(f"Already indexed semantic timestamp {closest_timestamp}, skipping")
                return
            entries_to_index = self.timestamp_to_triples.get(closest_timestamp, [])

        if not entries_to_index:
            self.indexed_entries = []
            self.indexed_time = until_time
            self.indexed_timestamp = closest_timestamp
            logger.debug("No semantic entries available after indexing filter")
            return

        self.triple_to_entities = {}
        entity_set: Set[str] = set()
        for entry in entries_to_index:
            subj, obj = entry.subject, entry.object
            if subj:
                entity_set.add(subj)
            if obj:
                entity_set.add(obj)
            self.triple_to_entities[entry.id] = (subj, obj)

        entity_list = sorted(entity_set)
        self.entity_to_vertex = {entity: i for i, entity in enumerate(entity_list)}
        self.graph = ig.Graph()
        self.graph.add_vertices(entity_list)

        pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
        for entry in entries_to_index:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            if not subj or not obj or subj == obj:
                continue
            a, b = sorted([subj, obj])
            weight = float(entry.confidence) * (1.0 + 0.10 * min(entry.support_count, 5))
            pair_weights[(a, b)] += weight

        if pair_weights:
            edges = []
            weights = []
            for (a, b), w in pair_weights.items():
                if a not in self.entity_to_vertex or b not in self.entity_to_vertex:
                    continue
                edges.append((self.entity_to_vertex[a], self.entity_to_vertex[b]))
                weights.append(w)
            if edges:
                self.graph.add_edges(edges)
                self.graph.es["weight"] = weights

        all_embeddings = self._get_entry_embeddings(entries_to_index)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = torch.tensor(all_embeddings, dtype=torch.float32, device=device)

        self.indexed_entries = entries_to_index
        self.indexed_time = until_time
        self.indexed_timestamp = closest_timestamp

        if self.index_mode == "flat_facts":
            logger.info(
                "Indexed %d cumulative semantic facts up to timestamp %s (query time: %s)",
                len(entries_to_index),
                closest_timestamp,
                until_time,
            )
        else:
            logger.info(
                "Indexed %d semantic facts from timestamp %s (query time: %s)",
                len(entries_to_index),
                closest_timestamp,
                until_time,
            )

    # -----------------------------------------------------
    # retrieval
    # -----------------------------------------------------

    def _min_max_norm(self, values: List[float]) -> List[float]:
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if abs(vmax - vmin) < 1e-8:
            return [1.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

    def retrieve(self, query: str, top_k: int = 10, as_context: bool = True) -> Union[List[SemanticTripleEntry], str]:
        if not self.indexed_entries or self.embeddings is None:
            logger.warning("No semantic facts indexed. Call index(until_time) before retrieve().")
            return "" if as_context else []

        device = self.embeddings.device
        query_embedding = self.embedding_model.encode_text(query)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device)

        similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        sim_values = similarities.detach().cpu().tolist()

        num_available = len(self.indexed_entries)
        top_seed_k = min(max(top_k * 2, 8), num_available)
        _, top_pos_indices = torch.topk(similarities, top_seed_k)
        top_seed_entries = [self.indexed_entries[pos] for pos in top_pos_indices.cpu().tolist()]

        if self.graph is None or self.graph.vcount() == 0 or self.graph.ecount() == 0:
            sorted_entries = sorted(zip(self.indexed_entries, sim_values), key=lambda x: x[1], reverse=True)[:top_k]
            result = [entry for entry, _ in sorted_entries]
            return self.retrieve_triples_as_str(result) if as_context else result

        personalization_entities: Set[str] = set()
        for entry in top_seed_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            if subj:
                personalization_entities.add(subj)
            if obj:
                personalization_entities.add(obj)

        if not personalization_entities:
            result = top_seed_entries[:top_k]
            return self.retrieve_triples_as_str(result) if as_context else result

        entity_list = [self.graph.vs[i]["name"] for i in range(self.graph.vcount())]
        reset = [
            1.0 / len(personalization_entities) if entity in personalization_entities else 0.0
            for entity in entity_list
        ]

        try:
            ppr_scores = self.graph.personalized_pagerank(
                directed=False,
                damping=0.85,
                reset=reset,
                weights=self.graph.es["weight"] if "weight" in self.graph.es.attributes() else None,
                implementation="prpack",
            )
        except Exception:
            ppr_scores = self.graph.personalized_pagerank(
                directed=False,
                damping=0.85,
                reset=reset,
                implementation="prpack",
            )

        entity_to_ppr = {entity_list[i]: float(ppr_scores[i]) for i in range(len(entity_list))}

        fact_ppr_scores = []
        fact_conf_scores = []
        for entry in self.indexed_entries:
            subj, obj = self.triple_to_entities.get(entry.id, ("", ""))
            ppr_score = entity_to_ppr.get(subj, 0.0) + entity_to_ppr.get(obj, 0.0)
            fact_ppr_scores.append(ppr_score)
            fact_conf_scores.append(float(entry.confidence))

        sim_norm = self._min_max_norm(sim_values)
        ppr_norm = self._min_max_norm(fact_ppr_scores)
        conf_norm = self._min_max_norm(fact_conf_scores)

        combined = []
        for idx, entry in enumerate(self.indexed_entries):
            score = 0.55 * sim_norm[idx] + 0.30 * ppr_norm[idx] + 0.15 * conf_norm[idx]
            combined.append((entry, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        result = [entry for entry, _ in combined[:top_k]]
        return self.retrieve_triples_as_str(result) if as_context else result

    def retrieve_triples_as_str(self, entries: List[SemanticTripleEntry]) -> str:
        return "\n".join(entry.to_display_str() for entry in entries)

    # -----------------------------------------------------
    # support / packets
    # -----------------------------------------------------

    def get_support_event_ids(self, entry: SemanticTripleEntry, limit: int = 2) -> List[str]:
        ids = list(getattr(entry, "evidence_event_ids", []) or [])
        if not ids and hasattr(entry, "support_docs"):
            ids = list(getattr(entry, "support_docs", []) or [])
        if not ids:
            ids = list(getattr(entry, "source_doc_ids", []) or [])
        if not ids:
            ids = list(getattr(entry, "provenance_root_ids", []) or [])

        deduped: List[str] = []
        seen = set()
        for x in ids:
            x = str(x)
            if x and x not in seen:
                seen.add(x)
                deduped.append(x)
            if len(deduped) >= limit:
                break
        return deduped

    def build_packet_text(self, entry: SemanticTripleEntry, support_event_limit: int = 2) -> str:
        lines = [f"Semantic Fact: {entry.to_display_str()}"]
        support_ids = self.get_support_event_ids(entry, limit=support_event_limit)
        if support_ids:
            lines.append("Support Event IDs: " + ", ".join(support_ids))
        if entry.support_days:
            lines.append("Support Days: " + ", ".join(entry.support_days[:5]))
        if entry.support_scales:
            lines.append("Support Scales: " + ", ".join(entry.support_scales[:5]))
        return "\n".join(lines)

    def retrieve_packets(
        self,
        query: str,
        top_k: int = 5,
        support_event_limit: int = 2,
    ) -> List[Dict[str, Any]]:
        entries = self.retrieve(query=query, top_k=top_k, as_context=False)
        packets: List[Dict[str, Any]] = []
        for entry in entries:
            packets.append({
                "packet_type": "semantic",
                "fact_id": entry.id,
                "text": self.build_packet_text(entry, support_event_limit=support_event_limit),
                "support_event_ids": self.get_support_event_ids(entry, limit=support_event_limit),
                "confidence": float(entry.confidence),
                "support_count": int(entry.support_count),
            })
        return packets

    # -----------------------------------------------------
    # misc
    # -----------------------------------------------------

    def cleanup(self) -> None:
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_index(self) -> None:
        self.graph = None
        self.embeddings = None
        self.indexed_entries = []
        self.indexed_time = 0
        self.indexed_timestamp = 0
        self.triple_to_entities = {}
        self.entity_to_vertex = {}
        logger.info("Semantic index reset - graph and embeddings cleared")

    def get_indexed_time(self) -> str:
        return _transform_timestamp(str(self.indexed_time))

    def get_indexed_timestamp(self) -> str:
        return _transform_timestamp(str(self.indexed_timestamp)) if self.indexed_timestamp > 0 else "Not indexed"

    def get_triple_by_id(self, triple_id: str) -> Optional[SemanticTripleEntry]:
        return self.triple_id_to_entry.get(triple_id)

    def get_indexed_count(self) -> int:
        return len(self.indexed_entries)
