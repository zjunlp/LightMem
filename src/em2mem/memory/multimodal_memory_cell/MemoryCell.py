import os
import json
import logging
import re
import hashlib
import threading
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.auto import tqdm

from ...llm import LLMModel, PromptTemplateManager
from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "do", "did", "does",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "this", "that", "these", "those", "it", "its"
}


@dataclass
class CaptionEntry:
    id: str
    doc_id: str
    text: str
    start_time: str
    end_time: str
    date: str
    granularity: str
    video_path: Optional[str] = None
    visual_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_int(self) -> Tuple[int, int]:
        day = self.date.replace('DAY', '').replace('Day', '')
        start_ts = int(day + self.start_time.zfill(8))
        end_ts = int(day + self.end_time.zfill(8))
        return start_ts, end_ts

    def to_display_str(self, include_visual_summary: bool = True) -> str:
        start_ts, end_ts = self.timestamp_int
        base = f"[{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}]\n{self.text}"
        if include_visual_summary and self.visual_summary:
            base += f"\nVisual: {self.visual_summary}"
        return base


@dataclass
class GraphEventSidecar:
    event_id: str
    doc_id: str
    granularity: str
    entity_labels: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    triplet_strings: List[str] = field(default_factory=list)
    prev_doc_id: Optional[str] = None
    next_doc_id: Optional[str] = None
    graph_tokens: set = field(default_factory=set)


def _transform_timestamp(ts_str: str) -> str:
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"


def _load_json(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _tokenize(text: str) -> set:
    toks = re.findall(r"[a-zA-Z0-9_/-]+", str(text).lower())
    return {t for t in toks if len(t) > 1 and t not in STOPWORDS}


def _safe_cache_tag(x: Optional[str]) -> str:
    if not x:
        return "default"
    x = str(x).strip()
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x or "default"


def _normalize_phrase(text: str) -> Optional[str]:
    text = str(text).strip().lower()
    if not text:
        return None
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:!?\"'`()[]{}")
    if not text or text in STOPWORDS:
        return None
    if re.fullmatch(r"[\d\s:.-]+", text):
        return None
    if len(text) < 2:
        return None
    return text


def _timestamp_to_seconds(ts_int: int) -> int:
    ts_str = str(ts_int)
    if len(ts_str) < 9:
        ts_str = ts_str.zfill(9)
    day = int(ts_str[0])
    hh = int(ts_str[1:3])
    mm = int(ts_str[3:5])
    ss = int(ts_str[5:7])
    return day * 86400 + hh * 3600 + mm * 60 + ss


def _md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class MemoryCell:
    GRANULARITY_ORDER = ["30sec", "3min", "10min", "1h"]

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_model: LLMModel,
        prompt_template_manager: PromptTemplateManager,
        granularities: Optional[List[str]] = None,
        cache_tag: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.prompt_template_manager = prompt_template_manager
        self.granularities = granularities or self.GRANULARITY_ORDER
        self.cache_tag = _safe_cache_tag(cache_tag)

        self.captions: Dict[str, List[CaptionEntry]] = {g: [] for g in self.granularities}
        self.caption_id_to_entry: Dict[str, CaptionEntry] = {}
        self.doc_id_to_entry: Dict[str, Dict[str, CaptionEntry]] = {g: {} for g in self.granularities}
        self.doc_pos_by_doc_id: Dict[str, Dict[str, int]] = {g: {} for g in self.granularities}
        self.text_to_entries: Dict[str, Dict[str, List[CaptionEntry]]] = {g: {} for g in self.granularities}

        self.triplets_by_doc: Dict[str, Dict[str, List[List[str]]]] = {g: {} for g in self.granularities}
        self.raw_triplets_by_doc: Dict[str, Dict[str, List[List[str]]]] = {g: {} for g in self.granularities}
        self.graph_sidecar: Dict[str, Dict[str, GraphEventSidecar]] = {g: {} for g in self.granularities}

        self.link_entity_to_doc_ids: Dict[str, Dict[str, Set[str]]] = {g: {} for g in self.granularities}
        self.link_entity_index_built: Dict[str, bool] = {g: False for g in self.granularities}

        # dense index state
        self.dense_cache_dir = os.path.join(".cache", "episodic_dense", self.cache_tag)
        self.doc_embeddings: Dict[str, Optional[np.ndarray]] = {g: None for g in self.granularities}
        self.doc_end_times: Dict[str, Optional[np.ndarray]] = {g: None for g in self.granularities}
        self.doc_retrieval_texts: Dict[str, List[str]] = {g: [] for g in self.granularities}
        self.dense_index_built: bool = False
        self._query_embedding_cache: Dict[str, np.ndarray] = {}
        self._thread_local = threading.local()

        # graph-aware rerank weights
        self.retrieval_text_score_weight = 0.18
        self.graph_score_weight = 0.22
        self.metadata_score_weight = 0.12
        self.raw_triplet_score_weight = 0.22

        self.visual_summary_retrieval_granularities = {"30sec", "3min"}
        self.critical_speech_retrieval_granularities = {"30sec", "3min", "10min", "1h"}
        self.max_critical_speech_lines_in_retrieval = 4

        # what goes into dense embedding text
        self.embed_metadata_for_granularities = {"30sec", "3min", "10min", "1h"}
        self.max_metadata_items_for_embedding = 6

        self.graph_expand_top_n = 4
        self.graph_expand_hops = 1
        self.graph_expand_decay = 0.60

        self.entity_expand_top_n = 4
        self.entity_expand_limit_per_seed = 6
        self.entity_expand_decay = 0.75

        # dense encoding controls
        self.dense_encode_batch_size = int(os.environ.get("EPISODIC_DENSE_BATCH_SIZE", 8))
        self.dense_encode_min_batch_size = 1

    # -----------------------------------------------------
    # Loading captions
    # -----------------------------------------------------

    def load_captions_from_files(self, caption_files: Dict[str, str]) -> None:
        for granularity, file_path in caption_files.items():
            if granularity not in self.granularities:
                logger.warning(f"Skipping granularity {granularity} - not configured")
                continue
            try:
                data = _load_json(file_path)
                self._process_caption_data(data, granularity)
                logger.info(f"Loaded {len(self.captions[granularity])} captions for {granularity}")
            except Exception as e:
                logger.error(f"Failed to load captions from {file_path}: {e}")
        self.dense_index_built = False

    def load_captions_from_data(self, caption_data: Dict[str, List[Dict[str, Any]]]) -> None:
        for granularity, data in caption_data.items():
            if granularity not in self.granularities:
                logger.warning(f"Skipping granularity {granularity} - not configured")
                continue
            self._process_caption_data(data, granularity)
            logger.info(f"Loaded {len(self.captions[granularity])} captions for {granularity}")
        self.dense_index_built = False

    def _make_doc_id(self, entry: Dict[str, Any], granularity: str, idx: int) -> str:
        if entry.get("doc_id"):
            return str(entry["doc_id"])
        date = str(entry.get("date", ""))
        start_time = str(entry.get("start_time", "")).zfill(8)
        end_time = str(entry.get("end_time", "")).zfill(8)
        if date and start_time and end_time:
            suffix = "" if granularity == "30sec" else f"_{granularity}"
            return f"{date}_{start_time}_{end_time}{suffix}"
        return f"{granularity}_{idx}"

    def _use_visual_summary_in_retrieval(self, granularity: str) -> bool:
        return granularity in self.visual_summary_retrieval_granularities

    def _use_critical_speech_in_retrieval(self, granularity: str) -> bool:
        return granularity in self.critical_speech_retrieval_granularities

    def _get_critical_speech_lines(self, entry: CaptionEntry) -> List[str]:
        metadata = entry.metadata or {}
        lines = metadata.get("critical_speech_lines", []) or []
        if not isinstance(lines, list):
            return []
        clean_lines: List[str] = []
        for line in lines:
            text = str(line).strip()
            if text:
                clean_lines.append(text)
        return clean_lines

    def _entry_retrieval_text(self, entry: CaptionEntry) -> str:
        parts = [entry.text]

        if self._use_visual_summary_in_retrieval(entry.granularity) and entry.visual_summary:
            parts.append(f"Visual: {entry.visual_summary}")

        if self._use_critical_speech_in_retrieval(entry.granularity):
            critical_lines = self._get_critical_speech_lines(entry)
            if critical_lines:
                clipped = critical_lines[: self.max_critical_speech_lines_in_retrieval]
                parts.append("Critical speech: " + " | ".join(clipped))

        return "\n".join([p for p in parts if p])

    def _entry_embedding_text(self, entry: CaptionEntry) -> str:
        parts = [self._entry_retrieval_text(entry)]
        if entry.granularity in self.embed_metadata_for_granularities:
            metadata = entry.metadata or {}
            compact_items: List[str] = []
            for field, prefix in [
                ("action_threads", "Actions"),
                ("object_threads", "Objects"),
                ("topic_threads", "Topics"),
                ("visual_object_threads", "VisualObjects"),
            ]:
                vals = metadata.get(field, []) or []
                flat: List[str] = []
                for x in vals:
                    if isinstance(x, dict):
                        for key in ["action", "object", "topic", "canonical_label", "label", "value", "speaker"]:
                            if key in x and x[key]:
                                flat.append(str(x[key]))
                                break
                    else:
                        flat.append(str(x))
                flat = [v.strip() for v in flat if str(v).strip()]
                if flat:
                    compact_items.append(f"{prefix}: " + " | ".join(flat[: self.max_metadata_items_for_embedding]))
            scene_summary = metadata.get("scene_summary", {}) or {}
            dominant_scene = scene_summary.get("dominant_scene") if isinstance(scene_summary, dict) else None
            if dominant_scene:
                compact_items.append(f"Scene: {dominant_scene}")
            if compact_items:
                parts.append("\n".join(compact_items))
        return "\n".join([p for p in parts if p])

    def _process_caption_data(self, data: List[Dict[str, Any]], granularity: str) -> None:
        entries: List[CaptionEntry] = []
        for idx, entry in enumerate(data):
            doc_id = self._make_doc_id(entry, granularity, idx)
            caption_id = doc_id

            metadata = {
                "action_threads": entry.get("action_threads", entry.get("main_actions", [])),
                "object_threads": entry.get("object_threads", entry.get("salient_objects", [])),
                "topic_threads": entry.get("topic_threads", entry.get("conversation_focus", [])),
                "speaker_stats": entry.get("speaker_stats", entry.get("speakers", [])),
                "scene_summary": entry.get("scene_summary", {}),
                "visual_object_threads": entry.get("visual_object_threads", entry.get("visual_objects", [])),
                "critical_speech_lines": list(entry.get("critical_speech_lines", []) or []),
                "source_doc_ids": list(entry.get("source_doc_ids", []) or []),
                "child_ids": list(entry.get("child_ids", []) or []),
            }

            caption_entry = CaptionEntry(
                id=caption_id,
                doc_id=doc_id,
                text=entry.get("text", entry.get("fine_caption", "")),
                start_time=str(entry.get("start_time", "")),
                end_time=str(entry.get("end_time", "")),
                date=str(entry.get("date", "")),
                granularity=granularity,
                video_path=entry.get("video_path"),
                visual_summary=entry.get("visual_summary", ""),
                metadata=metadata,
            )
            entries.append(caption_entry)

        entries.sort(key=lambda x: x.timestamp_int[1])

        self.captions[granularity] = entries
        self.doc_id_to_entry[granularity] = {}
        self.doc_pos_by_doc_id[granularity] = {}
        self.text_to_entries[granularity] = {}
        self.link_entity_index_built[granularity] = False
        self.link_entity_to_doc_ids[granularity] = {}

        for pos, caption_entry in enumerate(entries):
            self.caption_id_to_entry[caption_entry.id] = caption_entry
            self.doc_id_to_entry[granularity][caption_entry.doc_id] = caption_entry
            self.doc_pos_by_doc_id[granularity][caption_entry.doc_id] = pos
            self.text_to_entries[granularity].setdefault(caption_entry.text, []).append(caption_entry)

    # -----------------------------------------------------
    # Dense indexing
    # -----------------------------------------------------

    def _normalize_embedding_matrix(self, embs: np.ndarray) -> np.ndarray:
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs[None, :]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return embs / norms

    def _encode_texts(self, texts: List[str], desc: str = "Encoding dense embeddings") -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        batch_size = max(self.dense_encode_min_batch_size, self.dense_encode_batch_size)
        total = len(texts)
        i = 0
        all_embs: List[np.ndarray] = []

        pbar = tqdm(total=total, desc=desc, unit="docs")

        try:
            while i < total:
                cur_bs = min(batch_size, total - i)
                batch_texts = texts[i:i + cur_bs]

                try:
                    if hasattr(self.embedding_model, "encode_text"):
                        embs = self.embedding_model.encode_text(
                            batch_texts,
                            batch_size=cur_bs,
                        )
                    elif hasattr(self.embedding_model, "encode"):
                        embs = self.embedding_model.encode(batch_texts)
                    else:
                        raise AttributeError("EmbeddingModel must expose encode_text() or encode().")

                    embs = np.asarray(embs, dtype=np.float32)
                    if embs.ndim == 1:
                        embs = embs[None, :]

                    all_embs.append(embs)
                    i += cur_bs
                    pbar.update(cur_bs)

                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if batch_size <= self.dense_encode_min_batch_size:
                        raise

                    new_bs = max(self.dense_encode_min_batch_size, batch_size // 2)
                    logger.warning(
                        "CUDA OOM during dense encoding at doc %d/%d. "
                        "Reducing batch_size from %d to %d and retrying.",
                        i, total, batch_size, new_bs
                    )
                    batch_size = new_bs

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if batch_size <= self.dense_encode_min_batch_size:
                            raise

                        new_bs = max(self.dense_encode_min_batch_size, batch_size // 2)
                        logger.warning(
                            "RuntimeError OOM during dense encoding at doc %d/%d. "
                            "Reducing batch_size from %d to %d and retrying.",
                            i, total, batch_size, new_bs
                        )
                        batch_size = new_bs
                    else:
                        raise
        finally:
            pbar.close()

        embs = np.concatenate(all_embs, axis=0).astype(np.float32)
        return self._normalize_embedding_matrix(embs)

    def _granularity_cache_dir(self, granularity: str) -> str:
        return os.path.join(self.dense_cache_dir, granularity)

    def _build_cache_meta(self, granularity: str, texts: List[str]) -> Dict[str, Any]:
        return {
            "doc_ids": [e.doc_id for e in self.captions[granularity]],
            "text_hashes": [_md5_text(t) for t in texts],
            "end_times": [e.timestamp_int[1] for e in self.captions[granularity]],
        }

    def _try_load_dense_cache(self, granularity: str, texts: List[str]) -> bool:
        cache_dir = self._granularity_cache_dir(granularity)
        meta_path = os.path.join(cache_dir, "meta.json")
        emb_path = os.path.join(cache_dir, "doc_embeddings.npy")

        if not (os.path.exists(meta_path) and os.path.exists(emb_path)):
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected = self._build_cache_meta(granularity, texts)
            if meta != expected:
                return False
            embeddings = np.load(emb_path)
            if embeddings.shape[0] != len(self.captions[granularity]):
                return False
            self.doc_embeddings[granularity] = embeddings.astype(np.float32)
            self.doc_end_times[granularity] = np.asarray(expected["end_times"], dtype=np.int64)
            self.doc_retrieval_texts[granularity] = texts
            logger.info("Loaded dense episodic cache for %s: %d docs", granularity, embeddings.shape[0])
            return True
        except Exception as e:
            logger.warning("Failed to load dense episodic cache for %s: %s", granularity, e)
            return False

    def _save_dense_cache(self, granularity: str, texts: List[str], embeddings: np.ndarray) -> None:
        cache_dir = self._granularity_cache_dir(granularity)
        os.makedirs(cache_dir, exist_ok=True)
        meta = self._build_cache_meta(granularity, texts)
        with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        np.save(os.path.join(cache_dir, "doc_embeddings.npy"), embeddings)

    def build_dense_index(self, force_rebuild: bool = False) -> None:
        if self.dense_index_built and not force_rebuild:
            return

        os.makedirs(self.dense_cache_dir, exist_ok=True)

        for granularity in self.granularities:
            entries = self.captions.get(granularity, [])
            if not entries:
                logger.warning("No captions loaded for granularity %s", granularity)
                self.doc_embeddings[granularity] = None
                self.doc_end_times[granularity] = np.asarray([], dtype=np.int64)
                self.doc_retrieval_texts[granularity] = []
                continue

            texts = [self._entry_embedding_text(e) for e in entries]
            if not force_rebuild and self._try_load_dense_cache(granularity, texts):
                continue

            logger.info("Building dense episodic index for %s: %d docs", granularity, len(entries))
            embeddings = self._encode_texts(
                texts,
                desc=f"Dense index [{granularity}]",
            )
            self.doc_embeddings[granularity] = embeddings
            self.doc_end_times[granularity] = np.asarray([e.timestamp_int[1] for e in entries], dtype=np.int64)
            self.doc_retrieval_texts[granularity] = texts
            self._save_dense_cache(granularity, texts, embeddings)
            logger.info("Built dense episodic index for %s: %d docs", granularity, embeddings.shape[0])

        self.dense_index_built = True

    # -----------------------------------------------------
    # Sidecar loading
    # -----------------------------------------------------

    def load_sidecar_from_files(
        self,
        triplet_files: Optional[Dict[str, str]] = None,
        graph_files: Optional[Dict[str, str]] = None,
    ) -> None:
        if triplet_files:
            for granularity, file_path in triplet_files.items():
                if granularity not in self.granularities:
                    continue
                data = _load_json(file_path)
                self._process_triplet_data(data, granularity)

        if graph_files:
            for granularity, file_path in graph_files.items():
                if granularity not in self.granularities:
                    continue
                data = _load_json(file_path)
                self._process_graph_data(data, granularity)

    def load_sidecar_from_data(
        self,
        triplet_data: Optional[Dict[str, Dict[str, Any]]] = None,
        graph_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if triplet_data:
            for granularity, data in triplet_data.items():
                if granularity not in self.granularities:
                    continue
                self._process_triplet_data(data, granularity)

        if graph_data:
            for granularity, data in graph_data.items():
                if granularity not in self.granularities:
                    continue
                self._process_graph_data(data, granularity)

    def _process_triplet_data(self, data: Dict[str, Any], granularity: str) -> None:
        triplet_map = dict(data.get("triplet_map", {}))
        raw_triplet_map: Dict[str, List[List[str]]] = {}

        for unit in data.get("units", []) or []:
            if not isinstance(unit, dict):
                continue
            doc_id = str(unit.get("doc_id", "")).strip()
            if not doc_id:
                continue
            raw_triples = unit.get("openie_results", []) or unit.get("raw_triplets", []) or []
            if isinstance(raw_triples, list):
                raw_triplet_map[doc_id] = raw_triples
            if doc_id not in triplet_map and isinstance(unit.get("episodic_triplets", []), list):
                triplet_map[doc_id] = unit.get("episodic_triplets", []) or []

        self.triplets_by_doc[granularity] = triplet_map
        self.raw_triplets_by_doc[granularity] = raw_triplet_map
        logger.info(
            "Loaded sidecar triplets for %s: %d docs (%d docs with raw triplets)",
            granularity,
            len(self.triplets_by_doc[granularity]),
            len(self.raw_triplets_by_doc[granularity]),
        )
        self.link_entity_index_built[granularity] = False

    def _process_graph_data(self, data: Dict[str, Any], granularity: str) -> None:
        nodes = {node["id"]: node for node in data.get("nodes", [])}
        edges = data.get("edges", [])
        doc_id_to_event_id = data.get("doc_id_to_event_id", {})

        sidecar = {}
        for doc_id, event_id in doc_id_to_event_id.items():
            sidecar[doc_id] = GraphEventSidecar(
                event_id=event_id,
                doc_id=doc_id,
                granularity=granularity,
            )

        event_id_to_doc_id = {event_id: doc_id for doc_id, event_id in doc_id_to_event_id.items()}

        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            edge_type = edge["type"]
            event_id = edge.get("event_id")

            if edge_type == "before" and source in event_id_to_doc_id and target in event_id_to_doc_id:
                src_doc = event_id_to_doc_id[source]
                tgt_doc = event_id_to_doc_id[target]
                if src_doc in sidecar:
                    sidecar[src_doc].next_doc_id = tgt_doc
                if tgt_doc in sidecar:
                    sidecar[tgt_doc].prev_doc_id = src_doc
                continue

            if event_id not in event_id_to_doc_id:
                continue

            doc_id = event_id_to_doc_id[event_id]
            info = sidecar[doc_id]

            src_node = nodes.get(source, {})
            tgt_node = nodes.get(target, {})
            src_type = src_node.get("type")
            tgt_type = tgt_node.get("type")
            src_label = src_node.get("label", "")
            tgt_label = tgt_node.get("label", "")

            if source == event_id and tgt_type != "Event":
                if tgt_label:
                    info.entity_labels.append(tgt_label)
                if edge_type:
                    info.relation_types.append(edge_type)
                continue

            if src_type != "Event" and tgt_type != "Event":
                if src_label:
                    info.entity_labels.append(src_label)
                if tgt_label:
                    info.entity_labels.append(tgt_label)
                if edge_type:
                    info.relation_types.append(edge_type)
                if src_label and tgt_label and edge_type:
                    info.triplet_strings.append(f"{src_label} {edge_type} {tgt_label}")

        for doc_id, info in sidecar.items():
            event_node = nodes.get(info.event_id, {})
            token_source = []
            token_source.extend(info.entity_labels)
            token_source.extend(info.relation_types)
            token_source.extend(info.triplet_strings)
            if event_node.get("text"):
                token_source.append(event_node["text"])
            if event_node.get("visual_summary"):
                token_source.append(event_node["visual_summary"])
            if event_node.get("critical_speech_lines"):
                token_source.extend([str(x) for x in (event_node.get("critical_speech_lines") or [])])
            for field in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
                val = event_node.get(field, [])
                if isinstance(val, list):
                    token_source.extend([
                        json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
                        for x in val
                    ])
            token_source.append(json.dumps(event_node.get("scene_summary", {}), ensure_ascii=False))

            info.graph_tokens = set()
            for item in token_source:
                info.graph_tokens.update(_tokenize(item))

            info.entity_labels = sorted(set(info.entity_labels))
            info.relation_types = sorted(set(info.relation_types))
            info.triplet_strings = sorted(set(info.triplet_strings))

        self.graph_sidecar[granularity] = sidecar
        self.link_entity_index_built[granularity] = False
        logger.info(f"Loaded sidecar graph for {granularity}: {len(sidecar)} event nodes")

    # -----------------------------------------------------
    # Indexing / query-time filtering
    # -----------------------------------------------------

    def _set_thread_index_state(self, until_time: int) -> None:
        if not hasattr(self._thread_local, "indexed_time"):
            self._thread_local.indexed_time = 0
        self._thread_local.indexed_time = until_time
        prefixes: Dict[str, int] = {}
        indexed_entries: Dict[str, List[CaptionEntry]] = {}
        for granularity in self.granularities:
            end_times = self.doc_end_times.get(granularity)
            entries = self.captions.get(granularity, [])
            if end_times is None or len(entries) == 0:
                prefixes[granularity] = 0
                indexed_entries[granularity] = []
                continue
            count = int(np.searchsorted(end_times, until_time, side="right"))
            prefixes[granularity] = count
            indexed_entries[granularity] = entries[:count]
        self._thread_local.prefix_sizes = prefixes
        self._thread_local.indexed_entries = indexed_entries

    def index(self, until_time: int) -> None:
        if not self.dense_index_built:
            self.build_dense_index()
        self._set_thread_index_state(until_time)
        logger.info("Dense episodic index ready up to %s", _transform_timestamp(str(until_time)))

    def _get_thread_indexed_time(self) -> int:
        return int(getattr(self._thread_local, "indexed_time", 0) or 0)

    def _get_thread_prefix_size(self, granularity: str) -> int:
        prefixes = getattr(self._thread_local, "prefix_sizes", None)
        if prefixes is None:
            return 0
        return int(prefixes.get(granularity, 0))

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def _lookup_entry_from_index(self, granularity: str, idx: int) -> Optional[CaptionEntry]:
        entries = self.captions.get(granularity, [])
        if idx < 0 or idx >= len(entries):
            return None
        return entries[idx]

    def _entry_retrieval_tokens(self, entry: CaptionEntry) -> set:
        return _tokenize(self._entry_retrieval_text(entry))

    def _entry_metadata_tokens(self, entry: CaptionEntry) -> set:
        toks = set()
        toks.update(_tokenize(entry.text))
        toks.update(_tokenize(entry.visual_summary))
        for line in self._get_critical_speech_lines(entry):
            toks.update(_tokenize(line))
        metadata = entry.metadata or {}
        for k in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
            val = metadata.get(k, [])
            if isinstance(val, list):
                for x in val:
                    toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))
        scene_summary = metadata.get("scene_summary", {})
        if scene_summary:
            toks.update(_tokenize(json.dumps(scene_summary, ensure_ascii=False)))
        speaker_stats = metadata.get("speaker_stats", [])
        if isinstance(speaker_stats, list):
            for x in speaker_stats:
                toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))
        return toks

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-8:
            return [1.0 for _ in scores]
        return [(s - mn) / (mx - mn) for s in scores]

    def _overlap_score(self, query_tokens: set, candidate_tokens: set) -> float:
        if not query_tokens or not candidate_tokens:
            return 0.0
        inter = query_tokens & candidate_tokens
        return len(inter) / max(1, len(query_tokens))

    def _graph_aware_rerank(
        self,
        granularity: str,
        query: str,
        candidates: List[Tuple[CaptionEntry, float]],
    ) -> List[Tuple[CaptionEntry, float]]:
        if not candidates:
            return []

        query_tokens = _tokenize(query)
        base_scores = [score for _, score in candidates]
        base_scores = self._normalize_scores(base_scores)

        reranked = []
        for (entry, _), base_score in zip(candidates, base_scores):
            retrieval_text_score = self._overlap_score(query_tokens, self._entry_retrieval_tokens(entry))
            metadata_score = self._overlap_score(query_tokens, self._entry_metadata_tokens(entry))
            raw_triplet_score = self._overlap_score(query_tokens, self._entry_raw_triplet_tokens(entry.doc_id, granularity))
            graph_score = 0.0
            sidecar = self.graph_sidecar[granularity].get(entry.doc_id)
            if sidecar:
                graph_score = self._overlap_score(query_tokens, sidecar.graph_tokens)
            final_score = (
                base_score
                + self.retrieval_text_score_weight * retrieval_text_score
                + self.graph_score_weight * graph_score
                + self.metadata_score_weight * metadata_score
                + self.raw_triplet_score_weight * raw_triplet_score
            )
            reranked.append((entry, final_score))

        reranked.sort(key=lambda x: -x[1])
        return reranked

    def _collect_link_phrases(self, value: Any) -> List[str]:
        phrases: List[str] = []
        if value is None:
            return phrases
        if isinstance(value, str):
            norm = _normalize_phrase(value)
            if norm:
                phrases.append(norm)
            return phrases
        if isinstance(value, dict):
            for key in ["label", "name", "object", "entity", "item", "value", "mention", "text"]:
                if key in value:
                    phrases.extend(self._collect_link_phrases(value[key]))
            return phrases
        if isinstance(value, list):
            for item in value:
                phrases.extend(self._collect_link_phrases(item))
            return phrases
        return phrases

    def _get_link_entities_for_entry(self, entry: CaptionEntry, granularity: str) -> Set[str]:
        labels: Set[str] = set()

        sidecar = self.graph_sidecar.get(granularity, {}).get(entry.doc_id)
        if sidecar is not None:
            for label in sidecar.entity_labels:
                norm = _normalize_phrase(label)
                if norm:
                    labels.add(norm)
            for triplet_str in sidecar.triplet_strings:
                parts = [p.strip() for p in re.split(r"\s+(?:is|are|was|were|has|have|had|at|in|on|with|to|from|of)\s+", triplet_str, maxsplit=1)]
                for part in parts:
                    norm = _normalize_phrase(part)
                    if norm and len(norm.split()) <= 4:
                        labels.add(norm)

        metadata = entry.metadata or {}
        for field in ["object_threads", "visual_object_threads"]:
            labels.update(self._collect_link_phrases(metadata.get(field, [])))

        return {x for x in labels if x}

    def _ensure_link_entity_index(self, granularity: str) -> None:
        if self.link_entity_index_built.get(granularity, False):
            return

        index: Dict[str, Set[str]] = defaultdict(set)
        for entry in self.captions.get(granularity, []):
            for label in self._get_link_entities_for_entry(entry, granularity):
                index[label].add(entry.doc_id)

        self.link_entity_to_doc_ids[granularity] = dict(index)
        self.link_entity_index_built[granularity] = True
        logger.info(
            "Built object/entity link index for %s: %d labels",
            granularity,
            len(self.link_entity_to_doc_ids[granularity]),
        )

    def _expand_temporal_neighbors(
        self,
        granularity: str,
        ranked_candidates: List[Tuple[CaptionEntry, float]],
    ) -> List[Tuple[CaptionEntry, float]]:
        if not ranked_candidates:
            return []

        indexed_time = self._get_thread_indexed_time()
        expanded: Dict[str, Tuple[CaptionEntry, float]] = {}
        seeds = ranked_candidates[: max(1, self.graph_expand_top_n)]

        for entry, seed_score in seeds:
            current_doc_ids = [entry.doc_id]
            decay = float(seed_score)

            for _ in range(self.graph_expand_hops):
                next_doc_ids: List[str] = []
                decay *= self.graph_expand_decay
                for current_doc_id in current_doc_ids:
                    sidecar = self.graph_sidecar.get(granularity, {}).get(current_doc_id)
                    if sidecar is None:
                        continue
                    for neighbor_doc_id in [sidecar.prev_doc_id, sidecar.next_doc_id]:
                        if not neighbor_doc_id:
                            continue
                        neighbor_entry = self.get_caption_by_doc_id(neighbor_doc_id, granularity)
                        if neighbor_entry is None:
                            continue
                        if neighbor_entry.timestamp_int[1] > indexed_time:
                            continue
                        prev = expanded.get(neighbor_doc_id)
                        if prev is None or decay > prev[1]:
                            expanded[neighbor_doc_id] = (neighbor_entry, decay)
                        next_doc_ids.append(neighbor_doc_id)
                current_doc_ids = next_doc_ids
                if not current_doc_ids:
                    break

        return sorted(expanded.values(), key=lambda x: -x[1])

    def _expand_entity_neighbors(
        self,
        granularity: str,
        ranked_candidates: List[Tuple[CaptionEntry, float]],
    ) -> List[Tuple[CaptionEntry, float]]:
        if not ranked_candidates:
            return []

        indexed_time = self._get_thread_indexed_time()
        self._ensure_link_entity_index(granularity)
        index = self.link_entity_to_doc_ids.get(granularity, {})
        if not index:
            return []

        expanded: Dict[str, Tuple[CaptionEntry, float]] = {}
        seeds = ranked_candidates[: max(1, self.entity_expand_top_n)]

        for seed_entry, seed_score in seeds:
            seed_labels = self._get_link_entities_for_entry(seed_entry, granularity)
            if not seed_labels:
                continue

            candidate_overlap_counts: Dict[str, int] = defaultdict(int)
            for label in seed_labels:
                for neighbor_doc_id in index.get(label, set()):
                    if neighbor_doc_id == seed_entry.doc_id:
                        continue
                    candidate_overlap_counts[neighbor_doc_id] += 1

            seed_time = _timestamp_to_seconds(seed_entry.timestamp_int[0])
            scored_neighbors: List[Tuple[CaptionEntry, float]] = []

            for neighbor_doc_id, overlap_count in candidate_overlap_counts.items():
                neighbor_entry = self.get_caption_by_doc_id(neighbor_doc_id, granularity)
                if neighbor_entry is None:
                    continue
                if neighbor_entry.timestamp_int[1] > indexed_time:
                    continue

                neighbor_time = _timestamp_to_seconds(neighbor_entry.timestamp_int[0])
                time_gap = abs(neighbor_time - seed_time)
                temporal_proximity = 1.0 / (1.0 + time_gap / 300.0)
                overlap_strength = min(overlap_count, 3) / 3.0

                score = float(seed_score) * self.entity_expand_decay * (0.55 + 0.45 * overlap_strength) * temporal_proximity
                scored_neighbors.append((neighbor_entry, score))

            scored_neighbors.sort(key=lambda x: -x[1])
            for neighbor_entry, score in scored_neighbors[: self.entity_expand_limit_per_seed]:
                prev = expanded.get(neighbor_entry.doc_id)
                if prev is None or score > prev[1]:
                    expanded[neighbor_entry.doc_id] = (neighbor_entry, score)

        return sorted(expanded.values(), key=lambda x: -x[1])

    def expand_entry_to_30sec_doc_ids(self, entry: CaptionEntry) -> List[str]:
        if entry.granularity == "30sec":
            return [entry.doc_id]
        source_doc_ids = list(entry.metadata.get("source_doc_ids", []) or [])
        child_ids = list(entry.metadata.get("child_ids", []) or [])
        candidate_ids = source_doc_ids or child_ids
        if candidate_ids:
            return [str(x) for x in candidate_ids]
        return [entry.doc_id]

    def get_caption_by_doc_id(self, doc_id: str, granularity: Optional[str] = None) -> Optional[CaptionEntry]:
        if granularity is not None:
            return self.doc_id_to_entry.get(granularity, {}).get(doc_id)
        for g in self.granularities:
            if doc_id in self.doc_id_to_entry.get(g, {}):
                return self.doc_id_to_entry[g][doc_id]
        return None

    def get_triplets_by_doc_id(self, doc_id: str, granularity: str = "30sec") -> List[List[str]]:
        return self.triplets_by_doc.get(granularity, {}).get(doc_id, [])

    def get_raw_triplets_by_doc_id(self, doc_id: str, granularity: str = "30sec") -> List[List[str]]:
        return self.raw_triplets_by_doc.get(granularity, {}).get(doc_id, [])

    def _triplets_to_strings(self, triplets: List[List[str]]) -> List[str]:
        lines: List[str] = []
        for tri in triplets or []:
            if isinstance(tri, list) and len(tri) == 3:
                h, r, t = [str(x).strip() for x in tri]
                if h and r and t:
                    lines.append(f"{h} {r} {t}")
        return lines

    def _entry_raw_triplet_tokens(self, doc_id: str, granularity: str) -> set:
        toks = set()
        raw_triplets = self.get_raw_triplets_by_doc_id(doc_id, granularity)
        if not raw_triplets:
            raw_triplets = self.get_triplets_by_doc_id(doc_id, granularity)
        for line in self._triplets_to_strings(raw_triplets):
            toks.update(_tokenize(line))
        return toks

    def get_parent_caption(self, doc_id: str, parent_granularity: str = "3min") -> Optional[CaptionEntry]:
        child_entry = self.get_caption_by_doc_id(doc_id, "30sec")
        if child_entry is None:
            return None
        if parent_granularity not in self.doc_id_to_entry:
            return None

        child_start, child_end = child_entry.timestamp_int
        best_parent: Optional[CaptionEntry] = None
        best_span: Optional[int] = None

        for parent in self.captions.get(parent_granularity, []):
            if parent.date != child_entry.date:
                continue
            parent_start, parent_end = parent.timestamp_int
            if parent_start <= child_start and parent_end >= child_end:
                span = parent_end - parent_start
                if best_parent is None or best_span is None or span < best_span:
                    best_parent = parent
                    best_span = span

        return best_parent

    def normalize_doc_ids_to_30sec(self, doc_ids: List[str]) -> List[str]:
        normalized: List[str] = []
        seen: Set[str] = set()
        for raw_doc_id in doc_ids or []:
            doc_id = str(raw_doc_id).strip()
            if not doc_id:
                continue
            entry = self.get_caption_by_doc_id(doc_id)
            if entry is None:
                continue
            for child_id in self.expand_entry_to_30sec_doc_ids(entry):
                child = self.get_caption_by_doc_id(child_id, "30sec")
                if child is None:
                    continue
                if child.doc_id not in seen:
                    seen.add(child.doc_id)
                    normalized.append(child.doc_id)
        return normalized

    def expand_seed_doc_ids_with_neighbors(
        self,
        seed_doc_ids: List[str],
        radius: int = 2,
        indexed_time: Optional[int] = None,
    ) -> List[str]:
        seeds = self.normalize_doc_ids_to_30sec(seed_doc_ids)
        expanded: List[str] = []
        seen: Set[str] = set()
        if indexed_time is None:
            indexed_time = self._get_thread_indexed_time()

        for seed_doc_id in seeds:
            if seed_doc_id not in seen:
                seen.add(seed_doc_id)
                expanded.append(seed_doc_id)

            cur = seed_doc_id
            for _ in range(max(0, radius)):
                sidecar = self.graph_sidecar.get("30sec", {}).get(cur)
                prev_id = sidecar.prev_doc_id if sidecar is not None else None
                if not prev_id:
                    break
                prev_entry = self.get_caption_by_doc_id(prev_id, "30sec")
                if prev_entry is None:
                    break
                if indexed_time and prev_entry.timestamp_int[1] > indexed_time:
                    break
                if prev_id not in seen:
                    seen.add(prev_id)
                    expanded.append(prev_id)
                cur = prev_id

            cur = seed_doc_id
            for _ in range(max(0, radius)):
                sidecar = self.graph_sidecar.get("30sec", {}).get(cur)
                next_id = sidecar.next_doc_id if sidecar is not None else None
                if not next_id:
                    break
                next_entry = self.get_caption_by_doc_id(next_id, "30sec")
                if next_entry is None:
                    break
                if indexed_time and next_entry.timestamp_int[1] > indexed_time:
                    break
                if next_id not in seen:
                    seen.add(next_id)
                    expanded.append(next_id)
                cur = next_id
        return expanded

    def retrieve_ranked_from_doc_id_pool(
        self,
        query: str,
        doc_ids: List[str],
        family_info: Optional[Dict[str, Any]] = None,
        final_top_k: int = 8,
        neighbor_radius: int = 2,
        max_candidates: int = 64,
    ) -> List[Tuple[CaptionEntry, float]]:
        if not doc_ids:
            return []
        if not self.dense_index_built:
            self.build_dense_index()

        policy = self._family_policy(family_info)
        family = policy["question_family"]
        indexed_time = self._get_thread_indexed_time()
        query_tokens = _tokenize(query)
        seed_doc_ids = self.normalize_doc_ids_to_30sec(doc_ids)
        pool_doc_ids = self.expand_seed_doc_ids_with_neighbors(seed_doc_ids, radius=neighbor_radius, indexed_time=indexed_time)
        if max_candidates and len(pool_doc_ids) > max_candidates:
            pool_doc_ids = pool_doc_ids[:max_candidates]
        if not pool_doc_ids:
            return []

        doc_embeddings = self.doc_embeddings.get("30sec")
        dense_score_map: Dict[str, float] = {}
        if doc_embeddings is not None and len(pool_doc_ids) > 0:
            try:
                query_emb = self._get_query_embedding(query)
                scored: List[Tuple[str, float]] = []
                for doc_id in pool_doc_ids:
                    pos = self.doc_pos_by_doc_id.get("30sec", {}).get(doc_id)
                    if pos is None or pos >= len(doc_embeddings):
                        continue
                    score = float(np.dot(query_emb, doc_embeddings[pos]))
                    scored.append((doc_id, score))
                if scored:
                    vals = [s for _, s in scored]
                    vals_norm = self._normalize_scores(vals)
                    for (doc_id, _), v in zip(scored, vals_norm):
                        dense_score_map[doc_id] = v
            except Exception:
                dense_score_map = {}

        anchor_set = set(seed_doc_ids)
        results: List[Tuple[CaptionEntry, float]] = []
        for rank, doc_id in enumerate(pool_doc_ids):
            entry = self.get_caption_by_doc_id(doc_id, "30sec")
            if entry is None:
                continue
            if indexed_time and entry.timestamp_int[1] > indexed_time:
                continue
            dense_score = dense_score_map.get(doc_id, 0.0)
            direct_score = self._family_direct_score(entry, query_tokens, family)
            local_score = self._projection_local_score_30sec(query_tokens, doc_id)
            anchor_bonus = 1.0 if doc_id in anchor_set else 0.68
            temporal_bonus = 0.0
            if family == "temporal-recall":
                temporal_bonus = 0.10 * anchor_bonus
            final_score = 0.40 * dense_score + 0.35 * direct_score + 0.15 * local_score + 0.10 * anchor_bonus + temporal_bonus
            results.append((entry, final_score))

        results.sort(key=lambda x: -x[1])
        best_by_doc: Dict[str, Tuple[CaptionEntry, float]] = {}
        for entry, score in results:
            prev = best_by_doc.get(entry.doc_id)
            if prev is None or score > prev[1]:
                best_by_doc[entry.doc_id] = (entry, score)
        deduped = list(best_by_doc.values())
        deduped.sort(key=lambda x: -x[1])
        return deduped[:max(1, final_top_k)]

    # -----------------------------------------------------
    # Dense retrieval
    # -----------------------------------------------------

    def _get_query_embedding(self, query: str) -> np.ndarray:
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached
        emb = self._encode_texts([query], desc="Encoding query")[0]
        self._query_embedding_cache[query] = emb
        return emb

    def retrieve_captions_as_str(self, entries: List[CaptionEntry], include_visual_summary: bool = True) -> str:
        return "\n\n".join(entry.to_display_str(include_visual_summary=include_visual_summary) for entry in entries)

    def retrieve_ranked(
        self,
        query: str,
        top_k_per_granularity: Union[int, Dict[str, int]] = None,
        dedup_by_doc_id: bool = True,
    ) -> List[Tuple[CaptionEntry, float]]:
        if top_k_per_granularity is None:
            top_k_per_granularity = {"30sec": 10, "3min": 5, "10min": 5, "1h": 3}

        if not self.dense_index_built:
            self.build_dense_index()

        indexed_time = self._get_thread_indexed_time()
        if indexed_time == 0:
            logger.warning("No captions indexed. Call index(until_time) before retrieve().")
            return []

        query_emb = self._get_query_embedding(query)
        all_candidates: List[Tuple[CaptionEntry, float]] = []

        for granularity in self.granularities:
            if isinstance(top_k_per_granularity, dict):
                granularity_top_k = top_k_per_granularity.get(granularity, 5)
            else:
                granularity_top_k = top_k_per_granularity

            prefix_size = self._get_thread_prefix_size(granularity)
            if prefix_size <= 0:
                continue

            emb = self.doc_embeddings.get(granularity)
            if emb is None or emb.shape[0] == 0:
                continue

            active_emb = emb[:prefix_size]
            scores = np.matmul(active_emb, query_emb)
            if scores.size == 0:
                continue

            top_n = min(prefix_size, max(1, granularity_top_k * 2))
            if top_n >= scores.size:
                top_idx = np.argsort(-scores)
            else:
                partial = np.argpartition(-scores, top_n - 1)[:top_n]
                top_idx = partial[np.argsort(-scores[partial])]

            raw_candidates: List[Tuple[CaptionEntry, float]] = []
            for idx in top_idx.tolist():
                entry = self._lookup_entry_from_index(granularity, idx)
                if entry is None:
                    continue
                raw_candidates.append((entry, float(scores[idx])))

            reranked_candidates = self._graph_aware_rerank(
                granularity=granularity,
                query=query,
                candidates=raw_candidates,
            )

            selected_base = reranked_candidates[:granularity_top_k]
            expanded_temporal = self._expand_temporal_neighbors(granularity, selected_base)
            expanded_entity = self._expand_entity_neighbors(granularity, selected_base)

            logger.info(
                "Dense episodic %s: prefix=%d base=%d temporal_expanded=%d entity_expanded=%d",
                granularity,
                prefix_size,
                len(selected_base),
                len(expanded_temporal),
                len(expanded_entity),
            )

            all_candidates.extend(selected_base)
            all_candidates.extend(expanded_temporal)
            all_candidates.extend(expanded_entity)

        if dedup_by_doc_id:
            best_by_doc: Dict[str, Tuple[CaptionEntry, float]] = {}
            for entry, score in all_candidates:
                prev = best_by_doc.get(entry.doc_id)
                if prev is None or score > prev[1]:
                    best_by_doc[entry.doc_id] = (entry, score)
            all_candidates = list(best_by_doc.values())

        all_candidates.sort(key=lambda x: -x[1])
        return all_candidates

    def _family_policy(self, family_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        family = str((family_info or {}).get("question_family", "event")).strip().lower()
        graph_mode = str((family_info or {}).get("graph_mode", "default")).strip().lower()
        time_bias = str((family_info or {}).get("time_bias", "none")).strip().lower()

        policy = {
            "question_family": family,
            "graph_mode": graph_mode,
            "time_bias": time_bias,
            "seed_limit": 6,
            "max_hops": 2,
            "hop_decay": 0.72,
            "final_top_k": 12,
            "projection_budget": 192,
            "top_k_per_granularity": {"30sec": 10, "3min": 6, "10min": 4, "1h": 2},
        }

        if family == "source-trace":
            policy.update({
                "graph_mode": graph_mode if graph_mode != "default" else "backtrack_object_source",
                "time_bias": "backward" if time_bias == "none" else time_bias,
                "seed_limit": 5,
                "max_hops": 3,
                "hop_decay": 0.78,
                "final_top_k": 10,
                "projection_budget": 256,
                "top_k_per_granularity": {"30sec": 12, "3min": 8, "10min": 4, "1h": 1},
            })
        elif family == "temporal-recall":
            policy.update({
                "graph_mode": graph_mode if graph_mode != "default" else "temporal_walk",
                "time_bias": time_bias if time_bias in {"forward", "backward"} else "backward",
                "seed_limit": 5,
                "max_hops": 3,
                "hop_decay": 0.76,
                "final_top_k": 10,
                "projection_budget": 256,
                "top_k_per_granularity": {"30sec": 12, "3min": 8, "10min": 4, "1h": 1},
            })
        elif family == "action-owner":
            policy.update({
                "graph_mode": graph_mode if graph_mode != "default" else "actor_action_refine",
                "seed_limit": 5,
                "max_hops": 1,
                "hop_decay": 0.68,
                "final_top_k": 8,
                "projection_budget": 160,
                "top_k_per_granularity": {"30sec": 10, "3min": 6, "10min": 3, "1h": 1},
            })
        elif family == "participant-membership":
            policy.update({
                "graph_mode": graph_mode if graph_mode != "default" else "participant_cooccurrence_refine",
                "seed_limit": 5,
                "max_hops": 1,
                "hop_decay": 0.68,
                "final_top_k": 8,
                "projection_budget": 160,
                "top_k_per_granularity": {"30sec": 10, "3min": 7, "10min": 3, "1h": 1},
            })
        elif family == "plan-intention-decision":
            policy.update({
                "graph_mode": graph_mode if graph_mode != "default" else "topic_commitment_refine",
                "seed_limit": 5,
                "max_hops": 1,
                "hop_decay": 0.66,
                "final_top_k": 8,
                "projection_budget": 192,
                "top_k_per_granularity": {"30sec": 8, "3min": 8, "10min": 4, "1h": 2},
            })
        elif family == "attribute-content-purpose":
            policy.update({
                "graph_mode": graph_mode if graph_mode != "default" else "anchor_refine_then_visual",
                "seed_limit": 4,
                "max_hops": 1,
                "hop_decay": 0.62,
                "final_top_k": 6,
                "projection_budget": 128,
                "top_k_per_granularity": {"30sec": 8, "3min": 5, "10min": 2, "1h": 1},
            })
        elif family == "habit-preference":
            policy.update({
                "graph_mode": graph_mode if graph_mode != "default" else "habit_support_only",
                "seed_limit": 4,
                "max_hops": 0,
                "hop_decay": 0.0,
                "final_top_k": 6,
                "projection_budget": 96,
                "top_k_per_granularity": {"30sec": 4, "3min": 6, "10min": 4, "1h": 2},
            })

        return policy

    def _granularity_projection_weight(self, granularity: str) -> float:
        if granularity == "30sec":
            return 1.0
        if granularity == "3min":
            return 0.75
        if granularity == "10min":
            return 0.55
        if granularity == "1h":
            return 0.35
        return 0.30

    def _projection_local_score_30sec(self, query_tokens: Set[str], doc_id: str) -> float:
        entry = self.get_caption_by_doc_id(doc_id, "30sec")
        if entry is None:
            return 0.0
        retrieval_score = self._overlap_score(query_tokens, self._entry_retrieval_tokens(entry))
        metadata_score = self._overlap_score(query_tokens, self._entry_metadata_tokens(entry))
        raw_triplet_score = self._overlap_score(query_tokens, self._entry_raw_triplet_tokens(doc_id, "30sec"))
        return 0.45 * retrieval_score + 0.20 * metadata_score + 0.35 * raw_triplet_score

    def _project_ranked_candidates_to_30sec(
        self,
        query: str,
        ranked: List[Tuple[CaptionEntry, float]],
        max_total_candidates: int = 192,
    ) -> Dict[str, float]:
        projected: Dict[str, float] = defaultdict(float)
        local_cache: Dict[str, float] = {}
        indexed_time = self._get_thread_indexed_time()
        query_tokens = _tokenize(query)

        for rank, (entry, score) in enumerate(ranked):
            rank_bonus = 1.0 / (rank + 1)
            base = float(score) * rank_bonus * self._granularity_projection_weight(entry.granularity)
            target_doc_ids = []
            for doc_id in self.expand_entry_to_30sec_doc_ids(entry):
                child_entry = self.get_caption_by_doc_id(doc_id, "30sec")
                if child_entry is None:
                    continue
                if indexed_time > 0 and child_entry.timestamp_int[1] > indexed_time:
                    continue
                target_doc_ids.append(doc_id)
            if not target_doc_ids:
                continue

            denom = max(1.0, float(len(target_doc_ids)) ** 0.5)
            for doc_id in target_doc_ids:
                local_score = local_cache.get(doc_id)
                if local_score is None:
                    local_score = self._projection_local_score_30sec(query_tokens, doc_id)
                    local_cache[doc_id] = local_score
                projected[doc_id] += (base / denom) * (0.80 + 0.40 * local_score)

        if max_total_candidates and len(projected) > max_total_candidates:
            ranked_items = sorted(
                projected.items(),
                key=lambda x: -(x[1] + 0.35 * local_cache.get(x[0], 0.0)),
            )
            return dict(ranked_items[:max_total_candidates])

        return dict(projected)

    def _entry_graph_tokens_for_family(self, doc_id: str) -> Set[str]:
        entry = self.get_caption_by_doc_id(doc_id, "30sec")
        if entry is None:
            return set()
        toks = set()
        toks.update(_tokenize(entry.text))
        toks.update(_tokenize(entry.visual_summary))
        for line in self._get_critical_speech_lines(entry):
            toks.update(_tokenize(line))
        metadata = entry.metadata or {}
        for k in ["action_threads", "object_threads", "topic_threads", "visual_object_threads"]:
            val = metadata.get(k, [])
            if isinstance(val, list):
                for x in val:
                    toks.update(_tokenize(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)))
        sidecar = self.graph_sidecar.get("30sec", {}).get(doc_id)
        if sidecar is not None:
            toks.update(sidecar.graph_tokens)
            for tri in sidecar.triplet_strings:
                toks.update(_tokenize(tri))
        toks.update(self._entry_raw_triplet_tokens(doc_id, "30sec"))
        return toks

    def _family_neighbor_doc_ids(self, doc_id: str, policy: Dict[str, Any]) -> List[str]:
        sidecar = self.graph_sidecar.get("30sec", {}).get(doc_id)
        if sidecar is None:
            return []
        time_bias = policy.get("time_bias", "none")
        graph_mode = policy.get("graph_mode", "default")
        neighbors: List[str] = []
        if graph_mode in {"backtrack_object_source", "temporal_walk"}:
            if time_bias == "backward":
                neighbors = [sidecar.prev_doc_id] if sidecar.prev_doc_id else []
            elif time_bias == "forward":
                neighbors = [sidecar.next_doc_id] if sidecar.next_doc_id else []
            else:
                neighbors = [x for x in [sidecar.prev_doc_id, sidecar.next_doc_id] if x]
        else:
            neighbors = [x for x in [sidecar.prev_doc_id, sidecar.next_doc_id] if x]
        return [x for x in neighbors if x]

    def _family_direct_score(
        self,
        entry: CaptionEntry,
        query_tokens: Set[str],
        family: str,
    ) -> float:
        entry_tokens = self._entry_graph_tokens_for_family(entry.doc_id)
        direct_overlap = self._overlap_score(query_tokens, entry_tokens)
        metadata_overlap = self._overlap_score(query_tokens, self._entry_metadata_tokens(entry))
        raw_triplet_overlap = self._overlap_score(query_tokens, self._entry_raw_triplet_tokens(entry.doc_id, "30sec"))

        bonus = 0.0
        if family == "plan-intention-decision":
            critical = " ".join(self._get_critical_speech_lines(entry)).lower()
            if any(x in critical for x in ["will", "going to", "let's", "should", "come on", "see you", "need to"]):
                bonus += 0.18
        elif family == "action-owner":
            action_threads = json.dumps((entry.metadata or {}).get("action_threads", []), ensure_ascii=False).lower()
            if any(t in action_threads for t in list(query_tokens)[:6]):
                bonus += 0.15
        elif family == "participant-membership":
            speaker_stats = json.dumps((entry.metadata or {}).get("speaker_stats", []), ensure_ascii=False).lower()
            if any(t in speaker_stats for t in list(query_tokens)[:6]):
                bonus += 0.12
        elif family == "attribute-content-purpose":
            visual = (entry.visual_summary or "").lower()
            if any(x in visual for x in ["color", "inside", "contains", "holding", "look", "scene"]):
                bonus += 0.08

        return 0.55 * direct_overlap + 0.15 * metadata_overlap + 0.30 * raw_triplet_overlap + bonus

    def _family_graph_refine_30sec(
        self,
        query: str,
        family_info: Optional[Dict[str, Any]],
        anchor_scores: Dict[str, float],
    ) -> List[Tuple[CaptionEntry, float]]:
        if not anchor_scores:
            return []

        policy = self._family_policy(family_info)
        family = policy["question_family"]
        if policy.get("graph_mode") == "habit_support_only":
            ranked = []
            for doc_id, score in sorted(anchor_scores.items(), key=lambda x: -x[1])[: policy["final_top_k"]]:
                entry = self.get_caption_by_doc_id(doc_id, "30sec")
                if entry is not None:
                    ranked.append((entry, score))
            return ranked

        query_tokens = _tokenize(query)
        seed_doc_ids = [doc_id for doc_id, _ in sorted(anchor_scores.items(), key=lambda x: -x[1])[: policy["seed_limit"]]]
        accumulated: Dict[str, float] = defaultdict(float)
        visited_best_hop: Dict[str, int] = {}

        for seed_rank, seed_doc_id in enumerate(seed_doc_ids):
            seed_entry = self.get_caption_by_doc_id(seed_doc_id, "30sec")
            if seed_entry is None:
                continue
            seed_base = anchor_scores.get(seed_doc_id, 0.0) * (1.0 / (seed_rank + 1))
            accumulated[seed_doc_id] += seed_base + self._family_direct_score(seed_entry, query_tokens, family)
            frontier = [(seed_doc_id, 0, seed_base)]
            while frontier:
                current_doc_id, hop, current_score = frontier.pop(0)
                if hop >= policy["max_hops"]:
                    continue
                for neighbor_doc_id in self._family_neighbor_doc_ids(current_doc_id, policy):
                    neighbor_entry = self.get_caption_by_doc_id(neighbor_doc_id, "30sec")
                    if neighbor_entry is None:
                        continue
                    prev_hop = visited_best_hop.get(neighbor_doc_id)
                    if prev_hop is not None and prev_hop <= hop + 1:
                        continue
                    visited_best_hop[neighbor_doc_id] = hop + 1
                    decay = policy["hop_decay"] ** (hop + 1)
                    base = current_score * decay
                    direct = self._family_direct_score(neighbor_entry, query_tokens, family)
                    continuity = self._overlap_score(
                        self._entry_graph_tokens_for_family(current_doc_id),
                        self._entry_graph_tokens_for_family(neighbor_doc_id),
                    )
                    if family == "source-trace" and continuity <= 0.0 and direct <= 0.05:
                        continue
                    score = base + 0.35 * direct + 0.25 * continuity
                    accumulated[neighbor_doc_id] += score
                    frontier.append((neighbor_doc_id, hop + 1, score))

        final_items: List[Tuple[CaptionEntry, float]] = []
        for doc_id, score in sorted(accumulated.items(), key=lambda x: -x[1]):
            entry = self.get_caption_by_doc_id(doc_id, "30sec")
            if entry is None:
                continue
            final_score = score + 0.60 * anchor_scores.get(doc_id, 0.0)
            final_items.append((entry, final_score))

        best_by_doc: Dict[str, Tuple[CaptionEntry, float]] = {}
        for entry, score in final_items:
            prev = best_by_doc.get(entry.doc_id)
            if prev is None or score > prev[1]:
                best_by_doc[entry.doc_id] = (entry, score)
        deduped = list(best_by_doc.values())
        deduped.sort(key=lambda x: -x[1])
        return deduped[: policy["final_top_k"]]

    def retrieve_ranked_with_family(
        self,
        query: str,
        family_info: Optional[Dict[str, Any]] = None,
        top_k_per_granularity: Optional[Union[int, Dict[str, int]]] = None,
        dedup_by_doc_id: bool = True,
    ) -> List[Tuple[CaptionEntry, float]]:
        policy = self._family_policy(family_info)
        topk_cfg = top_k_per_granularity or policy["top_k_per_granularity"]
        ranked = self.retrieve_ranked(
            query=query,
            top_k_per_granularity=topk_cfg,
            dedup_by_doc_id=dedup_by_doc_id,
        )
        if not ranked:
            return []
        projected = self._project_ranked_candidates_to_30sec(
            query=query,
            ranked=ranked,
            max_total_candidates=policy.get("projection_budget", 192),
        )
        refined = self._family_graph_refine_30sec(query, family_info, projected)
        if refined:
            return refined
        # fallback: simple projection ranking
        fallback = []
        for doc_id, score in sorted(projected.items(), key=lambda x: -x[1])[: policy["final_top_k"]]:
            entry = self.get_caption_by_doc_id(doc_id, "30sec")
            if entry is not None:
                fallback.append((entry, score))
        return fallback

    def retrieve_with_family(
        self,
        query: str,
        family_info: Optional[Dict[str, Any]] = None,
        final_top_k: int = 5,
        as_context: bool = True,
    ) -> Union[List[CaptionEntry], str]:
        ranked = self.retrieve_ranked_with_family(query=query, family_info=family_info)
        if not ranked:
            return [] if not as_context else ""
        result_entries = [entry for entry, _ in ranked[:final_top_k]]
        if as_context:
            return self.retrieve_captions_as_str(result_entries, include_visual_summary=True)
        return result_entries

    def retrieve(
        self,
        query: str,
        top_k_per_granularity: Union[int, Dict[str, int]] = None,
        final_top_k: int = 3,
        as_context: bool = True,
    ) -> Union[List[CaptionEntry], str]:
        if top_k_per_granularity is None:
            top_k_per_granularity = {"30sec": 10, "3min": 5, "10min": 5, "1h": 3}

        ranked = self.retrieve_ranked(
            query=query,
            top_k_per_granularity=top_k_per_granularity,
            dedup_by_doc_id=True,
        )
        if not ranked:
            return [] if not as_context else ""

        result_entries = [entry for entry, _ in ranked[:final_top_k]]
        if as_context:
            return self.retrieve_captions_as_str(result_entries, include_visual_summary=True)
        return result_entries

    def reset_index(self) -> None:
        self._thread_local = threading.local()
        self._query_embedding_cache = {}
        for g in self.granularities:
            self.link_entity_index_built[g] = False
            self.link_entity_to_doc_ids[g] = {}
            if g not in self.doc_pos_by_doc_id:
                self.doc_pos_by_doc_id[g] = {}
        logger.info("Dense episodic query state reset")

    def get_indexed_time(self) -> str:
        indexed_time = self._get_thread_indexed_time()
        if indexed_time <= 0:
            return "DAY0 00:00:00"
        return _transform_timestamp(str(indexed_time))

    def get_caption_by_id(self, caption_id: str) -> Optional[CaptionEntry]:
        return self.caption_id_to_entry.get(caption_id)