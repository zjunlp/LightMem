import os
import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from PIL import Image

from ...embedding import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class VideoClipEntry:
    """Represents a single video clip / event-aligned visual unit."""
    id: str
    doc_id: str
    video_path: str
    start_time: str
    end_time: str
    date: str
    keyframe_paths: List[str] = field(default_factory=list)
    keyframe_caption: str = ""
    visual_objects: List[str] = field(default_factory=list)
    scene_summary: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    @property
    def timestamp_int(self) -> Tuple[int, int]:
        day = self.date.replace('DAY', '').replace('Day', '')
        start_ts = int(day + self.start_time.zfill(8))
        end_ts = int(day + self.end_time.zfill(8))
        return start_ts, end_ts

    def to_display_str(self) -> str:
        start_ts, end_ts = self.timestamp_int
        return f"{_transform_timestamp(str(start_ts))} - {_transform_timestamp(str(end_ts))}"


@dataclass
class FrameEntry:
    """Represents a single frame from a video or keyframe image."""
    video_path: str
    frame_index: int
    timestamp_sec: float
    frame: Optional[Image.Image] = None



def _transform_timestamp(ts_str: str) -> str:
    if len(ts_str) < 7:
        return ts_str
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"



def _load_json(file_path: str) -> Any:
    import json
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def _parse_time_range(time_range: str) -> Tuple[int, int]:
    import re
    pattern = r'DAY\s*(\d+)\s+(\d{1,2}):(\d{2}):(\d{2})'
    matches = re.findall(pattern, time_range, re.IGNORECASE)

    if len(matches) < 2:
        raise ValueError(
            f"Invalid time range format: {time_range}. Expected 'DAY X HH:MM:SS - DAY Y HH:MM:SS'"
        )

    start_day, start_hh, start_mm, start_ss = matches[0]
    start_ts = int(f"{start_day}{start_hh.zfill(2)}{start_mm.zfill(2)}{start_ss.zfill(2)}00")

    end_day, end_hh, end_mm, end_ss = matches[1]
    end_ts = int(f"{end_day}{end_hh.zfill(2)}{end_mm.zfill(2)}{end_ss.zfill(2)}00")

    return start_ts, end_ts



def _is_time_range_query(query: str) -> bool:
    import re
    pattern = r'DAY\s*\d+\s+\d{1,2}:\d{2}:\d{2}\s*-\s*DAY\s*\d+\s+\d{1,2}:\d{2}:\d{2}'
    return bool(re.search(pattern, query, re.IGNORECASE))


class VisualMemory:
    """
    Visual Memory module.

    Current recommended usage in the event-centric pipeline:
    - load 30sec evidence units containing doc_id / keyframe_paths / video_path
    - after text-based retrieval selects top event anchors, fetch their keyframes directly

    Embedding-based retrieval is kept as an optional compatibility path,
    but it is no longer required for the main pipeline.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model

        self.clips: List[VideoClipEntry] = []
        self.clip_id_to_entry: Dict[str, VideoClipEntry] = {}
        self.doc_id_to_entry: Dict[str, VideoClipEntry] = {}

        self.video_path_to_embedding: Dict[str, np.ndarray] = {}

        self.indexed_entries: List[VideoClipEntry] = []
        self.indexed_time: int = 0
        self.embeddings: Optional[torch.Tensor] = None
        self.index_to_pos: Dict[str, int] = {}

    # -----------------------------------------------------
    # Loading
    # -----------------------------------------------------

    def load_embeddings_from_file(self, embeddings_path: str) -> None:
        if not embeddings_path or not os.path.exists(embeddings_path):
            logger.warning(f"Visual embeddings file not found or not provided: {embeddings_path}")
            return

        with open(embeddings_path, 'rb') as f:
            self.video_path_to_embedding = pickle.load(f)

        logger.info(f"Loaded {len(self.video_path_to_embedding)} video embeddings from {embeddings_path}")

    def load_clips_from_file(self, clips_path: str) -> None:
        data = _load_json(clips_path)
        self.load_clips_from_data(data)

    def _infer_doc_id(self, entry: Dict[str, Any], idx: int) -> str:
        if entry.get("doc_id"):
            return str(entry["doc_id"])
        date = str(entry.get("date", ""))
        start_time = str(entry.get("start_time", "")).zfill(8)
        end_time = str(entry.get("end_time", "")).zfill(8)
        if date and start_time and end_time:
            return f"{date}_{start_time}_{end_time}"
        return f"visual_{idx}"

    def load_clips_from_data(self, data: List[Dict[str, Any]]) -> None:
        self.clips = []
        self.clip_id_to_entry = {}
        self.doc_id_to_entry = {}

        for idx, entry in enumerate(data):
            clip_id = f"visual_{idx}"
            doc_id = self._infer_doc_id(entry, idx)
            video_path = entry.get("video_path", "")
            embedding = self.video_path_to_embedding.get(video_path)

            clip_entry = VideoClipEntry(
                id=clip_id,
                doc_id=doc_id,
                video_path=video_path,
                start_time=str(entry.get("start_time", "")),
                end_time=str(entry.get("end_time", "")),
                date=str(entry.get("date", "")),
                keyframe_paths=list(entry.get("keyframe_paths", []) or []),
                keyframe_caption=str(entry.get("keyframe_caption", "") or ""),
                visual_objects=list(entry.get("visual_objects", []) or []),
                scene_summary=dict(entry.get("scene_summary", {}) or {}),
                embedding=embedding,
            )
            self.clips.append(clip_entry)
            self.clip_id_to_entry[clip_id] = clip_entry
            self.doc_id_to_entry[doc_id] = clip_entry

        self.clips.sort(key=lambda c: c.timestamp_int[0])
        logger.info(f"Loaded {len(self.clips)} visual clips / evidence events")

    # -----------------------------------------------------
    # Indexing
    # -----------------------------------------------------

    def index(self, until_time: int) -> None:
        if self.indexed_time >= until_time:
            logger.debug(f"Already indexed visual memory up to {self.indexed_time}, skipping")
            return

        # keep all entries up to time as indexed entries (even without embeddings)
        entries_to_index = [entry for entry in self.clips if entry.timestamp_int[1] <= until_time]
        self.indexed_entries = entries_to_index
        self.indexed_time = until_time

        # build embedding tensor only for entries with valid embeddings
        entries_with_embeddings = [entry for entry in entries_to_index if entry.embedding is not None]
        self.index_to_pos = {}
        if entries_with_embeddings:
            all_embeddings = []
            for pos, entry in enumerate(entries_with_embeddings):
                self.index_to_pos[entry.id] = pos
                all_embeddings.append(entry.embedding)
            self.embeddings = torch.tensor(
                np.array(all_embeddings),
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.embeddings = None

        logger.info(
            f"Indexed {len(entries_to_index)} visual clips up to {until_time} "
            f"({len(entries_with_embeddings)} with embeddings)"
        )

    # -----------------------------------------------------
    # Main retrieval interfaces
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        fps: float = 1.0,
        max_frames: int = 64,
        as_context: bool = True,
    ) -> Union[List[VideoClipEntry], List[FrameEntry], Dict[str, List[Image.Image]]]:
        if _is_time_range_query(query):
            frames = self._retrieve_by_time_range(
                time_range=query,
                fps=fps,
                max_frames=max_frames,
            )
            if as_context:
                if len(frames) > max_frames:
                    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int).tolist()
                    frames = [frames[i] for i in indices]
                images = [f.frame for f in frames if f.frame is not None]
                return {query: images}
            return frames

        return self._retrieve_by_similarity(
            query=query,
            top_k=top_k,
            fps=fps,
            max_frames=max_frames,
            as_context=as_context,
        )

    def get_clip_by_doc_id(self, doc_id: str) -> Optional[VideoClipEntry]:
        return self.doc_id_to_entry.get(doc_id)

    def get_event_images(
        self,
        doc_ids: List[str],
        max_images_per_event: int = 3,
        total_max_images: Optional[int] = None,
    ) -> Dict[str, List[Image.Image]]:
        """
        Return keyframes for selected event anchors.

        Preference order:
        1. keyframe_paths
        2. fallback to a few uniformly sampled frames from video_path
        """
        result: Dict[str, List[Image.Image]] = {}
        total_images = 0

        for doc_id in doc_ids:
            entry = self.doc_id_to_entry.get(doc_id)
            if entry is None:
                continue

            images: List[Image.Image] = []

            # load keyframes first
            if entry.keyframe_paths:
                for img_path in entry.keyframe_paths[:max_images_per_event]:
                    if not os.path.exists(img_path):
                        continue
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load keyframe {img_path}: {e}")

            # fallback to video extraction
            if not images and entry.video_path:
                frames = self._extract_frames(
                    entry.video_path,
                    fps=1.0,
                    max_frames=max_images_per_event,
                )
                images = [f.frame for f in frames if f.frame is not None]

            if images:
                if total_max_images is not None and total_images >= total_max_images:
                    break
                if total_max_images is not None and total_images + len(images) > total_max_images:
                    keep = max(0, total_max_images - total_images)
                    images = images[:keep]
                if images:
                    result[doc_id] = images
                    total_images += len(images)

        return result

    # -----------------------------------------------------
    # Optional embedding-based retrieval (compatibility)
    # -----------------------------------------------------

    def _retrieve_by_similarity(
        self,
        query: str,
        top_k: int = 5,
        fps: float = 1.0,
        max_frames: int = 64,
        as_context: bool = False,
    ) -> Union[List[VideoClipEntry], Dict[str, List[Image.Image]]]:
        if not self.indexed_entries or self.embeddings is None:
            logger.warning("No visual clips with embeddings indexed. Falling back to empty retrieval.")
            return {} if as_context else []

        if self.embedding_model is None:
            raise ValueError("embedding_model is required for similarity-based retrieval")

        indexed_with_embeddings = [entry for entry in self.indexed_entries if entry.embedding is not None]
        if not indexed_with_embeddings:
            return {} if as_context else []

        device = self.embeddings.device
        q_emb = self.embedding_model.encode_vis_query(query)
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)
        query_tensor = torch.tensor(q_emb, dtype=torch.float32, device=device)

        similarities = F.cosine_similarity(query_tensor, self.embeddings, dim=1)
        num_available = len(indexed_with_embeddings)
        k = min(top_k, num_available)
        _, top_indices = torch.topk(similarities, k)
        results = [indexed_with_embeddings[idx] for idx in top_indices.cpu().tolist()]

        if as_context:
            context: Dict[str, List[Image.Image]] = {}
            for clip in results:
                context[clip.doc_id] = self.get_event_images([clip.doc_id], max_images_per_event=3).get(clip.doc_id, [])
            return context

        return results

    def _retrieve_by_time_range(
        self,
        time_range: str,
        fps: float = 1.0,
        max_frames: int = 64,
    ) -> List[FrameEntry]:
        try:
            start_ts, end_ts = _parse_time_range(time_range)
        except ValueError as e:
            logger.error(str(e))
            return []

        matching_clips = []
        for clip in self.clips:
            clip_start, clip_end = clip.timestamp_int
            if clip_start <= end_ts and clip_end >= start_ts:
                matching_clips.append(clip)

        if not matching_clips:
            logger.warning(f"No clips found for time range {time_range}")
            return []

        matching_clips.sort(key=lambda c: c.timestamp_int[0])
        all_frames: List[FrameEntry] = []
        for clip in matching_clips:
            clip_start, clip_end = clip.timestamp_int
            overlap_start = max(start_ts, clip_start)
            overlap_end = min(end_ts, clip_end)
            start_sec = self._timestamp_diff_seconds(clip_start, overlap_start)
            end_sec = self._timestamp_diff_seconds(clip_start, overlap_end)
            frames = self._extract_frames(
                clip.video_path,
                fps=fps,
                max_frames=None,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            all_frames.extend(frames)

        if max_frames is not None and len(all_frames) > max_frames:
            indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int).tolist()
            all_frames = [all_frames[i] for i in indices]
        return all_frames

    def _timestamp_diff_seconds(self, ts_from: int, ts_to: int) -> float:
        def parse_ts(ts: int) -> Tuple[int, int, int, int]:
            ts_str = str(ts)
            day = int(ts_str[0])
            hh = int(ts_str[1:3])
            mm = int(ts_str[3:5])
            ss = int(ts_str[5:7])
            return day, hh, mm, ss

        d1, h1, m1, s1 = parse_ts(ts_from)
        d2, h2, m2, s2 = parse_ts(ts_to)
        total_sec_from = d1 * 86400 + h1 * 3600 + m1 * 60 + s1
        total_sec_to = d2 * 86400 + h2 * 3600 + m2 * 60 + s2
        return float(total_sec_to - total_sec_from)

    def _extract_frames(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: Optional[int] = 64,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> List[FrameEntry]:
        try:
            from decord import VideoReader, cpu
        except ImportError as e:
            raise ImportError("decord is required for frame extraction.") from e

        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return []

        frames: List[FrameEntry] = []
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            video_fps = vr.get_avg_fps()
            total_frames = len(vr)
            video_duration = total_frames / video_fps if video_fps > 0 else 0

            start_sec = 0.0 if start_sec is None else start_sec
            end_sec = video_duration if end_sec is None else end_sec
            start_sec = max(0.0, min(start_sec, video_duration))
            end_sec = max(start_sec, min(end_sec, video_duration))

            start_frame = int(start_sec * video_fps)
            end_frame = int(end_sec * video_fps)
            end_frame = min(end_frame, total_frames - 1)
            if start_frame >= end_frame:
                return []

            frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps)
            frame_interval = max(1, frame_interval)
            frame_indices = list(range(start_frame, end_frame + 1, frame_interval))
            if max_frames is not None and len(frame_indices) > max_frames:
                frame_indices = np.linspace(start_frame, end_frame, max_frames, dtype=int).tolist()
            if not frame_indices:
                return []

            video_frames = vr.get_batch(frame_indices).asnumpy()
            for i, frame_idx in enumerate(frame_indices):
                pil_frame = Image.fromarray(video_frames[i])
                timestamp_sec = frame_idx / video_fps if video_fps > 0 else 0
                frames.append(
                    FrameEntry(
                        video_path=video_path,
                        frame_index=frame_idx,
                        timestamp_sec=timestamp_sec,
                        frame=pil_frame,
                    )
                )
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            return []
        return frames

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def build_packets_for_events(
        self,
        doc_ids: List[str],
        max_images_per_event: int = 3,
    ) -> List[Dict[str, Any]]:
        packets: List[Dict[str, Any]] = []
        event_images = self.get_event_images(doc_ids, max_images_per_event=max_images_per_event)
        for doc_id in doc_ids:
            images = event_images.get(doc_id, [])
            if not images:
                continue
            packets.append({
                "packet_type": "visual",
                "anchor_doc_id": doc_id,
                "images": images,
                "num_images": len(images),
            })
        return packets


    def get_clip_by_id(self, clip_id: str) -> Optional[VideoClipEntry]:
        return self.clip_id_to_entry.get(clip_id)

    def get_clip_by_video_path(self, video_path: str) -> Optional[VideoClipEntry]:
        for clip in self.clips:
            if clip.video_path == video_path:
                return clip
        return None

    def cleanup(self) -> None:
        if self.embeddings is not None:
            del self.embeddings
            self.embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_index(self) -> None:
        self.embeddings = None
        self.indexed_entries = []
        self.indexed_time = 0
        self.index_to_pos = {}
        logger.info("Visual index reset - embeddings cleared")

    def get_indexed_time(self) -> str:
        return _transform_timestamp(str(self.indexed_time))

    def get_clips_count(self) -> int:
        return len(self.clips)

    def get_indexed_count(self) -> int:
        return len(self.indexed_entries)

    def get_clips_with_embeddings_count(self) -> int:
        return sum(1 for clip in self.clips if clip.embedding is not None)