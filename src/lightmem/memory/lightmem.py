import uuid
import re
import copy
import concurrent
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Tuple
from pydantic import ValidationError
from lightmem.configs.base import BaseMemoryConfigs
from lightmem.factory.pre_compressor.factory import PreCompressorFactory
from lightmem.factory.topic_segmenter.factory import TopicSegmenterFactory
from lightmem.factory.memory_manager.factory import MemoryManagerFactory
from lightmem.factory.text_embedder.factory import TextEmbedderFactory
from lightmem.factory.retriever.contextretriever.factory import ContextRetrieverFactory
from lightmem.factory.retriever.embeddingretriever.factory import EmbeddingRetrieverFactory
from lightmem.factory.memory_buffer.sensory_memory import SenMemBufferManager
from lightmem.factory.memory_buffer.short_term_memory import ShortMemBufferManager
from lightmem.memory.utils import MemoryEntry, assign_sequence_numbers_with_timestamps, save_memory_entries, convert_extraction_results_to_memory_entries
from lightmem.memory.prompts import METADATA_GENERATE_PROMPT, UPDATE_PROMPT, METADATA_GENERATE_PROMPT_locomo
from lightmem.configs.logging.utils import get_logger

GLOBAL_TOPIC_IDX = 0


class MessageNormalizer:

    _SESSION_RE = re.compile(
        r'(?P<date>\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*\((?P<weekday>[^)]+)\)\s*(?P<time>\d{1,2}:\d{2}(?::\d{2})?)'
    )

    def __init__(self, offset_ms: int = 1000):
        self.last_timestamp_map: Dict[str, datetime] = {}
        self.offset = timedelta(milliseconds=offset_ms)

    def _parse_session_timestamp(self, raw_ts: str) -> Tuple[datetime, str]:
        """
        Parse a session-level timestamp and return (base_datetime, weekday).
        Supports formats like "2023/05/20 (Sat) 00:44" (also accepts '-' as separator, and optional seconds).
        Raises ValueError if parsing fails.
        """
        m = self._SESSION_RE.search(raw_ts)
        if m:
            date_str = m.group('date').replace('-', '/')
            time_str = m.group('time')
            weekday = m.group('weekday')
            fmt = "%Y/%m/%d %H:%M:%S" if time_str.count(':') == 2 else "%Y/%m/%d %H:%M"
            base_dt = datetime.strptime(f"{date_str} {time_str}", fmt)
            return base_dt, weekday

        try:
            dt = datetime.fromisoformat(raw_ts)
            return dt, dt.strftime("%a")
        except Exception as e:
            raise ValueError(f"{str(e)}: Failed to parse session time format: '{raw_ts}'. Expected something like '2023/05/20 (Sat) 00:44'")

    def normalize_messages(self, messages: Any) -> List[Dict[str, Any]]:
        """
        Accepts str / dict / list[dict]:
          - If str -> treated as a single user message (if 'time_stamp' is required, use dict form)
          - If dict -> single message
          - If list -> multiple messages (each must be a dict and contain 'time_stamp')
        Returns: List[Dict] (each item is a copied and enriched message)
        """
        # Normalize input into a list
        if isinstance(messages, dict):
            messages_list = [messages]
        elif isinstance(messages, list):
            messages_list = messages
        elif isinstance(messages, str):
            raise ValueError("Please provide messages as dict or list[dict], and ensure each dict contains a 'time_stamp' field (session-level).")
        else:
            raise ValueError("messages must be dict or list[dict] (or str, but not recommended).")

        enriched_list: List[Dict[str, Any]] = []

        for msg in messages_list:
            if not isinstance(msg, dict):
                raise ValueError("Each item in messages list must be a dict.")
            raw_ts = msg.get("time_stamp")
            if not raw_ts:
                raise ValueError("Each message should contain a 'time_stamp' field (e.g., '2023/05/20 (Sat) 00:44').")

            base_dt, weekday = self._parse_session_timestamp(raw_ts)

            # Maintain incrementing time based on raw_ts as session key
            last_dt = self.last_timestamp_map.get(raw_ts)
            if last_dt is None:
                new_dt = base_dt
            else:
                new_dt = last_dt + self.offset

            self.last_timestamp_map[raw_ts] = new_dt

            enriched = copy.deepcopy(msg)
            enriched["session_time"] = raw_ts
            enriched["time_stamp"] = new_dt.isoformat(timespec="milliseconds")
            enriched["weekday"] = weekday

            enriched_list.append(enriched)

        return enriched_list


class LightMemory:
    def __init__(self, config: BaseMemoryConfigs = BaseMemoryConfigs()):
        
        """
        Initialize a LightMemory instance.

        This constructor initializes various memory-related components based on the provided configuration (`config`), 
        including the memory manager, optional pre-compressor, optional topic segmenter, text embedder, 
        and retrievers based on the configured strategies.

        This design supports flexible extension of the memory system, making it easy to integrate 
        different processing and retrieval capabilities.

        Args:
            config (BaseMemoryConfigs): The configuration object for the memory system, 
                containing initialization parameters for all submodules.

        Components initialized:
            - compressor (optional): Pre-compression model if pre_compress=True
            - segmenter (optional): Topic segmentation model if topic_segment=True
            - manager: Memory management model for metadata generation and text summarization
            - text_embedder (optional): Text embedding model if index_strategy is 'embedding' or 'hybrid'
            - retrieve_strategy (optional): Retrieval strategy ('context', 'embedding', or 'hybrid')
            - context_retriever (optional): Context-based retriever if retrieve_strategy is 'context' or 'hybrid'
            - embedding_retriever (optional): Embedding-based retriever if retrieve_strategy is 'embedding' or 'hybrid'
            - graph (optional): Graph memory store if graph_mem is enabled

        Note:
            - Multimodal embedder initialization is currently commented out
            - Graph memory initialization is conditional on graph_mem configuration
        """
        if config.logging is not None:
            config.logging.apply()
        
        self.logger = get_logger("LightMemory")
        self.logger.info("Initializing LightMemory with provided configuration")
        self.token_stats = {
            "add_memory_calls": 0,
            "add_memory_prompt_tokens": 0,
            "add_memory_completion_tokens": 0,
            "add_memory_total_tokens": 0,
            "update_calls": 0,
            "update_prompt_tokens": 0,
            "update_completion_tokens": 0,
            "update_total_tokens": 0,
            "embedding_calls": 0,
            "embedding_total_tokens": 0,
        }
        self.logger.info("Token statistics tracking initialized")
        
        self.config = config
        if self.config.pre_compress:
            self.logger.info("Initializing pre-compressor")
            self.compressor = PreCompressorFactory.from_config(self.config.pre_compressor)
        if self.config.topic_segment:
            self.logger.info("Initializing topic segmenter")
            self.segmenter = TopicSegmenterFactory.from_config(self.config.topic_segmenter, self.config.precomp_topic_shared, self.compressor)
            self.senmem_buffer_manager = SenMemBufferManager(max_tokens=self.segmenter.buffer_len, tokenizer=self.segmenter.tokenizer)
        self.logger.info("Initializing memory manager")
        self.manager = MemoryManagerFactory.from_config(self.config.memory_manager)
        self.shortmem_buffer_manager = ShortMemBufferManager(max_tokens = 512, tokenizer=getattr(self.manager, "tokenizer", self.manager.config.model))
        if self.config.index_strategy == 'embedding' or self.config.index_strategy == 'hybrid':
            self.logger.info("Initializing text embedder")
            self.text_embedder = TextEmbedderFactory.from_config(self.config.text_embedder)
        # if self.config.multimodal_embedder:
        self.retrieve_strategy = self.config.retrieve_strategy
        if self.retrieve_strategy in ["context", "hybrid"]:
            self.logger.info("Initializing context retriever")
            self.context_retriever = ContextRetrieverFactory.from_config(self.config.context_retriever)
        if self.retrieve_strategy in ["embedding", "hybrid"]:
            self.logger.info("Initializing embedding retriever")
            self.embedding_retriever = EmbeddingRetrieverFactory.from_config(self.config.embedding_retriever)
        if self.config.graph_mem:
            from .graph import GraphMem
            self.logger.info("Initializing graph memory")
            self.graph = GraphMem(self.config.graph_mem)
        self.logger.info("LightMemory initialization completed successfully")

    @classmethod
    def from_config(cls, config: Dict[str,Any]):
        try:
            configs = BaseMemoryConfigs(**config)
        except ValidationError as e:
            print(f"Configuration validation error: {e}")
            raise
        return cls(configs)
    
    
    def add_memory(
        self,
        messages,
        METADATA_GENERATE_PROMPT: Optional[str] = None,
        *,
        force_segment: bool = False, 
        force_extract: bool = False
    ):
        """
        Add new memory entries from message history.

        This method serves as the main pipeline for constructing new memory units from 
        incoming messages. It performs message normalization, optional pre-compression,
        segmentation, and knowledge extraction to produce structured memory entries.

        The process is as follows:
          1. Normalize input messages with standardized timestamps and session tracking.
          2. Optionally compress messages using the pre-defined compression model (if enabled).
          3. If topic segmentation is enabled, split messages into coherent segments and add them to the sentence-level buffer.
          4. Trigger memory extraction based on configured thresholds or forced flags.
          5. Optionally perform metadata summarization using an external model if enabled.
          6. Convert extracted results into `MemoryEntry` objects and update memory storage
             (either in online or offline mode depending on configuration).

        Args:
            messages (dict or List[dict]): Input message(s) to process.
            force_segment (bool, optional): If True, forces segmentation regardless of buffer conditions.
            force_extract (bool, optional): If True, forces memory extraction even if thresholds are not met.

        Returns:
            dict: A dictionary containing the intermediate results of the memory addition pipeline.
                  Typically includes:
                    - `"add_input_prompt"`: List of input prompts used for metadata generation (if enabled)
                    - `"add_output_prompt"`: Corresponding output results from metadata generation
                    - `"api_call_nums"`: Number of API calls made for extraction/summarization
                    - (In early termination cases) A segmentation result dict with keys such as
                      `"triggered"`, `"cut_index"`, `"boundaries"`, and `"emitted_messages"`

        Notes:
            - If `self.config.pre_compress` is True, messages will first be token-compressed before segmentation.
            - If `self.config.topic_segment` is disabled, the function returns early with segmentation info only.
            - Memory extraction results are wrapped into `MemoryEntry` objects containing timestamps,
              weekdays, and extracted factual content.
            - Depending on `self.config.update`, the function triggers either online or offline memory updates.
        """
        if METADATA_GENERATE_PROMPT is None:
            from lightmem.memory.prompts import METADATA_GENERATE_PROMPT as DEFAULT_PROMPT
            METADATA_GENERATE_PROMPT = DEFAULT_PROMPT
            
        call_id = f"add_memory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"force_segment={force_segment}, force_extract={force_extract}")
        result = {
            "add_input_prompt": [],
            "add_output_prompt": [],
            "api_call_nums": 0
        }
        self.logger.debug(f"[{call_id}] Raw input type: {type(messages)}")
        if isinstance(messages, list):
            self.logger.debug(f"[{call_id}] Raw input sample: {json.dumps(messages)}")
        normalizer = MessageNormalizer(offset_ms=500)
        msgs = normalizer.normalize_messages(messages)
        self.logger.debug(f"[{call_id}] Normalized messages sample: {json.dumps(msgs)}")
        if self.config.pre_compress:
            if hasattr(self.compressor, "tokenizer") and self.compressor.tokenizer is not None:
                args = (msgs, self.compressor.tokenizer)
            elif self.config.topic_segment and hasattr(self.segmenter, "tokenizer") and self.segmenter.tokenizer is not None:
                args = (msgs, self.segmenter.tokenizer)
            else:
                args = (msgs,)
            # fixed: empty 'content' in the 'messages' of 'compress(*args)'
            compressed_messages = self.compressor.compress(*args)
            cfg = getattr(self.compressor, "config", None)
            target_rate = None
            if cfg is not None:
                if hasattr(cfg, 'entropy_config') and isinstance(cfg.entropy_config, dict):
                    target_rate = cfg.entropy_config.get('compress_rate')
                elif hasattr(cfg, 'compress_config') and isinstance(cfg.compress_config, dict):
                    target_rate = cfg.compress_config.get('rate')
            self.logger.info(f"[{call_id}] Target compression rate: {target_rate}")
            self.logger.debug(f"[{call_id}] Compressed messages sample: {json.dumps(compressed_messages)}")
        else:
            compressed_messages = msgs
            self.logger.info(f"[{call_id}] Pre-compression disabled, using normalized messages")
        
        if not self.config.topic_segment:
            # TODO:
            self.logger.info(f"[{call_id}] Topic segmentation disabled, returning emitted messages")
            return {
                "triggered": True,
                "cut_index": len(msgs),
                "boundaries": [0, len(msgs)],
                "emitted_messages": msgs,
                "carryover_size": 0,
            }

        all_segments = self.senmem_buffer_manager.add_messages(compressed_messages, self.segmenter, self.text_embedder)

        if force_segment:
            all_segments = self.senmem_buffer_manager.cut_with_segmenter(self.segmenter, self.text_embedder, force_segment)
        
        if not all_segments:
            self.logger.debug(f"[{call_id}] No segments generated, returning empty result")
            return result # TODO

        self.logger.info(f"[{call_id}] Generated {len(all_segments)} segments")
        self.logger.debug(f"[{call_id}] Segments sample: {json.dumps(all_segments)}")

        extract_trigger_num, extract_list = self.shortmem_buffer_manager.add_segments(all_segments, self.config.messages_use, force_extract)

        if extract_trigger_num == 0:
            self.logger.debug(f"[{call_id}] Extraction not triggered, returning result")
            return result # TODO 
        
        global GLOBAL_TOPIC_IDX
        topic_id_mapping = []
        for api_call_segments in extract_list:
            api_call_topic_ids = []
            for topic_segment in api_call_segments:
                api_call_topic_ids.append(GLOBAL_TOPIC_IDX)
                GLOBAL_TOPIC_IDX += 1
            topic_id_mapping.append(api_call_topic_ids)
        self.logger.debug(f"topic_id_mapping: {topic_id_mapping}")
        self.logger.info(f"[{call_id}] Assigned global topic IDs: total={sum(len(x) for x in topic_id_mapping)}, mapping={topic_id_mapping}")
        self.logger.info(f"[{call_id}] Extraction triggered {extract_trigger_num} times, extract_list length: {len(extract_list)}")
        extract_list, timestamps_list, weekday_list, speaker_list, topic_id_map = assign_sequence_numbers_with_timestamps(extract_list, offset_ms=500, topic_id_mapping=topic_id_mapping)
        self.logger.debug(f"[{call_id}] Extract list sample: {json.dumps(extract_list)}")
        max_source_ids = [sum(1 for seg in batch for msg in seg if msg.get("role") == "user") - 1 for batch in extract_list]
        self.logger.info(f"[{call_id}] Batch max_source_ids: {max_source_ids}")
        if self.config.metadata_generate and self.config.text_summary:
            self.logger.info(f"[{call_id}] Starting metadata generation")
            extracted_results = self.manager.meta_text_extract(METADATA_GENERATE_PROMPT, extract_list, self.config.messages_use, topic_id_mapping)
        
            # =============API Consumption======================
            for idx, item in enumerate(extracted_results):
                if item is None:
                    continue
                
                if "usage" in item:
                    usage = item["usage"]
                    self.token_stats["add_memory_calls"] += 1
                    self.token_stats["add_memory_prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.token_stats["add_memory_completion_tokens"] += usage.get("completion_tokens", 0)
                    self.token_stats["add_memory_total_tokens"] += usage.get("total_tokens", 0)
                    
                    self.logger.info(
                        f"[{call_id}] API Call {idx} tokens - "
                        f"Prompt: {usage.get('prompt_tokens', 0)}, "
                        f"Completion: {usage.get('completion_tokens', 0)}, "
                        f"Total: {usage.get('total_tokens', 0)}"
                    )
                    
                self.logger.debug(f"[{call_id}] API Call {idx} raw output: {item['output_prompt']}")
                self.logger.debug(f"[{call_id}] API Call {idx} cleaned result: {item['cleaned_result']}")
                result["add_input_prompt"].append(item["input_prompt"])
                result["add_output_prompt"].append(item["output_prompt"])
                result["api_call_nums"] += 1

            # =======================================
            
            self.logger.info(f"[{call_id}] Metadata generation completed with {result['api_call_nums']} API calls")

        memory_entries = convert_extraction_results_to_memory_entries(
            extracted_results=extracted_results,
            timestamps_list=timestamps_list,
            weekday_list=weekday_list,
            speaker_list=speaker_list,
            topic_id_map=topic_id_map,
            max_source_ids=max_source_ids,
            logger=self.logger
        )
        self.logger.info(f"[{call_id}] Created {len(memory_entries)} MemoryEntry objects")
        for i, mem in enumerate(memory_entries):
            self.logger.debug(f"[{call_id}] MemoryEntry[{i}]: time={mem.time_stamp}, weekday={mem.weekday}, speaker_id={mem.speaker_id}, speaker_name={mem.speaker_name}, topic_id={mem.topic_id}, memory={mem.memory}")

        if self.config.update == "online":
            self.online_update(memory_entries)
        elif self.config.update == "offline":
            self.offline_update(memory_entries)
        
        self.logger.info(
            f"[{call_id}] Cumulative token stats - "
            f"Total API calls: {self.token_stats['add_memory_calls']}, "
            f"Total tokens: {self.token_stats['add_memory_total_tokens']}"
        )
        return result

    def online_update(self, memory_list: List):
        return None

    def offline_update(self, memory_list: List, construct_update_queue_trigger: bool = False, offline_update_trigger: bool = False):
        call_id = f"offline_update_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Received {len(memory_list)} memory entries")
        self.logger.info(f"[{call_id}] construct_update_queue_trigger={construct_update_queue_trigger}, offline_update_trigger={offline_update_trigger}")

        if self.config.index_strategy in ["context", "hybrid"]:
            self.logger.info(f"[{call_id}] Saving memory entries to file (strategy: {self.config.index_strategy})")
            save_memory_entries(memory_list, "memory_entries.json")

        if self.config.index_strategy in ["embedding", "hybrid"]:
            inserted_count = 0
            self.logger.info(f"[{call_id}] Starting embedding and insertion to vector database")
            for mem_obj in memory_list:
                embedding_vector = self.text_embedder.embed(mem_obj.memory)
                ids = mem_obj.id
                while self.embedding_retriever.exists(ids):
                    ids = str(uuid.uuid4())
                    mem_obj.id = ids
                payload = {
                    "time_stamp": mem_obj.time_stamp,
                    "float_time_stamp": mem_obj.float_time_stamp,
                    "weekday": mem_obj.weekday,
                    "topic_id": mem_obj.topic_id,
                    "topic_summary": mem_obj.topic_summary,
                    "category": mem_obj.category,
                    "subcategory": mem_obj.subcategory,
                    "memory_class": mem_obj.memory_class,
                    "memory": mem_obj.memory,
                    "original_memory": mem_obj.original_memory,
                    "compressed_memory": mem_obj.compressed_memory,
                    "speaker_id": mem_obj.speaker_id,
                    "speaker_name": mem_obj.speaker_name,
                }
                self.embedding_retriever.insert(
                    vectors = [embedding_vector],
                    payloads = [payload],
                    ids = [ids],
                )
                inserted_count += 1

            self.logger.info(f"[{call_id}] Successfully inserted {inserted_count} entries to vector database")
            if construct_update_queue_trigger:
                self.logger.info(f"[{call_id}] Triggering update queue construction")
                self.construct_update_queue_all_entries(
                    top_k=20,
                    keep_top_n=10
                )
            
            if offline_update_trigger:
                self.logger.info(f"[{call_id}] Triggering offline update for all entries")
                self.offline_update_all_entries(
                    update_sim_threshold = 0.8
                )

    def construct_update_queue_all_entries(self, top_k: int = 20, keep_top_n: int = 10, max_workers: int = 8):

        """
        Offline update all entries in parallel using multithreading.
        Each entry updates its own update_queue based on entries with earlier timestamps.

        Args:
            top_k (int): Number of nearest neighbors to consider for each entry.
            keep_top_n (int): Number of top entries to keep in update_queue.
            max_workers (int): Maximum number of threads to use.
        """
        call_id = f"construct_queue_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Parameters: top_k={top_k}, keep_top_n={keep_top_n}, max_workers={max_workers}")
        all_entries = self.embedding_retriever.get_all()
        self.logger.info(f"[{call_id}] Retrieved {len(all_entries)} entries from vector database")
        if not all_entries:
            self.logger.warning(f"[{call_id}] No entries found in database, skipping queue construction")
            self.logger.info(f"========== END {call_id} ==========")
            return
        updated_count = 0
        skipped_count = 0
        nonempty_queue_count = 0
        empty_queue_count = 0
        lock = threading.Lock()
        write_lock = threading.Lock()
        def _update_queue_construction(entry):
            nonlocal updated_count, skipped_count, nonempty_queue_count, empty_queue_count
            eid = entry["id"]
            payload = entry["payload"]
            vec = entry.get("vector")
            ts = payload.get("float_time_stamp", None)
            
            if vec is None or ts is None:
                self.logger.debug(f"[{call_id}] Skipping entry {eid}: missing vector={vec is None}, float_time_stamp={ts is None} ({ts})")
                with lock:
                    skipped_count += 1
                return

            hits = self.embedding_retriever.search(
                query_vector=vec,
                limit=top_k,
                filters={"float_time_stamp": {"lte": ts}}
            )

            candidates = []
            for h in hits:
                hid = h["id"]
                if hid == eid:
                    continue
                candidates.append({"id": hid, "score": h.get("score")})

            candidates.sort(key=lambda x: x["score"], reverse=True)
            update_queue = candidates[:keep_top_n]

            new_payload = dict(payload)
            new_payload["update_queue"] = update_queue

            if update_queue:
                with lock:
                    nonempty_queue_count += 1
                self.logger.debug(f"[{call_id}] Entry {eid} update_queue length={len(update_queue)} top_candidates=" + str(update_queue[:3]))
            else:
                with lock:
                    empty_queue_count += 1
                self.logger.debug(f"[{call_id}] Entry {eid} has no candidates after filtering (hits may be only itself)")

            with write_lock:
                self.embedding_retriever.update(vector_id=eid, vector=vec, payload=new_payload)

            with lock:
                updated_count += 1
        self.logger.info(f"[{call_id}] Starting parallel queue construction with {max_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(_update_queue_construction, all_entries)
        self.logger.info(
            f"[{call_id}] Queue construction completed: {updated_count} updated, {skipped_count} skipped, "
            f"nonempty_queues={nonempty_queue_count}, empty_queues={empty_queue_count}"
        )
        self.logger.info(f"========== END {call_id} ==========")

    def offline_update_all_entries(self, score_threshold: float = 0.9, max_workers: int = 5):
        """
        Perform offline updates for all entries based on their update_queue, in parallel.

        Args:
            score_threshold (float): Minimum similarity score for considering update candidates.
            max_workers (int): Maximum number of worker threads.
        """
        call_id = f"offline_update_all_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Parameters: score_threshold={score_threshold}, max_workers={max_workers}")
        all_entries = self.embedding_retriever.get_all()
        self.logger.info(f"[{call_id}] Retrieved {len(all_entries)} entries from vector database")
        if not all_entries:
            self.logger.warning(f"[{call_id}] No entries found in database, skipping offline update")
            self.logger.info(f"========== END {call_id} ==========")
            return
        processed_count = 0
        updated_count = 0
        deleted_count = 0
        skipped_count = 0
        lock = threading.Lock()
        write_lock = threading.Lock()
        update_token_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        token_lock = threading.Lock()
        def update_entry(entry):
            nonlocal processed_count, updated_count, deleted_count, skipped_count
            
            eid = entry["id"]
            payload = entry["payload"]

            candidate_sources = []
            for other in all_entries:
                update_queue = other["payload"].get("update_queue", [])
                for candidate in update_queue:
                    if candidate["id"] == eid and candidate["score"] >= score_threshold:
                        candidate_sources.append(other)
                        break

            if not candidate_sources:
                with lock:
                    skipped_count += 1
                return

            with lock:
                processed_count += 1

            updated_entry = self.manager._call_update_llm(UPDATE_PROMPT, entry, candidate_sources)

            if updated_entry is None:
                return
            # ====== token consumption ======
            usage = updated_entry["usage"]
            with token_lock:
                update_token_stats["calls"] += 1
                update_token_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                update_token_stats["completion_tokens"] += usage.get("completion_tokens", 0)
                update_token_stats["total_tokens"] += usage.get("total_tokens", 0)
                
            self.logger.debug(
                f"[{call_id}] Update LLM call for {eid} - "
                f"Tokens: {usage.get('total_tokens', 0)}"
            )
            # ==================== token consumption ====================
            action = updated_entry.get("action")
            if action == "delete":
                with write_lock:
                    self.embedding_retriever.delete(eid)
                with lock:
                    deleted_count += 1
                self.logger.debug(f"[{call_id}] Deleted entry: {eid}")
            elif action == "update":
                new_payload = dict(payload)
                new_payload["memory"] = updated_entry.get("new_memory")
                vector = entry.get("vector")
                with write_lock:
                    self.embedding_retriever.update(vector_id=eid, vector=vector, payload=new_payload)
                with lock:
                    updated_count += 1
                self.logger.debug(f"[{call_id}] Updated entry: {eid}")
        self.logger.info(f"[{call_id}] Starting parallel offline update with {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(update_entry, all_entries)
        with lock:
            self.token_stats["update_calls"] += update_token_stats["calls"]
            self.token_stats["update_prompt_tokens"] += update_token_stats["prompt_tokens"]
            self.token_stats["update_completion_tokens"] += update_token_stats["completion_tokens"]
            self.token_stats["update_total_tokens"] += update_token_stats["total_tokens"]    
        self.logger.info(f"[{call_id}] Offline update completed:")
        self.logger.info(f"[{call_id}]   - Processed: {processed_count} entries")
        self.logger.info(f"[{call_id}]   - Updated: {updated_count} entries")
        self.logger.info(f"[{call_id}]   - Deleted: {deleted_count} entries")
        self.logger.info(f"[{call_id}]   - Skipped (no candidates): {skipped_count} entries")
        self.logger.info(
            f"[{call_id}]   - Update API calls: {update_token_stats['calls']}, "
            f"Total tokens: {update_token_stats['total_tokens']}"
        )
        self.logger.info(f"========== END {call_id} ==========")
    
    def retrieve(self, query: str, limit: int = 10, filters: Optional[dict] = None) -> list[str]:
        """
        Retrieve relevant entries and return them as formatted strings.

        Args:
            query (str): The natural language query string.
            limit (int, optional): Number of results to return. Defaults to 10.
            filters (dict, optional): Optional filters to narrow down the search. Defaults to None.

        Returns:
            list[str]: A list of formatted strings containing time_stamp, weekday, and memory.
        """
        call_id = f"retrieve_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.logger.info(f"========== START {call_id} ==========")
        self.logger.info(f"[{call_id}] Query: {query}")
        self.logger.info(f"[{call_id}] Parameters: limit={limit}, filters={filters}")
        self.logger.debug(f"[{call_id}] Generating embedding for query")
        query_vector = self.text_embedder.embed(query)
        self.logger.debug(f"[{call_id}] Query embedding dimension: {len(query_vector)}")
        self.logger.info(f"[{call_id}] Searching vector database")
        results = self.embedding_retriever.search(
            query_vector=query_vector,
            limit=limit,
            filters=filters,
            return_full=True,
        )
        self.logger.info(f"[{call_id}] Found {len(results)} results")
        formatted_results = []
        for r in results:
            payload = r.get("payload", {})
            time_stamp = payload.get("time_stamp", "")
            weekday = payload.get("weekday", "")
            memory = payload.get("memory", "")
            formatted_results.append(f"{time_stamp} {weekday} {memory}")
            
        result_string = "\n".join(formatted_results)
        self.logger.info(f"[{call_id}] Formatted {len(formatted_results)} results into output string")
        self.logger.debug(f"[{call_id}] Output string length: {len(result_string)} characters")
        self.logger.info(f"========== END {call_id} ==========")
        return result_string

    def get_token_statistics(self):
        embedder_stats = {"total_calls": 0, "total_tokens": None}
        if hasattr(self, 'text_embedder') and hasattr(self.text_embedder, 'get_stats'):
            embedder_stats = self.text_embedder.get_stats()
        
        stats = {
            "summary": {
                "total_llm_calls": self.token_stats["add_memory_calls"] + self.token_stats["update_calls"],
                "total_llm_tokens": self.token_stats["add_memory_total_tokens"] + self.token_stats["update_total_tokens"],
                "total_embedding_calls": embedder_stats["total_calls"],
                "total_embedding_tokens": embedder_stats["total_tokens"],
            },
            "llm": {
                "add_memory": {
                    "calls": self.token_stats["add_memory_calls"],
                    "prompt_tokens": self.token_stats["add_memory_prompt_tokens"],
                    "completion_tokens": self.token_stats["add_memory_completion_tokens"],
                    "total_tokens": self.token_stats["add_memory_total_tokens"],
                },
                "update": {
                    "calls": self.token_stats["update_calls"],
                    "prompt_tokens": self.token_stats["update_prompt_tokens"],
                    "completion_tokens": self.token_stats["update_completion_tokens"],
                    "total_tokens": self.token_stats["update_total_tokens"],
                },
            },
            "embedding": {
                "total_calls": embedder_stats["total_calls"],
                "total_tokens": embedder_stats["total_tokens"],
                "note": "Includes topic segmentation + memory indexing. Local models show None for tokens."
            }
        }
        
        return stats
