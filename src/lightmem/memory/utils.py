import os
import re
import json
from datetime import datetime
from typing import List, Dict, Literal, Optional, Any, Tuple, Union
import tiktoken
import uuid
from dataclasses import dataclass, field
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Optional, Union, Dict

@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time_stamp: str = field(default_factory=lambda: datetime.now().isoformat())
    float_time_stamp: float = 0
    weekday: str = ""
    category: str = ""
    subcategory: str = ""
    memory_class: str = ""
    memory: str = ""
    original_memory: str = ""
    compressed_memory: str = ""
    topic_id: Optional[int] = None
    topic_summary: str = ""
    speaker_id: str = ""
    speaker_name: str = ""
    hit_time: int = 0
    update_queue: List = field(default_factory=list)
    consolidated: bool = False
    
def clean_response(response: str) -> List[Dict[str, Any]]:
    """
    Cleans the model response by:
    1. Removing enclosing code block markers (```[language] ... ```).
    2. Parsing the JSON content safely.
    3. Returning the value of the "data" key if present, otherwise trying to return the parsed list/dict.
    """
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response.strip())
    cleaned = match.group(1).strip() if match else response.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}")
        return []

    if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        return parsed["data"]

    if isinstance(parsed, list):
        return parsed

    return []


def assign_sequence_numbers_with_timestamps(extract_list, offset_ms: int = 500, topic_id_mapping: List[List[int]] = None):
    from datetime import datetime, timedelta
    from collections import defaultdict
    import re
    
    current_index = 0
    timestamps_list = []
    weekday_list = []
    speaker_list = []
    message_refs = []
    
    for segments in extract_list:
        for seg in segments:
            for message in seg:
                session_time = message.get('session_time', '')
                message_refs.append((message, session_time))
    
    session_groups = defaultdict(list)
    for msg, sess_time in message_refs:
        session_groups[sess_time].append(msg)
    
    for sess_time, messages in session_groups.items():
        cleaned_time = re.sub(r'\s*\([A-Za-z]+\)\s*', ' ', sess_time).strip()
        
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",      
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",      
            "%Y/%m/%d"
        ]
        
        base_dt = None
        for fmt in formats:
            try:
                base_dt = datetime.strptime(cleaned_time, fmt)
                break
            except ValueError:
                continue
                
        if base_dt is None:
            try:
                base_dt = datetime.fromisoformat(cleaned_time.replace('/', '-'))
            except:
                raise ValueError(f"Time format '{sess_time}' not supported. Expected formats: YYYY-MM-DD, YYYY/MM/DD, with optional HH:MM or HH:MM:SS")
            
        for i, msg in enumerate(messages):
            offset = timedelta(milliseconds=offset_ms * i)
            new_dt = base_dt + offset
            msg['time_stamp'] = new_dt.isoformat(timespec='milliseconds')
    
    for segments in extract_list:
        for seg in segments:
            for message in seg:
                message["sequence_number"] = current_index
                timestamps_list.append(message["time_stamp"])
                weekday_list.append(message["weekday"])
                speaker_info = {
                    'speaker_id': message.get('speaker_id', 'unknown'),
                    'speaker_name': message.get('speaker_name', 'Unknown')
                }
                speaker_list.append(speaker_info)
                current_index += 1

    sequence_to_topic = {}
    if topic_id_mapping:
        for api_idx, api_call_segments in enumerate(extract_list):
            for topic_idx, topic_segment in enumerate(api_call_segments):
                tid = topic_id_mapping[api_idx][topic_idx]
                for msg in topic_segment:
                    seq = msg.get("sequence_number")
                    sequence_to_topic[seq] = tid

    return extract_list, timestamps_list, weekday_list, speaker_list, sequence_to_topic

# TODO：merge into context retriever
def save_memory_entries(memory_entries, file_path="memory_entries.json"):
    def entry_to_dict(entry):
        return {
            "id": entry.id,
            "time_stamp": entry.time_stamp,
            "topic_id": entry.topic_id,
            "topic_summary": entry.topic_summary,
            "category": entry.category,
            "subcategory": entry.subcategory,
            "memory_class": entry.memory_class,
            "memory": entry.memory,
            "original_memory": entry.original_memory,
            "compressed_memory": entry.compressed_memory,
            "hit_time": entry.hit_time,
            "update_queue": entry.update_queue,
            "float_time_stamp": getattr(entry, "float_time_stamp", 0),  
            "weekday": getattr(entry, "weekday", ""),  
            "speaker_id": getattr(entry, "speaker_id", ""),  
            "speaker_name": getattr(entry, "speaker_name", ""),  
            "consolidated": getattr(entry, "consolidated", False),  
        }

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {str(e)}")
                existing_data = []
    else:
        existing_data = []

    new_data = [entry_to_dict(e) for e in memory_entries]
    existing_data.extend(new_data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

# TODO：more support for any models
def resolve_tokenizer(tokenizer_or_name: Union[str, Any]) -> Union[tiktoken.Encoding, Any]:
    """
    Resolve the tokenizer for a given model name or tokenizer instance.
    """

    # --- Case: already a tokenizer object (transformers local model) ---
    if isinstance(tokenizer_or_name, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        return tokenizer_or_name

    # --- Case: OpenAI tiktoken model name ---
    try:
        return tiktoken.encoding_for_model(tokenizer_or_name)
    except:
        pass

    # --- Case: user-defined patterns (Qwen etc.) ---
    patterns = [
        (r"^qwen3", "o200k_base"),
        # Add more patterns as needed...
    ]
    for pattern, encoding_name in patterns:
        if isinstance(tokenizer_or_name, str) and re.match(pattern, tokenizer_or_name):
            return tiktoken.get_encoding(encoding_name)

    # --- Case: fallback ---
    return tiktoken.get_encoding("o200k_base")

def convert_extraction_results_to_memory_entries(
    extracted_results: List[Optional[Dict]],
    timestamps_list: List,
    weekday_list: List,
    speaker_list: List = None,
    topic_id_map: Dict[int, int] = None,
    max_source_ids: List[int] = None, 
    logger = None
) -> List[MemoryEntry]:
    """
    Convert extraction results to MemoryEntry objects.

    Args:
        extracted_results: Results from meta_text_extract, each containing cleaned_result
        timestamps_list: List of timestamps indexed by sequence_number
        weekday_list: List of weekdays indexed by sequence_number
        speaker_list: List of speaker information
        topic_id_map: Optional mapping of sequence_number -> topic_id (preferred)
        logger: Optional logger for debug info

    Returns:
        List of MemoryEntry objects with assigned topic_id and timestamps
    """
    memory_entries = []

    extracted_memory_entry = [
        item["cleaned_result"]
        for item in extracted_results
        if item and item.get("cleaned_result")
    ]

    for batch_idx, topic_memory in enumerate(extracted_memory_entry):
        if not topic_memory:
            continue
        
        max_valid_sid = max_source_ids[batch_idx] if max_source_ids and batch_idx < len(max_source_ids) else None
        
        for topic_idx, fact_list in enumerate(topic_memory):
            if not isinstance(fact_list, list):
                fact_list = [fact_list]

            for fact_entry in fact_list:
                original_sid = int(fact_entry.get("source_id", 0))
                sid = original_sid
                
                if max_valid_sid is not None and sid > max_valid_sid:
                    sid = max_valid_sid  
                    logger.warning(
                        f"LLM returned invalid source_id={original_sid} "
                        f"(valid range: [0, {max_valid_sid}]) in batch {batch_idx}. "
                        f"Auto-corrected to source_id={sid}. "
                        f"Fact: {fact_entry.get('fact', '')[:100]}..."
                    )
                
                seq_candidate = sid * 2
                
                if seq_candidate not in topic_id_map:
                    logger.error(
                        f"sequence {seq_candidate} (from corrected source_id={sid}) "
                        f"not found in topic_id_map. "
                        f"Available range: {min(topic_id_map.keys())}-{max(topic_id_map.keys())}. "
                        f"Skipping this fact."
                    )
                    continue
                
                resolved_topic_id = topic_id_map[seq_candidate]
                
                mem_obj = _create_memory_entry_from_fact(
                    fact_entry,
                    timestamps_list,
                    weekday_list,
                    speaker_list,
                    topic_id=resolved_topic_id,
                    topic_summary="",
                    logger=logger,
                )

                if mem_obj:
                    memory_entries.append(mem_obj)

    return memory_entries


def _create_memory_entry_from_fact(
    fact_entry: Dict,
    timestamps_list: List,
    weekday_list: List,
    speaker_list: List = None,
    topic_id: int = None,  
    topic_summary: str = "",
    logger = None
) -> Optional[MemoryEntry]:
    """
    Helper function to create a MemoryEntry from a fact entry.
    
    Args:
        fact_entry: Dict containing source_id and fact
        timestamps_list: List of timestamps indexed by sequence_number
        weekday_list: List of weekdays indexed by sequence_number
        speaker_list: List of speaker information
        topic_id: Topic ID for this memory entry
        topic_summary: Topic summary for this memory entry (reserved for future use)
        logger: Optional logger for warnings
        
    Returns:
        MemoryEntry object or None if creation fails
    """
    source_id = int(fact_entry.get("source_id", 0))
    sequence_n = source_id * 2

    try:
        time_stamp = timestamps_list[sequence_n]
        
        if not isinstance(time_stamp, float):
            from datetime import datetime
            float_time_stamp = datetime.fromisoformat(time_stamp).timestamp()
        else:
            float_time_stamp = time_stamp
            
        weekday = weekday_list[sequence_n]
        speaker_info = speaker_list[sequence_n]
        speaker_id = speaker_info.get('speaker_id', 'unknown')
        speaker_name = speaker_info.get('speaker_name', 'Unknown')
        
    except (IndexError, TypeError, ValueError) as e:
        if logger:
            logger.warning(
                f"Error getting timestamp for sequence {sequence_n}: {e}"
            )
        time_stamp = None
        float_time_stamp = None
        weekday = None
        speaker_id = 'unknown'
        speaker_name = 'Unknown'
    
    mem_obj = MemoryEntry(
        time_stamp=time_stamp,
        float_time_stamp=float_time_stamp,
        weekday=weekday,
        memory=fact_entry.get("fact") or fact_entry.get("relation", ""),
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        topic_id=topic_id,
        topic_summary=topic_summary,
        consolidated=False, 
    )
    
    return mem_obj


def normalize_extraction_prompts(
    prompts: Optional[Union[str, Dict[str, str]]],
    extraction_mode: str = "flat",
    logger = None
) -> Optional[Dict[str, str]]:
    if prompts is None:
        logger.debug(f"No custom prompts provided, will use defaults for mode: {extraction_mode}")
        return None
    if isinstance(prompts, str):
        logger.debug("Legacy string prompt detected, converting to dict format")
        return {"factual": prompts}
    if isinstance(prompts, dict):
        logger.debug(f"Using dict prompts with keys: {list(prompts.keys())}")
        return prompts
    raise TypeError(
        f"METADATA_GENERATE_PROMPT must be str, dict, or None, "
        f"got {type(prompts).__name__}"
    )


def process_extraction_results(
    extracted_results: List[Optional[Dict]],
    token_stats: Dict[str, int],
    result_dict: Dict[str, Any],
    call_id: str,
    logger = None
) -> None:
    for idx, item in enumerate(extracted_results):
        if item is None:
            continue
        if "usage" in item:
            usage = item["usage"]
            token_stats["add_memory_calls"] += 1
            token_stats["add_memory_prompt_tokens"] += usage.get("prompt_tokens", 0)
            token_stats["add_memory_completion_tokens"] += usage.get("completion_tokens", 0)
            token_stats["add_memory_total_tokens"] += usage.get("total_tokens", 0)
            logger.info(
                f"[{call_id}] API Call {idx} tokens - "
                f"Prompt: {usage.get('prompt_tokens', 0)}, "
                f"Completion: {usage.get('completion_tokens', 0)}, "
                f"Total: {usage.get('total_tokens', 0)}"
            )
        logger.debug(f"[{call_id}] API Call {idx} raw output: {item.get('output_prompt', 'N/A')}")
        logger.debug(f"[{call_id}] API Call {idx} cleaned result: {item.get('cleaned_result', [])}")
        
        result_dict["add_input_prompt"].append(item.get("input_prompt", []))
        result_dict["add_output_prompt"].append(item.get("output_prompt", ""))
        result_dict["api_call_nums"] += 1

def retrieve_supplementary_entries(
    buffer_entries: List,
    retriever,
    text_embedder,
    top_k: int = 15,
    retrieval_scope: Literal["global", "historical"] = "global",
    additional_filters: Optional[Dict] = None, 
    logger = None
) -> List[Dict]:
    logger.debug(
        f"Retrieving supplementary entries: top_k={top_k}, "
        f"scope={retrieval_scope}"
    )
    buffer_text_parts = []
    for entry in buffer_entries:
        payload = entry["payload"]  
        buffer_text_parts.append(payload["memory"])
    
    aggregated_text = "\n".join(buffer_text_parts)
    query_vector = text_embedder.embed(aggregated_text)
    buffer_ids = [e["id"] for e in buffer_entries]  
    filters = additional_filters.copy() if additional_filters else {}
    
    if "float_time_stamp" not in filters:
        if retrieval_scope == "historical":
            min_timestamp = min(e["payload"]["float_time_stamp"] for e in buffer_entries)  
            filters["float_time_stamp"] = {"lt": min_timestamp}
    
    seed_results = retriever.search(
        query_vector=query_vector,
        limit=top_k,  
        filters=filters if filters else None,
        exclude_ids=buffer_ids,  
        return_full=True
    )
    seed_entries = seed_results
    logger.debug(f"Retrieved {len(seed_entries)} seed entries")
    
    supplementary_entries = []
    seen_ids = set()
    
    for seed in seed_entries:
        if seed["id"] not in seen_ids:  
            supplementary_entries.append(seed)
            seen_ids.add(seed["id"])
            seed_ts = seed["payload"]["time_stamp"] 
            logger.debug(f"[Retrieve] Seed entry found: {seed_ts}")
            
            same_time_entries_raw, _ = retriever.scroll(  
                scroll_filter={"time_stamp": seed_ts},
                limit=1000
            )
            for other in same_time_entries_raw:
                if other.id not in seen_ids and other.id not in buffer_ids:  
                    supplementary_entries.append({
                        "id": other.id, 
                        "payload": dict(other.payload)
                    })
                    seen_ids.add(other.id)
                    logger.debug(f"[Retrieve]   └─ Associated entry added: {other.payload['time_stamp']}")
                    
    supplementary_entries.sort(key=lambda e: e["payload"]["float_time_stamp"])
    logger.debug(
        f"After event reconstruction: {len(supplementary_entries)} entries "
        f"({len(seed_entries)} seeds → {len(supplementary_entries)} total)"
    )

    return supplementary_entries

def format_entries_for_prompt(
    entries: List[Dict],
    include_type_tag: bool = True
) -> str:
    if not entries:
        return ""
    
    lines = []
    for entry in entries:
        payload = entry["payload"]
        speaker = payload.get("speaker_name") or payload.get("speaker_id") or "?"
        timestamp = payload.get("time_stamp", "")
        weekday = payload.get("weekday", "")
        memory = payload.get("memory", "")
        type_tag = ""
        if include_type_tag and payload.get("entry_type"):
            type_tag = f"[{payload['entry_type'].upper()}] "
        time_tag = f"[{timestamp}, {weekday}]" if timestamp and weekday else f"[{timestamp}]"
        lines.append(f"{type_tag}{time_tag} {speaker}: {memory}")
    return "\n".join(lines)

def call_summary_llm(
    manager,
    buffer_text: str,
    supplementary_text: str,
    time_range: str,
    speakers: List[str],
    custom_prompt: Optional[str] = None,  
    token_stats: Dict[str, int] = None,
    logger = None
) -> str:
    from lightmem.memory.prompts import LoCoMo_Cross_Event_Consolidation
    logger.debug("Calling LLM for summary generation")
    speakers_str = ", ".join(sorted(speakers))
    prompt_template = custom_prompt if custom_prompt else LoCoMo_Cross_Event_Consolidation
    
    if logger and custom_prompt:
        logger.debug("Using custom summary prompt")
    elif logger:
        logger.debug("Using default LoCoMo_Cross_Event_Consolidation prompt")
    
    prompt = prompt_template.format(
        bucket=time_range,
        speakers=speakers_str,
        aggregated_text=buffer_text,
        supplementary_context=supplementary_text or "No additional context available."
    )
    
    messages = [
        {
            "role": "system", 
            "content": "You are a professional conversation summarization assistant with temporal awareness."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    response, usage_info = manager.generate_response(messages)
    if token_stats is not None:
        token_stats["summarize_calls"] += 1
        token_stats["summarize_prompt_tokens"] += usage_info.get("prompt_tokens", 0)
        token_stats["summarize_completion_tokens"] += usage_info.get("completion_tokens", 0)
        token_stats["summarize_total_tokens"] += usage_info.get("total_tokens", 0)
    
    if logger:
        logger.debug(
            f"Summary generated: {len(response)} chars, "
            f"tokens: {usage_info.get('total_tokens', 0)}"
        )
    
    return response

def store_summary(
    summary_text: str,
    buffer_entries: List[Dict],
    seed_entries: List[Dict],
    summary_retriever,
    text_embedder,
    logger = None
) -> str:
    summary_id = str(uuid.uuid4())
    logger.debug(f"Storing summary with id: {summary_id}")
    embedding_vector = text_embedder.embed(summary_text)
    payload = {
        "summary": summary_text,
        "time_range": {
            "start": buffer_entries[0]["payload"]["time_stamp"],
            "end": buffer_entries[-1]["payload"]["time_stamp"],
            "start_float": buffer_entries[0]["payload"]["float_time_stamp"],
            "end_float": buffer_entries[-1]["payload"]["float_time_stamp"]
        },
        "covered_entry_ids": [e["id"] for e in buffer_entries],
        "seed_entry_ids": [e["id"] for e in seed_entries] if seed_entries else [],
        "created_at": datetime.now().isoformat(),
        "entry_count": len(buffer_entries),
        "seed_count": len(seed_entries)
    }
    summary_retriever.insert(
        vectors=[embedding_vector],
        payloads=[payload],
        ids=[summary_id]
    )
    logger.debug(
        f"Summary stored: {len(buffer_entries)} buffer entries + "
        f"{len(seed_entries)} seed entries"
    )

    return summary_id

def initialize_time_pointer(retriever, call_id, logger):
    logger.info(f"[{call_id}] Initializing time pointer")
    all_unconsolidated, _ = retriever.scroll(
        scroll_filter={"consolidated": False},
        limit=1000,
        with_payload=True,
        with_vectors=False
    )
    if len(all_unconsolidated) == 0:
        logger.info(f"[{call_id}] No unconsolidated entries")
        return None
    all_unconsolidated.sort(key=lambda x: x.payload["float_time_stamp"])
    earliest = all_unconsolidated[0]
    return earliest.payload["float_time_stamp"] 


def get_window_entries(
    retriever,
    current_time: float,
    time_window: int,
    call_id: str,
    logger = None
) -> Tuple[Optional[List], bool, Optional[float]]:
    end_time = current_time + time_window
    filters = {
        "consolidated": False,
        "float_time_stamp": {"gte": current_time, "lte": end_time}
    }
    
    logger.debug(
        f"[{call_id}] Window: "
        f"{datetime.fromtimestamp(current_time).isoformat()} - "
        f"{datetime.fromtimestamp(end_time).isoformat()}"
    )
    
    Cbuf_raw, _ = retriever.scroll(scroll_filter=filters, limit=10000)
    
    if not Cbuf_raw:
        future_raw, _ = retriever.scroll(
            scroll_filter={"consolidated": False, "float_time_stamp": {"gt": end_time}},
            limit=10000
        )
        
        if future_raw:
            all_futures = [f.payload["float_time_stamp"] for f in future_raw]
            new_time = min(all_futures) 
            logger.debug(f"[{call_id}] Chronologically jumped to {datetime.fromtimestamp(new_time).isoformat()}")
            return None, True, new_time
        else:
            logger.debug(f"[{call_id}] No more data")
            return None, False, None  
    
    Cbuf = [{"id": e.id, "payload": dict(e.payload), "vector": e.vector if hasattr(e, 'vector') else None} for e in Cbuf_raw]
    Cbuf.sort(key=lambda x: x["payload"]["float_time_stamp"])
    return Cbuf, True, None

def mark_entries_and_get_next_time(
    retriever,
    entries: List[Dict],
    call_id: str,
    logger = None
) -> float:
    for entry in entries:
        updated_payload = entry["payload"].copy() 
        updated_payload["consolidated"] = True
        updated_payload["consolidation_time"] = datetime.now().isoformat()
        
        retriever.update(
            vector_id=entry["id"],  
            payload=updated_payload
        )
    
    next_time = entries[-1]["payload"]["float_time_stamp"]  
    if logger:
        logger.debug(
            f"[{call_id}] Time → "
            f"{datetime.fromtimestamp(next_time).isoformat()}"
        )
    
    return next_time

def check_has_more_entries(
    retriever,
    current_time: float
) -> bool:
    remaining, _ = retriever.scroll(  
        scroll_filter={
            "consolidated": False,
            "float_time_stamp": {"gt": current_time}
        },
        limit=1
    )
    return len(remaining) > 0

def build_summary_item(
    summary_text: str,
    summary_id: str,
    buffer_entries: List,
    seed_entries: List
) -> Dict:
    return {
        "summary": summary_text,
        "summary_id": summary_id,
        "time_range": {
            "start": buffer_entries[0]["payload"]["time_stamp"],
            "end": buffer_entries[-1]["payload"]["time_stamp"],
            "start_float": buffer_entries[0]["payload"]["float_time_stamp"],
            "end_float": buffer_entries[-1]["payload"]["float_time_stamp"]
        },
        "entry_count": len(buffer_entries),
        "seed_count": len(seed_entries)
    }


def build_single_result(
    summary_text: str,
    summary_id: str,
    buffer_entries: List,
    seed_entries: List,
    has_more: bool
) -> Dict:
    return {
        "summary": summary_text,
        "covered_entries": [e["id"] for e in buffer_entries],
        "seed_entries": [e["id"] for e in seed_entries],
        "summary_id": summary_id,
        "time_range": {
            "start": buffer_entries[0]["payload"]["time_stamp"],
            "end": buffer_entries[-1]["payload"]["time_stamp"],
            "start_float": buffer_entries[0]["payload"]["float_time_stamp"],
            "end_float": buffer_entries[-1]["payload"]["float_time_stamp"]
        },
        "has_more": has_more
    }


def build_batch_result(
    summaries: List,
    total_entries: int,
    call_id: str,
    logger = None
) -> Dict:
    logger.info(f"[{call_id}] Completed: {len(summaries)} summaries, {total_entries} entries")
    return {
        "summaries": summaries,
        "total_summaries": len(summaries),
        "total_entries": total_entries,
        "time_range": {
            "start": summaries[0]["time_range"]["start"] if summaries else None,
            "end": summaries[-1]["time_range"]["end"] if summaries else None
        }
    }


def build_empty_result(process_all: bool, has_more: bool = False) -> Dict:
    if process_all:
        return {
            "summaries": [],
            "total_summaries": 0,
            "total_entries": 0,
            "time_range": None
        }
    else:
        return {
            "summary": None,
            "covered_entries": [],
            "seed_entries": [],
            "summary_id": None,
            "time_range": None,
            "has_more": has_more
        }
