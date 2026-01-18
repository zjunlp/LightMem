import os
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
import tiktoken
import uuid
from dataclasses import dataclass, field
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer


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
    for msg, sess_time in session_groups:
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
                    if logger:
                        logger.warning(
                            f"LLM returned invalid source_id={original_sid} "
                            f"(valid range: [0, {max_valid_sid}]) in batch {batch_idx}. "
                            f"Auto-corrected to source_id={sid}. "
                            f"Fact: {fact_entry.get('fact', '')[:100]}..."
                        )
                
                seq_candidate = sid * 2
                
                if seq_candidate not in topic_id_map:
                    if logger:
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
        memory=fact_entry.get("fact", ""),
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        topic_id=topic_id,
        topic_summary=topic_summary,
    )
    
    return mem_obj
