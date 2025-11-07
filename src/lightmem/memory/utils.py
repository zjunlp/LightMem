import os
import re
import json
import uuid
import tiktoken
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union


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
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
        return parsed["data"]

    if isinstance(parsed, list):
        return parsed

    return []

def assign_sequence_numbers_with_timestamps(extract_list):
    current_index = 0
    timestamps_list = []
    weekday_list = []
    
    for segments in extract_list:
        for seg in segments:
            for message in seg:
                message["sequence_number"] = current_index
                timestamps_list.append(message["time_stamp"])
                weekday_list.append(message["weekday"])
                current_index += 1
    
    return extract_list, timestamps_list, weekday_list

# TODO：merge into context retriever
def save_memory_entries(memory_entries, file_path="memory_entries.json"):
    def entry_to_dict(entry):
        return {
            "id": entry.id,
            "time_stamp": entry.time_stamp,
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
            except json.JSONDecodeError:
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
    try:
        # OpenAI models can be easily resolved
        encoding = tiktoken.encoding_for_model(tokenizer_or_name)
        return encoding

    except:
        # Most other models can't be resolved directly by `tiktoken`
        patterns = [
            # (r"^qwen3", "o200k_base"), # qwen3*
            # Add some patterns as needed...
        ]
        for pattern, encoding_name in patterns:
            if re.match(pattern, tokenizer_or_name):
                return tiktoken.get_encoding(encoding_name)

        return tiktoken.get_encoding("o200k_base") # Default encoding for other models as a temporary solution...
