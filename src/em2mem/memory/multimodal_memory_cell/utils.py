import json
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from hashlib import md5

@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]

class NerRawOutput(BaseModel):
    named_entities: List[str]

class TripleRawOutput(BaseModel):
    triples: List[List[str]]

@dataclass
class NerOutput:
    chunk_id: str
    unique_entities: List[str]
    metadata: Dict[str, Any]

@dataclass
class TripleOutput:
    chunk_id: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


def compute_mdhash_id(content: str, prefix: Optional[str] = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()

def fix_broken_generated_json(json_str: str) -> str:
    """
    Fixes a malformed JSON string by:
    - Removing the last comma and any trailing content.
    - Iterating over the JSON string once to determine and fix unclosed braces or brackets.
    - Ensuring braces and brackets inside string literals are not considered.

    If the original json_str string can be successfully loaded by json.loads(), will directly return it without any modification.
    
    Args:
        json_str (str): The malformed JSON string to be fixed.

    Returns:
        str: The corrected JSON string.
    """

    def find_unclosed(json_str):
        """
        Identifies the unclosed braces and brackets in the JSON string.

        Args:
            json_str (str): The JSON string to analyze.

        Returns:
            list: A list of unclosed elements in the order they were opened.
        """
        unclosed = []
        inside_string = False
        escape_next = False

        for char in json_str:
            if inside_string:
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"':
                    inside_string = False
            else:
                if char == '"':
                    inside_string = True
                elif char in '{[':
                    unclosed.append(char)
                elif char in '}]':
                    if unclosed and ((char == '}' and unclosed[-1] == '{') or (char == ']' and unclosed[-1] == '[')):
                        unclosed.pop()

        return unclosed

    try:
        # Try to load the JSON to see if it is valid
        json.loads(json_str)
        return json_str  # Return as-is if valid
    except json.JSONDecodeError as e:
        pass

    # Step 1: Remove trailing content after the last comma.
    last_comma_index = json_str.rfind(',')
    if last_comma_index != -1:
        json_str = json_str[:last_comma_index]

    # Step 2: Identify unclosed braces and brackets.
    unclosed_elements = find_unclosed(json_str)

    # Step 3: Append the necessary closing elements in reverse order of opening.
    closing_map = {'{': '}', '[': ']'}
    for open_char in reversed(unclosed_elements):
        json_str += closing_map[open_char]

    return json_str


def filter_invalid_triples(triples: List[List[str]]) -> List[List[str]]:
    """
    Filters out invalid and duplicate triples from a list of triples.

    A valid triple meets the following criteria:
    1. It contains exactly three elements.
    2. It is unique within the list (no duplicates in the output).

    The function ensures:
    - Each valid triple is converted to a list of strings.
    - The order of unique, valid triples is preserved.
    - Do not apply any text preprocessing techniques or rules within this function.
    
    Args:
        triples (List[List[str]]): 
            A list of triples (each a list of strings or elements that can be converted to strings).

    Returns:
        List[List[str]]: 
            A list of unique, valid triples, each represented as a list of strings.
    """
    unique_triples = set()
    valid_triples = []

    for triple in triples:
        if len(triple) != 3: continue  # Skip triples that do not have exactly 3 elements

        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)

    return valid_triples