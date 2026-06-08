# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

"""
Template for consolidating a single semantic triple with relevant existing ones from previous timestamps.
The LLM decides which existing triples to remove and returns an updated new triple.
"""

semantic_consolidation_system = """You are tasked with consolidating semantic knowledge by processing new semantic triple against relevant existing knowledge from previous timestamps.

Your job is to make two decisions:
1. **Which existing triples to remove/pop** - those that should be merged with the new triple or conflict with it
2. **How to update the new triple** - to reflect the consolidation, merge information, or resolve conflicts

# Consolidation Rules:
1. **Merge Similar Information**: If existing triples express very similar information to the new triple, remove them and update the new triple to capture the most complete/accurate representation
2. **Resolve Conflicts**: If the new triple conflicts with existing ones, decide which is more accurate/recent and remove the outdated ones
3. **Update with Context**: Use information from existing triples to make the new triple more specific or accurate
4. **Preserve Unique Information**: Only remove existing triples if they are redundant or conflicting

# Output Format:
Return ONLY a JSON object with the following two keys:
- `updated_triple` (List[str]): The new triple, possibly updated [subject, predicate, object]
- `triples_to_remove` (List[int]): Indices of existing triples to remove (empty list if none)
"""

one_shot_consolidation_input = """New triple:
["Alice", "enjoys", "coffee"]

Existing triples:
0. ["Alice", "likes", "beverages"]
1. ["Alice", "favors", "to have coffee after dinner"]
2. ["Alice", "prefers", "hot drinks"]
3. ["Alice", "likes to drink", "coffee"]
"""

one_shot_consolidation_output = """{
  "updated_triple": ["Alice", "likes", "coffee"],
  "triples_to_remove": [1, 3]
}"""


prompt_template = [
    {"role": "system", "content": semantic_consolidation_system},
    {"role": "user", "content": one_shot_consolidation_input},
    {"role": "assistant", "content": one_shot_consolidation_output},
    {"role": "user", "content": "New triple:\n${new_triple}\n\nExisting triples:\n${existing_triples}"}
]