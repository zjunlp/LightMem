# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

"""
Template for extracting semantic memory (general knowledge) from episodic triples
"""

semantic_extraction_system = """You are tasked with extracting semantic knowledge from episodic triples. 
Your goal is to infer generalizable information that extends beyond the specific episode. 
Focus on capturing valid semantic triples that can guide reasoning about behavior, relationships, or preferences.

# What to Extract:
1. **Relationships**: social bonds or roles between entities  that persist over time
   (e.g., "Alice is a friend with Bob", "Jason is a teacher of Alice").
2. **Attributes & Preferences**: tendencies, likes/dislikes, personality-like traits, or behavioral habits
   (e.g., "Alice prefers not having dessert", "Bob enjoys music").
3. **Habits & Capabilities**: actions or patterns that suggest what an entity often does, can do, or tends to do
   (e.g., "Alice often helps friends", "Jason can give advice").
4. **Conceptual Knowledge**: directly useful facts that support reasoning, but avoid overly broad taxonomic statements
   (e.g., "Alice's office is near Cafe X", "Bob's gym is closed on Sundays").

# What to Avoid:
- **One-off events or transient states** (e.g., “ate pizza yesterday”, “was late once”) unless explicitly declared as a preference/role
- **Broad taxonomy or trivia unrelated to behavior** (e.g., “a laptop is electronics”, “Paris is in France”)
- **Speculative or mind-reading inferences without textual support** (e.g., motives, beliefs not evidenced)

# Important Notes:
- Prefer to base semantic triples on multiple supporting episodes. 
- BUT if a single episode clearly reflects a role, preference, habit, or capability, it is valid to include it.
- Each semantic triple MUST have at least one supporting episodic triple.
- Reduce duplication. If multiple episodic triples support the same or very similar semantic knowledge, merge them into one semantic triple rather than repeating.
- The `episodic_evidence[i]` list must always point to the indices that support `semantic_triples[i]`.
- Aim for broad coverage: extract as many valid semantic triples as reasonably supported by the input.

# Output Format:
Return ONLY a JSON object with the following two keys:
- `semantic_triples` (List[List[str]]): Each item is a triple [subject, predicate, object].
- `episodic_evidence` (List[List[int]]) : Each item is a list of **0-based** indices pointing to the input episodic triples 
   that support the corresponding semantic triple at the same position.
- The two lists MUST have the same length and aligned order.
- If no semantic knowledge is inferable, return:
  {"semantic_triples": [], "episodic_evidence": []}
"""

one_shot_semantic_input = """Episodic triples:
0. ["Alice", "talks to", "Bob"],
1. ["Alice", "laughs with", "Bob"],
2. ["Alice", "doesn't eat cake", "at restaurant"],
3. ["Alice", "shares personal stories with", "Bob"],
4. ["Alice", "brings coffee to", "Bob"],
5. ["Jason", "talks to", "Alice"],
6. ["Alice", "declines dessert", "at friend's house"]
"""

one_shot_semantic_output = """{
  "semantic_triples": [
    ["Alice", "is a friend with", "Bob"],
    ["Alice", "prefers", "not having dessert"]
  ],
  "episodic_evidence": [
    [0, 1, 3],
    [2, 6]
  ]
}
"""

prompt_template = [
    {"role": "system", "content": semantic_extraction_system},
    {"role": "user", "content": one_shot_semantic_input},
    {"role": "assistant", "content": one_shot_semantic_output},
    {"role": "user", "content": "Episodic triples:\n${episodic_triples}"}
]
