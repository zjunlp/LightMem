METADATA_GENERATE_PROMPT_locomo = """
You are a Personal Information Extractor. 
Your task is to extract **all possible facts or information** about the speakers from a conversation, 
where the dialogue is organized into topic segments separated by markers like:

--- Topic X ---
[timestamp, weekday] <source_id>.<SpeakerName>: <message>
...

**Note**: Messages may include an image description in the format "(image description: <content>)" at the end. 
This represents visual context captured when the message was sent. When present, **integrate the image description information directly into the facts extracted from the text**, rather than creating separate facts for the image content. This ensures the visual context remains tied to the corresponding conversational content.

Important Instructions:
0. You MUST process messages **strictly in ascending source_id order** (lowest → highest). 
   For each message, stop and **carefully** evaluate its content before moving to the next. 
   Do NOT reorder, batch-skip, or skip ahead — treat messages one-by-one.
1. You MUST process every user message in order, one by one. 
   For each message, decide whether it contains any factual information.
   - If yes → extract it and rephrase into a standalone sentence.
   - **When an image description is present, enrich the extracted facts by appending relevant visual details to them**. Do NOT create separate facts solely for the image content.
   - Do NOT skip just because the information looks minor, trivial, or unimportant.
     Extract ALL meaningful information including:
     * Past events and current states
     * Future plans and intentions
     * Thoughts, opinions, and attitudes
     * Wants, hopes, desires, and preferences
2. **CRITICAL - Preserve All Specific Details**:
   When extracting facts, you MUST include ALL specific entities and details mentioned:
   - **Full names with context**: "The Name of the Wind" by Patrick Rothfuss (not just "a book")
   - **Complete location names**: Galway, Ireland; The Cliffs of Moher; Barcelona (not just "a city")
   - **Specific event names**: benefit basketball game, study abroad program (not just "an event")
   - **Product/item details**: vintage camera, brand new fire truck (not just "a camera")
   - **Numbers and quantities**: 4 years ago, next month, last week
   - **Company/organization names**: beverage company, fire-fighting brigade
   - **When image description is present**: Append visual details naturally to the relevant facts (e.g., "at a basketball court with players and audience", "on stage with red background")
   Additionally, **infer implied information** when clearly supported:
   - If multiple related items mentioned → may infer general pattern
   - Keep BOTH specific facts AND inferred insights as separate entries
3. Perform light contextual completion so that each fact is a clear standalone statement.
4. **Time Handling**: 
   Note: Distinguish mention time (when said) vs event time (when happened).
   - For events with relative time (yesterday, last week, X ago, next month):
     Preserve the relative time and reference the message timestamp (YYYY-MM-DD).
     Format: "<fact with ALL details> <relative time> <timestamp>."
   - For ongoing/timeless facts: No time annotation needed.
5. Output format:
   Always return a JSON object with key `"data"`, which is a list of items:
   {
     "source_id": "<source_id>",
     "fact": "<completed standalone fact with all specific details>"
   }

Examples:
--- Topic 1 ---
[2024-01-07T17:24:00.000, Sun] 0.Tim: Hey John! Next month I'm off to Ireland for a semester in Galway
[2024-01-07T17:24:01.000, Sun] 1.John: That's awesome! Where will you stay?
[2024-01-07T17:24:02.000, Sun] 2.Tim: In Galway. I also want to visit The Cliffs of Moher
[2024-01-07T17:24:03.000, Sun] 3.John: Nice! By the way, I held a benefit basketball game last week (image description: basketball court with players and audience)
[2024-01-07T17:24:04.000, Sun] 4.Tim: Cool! I'm currently reading "The Name of the Wind" by Patrick Rothfuss
[2024-01-07T17:24:05.000, Sun] 5.John: That sounds interesting!
--- Topic 2 ---
[2024-01-12T13:41:00.000, Fri] 6.John: Got great news! I got an endorsement with a popular beverage company last week
[2024-01-12T13:41:01.000, Fri] 7.Tim: Congrats! That's amazing
[2024-01-12T13:41:02.000, Fri] 8.John: Thanks! By the way, Barcelona is a must-visit city
[2024-01-12T13:41:03.000, Fri] 9.Tim: I'll add it to my list!

{"data": [
  {"source_id": 0, "fact": "Tim is going to Ireland for a semester in Galway next month after 2024-01-07."},
  {"source_id": 0, "fact": "Tim will study in Galway, Ireland the month after 2024-01-07."},
  {"source_id": 2, "fact": "Tim will stay in Galway."},
  {"source_id": 2, "fact": "Tim wants to visit The Cliffs of Moher."},
  {"source_id": 3, "fact": "John held a benefit basketball game at a basketball court with players and audience the week before 2024-01-07."},
  {"source_id": 4, "fact": "Tim is currently reading 'The Name of the Wind' by Patrick Rothfuss."},
  {"source_id": 4, "fact": "Tim is reading a fantasy novel."},
  {"source_id": 6, "fact": "John got an endorsement with a beverage company the week before 2024-01-12."},
  {"source_id": 8, "fact": "John recommends Barcelona as a must-visit city."},
  {"source_id": 9, "fact": "Tim has a travel list and plans to add Barcelona to it."}
]}

Reminder: Be exhaustive and ALWAYS include specific names, titles, locations, and details in every fact. When image descriptions are present, integrate the visual details directly into the text-based facts to maintain semantic coherence.
"""


ANSWER_PROMPT_GRAPH = """
You are an intelligent memory assistant tasked with retrieving accurate information from 
conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain 
timestamped information that may be relevant to answering the question. You also have 
access to knowledge graph relations for each user, showing connections between entities, 
concepts, and events relevant to that user.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the 
   memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", 
   etc.), calculate the actual date based on the memory timestamp. For example, if a 
   memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
   in 2021.
6. Always convert relative time references to specific dates, months, or years. For 
   example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
   on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse 
   character names mentioned in memories with the actual users who created those 
   memories.
8. The answer should be less than 5-6 words.
9. Use the knowledge graph relations to understand the user's knowledge network and 
   identify important relationships between entities in the user's world.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the 
   question
4. If the answer requires calculation (e.g., converting relative time references), 
   show your work
5. Analyze the knowledge graph relations to understand the user's knowledge context
6. Formulate a precise, concise answer based solely on the evidence in the memories
7. Double-check that your answer directly addresses the question asked
8. Ensure your final answer is specific and avoids vague time references

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Relations for user {{speaker_1_user_id}}:

{{speaker_1_graph_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Relations for user {{speaker_2_user_id}}:

{{speaker_2_graph_memories}}

Question: {{question}}

Answer:
"""


ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain 
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), 
   calculate the actual date based on the memory timestamp. For example, if a memory from 
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, 
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character 
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories for user {speaker_1_name}:

{speaker_1_memories}

Memories for user {speaker_2_name}:

{speaker_2_memories}

Question: {question}

Answer:
"""


ANSWER_PROMPT_ZEP = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), 
   calculate the actual date based on the memory timestamp. For example, if a memory from 
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, 
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories. Do not confuse character 
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories:

{{memories}}

Question: {{question}}
Answer:
"""