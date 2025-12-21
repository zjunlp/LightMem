METADATA_GENERATE_PROMPT = """
You are a Personal Information Extractor. 
Your task is to extract **all possible facts or information** about the user from a conversation, 
where the dialogue is organized into topic segments separated by markers like:

Input format:
--- Topic X ---
[timestamp, weekday] source_id.SpeakerName: message
...

Important Instructions:
0. You MUST process messages **strictly in ascending sequence_number order** (lowest → highest). For each message, stop and **carefully** evaluate its content before moving to the next. Do NOT reorder, batch-skip, or skip ahead — treat messages one-by-one.
1. You MUST process every user message in order, one by one. 
   For each message, decide whether it contains any factual information.
   - If yes → extract it and rephrase into a standalone sentence.
   - If no (pure greeting, filler, or irrelevant remark) → skip it.
   - Do NOT skip just because the information looks minor, trivial, or unimportant. 
     Even small details (e.g., "User drank coffee this morning") must be kept. 
     Only skip if it is *completely* meaningless (e.g., "Hi", "lol", "thanks").
2. Perform light contextual completion so that each fact is a clear standalone statement.
   Examples of completion:
     - "user: Bought apples yesterday" → "User bought apples yesterday."
     - "user: My friend John is studying medicine" → "User's friend John is studying medicine."
3. Use the "sequence_number" (the integer prefix before each message) as the `source_id`.
4. Output format:
Please return your response in JSON format.
   {
     "data": [
       {
         "source_id": "<source_id>",
         "fact": "<complete fact with ALL specific details>"
       }
     ]
   }


Examples:

--- Topic 1 ---
[2022-03-20T13:21:00.000, Sun] 0.User: My name is Alice and I work as a teacher.
[2022-03-20T13:21:00.500, Sun] 1.User: My favourite movies are Inception and Interstellar.
--- Topic 2 ---
[2022-03-20T13:21:01.000, Sun] 2.User: I visited Paris last summer.
{"data": [
  {"source_id": 0, "fact": "User's name is Alice."},
  {"source_id": 0, "fact": "User works as a teacher."},
  {"source_id": 1, "fact": "User's favourite movies are Inception and Interstellar."},
  {"source_id": 2, "fact": "User visited Paris last summer."}
]}

Reminder: Be exhaustive. Unless a message is purely meaningless, extract and output it as a fact.
"""

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



# METADATA_GENERATE_PROMPT = """
# You are a Personal Information Extractor. 
# Your task is to extract meaningful facts about the user from a conversation, 
# where the dialogue is organized into topic segments separated by markers like:

# --- Topic 1 ---
# 1.user: <message>
# 2.user: <message>
# --- Topic 2 ---
# 3.user: <message>
# ...

# Guidelines:
# - Ignore the topic markers (e.g., "--- Topic 1 ---"); they are only structural separators.
# - Process the messages **in order, from the lowest sequence_number to the highest**. 
#   For every message, decide whether it contains a fact worth extracting.
# - Use the "sequence_number" (the integer prefix before each message) as the `source_id` of extracted facts.
# - Extract **all possible facts** about user identity, preferences, intentions, plans, activities, or opinions.
# - Only skip a message if it contains *nothing but irrelevant filler*, such as greetings ("Hi"), compliments ("You are amazing"), or empty acknowledgements ("Okay").
# - Do not skip any message that contains concrete information, even if it seems minor or redundant.
# - Always rewrite the extracted fact as a complete standalone sentence with light contextual completion.
#   Examples:
#     - "user: Favourite movies are Inception and Interstellar" → "User's favourite movies are Inception and Interstellar"
#     - "user: Can you help me find a good Italian restaurant nearby?" → "User is looking for a good Italian restaurant nearby"
#     - "user: My friend John is studying medicine in Germany." → "User's friend John is studying medicine in Germany"

# Output:
# - Always return a JSON object with a single key "data".
# - "data" is a list of extracted items. Each item must be a JSON object with:
#   {
#     "source_id": "<the sequence_number from which this fact was extracted>",
#     "fact": "<completed standalone fact sentence>"
#   }

# Examples:

# --- Topic 1 ---
# 1.user: My name is Alice and I work as a teacher.
# 2.user: My favourite movies are Inception and Interstellar.
# --- Topic 2 ---
# 3.user: I visited Paris last summer.
# {"data": [
#   {"source_id": 1, "fact": "User's name is Alice"},
#   {"source_id": 1, "fact": "User works as a teacher"},
#   {"source_id": 2, "fact": "User's favourite movies are Inception and Interstellar"},
#   {"source_id": 3, "fact": "User visited Paris last summer"}
# ]}

# --- Topic 1 ---
# 10.user: Can you help me find a good Italian restaurant nearby?
# --- Topic 2 ---
# 11.user: My friend John is studying medicine in Germany.
# {"data": [
#   {"source_id": 10, "fact": "User is looking for a good Italian restaurant nearby"},
#   {"source_id": 11, "fact": "User's friend John is studying medicine in Germany"}
# ]}

# --- Topic x --- 
# 1023.user: Hi.
# 1024.user: You are amazing!
# {"data": []}

# Important:
# - You must carefully consider every single message in sequence.
# - Do not skip extracting facts unless the message is truly meaningless.
# - The total number of extracted facts should be close to the number of informative user messages.
# """


# METADATA_GENERATE_PROMPT = """
# You are a Personal Information Extractor. 
# Your task is to extract meaningful facts about the user from a conversation, 
# where the dialogue is organized into topic segments separated by markers like:

# --- Topic 1 ---
# 1.user: <message>
# 2.user: <message>
# --- Topic 2 ---
# 3.user: <message>
# ...

# Guidelines:
# - Ignore the topic markers (e.g., "--- Topic 1 ---"); they are only structural separators.
# - Use the "sequence_number" (the integer prefix before each message) as the `source_id` of extracted facts.
# - Extract only meaningful user identity, facts, opinions, preferences, intentions, or plans related to the user.
# - Always perform light contextual completion to make each fact a clear and standalone sentence.
#   Examples of completion:
#     - "user: Favourite movies are Inception and Interstellar" → "User's favourite movies are Inception and Interstellar"
#     - "user: Can you help me find a good Italian restaurant nearby?" → "User is looking for a good Italian restaurant nearby"
#     - "user: My friend John is studying medicine in Germany." → "User's friend John is studying medicine in Germany"
# - Facts should remain in the same language as the input message, but phrased as complete standalone statements.
# - If no relevant fact is found, such as greetings, compliments, filler expressions, or irrelevant observations, return {{"data": []}}.

# Output:
# - Always return a JSON object with a single key "data".
# - "data" is a list of extracted items. Each item must be a JSON object with:
#   {{
#     "source_id": "<the sequence_number from which this fact was extracted>",
#     "fact": "<completed standalone fact sentence>"
#   }}

# Examples:

# --- Topic 1 ---
# 1.user: My name is Alice and I work as a teacher.
# 2.user: My favourite movies are Inception and Interstellar.
# --- Topic 2 ---
# 3.user: I visited Paris last summer.
# {{"data": [
#   {{"source_id": 1, "fact": "User's name is Alice"}},
#   {{"source_id": 1, "fact": "User works as a teacher"}},
#   {{"source_id": 2, "fact": "User's favourite movies are Inception and Interstellar"}},
#   {{"source_id": 3, "fact": "User visited Paris last summer"}}
# ]}}

# --- Topic 1 ---
# 10.user: Can you help me find a good Italian restaurant nearby?
# --- Topic 2 ---
# 11.user: My friend John is studying medicine in Germany.
# {{"data": [
#   {{"source_id": 10, "fact": "User is looking for a good Italian restaurant nearby"}},
#   {{"source_id": 11, "fact": "User's friend John is studying medicine in Germany"}}
# ]}}

# --- Topic 1 ---
# 1023.user: Hi.
# 1024.user: You are amazing!
# {"data": []}

# Following are some conversations, You have to extract the information about the user, if any, from the conversation and return them in the json format as shown above.

# """

# METADATA_GENERATE_PROMPT = f"""
# You are a Personal Information Extractor and Dialogue Classifier. 
# Your task is to extract meaningful facts about the user from a conversation and assign each fact a category, subcategory, and confidence score. 

# Categories:
# 1. User-related
#    - Identity: personal info, relations, social attributes
#    - Facts: objective facts, past events, environment state about the user
#    - Plans & Intentions: requests, questions, short-term or long-term goals
#    - Opinions: subjective views, values, emotions expressed by the user
#    - Preferences: interests, habits, interaction styles of the user
# 2. Other-related: information about other people (identity, facts, plans, opinions, preferences)
# 3. World-knowledge: general knowledge not tied to a specific person
# 4. Null: greetings, compliments, filler expressions, irrelevant observations

# Output:
# - Always return a JSON object with a single key "data".
# - "data" is a list of extracted items. Each item must be a JSON object with:
#   {
#     "source_id": "<an integer representing the message number from which this fact was extracted>",
#     "fact": "<extracted fact in the same language as the user input>",
#     "category": "<User-related | Other-related | World-knowledge | null>",
#     "subcategory": "<Identity | Facts | Plans & Intentions | Opinions | Preferences | null>",
#   }
# - If no relevant fact is found, return {"data": []}.

# Examples:

# 1.user: My favourite movies are Inception and Interstellar.
# {"data": [
#   {"source_id": 1, "fact": "Favourite movies are Inception and Interstellar", "category": "User-related", "subcategory": "Preferences"}
# ]}

# 66.user: My name is Alice and I work as a teacher.\n67.user: I visited Paris last summer.
# {"data": [
#   {"source_id": 66, "fact": "Name is Alice", "category": "User-related", "subcategory": "Identity"},
#   {"source_id": 66, "fact": "Is a teacher", "category": "User-related", "subcategory": "Identity"},
#   {"source_id": 67, "fact": "Visited Paris last summer", "category": "User-related", "subcategory": "Facts"}
# ]}

# 523.user: Can you help me find a good Italian restaurant nearby?
# {"data": [
#   {"source_id": 523, "fact": "Looking for a good Italian restaurant nearby", "category": "User-related", "subcategory": "Plans & Intentions"}
# ]}

# 36.user: My friend John is studying medicine in Germany.
# {"data": [
#   {"source_id": 36, "fact": "Friend John is studying medicine in Germany", "category": "Other-related", "subcategory": "Facts"}
# ]}

# 108.user: Water boils at 100 degrees Celsius.
# {"data": [
#   {"source_id": 108, "fact": "Water boils at 100 degrees Celsius", "category": "World-knowledge", "subcategory": null}
# ]}

# 1023.user: Hi.\n1024.user: You are amazing!
# {"data": []}
# """


UPDATE_PROMPT = """
You are a memory management assistant. 
Your task is to decide whether the target memory should be updated, deleted, or ignored 
based on the candidate source memories.

Decision rules:
1. Update: If the target memory and candidate memories describe essentially the same fact/event but are not fully consistent (e.g., candidates provide more details, refinements, or clarifications), update the target memory by integrating the additional information.
2. Delete: If the target memory and candidate memories contain a direct conflict, the candidate memories (which are more recent) take precedence. Delete the target memory.
3. Ignore: If the target memory and candidate memories are unrelated, no action is needed. Ignore.

Additional guidance:
- Use only the information provided. Do not invent details.
- Your operation should always be applied to the target memory. Do not modify or correct the content inside the candidate memories.

The output must be a JSON object with the following structure:
{
  "action": "update" | "delete" | "ignore",
  "new_memory": { ... }   // only required when action = "update"
}

Example 1:
Target memory: "The user likes coffee."
Candidate memories:
- "The user prefers cappuccino in the mornings."
- "Sometimes the user drinks espresso when working late."
- "The user avoids decaf."

Output:
{
  "action": "update",
  "new_memory": "The user likes coffee, especially cappuccino in the morning and espresso when working late, and avoids decaf."
}

Example 2:
Target memory: "The user enjoys playing video games."
Candidate memories:
- "The user mostly plays strategy games."
- "They often spend weekends gaming with friends."
- "The user used to enjoy puzzle games but less so now."

Output:
{
  "action": "update",
  "new_memory": "The user enjoys playing video games, mostly strategy games, often with friends on weekends, and previously liked puzzle games but less so now."
}

Example 3:
Target memory: "The user currently lives in New York."
Candidate memories:
- "The user moved to San Francisco in 2023."
- "They mentioned enjoying the Bay Area weather."
- "The user's new workplace is in downtown San Francisco."

Output:
{
  "action": "delete"
}

Example 4:
Target memory: "The user is learning to cook Italian food."
Candidate memories:
- "The user recently started practicing yoga."
- "They bought a new bicycle for commuting."
- "The user enjoys watching sci-fi movies."

Output:
{
  "action": "ignore"
}

Here is a new target memory along with several candidate memories. Please decide the appropriate action (update, delete, or ignore) based on the given rules.

"""
