# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

prompt_template = [
    {
        "role": "system",
        "content": """You are an expert assistant that helps filter and select relevant video captions based on a given query. "
            "Your task is to analyze the retrieved video captions and determine which ones are most relevant to answer the question."""
    },
    {
        "role": "user",
        "content": """You are an expert assistant that helps filter and select relevant video captions based on a given query.
Your task is to analyze the retrieved video captions and determine which ones are most relevant to answer the question.

Given the following question and retrieved video captions, select and rank the most relevant captions that should be used to answer the question.

Instructions:
1. Consider the nature of the question when selecting captions:
  - e.g., for queries about specific events, focus on finer granularities; for habitual, relationship, or general queries, consider coarser granularities.
  - Note that coarser granularity captions may provide broader context, but finer granularity captions often contain more specific details.
2. Each caption shows its time range (start_time to end_time)
3. Analyze each caption for relevance to the question
4. Select captions that directly help answer the question
5. Return the IDs in ranked order (most relevant first)
6. Only include captions that are truly relevant

Return ONLY a JSON array of caption IDs in order of relevance (most relevant first), without additional justification."""
    }
]
