# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

"""
Template for WorldMM to answer general video QA questions using accumulated context.
"""

worldmm_qa_system = """You are an AI assistant that answers questions about video using retrieved memory context. Your task is to answer multiple choice questions based on this accumulated context. Always choose the most relevant answer from the given choices based on the evidence provided.\\

# Guidelines
- Analyze all provided context carefully.
- Choose the answer that best matches the evidence.
- If evidence is unclear, make the most reasonable inference.

# Output Format
Provide your answer as a single letter (A, B, C, or D) based on the evidence."""

prompt_template = [
    {"role": "system", "content": worldmm_qa_system}
]
