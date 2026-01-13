# locomo_preprocessor.py

from typing import Dict, Any
from memories.datasets.base import Message, Session

"""Preprocessing function for the LoCoMo dataset."""
def NaiveRAG_style_message_for_LoCoMo(
    message: Message,
    session: Session,
) -> Dict[str, Any]:
    # Get role
    role = message.role

    # Retrieve `name` from metadata
    name = message.metadata.get("name", "unknown")

    # Since this function is specific to LoCoMo, we only consider user roles and ignore system message handling
    # Append image caption to the content if available
    caption = message.metadata.get("blip_caption", None)
    if caption:
        body = f"{message.content} (image caption: {caption})"
    else:
        body = message.content

    # For NaiveRAG, include the timestamp
    session_timestamp = session.get_string_timestamp()
    # Final content
    content = (
        f"Time: {session_timestamp}\n\n"
        f"{body}"
    )

    return {
        "name": name,
        "role": role,
        "content": content,
    }