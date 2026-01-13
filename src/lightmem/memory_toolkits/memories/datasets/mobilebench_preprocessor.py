# mobilebench_preprocessor.py

from typing import Dict, Any
from memories.datasets.base import Message, Session

"""Preprocessing function for the MobileBench dataset."""
def style_message_for_MobileBench(
    message: Message,
    session: Session,
) -> Dict[str, Any]:
    # Get role
    role = message.role

    # Retrieve `name` from metadata
    name = message.metadata.get("name", "unknown")

    # Decide Human/System (example strategy):
    #   - If role is "user" -> Human Message
    #   - Otherwise -> System Message (e.g., Calendar/Search/Agent)
    if role.lower() == "user":
        header = "Human Message"
    else:
        header = "System Message"
    
    message_timestamp = message.timestamp
    content = (
        f"===== {header} =====\n"
        f"Name: {name}\n"
        f"Time: {message_timestamp}\n\n"
        f"{message.content}"
    )

    return {
        "role": role,
        "content": content,
    }