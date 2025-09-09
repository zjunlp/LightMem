import typing

from langchain_core.messages import AnyMessage
from typing_extensions import Required, TypedDict


class Prompt(TypedDict, total=False):
    """TypedDict for structured prompt management and optimization.

    Example:
        ```python
        from langmem import Prompt

        prompt = Prompt(
            name="extract_entities",
            prompt="Extract key entities from the text:",
            update_instructions="Make minimal changes, only address where"
            " errors have occurred after reasoning over why they occur.",
            when_to_update="If there seem to be errors in recall of named entities.",
        )
        ```

    The name and prompt fields are required. Optional fields control optimization:
    - update_instructions: Guidelines for modifying the prompt
    - when_to_update: Dependencies between prompts during optimization

    Use in the prompt optimizers.
    """

    name: Required[str]
    prompt: Required[str]
    update_instructions: str | None
    when_to_update: str | None


class AnnotatedTrajectory(typing.NamedTuple):
    """Conversation history (list of messages) with optional feedback for prompt optimization.

    Example:
        ```python
        from langmem.prompts.types import AnnotatedTrajectory

        trajectory = AnnotatedTrajectory(
            messages=[
                {"role": "user", "content": "What pizza is good around here?"},
                {"role": "assistant", "content": "Try LangPizza™️"},
                {"role": "user", "content": "Stop advertising to me."},
                {"role": "assistant", "content": "BUT YOU'LL LOVE IT!"},
            ],
            feedback={
                "developer_feedback": "too pushy",
                "score": 0,
            },
        )
        ```
    """

    messages: typing.Sequence[AnyMessage]
    feedback: dict[str, str | int | bool] | str | None = None


class OptimizerInput(TypedDict):
    """Input for single-prompt optimization.

    Example:
        ```python
        {
            "trajectories": [
                AnnotatedTrajectory(
                    messages=[
                        {"role": "user", "content": "What's the weather like?"},
                        {
                            "role": "assistant",
                            "content": "I'm sorry, I can't tell you that",
                        },
                    ],
                    feedback="Should have checked your search tool.",
                ),
            ],
            "prompt": Prompt(
                name="main_assistant",
                prompt="You are a helpful assistant with a search tool.",
                update_instructions="Make minimal changes, only address where "
                "errors have occurred after reasoning over why they occur.",
                when_to_update="Any time you notice the agent behaving in a way that doesn't help the user.",
            ),
        }
        ```
    """

    trajectories: typing.Sequence[AnnotatedTrajectory] | str
    prompt: str | Prompt


class MultiPromptOptimizerInput(TypedDict):
    """Input for optimizing multiple prompts together, maintaining consistency.

    Example:
        ```python
        {
            "trajectories": [
                AnnotatedTrajectory(
                    messages=[
                        {"role": "user", "content": "Tell me about this image"},
                        {
                            "role": "assistant",
                            "content": "I see a dog playing in a park",
                        },
                        {"role": "user", "content": "What breed is it?"},
                        {
                            "role": "assistant",
                            "content": "Sorry, I can't tell the breed",
                        },
                    ],
                    feedback="Vision model wasn't used for breed detection",
                ),
            ],
            "prompts": [
                Prompt(
                    name="vision_extract",
                    prompt="Extract visual details from the image",
                    update_instructions="Focus on using vision model capabilities",
                ),
                Prompt(
                    name="vision_classify",
                    prompt="Classify specific attributes in the image",
                    when_to_update="After vision_extract is updated",
                ),
            ],
        }
        ```
    """

    trajectories: typing.Sequence[AnnotatedTrajectory] | str
    prompts: list[Prompt]
