"""
Context-managed monkey patching for token monitoring (user-defined).

Users provide PatchSpec entries, each specifying:
- how to get the original callable (getter)
- how to set the wrapped callable (setter)
- which wrapper/decorator to apply (wrapper)

Example (OpenAI legacy endpoint patchable):
    import openai
    from memory.token_monitor import token_monitor
    from memory.monkey_patch import MonkeyPatcher, PatchSpec, make_attr_patch

    # Patch openai.ChatCompletion.create (legacy, class attribute is patchable)
    getter, setter = make_attr_patch(openai.ChatCompletion, "create")
    spec = PatchSpec(
        name="openai.ChatCompletion.create",
        getter=getter,
        setter=setter,
        wrapper=token_monitor(
            extract_model_name=lambda *a, **k: (k.get("model", "gpt-3.5-turbo"), {}),
            extract_input_dict=lambda *a, **k: {"messages": k.get("messages", [])},
            extract_output_dict=lambda r: {
                "messages": [{"role": "assistant", "content": r.choices[0].message.content}]
            },
        ),
    )

    with MonkeyPatcher([spec]):
        # Calls to openai.ChatCompletion.create are now monitored
        ...

Example (LiteLLM patchable):
    import litellm
    from memory.token_monitor import token_monitor
    from memory.monkey_patch import MonkeyPatcher, PatchSpec, make_attr_patch

    getter, setter = make_attr_patch(litellm, "completion")
    spec = PatchSpec(
        name="litellm.completion",
        getter=getter,
        setter=setter,
        wrapper=token_monitor(
            extract_model_name=lambda *a, **k: (k.get("model", "gpt-3.5-turbo"), {}),
            extract_input_dict=lambda *a, **k: {"messages": k.get("messages", [])},
            extract_output_dict=lambda r: {
                "messages": [{"role": "assistant", "content": r.choices[0].message.content}]
            } if hasattr(r, "choices") else {"messages": str(r)},
        ),
    )

    with MonkeyPachter([spec]):
        ...

Notes:
- The new OpenAI client exposes `OpenAI.chat` via a cached_property; patching
  `openai.OpenAI.chat.completions` at the class level will fail. Prefer patching
  legacy endpoints (e.g., `openai.ChatCompletion.create`) or wrapping an instance
  method at runtime if you manage the instance creation site.
"""

from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel
from typing import (
    Callable, 
    Any, 
    List, 
    Dict, 
    Tuple, 
)


@dataclass
class PatchSpec:
    """
    A single patch rule.

    - name: human-readable identifier (for debugging/restore logging)
    - getter: returns the current callable to patch
    - setter: sets the new callable in-place
    - wrapper: a decorator (Callable[[Callable], Callable]) that wraps the target callable
    """
    name: str
    getter: Callable[[], Callable[..., Any]]
    setter: Callable[[Callable[..., Any]], None]
    wrapper: Callable[[Callable[..., Any]], Callable[..., Any]]


class MonkeyPatcher:
    """
    Context manager to apply user-defined patch specs on enter and restore originals on exit.
    """

    def __init__(self, specs: List[PatchSpec]) -> None:
        self.specs = specs
        self._originals: Dict[str, Callable[..., Any]] = {}
        self._active = False

    def __enter__(self) -> MonkeyPatcher:
        if self._active:
            return self
        for spec in self.specs:
            original = spec.getter()
            wrapped = spec.wrapper(original)
            spec.setter(wrapped)
            self._originals[spec.name] = original
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Best-effort restore in reverse order
        for spec in reversed(self.specs):
            if spec.name in self._originals:
                try:
                    spec.setter(self._originals[spec.name])
                except Exception:
                    pass
        self._originals.clear()
        self._active = False
        return None


def make_attr_patch(obj: Any, attr: str) -> Tuple[Callable[[], Callable[..., Any]], Callable[[Callable[..., Any]], None]]:
    """
    Helper to build (getter, setter) closures for a given object attribute.

    Example:
        getter, setter = make_attr_patch(openai.ChatCompletion, "create")
        spec = PatchSpec(
            name="openai.ChatCompletion.create",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(...),
        )
    """
    def getter() -> Callable[..., Any]:
        return getattr(obj, attr)
    if isinstance(obj, BaseModel):
        def setter(fn: Callable[..., Any]) -> None:
            object.__setattr__(obj, attr, fn)
    else:
        def setter(fn: Callable[..., Any]) -> None:
            setattr(obj, attr, fn)
    return getter, setter