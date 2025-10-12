from __future__ import annotations
from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
    field_serializer,
    ConfigDict,
)
from abc import ABC, abstractmethod
from types import MappingProxyType
from datetime import datetime
from functools import total_ordering
import random 
from typing import (
    List, 
    Any, 
    Iterator,
    Tuple,
    Mapping, 
    Dict, 
    Optional, 
)

TIMESTAMP_FORMAT = "%Y-%m-%d (%a) %H:%M"

def _parse_timestamp_input(value: Any) -> datetime:
    """Coerce incoming value to datetime.

    Accepts datetime, or string in either `TIMESTAMP_FORMAT` or ISO 8601.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Try custom format first
        try:
            return datetime.strptime(value, TIMESTAMP_FORMAT)
        except Exception:
            pass
        # Fallback: ISO 8601
        try:
            return datetime.fromisoformat(value)
        except Exception:
            pass
    raise TypeError(
        f"timestamp must be datetime or str; got {type(value).__name__}"
    )

def _deep_freeze(value: Any) -> Any:
    """Recursively convert containers to immutable variants.
    - dict -> MappingProxyType of a new dict with frozen values
    - list/tuple -> tuple of frozen items
    - set -> frozenset of frozen items
    Other objects returned as-is.
    """
    if isinstance(value, dict):
        frozen_dict = {k: _deep_freeze(v) for k, v in value.items()}
        return MappingProxyType(frozen_dict)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(v) for v in value)
    return value

class _TimestampOrderingMixin:
    """Mixin to compare instances by a datetime timestamp.

    Subclasses must implement `_timestamp_for_ordering()` returning a datetime.
    """

    def _timestamp_for_ordering(self) -> datetime:
        raise NotImplementedError(
            "Subclasses must implement `_timestamp_for_ordering()` returning a datetime."
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._timestamp_for_ordering() == other._timestamp_for_ordering()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._timestamp_for_ordering() < other._timestamp_for_ordering()
    
    def get_string_timestamp(self) -> str:
        return self.timestamp.strftime(TIMESTAMP_FORMAT)

@total_ordering
class Message(_TimestampOrderingMixin, BaseModel):
    """A message in a session."""
    model_config = ConfigDict(frozen=True)
    role: str = Field(..., description="The role of the message.")
    content: str = Field(..., description="The content of the message.")
    timestamp: datetime = Field(..., description="The timestamp of the message.")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="The metadata of the message.")

    @model_validator(mode="after")
    def _freeze_metadata(self) -> Message:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _deep_freeze(self.metadata))
        return self

    @model_validator(mode="before")
    def _coerce_timestamp_before(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "timestamp" in values:
            values["timestamp"] = _parse_timestamp_input(values["timestamp"])
        return values

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(v)

    @field_serializer("timestamp")
    def _serialize_timestamp(self, v: datetime) -> str:
        return v.strftime(TIMESTAMP_FORMAT)

    def _timestamp_for_ordering(self) -> datetime:
        return self.timestamp

@total_ordering
class QuestionAnswerPair(_TimestampOrderingMixin, BaseModel):
    """A question and answer pair."""
    # Note that, in some cases, the QA pair is regarded as two messages in a session. 
    model_config = ConfigDict(frozen=True)
    role: str = Field(..., description="The role who asks the question.")
    question: str = Field(..., description="The question.")
    answer_list: Tuple[str, ...] = Field(..., description="The answer list.", min_length=1)
    timestamp: datetime = Field(..., description="The timestamp of the question and answer pair.")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="The metadata of the question and answer pair.")

    @model_validator(mode="after")
    def _freeze_metadata(self) -> QuestionAnswerPair:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _deep_freeze(self.metadata))
        return self

    @model_validator(mode="before")
    def _coerce_timestamp_before(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "timestamp" in values:
            values["timestamp"] = _parse_timestamp_input(values["timestamp"])
        return values

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(v)

    @field_serializer("timestamp")
    def _serialize_timestamp(self, v: datetime) -> str:
        return v.strftime(TIMESTAMP_FORMAT)

    def _timestamp_for_ordering(self) -> datetime:
        return self.timestamp

@total_ordering
class Session(_TimestampOrderingMixin, BaseModel):
    """A session."""
    model_config = ConfigDict(frozen=True)
    messages: Tuple[Message | QuestionAnswerPair, ...] = Field(
        ..., 
        description="The messages in the session.",
        min_length=1,
    )
    timestamp: datetime = Field(..., description="The timestamp of the session.")
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="The metadata of the session.")

    @model_validator(mode="after")
    def _sort_messages_by_timestamp(self) -> Session:
        # Stable ascending sort; then freeze to tuple
        sorted_msgs = sorted(self.messages, key=lambda m: m.timestamp)
        object.__setattr__(self, "messages", tuple(sorted_msgs))
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _deep_freeze(self.metadata))
        return self

    @model_validator(mode="before")
    def _coerce_timestamp_before(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "timestamp" in values:
            values["timestamp"] = _parse_timestamp_input(values["timestamp"])
        return values

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(v)

    @field_serializer("timestamp")
    def _serialize_timestamp(self, v: datetime) -> str:
        return v.strftime(TIMESTAMP_FORMAT)

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message | QuestionAnswerPair]:
        return iter(self.messages)

    def __getitem__(self, index: int) -> Message | QuestionAnswerPair:
        return self.messages[index]

    def _timestamp_for_ordering(self) -> datetime:
        return self.timestamp

class Trajectory(BaseModel):
    """A trajectory."""
    model_config = ConfigDict(frozen=True)
    sessions: Tuple[Session, ...] = Field(
        ..., 
        description="The sessions in the trajectory.", 
        min_length=1,
    )
    metadata: Mapping[str, Any] = Field(default_factory=dict, description="The metadata of the trajectory.")

    @model_validator(mode="after")
    def _sort_sessions_by_timestamp(self) -> Trajectory:
        # Stable ascending sort; then freeze to tuple
        sorted_sessions = sorted(self.sessions, key=lambda s: s.timestamp)
        object.__setattr__(self, "sessions", tuple(sorted_sessions))
        if self.metadata is not None:
            object.__setattr__(self, "metadata", _deep_freeze(self.metadata))
        return self

    @field_serializer("metadata")
    def _serialize_metadata(self, v: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(v)

    def __len__(self) -> int:
        return len(self.sessions)

    def __iter__(self) -> Iterator[Session]:
        return iter(self.sessions)

    def __getitem__(self, index: int) -> Session:
        return self.sessions[index]

class MemoryDataset(BaseModel, ABC):
    """A memory dataset."""
    trajectories: List[Trajectory] = Field(
        ..., 
        description="The trajectories in the dataset.",
        min_length=1,
    )
    question_answer_pair_lists: List[List[QuestionAnswerPair]] = Field(
        ..., 
        description="The question and answer pairs for each trajectory in the dataset. "
        "The length of the list is the same as the number of trajectories.",
        min_length=1,
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="The metadata of the dataset.")

    @model_validator(mode="after")
    def _validate_lengths(self) -> MemoryDataset:
        if len(self.trajectories) != len(self.question_answer_pair_lists):
            raise ValueError(
                "Length mismatch: `trajectories` and `question_answer_pair_lists` must have the same length."
            )
    
        return self
    
    @model_validator(mode="after")
    def _process_metadata(self) -> MemoryDataset:
        if not self.metadata:
            self.metadata = self._generate_metadata()
        return self

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self) -> Iterator[Tuple[Trajectory, List[QuestionAnswerPair]]]:
        return iter(zip(self.trajectories, self.question_answer_pair_lists))

    def __getitem__(self, index: int) -> Tuple[Trajectory, List[QuestionAnswerPair]]:
        return self.trajectories[index], self.question_answer_pair_lists[index]
    
    def get_trajectories(self) -> List[Trajectory]:
        """Get the trajectories in the dataset."""
        return self.trajectories
    
    def get_question_answer_pair_lists(self) -> List[List[QuestionAnswerPair]]:
        """Get the question and answer pairs for each trajectory in the dataset."""
        return self.question_answer_pair_lists
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset."""
        rng = random.Random(seed)
        indices = list(range(len(self)))
        rng.shuffle(indices)
        self.trajectories = [self.trajectories[i] for i in indices]
        self.question_answer_pair_lists = [self.question_answer_pair_lists[i] for i in indices]

    def sample(self, size: int, seed: Optional[int] = None) -> MemoryDataset:
        """Sample the dataset."""
        if len(self) < size:
            raise ValueError(
                f"Cannot sample {size} items from dataset of length {len(self)}."
            )
        rng = random.Random(seed)
        indices = rng.sample(range(len(self)), size)
        return self.__class__(
            trajectories=[self.trajectories[i] for i in indices],
            question_answer_pair_lists=[self.question_answer_pair_lists[i] for i in indices],
        )
    
    @classmethod
    @abstractmethod
    def read_raw_data(cls, path: str) -> MemoryDataset:
        """Read the raw data from the path."""
        raise NotImplementedError("Subclasses must implement `read_raw_data()`.")

    @abstractmethod
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate the metadata of the dataset."""
        raise NotImplementedError("Subclasses must implement `_generate_metadata()`.")
        
    def __repr__(self) -> str:
        def fmt_scalar(v: Any, width: int = 100) -> str:
            s = repr(v)
            return s if len(s) <= width else s[: width - 3] + "..."

        def render_dict(d: Dict[str, Any], indent: int = 2, width: int = 100) -> list[str]:
            if not d:
                return []
            keys = sorted(map(str, d.keys()))
            key_w = max(len(k) for k in keys)
            lines: list[str] = []
            for k in keys:
                v = d[k]
                # Use ljust to align the ':' column for all keys at this level
                pad = " " * indent + k.ljust(key_w) + ":"
                if isinstance(v, dict):
                    lines.append(pad)
                    lines.extend(render_dict(v, indent + 8, width))
                elif isinstance(v, (list, tuple, set)):
                    seq = list(v)
                    if not seq:
                        lines.append(pad + " []")
                    else:
                        lines.append(pad)
                        for x in seq:
                            if isinstance(x, dict):
                                lines.append(" " * (indent + 8) + "-")
                                lines.extend(render_dict(x, indent + 12, width))
                            else:
                                lines.append(" " * (indent + 8) + "- " + fmt_scalar(x, width))
                else:
                    lines.append(pad + " " + fmt_scalar(v, width))
            return lines

        header = f"{self.__class__.__name__} Metadata"
        bar = "─" * len(header)
        body_lines = render_dict(self.metadata, indent=2, width=100)

        return header + "\n" + bar + ("\n" + "\n".join(body_lines) if body_lines else "")
    