from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MediaPart(StrictModel):
    type: Literal["image_url"]
    image_url: str = Field(min_length=1)
    caption: str | None = None

    @field_validator("image_url", mode="before")
    @classmethod
    def normalize_image_url(cls, value: Any) -> Any:
        # Accept both the AML string form and the OpenAI multimodal form.
        if isinstance(value, dict):
            value = value.get("url")
        return value.strip() if isinstance(value, str) else value

    @field_validator("caption")
    @classmethod
    def caption_must_not_be_blank(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


class Message(StrictModel):
    role: str = Field(min_length=1, max_length=64)
    content: str = ""
    media: list[MediaPart] = Field(default_factory=list)
    timestamp: int | str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: str | None) -> str:
        return str(value or "").strip()

    def model_post_init(self, __context: Any) -> None:
        if not self.content and not self.media:
            raise ValueError("message needs non-blank content or media")


class AddRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=512)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=512)
    session_id: str = Field(min_length=1, max_length=512)


class AddResponse(StrictModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(StrictModel):
    query: str | list[dict[str, Any]] = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1, max_length=512)
    top_k: int = Field(ge=1, le=1000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if isinstance(value, str):
            text = value
        else:
            text = "\n".join(
                str(part.get("text") or "")
                for part in value
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if not text.strip():
            raise ValueError("query must not be blank")
        return value


class SearchItem(StrictModel):
    id: str
    content: str
    media: list[MediaPart] = Field(default_factory=list)
    score: float | None = None
    created_at: datetime | None = None


class SearchResponse(StrictModel):
    data: list[SearchItem]
