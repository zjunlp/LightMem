from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1)
    timestamp: int | None = None

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


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
    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1, max_length=512)
    top_k: int = Field(ge=1, le=1000)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchItem(StrictModel):
    id: str
    content: str
    score: float | None = None
    created_at: datetime | None = None


class SearchResponse(StrictModel):
    data: list[SearchItem]
