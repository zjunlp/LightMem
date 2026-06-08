from dataclasses import dataclass
from pydantic import BaseModel, field_validator, model_validator
from typing import List

class SemanticRawOutput(BaseModel):
    semantic_triples: List[List[str]]
    episodic_evidence: List[List[int]]

    @field_validator("semantic_triples")
    @classmethod
    def validate_semantic_triples(cls, v):
        bad = [(i, t) for i, t in enumerate(v) if not isinstance(t, list) or len(t) != 3]
        if bad:
            raise ValueError(f"Bad semantic_triples entries: {bad[:5]}")
        return v

    @field_validator("episodic_evidence")
    @classmethod
    def validate_evidence_items(cls, v):
        # 只校验类型，不做长度一致性
        if not all(isinstance(lst, list) and all(isinstance(i, int) for i in lst) for lst in v):
            raise ValueError("episodic_evidence must be list[list[int]]")
        return v

    @model_validator(mode="after")
    def align_lengths(self):
        n = len(self.semantic_triples or [])
        m = len(self.episodic_evidence or [])

        if m < n:
            self.episodic_evidence = (self.episodic_evidence or []) + ([[]] * (n - m))
        elif m > n:
            self.episodic_evidence = self.episodic_evidence[:n]

        return self


class ConsolidationRawOutput(BaseModel):
    updated_triple: List[str]
    triples_to_remove: List[int]

    @field_validator("updated_triple")
    def validate_updated_triple(cls, v):
        if len(v) != 3:
            raise ValueError("Updated triple must contain exactly 3 elements.", v)
        return v

    @field_validator("triples_to_remove")
    def validate_triples_to_remove(cls, v):
        if not all(isinstance(i, int) for i in v):
            raise ValueError("All indices in triples_to_remove must be integers.", v)
        return v


@dataclass
class SemanticOutput:
    chunk_id: str
    semantic_triples: List[List[str]]
    episodic_evidence: List[List[int]]