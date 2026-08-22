"""Structured move-plan schema (constrained-decoding target).

Same idea as DoM's DAG-constrained decoding: the organizer emits a schema-
guaranteed plan rather than prose that then has to be parsed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MoveOp(BaseModel):
    src: str
    dst: str
    reason: str = ""


class MovePlan(BaseModel):
    plan_id: str
    ops: list[MoveOp] = Field(min_length=1)
    destructive: Literal[True] = True
    requires_approval: Literal[True] = True

    def json_schema_for_decode(self) -> dict:
        return self.model_json_schema()


class ArtifactFileSpec(BaseModel):
    path: str
    role: str = "code"


class ArtifactSpec(BaseModel):
    """Constrained-decoding target for code / research artifacts."""

    plan_id: str
    goal: str
    template: Literal[
        "python-lib", "pytest", "jupyter-analysis", "streamlit", "research-repro"
    ]
    file_path: str
    citations: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    files: list[ArtifactFileSpec] = Field(default_factory=list)
    requires_approval: Literal[True] = True
    writes_tree: Literal[False] = False
