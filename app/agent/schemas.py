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
