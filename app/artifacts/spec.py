"""Spec card before any code is emitted.

Industry 2025–2026: Codex / Cursor / Tessl-style spec-driven generation.
A spec names citations (path + byte range), tests, template, and I/O.
Generating without a spec is how you get an uncited notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.retrieval.types import RetrievalHit

TEMPLATES = (
    "python-lib",
    "pytest",
    "jupyter-analysis",
    "streamlit",
    "research-repro",
)


@dataclass
class SpecCard:
    goal: str
    template: str
    citations: list[dict] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    file_path: str = "artifact.py"

    def as_dict(self) -> dict:
        return {
            "goal": self.goal,
            "template": self.template,
            "citations": self.citations,
            "tests": self.tests,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "file_path": self.file_path,
        }

    def ready(self) -> bool:
        return bool(self.goal and self.template in TEMPLATES and self.citations)


def _pick_template(query: str) -> str:
    q = (query or "").lower()
    if "streamlit" in q or "dashboard" in q:
        return "streamlit"
    if "notebook" in q or "jupyter" in q or "plot" in q:
        return "jupyter-analysis"
    if "test" in q or "pytest" in q:
        return "pytest"
    if "reproduc" in q or "from the paper" in q or "paper2code" in q:
        return "research-repro"
    return "python-lib"


def spec_from_query(
    query: str,
    hits: Sequence[RetrievalHit],
    *,
    n_cite: int = 4,
) -> SpecCard:
    citations = []
    for hit in hits[:n_cite]:
        citations.append(
            {
                "path": hit.chunk.path,
                "start_byte": hit.chunk.start_byte,
                "end_byte": hit.chunk.end_byte,
                "chunk_id": hit.chunk.chunk_id,
            }
        )
    tests = []
    q = query or ""
    if re.search(r"fusion", q, re.I):
        tests.append("assert fuse([1, 2], True) == [1, 2]")
        tests.append("assert fuse([1, 2], False) == 1")
    if re.search(r"learning.?rate|3e-4", q, re.I):
        tests.append("assert '3e-4' in str(LR) or LR == 3e-4 or abs(float(LR) - 3e-4) < 1e-9")
    template = _pick_template(q)
    suffix = {
        "python-lib": "lib.py",
        "pytest": "test_artifact.py",
        "jupyter-analysis": "analysis.py",
        "streamlit": "app.py",
        "research-repro": "reproduce.py",
    }[template]
    return SpecCard(
        goal=q.strip(),
        template=template,
        citations=citations,
        tests=tests,
        inputs=["retrieved chunks"],
        outputs=["executable module", "pytest asserts"],
        file_path=f"artifacts/{suffix}",
    )
