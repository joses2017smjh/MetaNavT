"""Gold-set schema: question, answer, retrieval targets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CATEGORIES = (
    "simple_factual",
    "conditional",
    "comparative",
    "aggregation",
    "multi_hop",
    "staleness",
    "semantic",
    "exact_path",
)


@dataclass
class GoldQuestion:
    id: str
    question: str
    category: str
    answer: str
    relevant_paths: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    notes: str = ""

    def relevant_ids(self) -> list[str]:
        """Primary retrieval targets: paths, falling back to chunk ids."""
        return list(self.relevant_paths) or list(self.relevant_chunk_ids)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "answer": self.answer,
            "relevant_paths": self.relevant_paths,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "GoldQuestion":
        return cls(
            id=str(raw["id"]),
            question=raw["question"],
            category=raw.get("category", "semantic"),
            answer=str(raw.get("answer", "")),
            relevant_paths=list(raw.get("relevant_paths") or []),
            relevant_chunk_ids=list(raw.get("relevant_chunk_ids") or []),
            notes=str(raw.get("notes") or ""),
        )


def load_gold(path: str | Path) -> list[GoldQuestion]:
    path = Path(path)
    questions: list[GoldQuestion] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            q = GoldQuestion.from_dict(raw)
            if q.category not in CATEGORIES:
                raise ValueError(f"{path}:{line_no} unknown category {q.category!r}")
            questions.append(q)
    return questions


def write_gold(path: str | Path, questions: Iterable[GoldQuestion]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
