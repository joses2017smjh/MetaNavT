"""Reference-free RAGAS-style metrics. LLM judge is optional.

faithfulness, context precision, context recall, answer relevancy.
When no LLM is provided, a token-overlap heuristic is used so the harness
still emits a number (labeled heuristic, not RAGAS-official).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from app.retrieval.bm25 import tokenize


Judge = Callable[[str], str]


def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer tokens that appear in retrieved context."""
    if not answer.strip():
        return 0.0
    blob = " ".join(contexts)
    return _overlap(answer, blob)


def answer_relevancy(question: str, answer: str) -> float:
    return _overlap(answer, question)


def context_precision(relevant_paths: Sequence[str], retrieved_paths: Sequence[str], k: int = 10) -> float:
    if not retrieved_paths[:k]:
        return 0.0
    rel = set(relevant_paths)
    hits = sum(1 for p in retrieved_paths[:k] if p in rel)
    return hits / min(k, len(retrieved_paths[:k]))


def context_recall(relevant_paths: Sequence[str], retrieved_paths: Sequence[str], k: int = 50) -> float:
    rel = set(relevant_paths)
    if not rel:
        return 0.0
    return len(rel & set(retrieved_paths[:k])) / len(rel)


@dataclass
class EndToEndScores:
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float
    judge: str

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "judge": self.judge,
        }


def aggregate_e2e(rows: Sequence[dict], judge: str = "heuristic") -> EndToEndScores:
    if not rows:
        return EndToEndScores(0, 0, 0, 0, judge)
    n = len(rows)
    return EndToEndScores(
        faithfulness=sum(r["faithfulness"] for r in rows) / n,
        context_precision=sum(r["context_precision"] for r in rows) / n,
        context_recall=sum(r["context_recall"] for r in rows) / n,
        answer_relevancy=sum(r["answer_relevancy"] for r in rows) / n,
        judge=judge,
    )
