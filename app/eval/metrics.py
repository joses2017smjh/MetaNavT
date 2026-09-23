"""BEIR-style retrieval metrics.

Always reported together: nDCG@10, Recall@5/10/50, MRR@10.
Relevance is binary at the document-path (or chunk-id) level.

Recall@k on a 61-file corpus is mostly a function of list length: a uniformly
random list of M distinct files scores an expected min(k, M) / N. `score_query`
therefore also returns the number of unique paths retrieved and that random
baseline, so every Recall@k can be read next to what a random list of the same
length would score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

RECALL_KS = (5, 10, 50)


def _to_set(values: Iterable[str]) -> set[str]:
    return {str(v) for v in values}


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = _to_set(relevant)
    if not rel:
        return 0.0
    hits = _to_set(retrieved[:k]) & rel
    return len(hits) / len(rel)


def mrr_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = _to_set(relevant)
    if not rel:
        return 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    total = 0.0
    for i, gain in enumerate(gains[:k], start=1):
        total += gain / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = _to_set(relevant)
    if not rel:
        return 0.0
    gains = [1.0 if item in rel else 0.0 for item in retrieved[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = dcg_at_k([1.0] * min(k, len(rel)), k)
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def unique_count_at_k(retrieved: Sequence[str], k: int) -> int:
    """Distinct items in the top-k of the list (the list length that Recall@k really measures)."""
    return len(dict.fromkeys(str(v) for v in retrieved[:k]))


def random_recall_at_k(n_unique_retrieved: int, n_files: int, k: int) -> float:
    """Expected Recall@k of a uniformly random list of `n_unique_retrieved` distinct files.

    Each relevant file lands in the top-k of such a list with probability
    min(k, M) / N, so the expectation is that ratio regardless of how many files
    are relevant. Exact, so no sampling is needed.
    """
    if n_files <= 0:
        return 0.0
    return min(k, n_unique_retrieved) / n_files


def score_query(retrieved: Sequence[str], relevant: Iterable[str], n_files: int | None = None) -> dict[str, float]:
    """All per-query metrics for one retrieved list (paths, deduplicated by the caller)."""
    row: dict[str, float] = {
        "ndcg@10": ndcg_at_k(retrieved, relevant, 10),
        "mrr@10": mrr_at_k(retrieved, relevant, 10),
        "unique_paths@50": float(unique_count_at_k(retrieved, 50)),
    }
    for k in RECALL_KS:
        row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
    if n_files:
        m = unique_count_at_k(retrieved, 50)
        for k in RECALL_KS:
            row[f"random_recall@{k}"] = random_recall_at_k(m, n_files, k)
    return row


@dataclass
class RetrievalScores:
    recall_50: float
    ndcg_10: float
    mrr_10: float
    n_queries: int
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    recall_5: float = 0.0
    recall_10: float = 0.0
    mean_unique_paths_50: float = 0.0
    n_files: int | None = None
    random_recall: dict[str, float] = field(default_factory=dict)  # "random_recall@k" -> mean
    per_query: list[dict[str, float]] = field(default_factory=list)  # gold order; not serialised here

    def as_dict(self) -> dict:
        out = {
            "ndcg@10": round(self.ndcg_10, 4),
            "recall@5": round(self.recall_5, 4),
            "recall@10": round(self.recall_10, 4),
            "recall@50": round(self.recall_50, 4),
            "mrr@10": round(self.mrr_10, 4),
            "mean_unique_paths@50": round(self.mean_unique_paths_50, 2),
            "n_queries": self.n_queries,
            "by_category": {
                cat: {k: round(v, 4) for k, v in scores.items()}
                for cat, scores in self.by_category.items()
            },
        }
        if self.n_files:
            out["n_files"] = self.n_files
            for key, value in self.random_recall.items():
                out[key] = round(value, 4)
        return out


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def aggregate_retrieval(
    per_query: Sequence[dict],
    n_files: int | None = None,
) -> RetrievalScores:
    """Average metrics over queries. Each dict needs retrieved, relevant, optional category, optional id."""
    if not per_query:
        return RetrievalScores(0.0, 0.0, 0.0, 0)

    scored: list[dict[str, float]] = []
    cats: dict[str, list[dict[str, float]]] = {}
    for row in per_query:
        s = score_query(list(row["retrieved"]), list(row["relevant"]), n_files=n_files)
        if row.get("id") is not None:
            s["id"] = row["id"]
        scored.append(s)
        cat = row.get("category")
        if cat:
            cats.setdefault(cat, []).append(s)

    by_category = {}
    for cat, rows in cats.items():
        by_category[cat] = {
            "ndcg@10": _mean(rows, "ndcg@10"),
            "recall@10": _mean(rows, "recall@10"),
            "recall@50": _mean(rows, "recall@50"),
            "mrr@10": _mean(rows, "mrr@10"),
            "n": float(len(rows)),
        }

    random_recall = (
        {f"random_recall@{k}": _mean(scored, f"random_recall@{k}") for k in RECALL_KS} if n_files else {}
    )
    return RetrievalScores(
        recall_50=_mean(scored, "recall@50"),
        ndcg_10=_mean(scored, "ndcg@10"),
        mrr_10=_mean(scored, "mrr@10"),
        n_queries=len(scored),
        by_category=by_category,
        recall_5=_mean(scored, "recall@5"),
        recall_10=_mean(scored, "recall@10"),
        mean_unique_paths_50=_mean(scored, "unique_paths@50"),
        n_files=n_files,
        random_recall=random_recall,
        per_query=scored,
    )
