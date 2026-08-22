"""BEIR-style retrieval metrics.

Always reported together: Recall@50, nDCG@10, MRR@10.
Relevance is binary at the document-path (or chunk-id) level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


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


@dataclass
class RetrievalScores:
    recall_50: float
    ndcg_10: float
    mrr_10: float
    n_queries: int
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "recall@50": round(self.recall_50, 4),
            "ndcg@10": round(self.ndcg_10, 4),
            "mrr@10": round(self.mrr_10, 4),
            "n_queries": self.n_queries,
            "by_category": {
                cat: {k: round(v, 4) for k, v in scores.items()}
                for cat, scores in self.by_category.items()
            },
        }


def aggregate_retrieval(
    per_query: Sequence[dict],
) -> RetrievalScores:
    """Average metrics over queries. Each dict needs retrieved, relevant, optional category."""
    if not per_query:
        return RetrievalScores(0.0, 0.0, 0.0, 0)

    rec, ndcg, mrr = [], [], []
    cats: dict[str, list[tuple[float, float, float]]] = {}
    for row in per_query:
        retrieved = list(row["retrieved"])
        relevant = list(row["relevant"])
        r = recall_at_k(retrieved, relevant, 50)
        n = ndcg_at_k(retrieved, relevant, 10)
        m = mrr_at_k(retrieved, relevant, 10)
        rec.append(r)
        ndcg.append(n)
        mrr.append(m)
        cat = row.get("category")
        if cat:
            cats.setdefault(cat, []).append((r, n, m))

    by_category = {}
    for cat, triples in cats.items():
        by_category[cat] = {
            "recall@50": sum(t[0] for t in triples) / len(triples),
            "ndcg@10": sum(t[1] for t in triples) / len(triples),
            "mrr@10": sum(t[2] for t in triples) / len(triples),
            "n": float(len(triples)),
        }

    nq = len(per_query)
    return RetrievalScores(
        recall_50=sum(rec) / nq,
        ndcg_10=sum(ndcg) / nq,
        mrr_10=sum(mrr) / nq,
        n_queries=nq,
        by_category=by_category,
    )
