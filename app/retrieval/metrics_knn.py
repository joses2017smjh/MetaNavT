"""Tiny knn helpers used by the pgvector sweep (no sklearn)."""

from __future__ import annotations


def recall_at_k_ids(retrieved: list[int], relevant: list[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / float(len(relevant))
