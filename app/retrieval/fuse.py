"""Reciprocal Rank Fusion (RRF).

score = Σ 1/(k + rank) with k=60 by default (Cormack et al. 2009).
Ranks are 1-indexed so the first hit contributes 1/(k+1).
rrf_score_map optionally weights each list: Σ w/(k + rank), every w 1.0 by default.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Sequence, TypeVar

T = TypeVar("T")

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    k: int = RRF_K,
    id_fn=None,
) -> list[tuple[T, float]]:
    """Fuse ranked lists into (item, fused_score) sorted descending.

    `id_fn` maps an item to a hashable identity. Defaults to the item itself
    when it is hashable, otherwise `id()`.
    """
    fused: dict[Hashable, float] = {}
    first_item: dict[Hashable, T] = {}

    def _ident(item: T) -> Hashable:
        if id_fn is not None:
            return id_fn(item)
        try:
            hash(item)
            return item  # type: ignore[return-value]
        except TypeError:
            return id(item)

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = _ident(item)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            if key not in first_item:
                first_item[key] = item

    return sorted(
        ((first_item[key], score) for key, score in fused.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )


def rrf_score_map(
    ranked_lists: Sequence[Iterable[Hashable]],
    k: int = RRF_K,
    *,
    weights: Sequence[float] | None = None,
) -> dict[Hashable, float]:
    """Same fusion, returning id -> score (when items are already ids).

    `weights`: one positive weight per list; an id at rank r of list i adds
    weights[i] / (k + r). None weights every list 1.0, and 1.0 / (k + r) is the
    unweighted term to the last bit, so the default fusion (scores and key
    order) is the one without weights. Multiplying every weight by one factor
    scales every score by it, so only the ratio of the weights matters for the
    order (app/eval/rrf_sweep.py fixes BM25's at 1.0 and tunes the dense list's).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError(f"{len(weights)} weights for {len(ranked_lists)} ranked lists")
    elif not all(math.isfinite(w) and w > 0 for w in weights):
        raise ValueError(f"RRF list weights must be positive and finite, got {list(weights)}")
    fused: dict[Hashable, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, key in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k + rank)
    return fused
