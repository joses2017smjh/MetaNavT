"""Search-R1 outcome reward. Do not use the overlap teacher as reward."""

from __future__ import annotations

from typing import Sequence

from app.eval.jury import KAPPA_GATE
from app.eval.metrics import ndcg_at_k
from app.rl.search_r1 import Trajectory


def exact_match(pred: str, gold: str) -> float:
    p, g = (pred or "").strip().lower(), (gold or "").strip().lower()
    if not g:
        return 0.0
    if p == g or g in p:
        return 1.0
    return 0.0


def search_r1_reward(
    traj: Trajectory,
    *,
    gold: str,
    gold_paths: Sequence[str],
    jury_score: float | None = None,
    kappa: float | None = None,
) -> dict:
    """R = 1.0*EM + 0.5*nDCG@10 + 0.3*jury(if κ≥0.6) - 0.1*n_search - 1.0*uncited_or_empty."""
    em = exact_match(traj.answer, gold)
    ndcg = ndcg_at_k(list(traj.retrieved_paths), gold_paths, 10)
    jury_term = 0.0
    jury_applied = False
    if jury_score is not None and kappa is not None and kappa >= KAPPA_GATE:
        jury_term = 0.3 * float(jury_score)
        jury_applied = True
    empty = 1.0 if (not traj.cited or not (traj.answer or "").strip()) else 0.0
    reward = (
        1.0 * em
        + 0.5 * ndcg
        + jury_term
        - 0.1 * traj.n_search
        - 1.0 * empty
    )
    return {
        "reward": round(float(reward), 6),
        "em": em,
        "ndcg@10": round(float(ndcg), 6),
        "jury_term": round(jury_term, 6),
        "jury_applied": jury_applied,
        "n_search": traj.n_search,
        "uncited_or_empty": empty,
    }
