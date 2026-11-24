"""Search-R1 outcome reward. Do not use the overlap teacher as reward.

R = 1.0 * exact match + 0.5 * nDCG@10(retrieved paths, gold paths) - 0.1 * n_search - 1.0 * uncited_or_empty
(+ 0.3 * jury vote when a jury score and a kappa at or above the gate are passed in; the M44 training run passes
neither, so its reward has no jury term).

Exact match here is strict (M44): the answer with its bracketed citations removed equals the gold answer after
normalise(): lower case, whitespace collapsed, surrounding quotes and a final full stop dropped, and every standalone
number written one way, so 0.0003, 3e-04 and 3e-4 are the same answer. Until M44 this function was plain containment
(`gold in answer`), which finds gold "2" in "run 42" and pays a policy for listing every candidate value. The
repository's other reference, app.eval.judge.exact_match_label(bounded=True), is bounded_match() here: the gold is
somewhere in the answer and not inside a longer number or word. It is reported beside exact match as a guard column,
never used as the reward.
"""

from __future__ import annotations

import re
from typing import Sequence

from app.eval.jury import KAPPA_GATE
from app.eval.judge import _NUM_RE, _standalone, exact_match_label
from app.eval.metrics import ndcg_at_k
from app.rl.search_r1 import Trajectory

CITATION_RE = re.compile(r"\[([^\[\]]*)\]")
_WRAPPERS = "\"'`"


def answer_value(answer: str) -> str:
    """The answer without its bracketed citations: `3e-4 [configs/run_047.yaml]` -> `3e-4`."""
    return " ".join(CITATION_RE.sub(" ", answer or "").split())


def cited_paths(answer: str) -> list[str]:
    """What the answer cites: the text of each bracketed span, in order."""
    return [m.strip() for m in CITATION_RE.findall(answer or "") if m.strip()]


def normalise(text: str) -> str:
    """The form two answers are compared in: lower case, whitespace collapsed, surrounding quotes and one final full
    stop dropped, every standalone number as repr(float) ("3e-04" and "0.0003" both become "0.0003"; the 2 of
    "dinov2" and the 047 of "run_047" are parts of words and stay)."""
    text = " ".join((text or "").lower().split())
    while len(text) >= 2 and text[0] == text[-1] and text[0] in _WRAPPERS:
        text = text[1:-1].strip()
    if text.endswith("."):
        text = text[:-1].rstrip()

    def canonical(m: re.Match) -> str:
        if not _standalone(text, m.start(), m.end()):
            return m.group(0)
        try:
            return repr(float(m.group(0)))
        except ValueError:
            return m.group(0)

    return _NUM_RE.sub(canonical, text)


def exact_match(pred: str, gold: str) -> float:
    """1.0 when the answer's value (citations removed) is the gold answer after normalise(), else 0.0."""
    if not (gold or "").strip():
        return 0.0
    value = answer_value(pred)
    return 1.0 if value and normalise(value) == normalise(gold) else 0.0


def bounded_match(pred: str, gold: str) -> float:
    """The guard column: app.eval.judge.exact_match_label(bounded=True) says "correct" for the answer's value."""
    if not (gold or "").strip():
        return 0.0
    return 1.0 if exact_match_label(answer_value(pred), gold, bounded=True) == "correct" else 0.0


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
    empty = 1.0 if (not traj.cited or not answer_value(traj.answer)) else 0.0
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
