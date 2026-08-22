"""Phase 6.4–6.5: jury of heterogeneous judges + Cohen's κ gate.

Verga et al. 2024 *Replacing Judges with Juries*: majority vote of small judges.
Do not print a judge number on the README unless kappa_vs_gold >= KAPPA_GATE
on the simple_factual slice.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from app.eval.judge import (
    LABELS,
    HeuristicJudge,
    LlmJudge,
    PointwiseVerdict,
    exact_match_label,
    label_to_score,
    parse_label,
)

KAPPA_GATE = 0.6


def cohens_kappa(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0
    yt, yp = list(y_true)[:n], list(y_pred)[:n]
    labels = sorted(set(yt) | set(yp) | set(LABELS))
    po = sum(1 for a, b in zip(yt, yp) if a == b) / n
    pe = 0.0
    for lab in labels:
        pe += (yt.count(lab) / n) * (yp.count(lab) / n)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


@dataclass
class JuryVerdict:
    label: str
    score: float
    votes: dict
    members: list[str]
    as_readme: bool = False

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "votes": self.votes,
            "members": self.members,
            "as_readme": self.as_readme,
        }


class Jury:
    """Majority vote over named judges. Tie → partial."""

    def __init__(self, members: list[tuple[str, object]] | None = None):
        self.members = members or [("heuristic", HeuristicJudge())]

    def vote(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
        gold: str = "",
    ) -> JuryVerdict:
        labels = []
        names = []
        for name, judge in self.members:
            names.append(name)
            if hasattr(judge, "judge_answer"):
                v: PointwiseVerdict = judge.judge_answer(question, answer, contexts, gold)
                labels.append(v.label)
            elif callable(judge):
                labels.append(parse_label(judge(question) or ""))
            else:
                labels.append("partial")
        counts = Counter(labels)
        top = counts.most_common()
        if len(top) >= 2 and top[0][1] == top[1][1]:
            winner = "partial"
        else:
            winner = top[0][0]
        return JuryVerdict(
            label=winner,
            score=label_to_score(winner),
            votes=dict(counts),
            members=names,
        )


def default_jury(llm_complete: Callable[[str], str] | None = None) -> Jury:
    members: list[tuple[str, object]] = [("heuristic", HeuristicJudge())]
    if llm_complete:
        members.append(("llm-a", LlmJudge(complete=llm_complete, name="llm-a")))
        members.append(("llm-b", LlmJudge(complete=llm_complete, name="llm-b")))
    return Jury(members)


def select_best_of_n(
    traces: Sequence[dict],
    gold_paths: Sequence[str],
) -> dict:
    """Pick the trace with the most gold-path citations (best-of-N + verifier)."""
    gold = set(gold_paths)
    if not traces:
        return {}

    def key(t: dict) -> tuple[int, int]:
        paths = t.get("paths") or t.get("citations") or []
        overlap = len(set(paths) & gold)
        return (overlap, -int(t.get("n_search_calls") or 0))

    return max(traces, key=key)


@dataclass
class KappaReport:
    kappa: float
    n: int
    gated: bool
    gold_labels: list[str] = field(default_factory=list)
    judge_labels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "kappa": round(self.kappa, 4),
            "n": self.n,
            "gated": self.gated,
            "gate": KAPPA_GATE,
            "readme_ok": self.kappa >= KAPPA_GATE,
        }


def kappa_vs_gold(
    answers: Sequence[str],
    golds: Sequence[str],
    judge_labels: Sequence[str],
) -> KappaReport:
    gold_labels = [exact_match_label(a, g) for a, g in zip(answers, golds)]
    kappa = cohens_kappa(gold_labels, list(judge_labels)[: len(gold_labels)])
    return KappaReport(
        kappa=kappa,
        n=len(gold_labels),
        gated=kappa >= KAPPA_GATE,
        gold_labels=gold_labels,
        judge_labels=list(judge_labels)[: len(gold_labels)],
    )
