"""Phase 10: Search-R1 environment — parse tagged rollouts, mask retrieved tokens.

Jin et al. 2025 (arXiv:2503.09516). The policy emits <search> / <answer>;
the env fills <information>. Retrieved tokens are masked in the loss.
Full veRL/GPU training is not required for CI — see app.rl.grpo.dummy_step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from app.agent.retrieval_loop import extractive_answer
from app.retrieval.bm25 import tokenize
from app.retrieval.hybrid import InMemoryHybridIndex

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
INFO_RE = re.compile(r"<information>(.*?)</information>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
INFO_SPAN_RE = re.compile(r"<information>.*?</information>", re.IGNORECASE | re.DOTALL)


@dataclass
class Trajectory:
    raw: str
    searches: list[str]
    informations: list[str]
    answer: str
    retrieved_paths: list[str] = field(default_factory=list)
    cited: bool = False
    train_mask: list[bool] = field(default_factory=list)

    @property
    def n_search(self) -> int:
        return len(self.searches)

    @property
    def tokens(self) -> list[str]:
        return tokenize(self.raw)


def parse_trajectory(raw: str) -> Trajectory:
    searches = [s.strip() for s in SEARCH_RE.findall(raw or "") if s.strip()]
    infos = [s.strip() for s in INFO_RE.findall(raw or "")]
    answers = [s.strip() for s in ANSWER_RE.findall(raw or "")]
    answer = answers[-1] if answers else ""
    cited = bool(re.search(r"\[[^\]]+\]", answer) or re.search(r"\S+\.\w{2,5}", answer))
    if infos:
        cited = cited or bool(infos[0].strip())
    mask = token_train_mask(raw or "")
    return Trajectory(
        raw=raw or "",
        searches=searches,
        informations=infos,
        answer=answer,
        cited=cited and bool(answer),
        train_mask=mask,
    )


def token_train_mask(raw: str) -> list[bool]:
    """True = include in the loss (generated). Retrieved <information> spans are False."""
    if not raw:
        return []
    pieces: list[tuple[str, bool]] = []
    pos = 0
    for m in INFO_SPAN_RE.finditer(raw):
        if m.start() > pos:
            pieces.append((raw[pos : m.start()], True))
        pieces.append((m.group(0), False))
        pos = m.end()
    if pos < len(raw):
        pieces.append((raw[pos:], True))
    mask: list[bool] = []
    for text, trainable in pieces:
        n = len(tokenize(text))
        mask.extend([trainable] * n)
    return mask


def format_information(hits, n: int = 4) -> str:
    lines = []
    for hit in hits[:n]:
        lines.append(
            f"{hit.chunk.path} [{hit.chunk.start_byte}:{hit.chunk.end_byte}] "
            f"{(hit.chunk.text or '')[:400]}"
        )
    return "\n".join(lines)


class SearchR1Env:
    """File-tree search env. Product is a personal corpus, not web search."""

    def __init__(self, index: InMemoryHybridIndex, *, max_searches: int = 4):
        self.index = index
        self.max_searches = max_searches

    def search(self, query: str) -> tuple[str, list[str]]:
        result = self.index.retrieve(query)
        return format_information(result.hits), result.paths

    def rollout(self, question: str, policy: Callable[[str], str] | None = None) -> Trajectory:
        """One trajectory. Default policy: one search, then extractive answer."""
        if policy is None:
            return self._heuristic_rollout(question)
        raw = policy(question) or ""
        parsed = parse_trajectory(raw)
        if parsed.searches and not parsed.informations:
            blobs = []
            paths: list[str] = []
            for q in parsed.searches[: self.max_searches]:
                info, p = self.search(q)
                blobs.append(f"<information>{info}</information>")
                paths.extend(p)
            filled = raw
            if "<information>" not in filled.lower():
                filled = filled.replace("</search>", "</search>\n" + "\n".join(blobs), 1)
            parsed = parse_trajectory(filled)
            parsed.retrieved_paths = list(dict.fromkeys(paths))
            parsed.cited = parsed.cited or bool(parsed.retrieved_paths)
        return parsed

    def _heuristic_rollout(self, question: str) -> Trajectory:
        info, paths = self.search(question)
        hits = self.index.retrieve(question).hits
        answer = extractive_answer(question, hits)
        raw = (
            f"<search>{question}</search>\n"
            f"<information>{info}</information>\n"
            f"<answer>{answer}</answer>"
        )
        traj = parse_trajectory(raw)
        traj.retrieved_paths = paths
        traj.cited = bool(paths) and bool(answer.strip())
        return traj
