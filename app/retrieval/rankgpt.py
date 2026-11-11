"""Phase 7: listwise LLM rerank (RankGPT-style) + honest BGE-m3 loader.

Sun et al. RankGPT: the model outputs a permutation of the top-n.
CI uses a deterministic permutation from overlap so the row exists without a GPU.
The bench payload records whether the real cross-encoder loaded.

A reply is read as a ranking only when the whole reply is a chain of bracketed
passage ids, "[2] > [1] > [3]", that names every passage of the window exactly
once (read_permutation). Anything else keeps the fused order, and the reason is
recorded (PARSE_STATUSES), so prose that mentions numbers or file names such as
run_047 can never be read as ranks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from app.retrieval.hybrid import Chunk
from app.retrieval.rerank import (
    CrossEncoderReranker,
    exact_token_hits,
    jaccard_overlap,
)


def bge_reranker(model_name: str = "BAAI/bge-reranker-v2-m3") -> tuple[object, bool]:
    """Return (rerank_fn, loaded_real_model); when the model did not load, rerank_fn is the labelled overlap fallback."""
    ce = CrossEncoderReranker(model_name=model_name, allow_fallback=True)
    loaded = ce.model is not None
    return ce, loaded


PASSAGE_CHARS = 280  # characters of each passage's text in the prompt (after its path)
# What a reply was, in the order read_permutation checks it. Only "permutation" changes the order.
PARSE_STATUSES = ("permutation", "no_completion", "not_a_chain", "out_of_range", "repeated_id", "partial")
_CHAIN = re.compile(r"\[\d+\](?:\s*>\s*\[\d+\])*")
_ID = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class ParsedPermutation:
    order: list[int]  # 0-based; the fused order 0..n-1 unless status == "permutation"
    status: str  # one of PARSE_STATUSES


def read_permutation(raw: str | None, n: int) -> ParsedPermutation:
    """Read a listwise reply over n passages numbered [1]..[n].

    The reply, stripped of surrounding whitespace, must be '[3] > [1] > [2]'
    naming each of the n ids once. Otherwise the fused order is kept and the
    status says why: no_completion (empty), not_a_chain (prose, bare numbers,
    any other text), out_of_range, repeated_id, or partial (a valid chain that
    leaves passages out; counted, not completed).
    """
    fused = list(range(n))
    text = (raw or "").strip()
    if not text:
        return ParsedPermutation(fused, "no_completion")
    if not _CHAIN.fullmatch(text):
        return ParsedPermutation(fused, "not_a_chain")
    ids = [int(x) - 1 for x in _ID.findall(text)]
    if any(not 0 <= i < n for i in ids):
        return ParsedPermutation(fused, "out_of_range")
    if len(set(ids)) != len(ids):
        return ParsedPermutation(fused, "repeated_id")
    if len(ids) != n:
        return ParsedPermutation(fused, "partial")
    return ParsedPermutation(ids, "permutation")


def parse_permutation(raw: str | None, n: int) -> list[int]:
    """0-based order from '[3] > [1] > [2]'; the fused order for anything else (see read_permutation)."""
    return read_permutation(raw, n).order


def listwise_prompt(query: str, passages: Sequence[str]) -> str:
    """The RankGPT-style instruction (Sun et al.), repeated after the passages, asking for the permutation only."""
    n = len(passages)
    lines = [f"[{i}] {text}" for i, text in enumerate(passages, start=1)]
    return (
        f"Rank the {n} passages below by their relevance to the query. Each passage has an id from [1] to [{n}].\n"
        f"Query: {query}\n" + "\n".join(lines) + "\n"
        f"Answer with the ranking of all {n} passages, most relevant first, as ids in the form [2] > [1] > [3]. "
        "Write only the ranking: no words, no explanation."
    )


class RankGPTReranker:
    """Listwise permutation rerank over the fused top-n (default 20)."""

    def __init__(
        self,
        complete: Callable[[str], str] | None = None,
        window: int = 20,
    ):
        self.complete = complete
        self.window = window
        self.last_parse: ParsedPermutation | None = None  # how the last reply was read (None on the heuristic path)

    def _heuristic_perm(self, query: str, pairs: Sequence[tuple[Chunk, float]]) -> list[int]:
        scored = []
        for i, (chunk, rrf) in enumerate(pairs):
            s = exact_token_hits(query, chunk.text) + jaccard_overlap(query, chunk.text) + 0.05 * rrf
            scored.append((s, i))
        scored.sort(reverse=True)
        return [i for _, i in scored]

    def __call__(
        self, query: str, pairs: Sequence[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        self.last_parse = None
        window = list(pairs[: self.window])
        rest = list(pairs[self.window :])
        if not window:
            return list(pairs)
        if self.complete is None:
            order = self._heuristic_perm(query, window)
        else:
            passages = [f"{chunk.path}\n{(chunk.text or '')[:PASSAGE_CHARS]}" for chunk, _ in window]
            self.last_parse = read_permutation(self.complete(listwise_prompt(query, passages)), len(window))
            order = self.last_parse.order
        ranked = []
        for rank, idx in enumerate(order):
            chunk, _ = window[idx]
            ranked.append((chunk, float(len(order) - rank)))
        return ranked + rest
