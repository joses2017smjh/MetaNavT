"""Phase 7: listwise LLM rerank (RankGPT-style) + honest BGE-m3 loader.

Sun et al. RankGPT: the model outputs a permutation of the top-n.
CI uses a deterministic permutation from overlap so the row exists without a GPU.
The bench payload records whether the real cross-encoder loaded.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from app.retrieval.hybrid import Chunk
from app.retrieval.rerank import (
    CrossEncoderReranker,
    exact_token_hits,
    jaccard_overlap,
)


def bge_reranker(model_name: str = "BAAI/bge-reranker-v2-m3") -> tuple[object, bool]:
    """Return (rerank_fn, loaded_real_model)."""
    ce = CrossEncoderReranker(model_name=model_name)
    loaded = ce.model is not None
    return ce, loaded


def parse_permutation(raw: str, n: int) -> list[int]:
    """Parse '[3] > [1] > [2]' or '3 1 2' into 0-based indices."""
    nums = [int(x) - 1 for x in re.findall(r"\d+", raw or "")]
    seen = []
    for i in nums:
        if 0 <= i < n and i not in seen:
            seen.append(i)
    for i in range(n):
        if i not in seen:
            seen.append(i)
    return seen


class RankGPTReranker:
    """Listwise permutation rerank over the fused top-n (default 20)."""

    def __init__(
        self,
        complete: Callable[[str], str] | None = None,
        window: int = 20,
    ):
        self.complete = complete
        self.window = window

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
        window = list(pairs[: self.window])
        rest = list(pairs[self.window :])
        if not window:
            return list(pairs)
        if self.complete is None:
            order = self._heuristic_perm(query, window)
        else:
            lines = []
            for i, (chunk, _) in enumerate(window, start=1):
                lines.append(f"[{i}] {chunk.path}\n{(chunk.text or '')[:280]}")
            prompt = (
                "Rank these passages for the query. Output a permutation like [2] > [1] > [3].\n"
                f"Query: {query}\n" + "\n".join(lines)
            )
            order = parse_permutation(self.complete(prompt), len(window))
        ranked = []
        for rank, idx in enumerate(order):
            chunk, _ = window[idx]
            ranked.append((chunk, float(len(order) - rank)))
        return ranked + rest
