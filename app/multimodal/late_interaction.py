"""ColBERT / ColPali-style late interaction: MaxSim over token/patch embeddings.

score(q, d) = Σ_i max_j (q_i · d_j)

Storage is typically 10-20x a single-vector index — report as a cost/recall
trade-off, not as the default production path.
"""

from __future__ import annotations

import numpy as np

from app.retrieval.embedders import l2_normalize


def maxsim(query_tok: np.ndarray, doc_tok: np.ndarray) -> float:
    """query_tok: (Q, D), doc_tok: (N, D). Both should be L2-normalized."""
    if query_tok.size == 0 or doc_tok.size == 0:
        return 0.0
    q = l2_normalize(np.asarray(query_tok, dtype=np.float32))
    d = l2_normalize(np.asarray(doc_tok, dtype=np.float32))
    sims = q @ d.T  # (Q, N)
    return float(sims.max(axis=1).sum())


def maxsim_search(
    query_tok: np.ndarray,
    corpus: list[tuple[str, np.ndarray]],
    k: int = 10,
) -> list[tuple[str, float]]:
    scored = [(doc_id, maxsim(query_tok, tokens)) for doc_id, tokens in corpus]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def storage_multiplier(seq_len: int) -> int:
    """Approximate storage vs one vector per chunk."""
    return max(1, int(seq_len))
