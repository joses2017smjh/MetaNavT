"""HyDE: Hypothetical Document Embedding (Gao et al. 2023).

Generate a hypothetical answer to the query, embed that instead of the raw
query.  The hypothesis lives in the same embedding space as the documents,
so cosine similarity catches semantic matches that raw-query terms miss.

CI path: the heuristic expander builds a pseudo-document from query tokens
without calling an LLM, so `make bench` stays GPU-free.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

from app.retrieval.embedders import Embedder


def heuristic_hypothesis(query: str) -> str:
    """Build a pseudo-document from query tokens (no LLM needed).

    Repeats key terms and adds filler so the embedding concentrates on the
    query's salient tokens rather than the question phrasing.
    """
    q = (query or "").strip()
    if not q:
        return ""
    tokens = re.findall(r"[A-Za-z0-9_./\-]+", q)
    key_tokens = [t for t in tokens if len(t) > 2 and t.lower() not in _STOP]
    if not key_tokens:
        key_tokens = tokens[:5]
    repeated = " ".join(key_tokens * 3)
    return f"This document contains information about {' '.join(key_tokens)}. {repeated}"


def llm_hypothesis(
    query: str,
    complete: Callable[[str], str],
    max_tokens: int = 128,
) -> str:
    """Generate a hypothetical answer using an LLM."""
    prompt = (
        "Write a short paragraph (2-3 sentences) that would be a good "
        "document answering the following question. Do not say 'I don't know'. "
        "Write as if you are the document itself, stating facts directly.\n\n"
        f"Question: {query}\n\nDocument:"
    )
    raw = (complete(prompt) or "").strip()
    if not raw:
        return heuristic_hypothesis(query)
    return raw


def hyde_embed(
    query: str,
    embedder: Embedder,
    complete: Callable[[str], str] | None = None,
    n_hypotheses: int = 1,
) -> np.ndarray:
    """Embed the query via HyDE: generate hypothesis(es), embed, average.

    Returns a single (dim,) vector ready for cosine search.
    """
    hypotheses: list[str] = []
    for _ in range(n_hypotheses):
        if complete is not None:
            hypotheses.append(llm_hypothesis(query, complete))
        else:
            hypotheses.append(heuristic_hypothesis(query))
    hypotheses.append(query)
    vecs = embedder.encode(hypotheses)
    mean_vec = vecs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec


_STOP = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "because", "but", "and", "or",
        "if", "while", "about", "what", "which", "who", "whom", "this",
        "that", "these", "those", "i", "me", "my", "we", "our", "you",
        "your", "he", "him", "his", "she", "her", "it", "its", "they",
        "them", "their",
    }
)
