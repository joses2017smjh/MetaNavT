"""Late chunking: embed the full document, then mean-pool token embeddings per span.

Jina-style. Needs a long-context embedding model at index time; the pooling
math is model-agnostic and is what this module tests.
"""

from __future__ import annotations

import numpy as np

from app.chunking.structure import Span
from app.retrieval.embedders import l2_normalize


def mean_pool_spans(
    token_embeddings: np.ndarray,
    spans_token: list[tuple[int, int]],
) -> np.ndarray:
    """token_embeddings: (seq_len, dim). spans are [start, end) token indices."""
    if token_embeddings.ndim != 2:
        raise ValueError("token_embeddings must be (seq, dim)")
    dim = token_embeddings.shape[1]
    seq = token_embeddings.shape[0]
    out = np.zeros((len(spans_token), dim), dtype=np.float32)
    for i, (start, end) in enumerate(spans_token):
        a = max(0, min(start, seq))
        b = max(a + 1, min(end, seq))
        out[i] = token_embeddings[a:b].mean(axis=0)
    return l2_normalize(out)


def char_spans_to_token_spans(
    text: str,
    char_spans: list[Span],
    tokens: list[str],
) -> list[tuple[int, int]]:
    """Greedy map character spans onto a whitespace tokenization of `text`."""
    # Reconstruct token char offsets from tokens joined by single spaces if needed.
    # We tokenize the original text with the same regex used for BM25-ish splits.
    import re

    token_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        token_spans.append((match.start(), match.end()))
    if tokens:
        # prefer provided tokens if counts match
        if len(tokens) != len(token_spans):
            token_spans = token_spans[: len(tokens)] if tokens else token_spans

    mapped = []
    for span in char_spans:
        start_tok = 0
        end_tok = len(token_spans)
        for i, (cs, ce) in enumerate(token_spans):
            if ce > span.start:
                start_tok = i
                break
        for i, (cs, ce) in enumerate(token_spans):
            if cs >= span.end:
                end_tok = i
                break
        else:
            end_tok = len(token_spans)
        mapped.append((start_tok, max(start_tok + 1, end_tok)))
    return mapped


def late_chunk_vectors(
    token_embeddings: np.ndarray,
    text: str,
    char_spans: list[Span],
) -> np.ndarray:
    tok_spans = char_spans_to_token_spans(text, char_spans, tokens=[])
    return mean_pool_spans(token_embeddings, tok_spans)
