"""Okapi BM25 over an in-memory corpus.

Used by the bench harness so retrieval metrics do not require ParadeDB.
Production still uses pg_search / ts_rank_cd via VectorStoreManager.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = defaultdict(int)
        self.tf: list[Counter] = []
        self.n: int = 0

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must be the same length")
        self.doc_ids = list(doc_ids)
        self.tf = []
        self.doc_len = []
        self.df = defaultdict(int)
        for text in texts:
            tokens = tokenize(text)
            counts = Counter(tokens)
            self.tf.append(counts)
            self.doc_len.append(len(tokens))
            for term in counts:
                self.df[term] += 1
        self.n = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        return self

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log(1.0 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def score_doc(self, query_tokens: Sequence[str], doc_index: int) -> float:
        if self.n == 0:
            return 0.0
        score = 0.0
        dl = self.doc_len[doc_index]
        tf = self.tf[doc_index]
        denom_norm = self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
        q_counts = Counter(query_tokens)
        for term, qtf in q_counts.items():
            f = tf.get(term, 0)
            if f == 0:
                continue
            score += self.idf(term) * (f * (self.k1 + 1.0) / (f + denom_norm)) * qtf
        return score

    def search(self, query: str, k: int = 50) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens or self.n == 0:
            return []
        scored = [(self.doc_ids[i], self.score_doc(tokens, i)) for i in range(self.n)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [(doc_id, score) for doc_id, score in scored[:k] if score > 0]
