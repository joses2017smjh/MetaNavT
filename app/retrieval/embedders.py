"""Deterministic embedders for the local bench.

hash     — character n-gram hashing trick (no downloads, CI-safe)
tfidf    — sklearn TF-IDF + TruncatedSVD dense proxy
st       — sentence-transformers (real numbers; optional)
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


class HashEmbedder:
    """Signed hashing trick over character 3-grams. Deterministic, no deps besides numpy."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            blob = (text or "").lower()
            for n in (3, 4):
                if len(blob) < n:
                    continue
                for j in range(len(blob) - n + 1):
                    gram = blob[j : j + n]
                    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                    idx = int.from_bytes(digest[:4], "little") % self.dim
                    sign = 1.0 if digest[4] & 1 else -1.0
                    out[i, idx] += sign
        return l2_normalize(out)


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._vectorizer = None
        self._svd = None
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        n = max(len(texts), 1)
        ngram = (1, 2)
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram,
            min_df=1,
            max_features=4096,
            token_pattern=r"[A-Za-z0-9_./-]+",
        )
        sparse = self._vectorizer.fit_transform(texts)
        n_comp = min(self.dim, max(1, min(sparse.shape) - 1), n - 1)
        if n_comp < 1:
            self._svd = None
            self.dim = sparse.shape[1] or 1
            self._fitted = True
            return self
        self._svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self._svd.fit(sparse)
        self.dim = n_comp
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._vectorizer is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before encode()")
        sparse = self._vectorizer.transform(texts)
        if self._svd is None:
            dense = sparse.toarray().astype(np.float32)
        else:
            dense = self._svd.transform(sparse).astype(np.float32)
        return l2_normalize(dense)


class SentenceTransformerEmbedder:
    name = "st"

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{model_name}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


def cosine_scores(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    q = query_vec.reshape(1, -1)
    q = l2_normalize(q)
    docs = l2_normalize(doc_mat)
    return (docs @ q.T).ravel()
