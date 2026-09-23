"""LlamaIndex adapter for the bench's deterministic hash embedder.

The bench (app/retrieval/embedders.HashEmbedder) embeds text with a signed
hashing trick over character n-grams: no model weights, no downloads, the same
vector on every machine. This adapter exposes it as a LlamaIndex BaseEmbedding
so the Postgres/pgvector path can be indexed and queried in CI and in
`docker compose up` without pulling a HuggingFace model.

It is a fallback, and results produced with it are labelled as such: set
EMBEDDING_PROVIDER=huggingface (and install the `ml` extra) for real vectors.
"""

from __future__ import annotations

from typing import Any, List

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr

from app.retrieval.embedders import HashEmbedder

DEFAULT_HASH_DIM = 256


class HashEmbedding(BaseEmbedding):
    """Deterministic character n-gram hashing embedder (see module docstring)."""

    dim: int = DEFAULT_HASH_DIM
    _embedder: HashEmbedder = PrivateAttr()

    def __init__(self, dim: int = DEFAULT_HASH_DIM, **kwargs: Any) -> None:
        super().__init__(model_name=f"hash-{dim}", dim=dim, **kwargs)
        self._embedder = HashEmbedder(dim=dim)

    @classmethod
    def class_name(cls) -> str:
        return "HashEmbedding"

    def _encode(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.encode(texts).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
