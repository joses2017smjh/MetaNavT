"""Shared retrieval datatypes with no numpy / sklearn imports."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    path: str
    text: str
    start_byte: int
    end_byte: int
    mtime: float = 0.0
    content_hash: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.path


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int
    bm25: float | None = None
    dense: float | None = None
    rrf: float | None = None
    rerank: float | None = None
