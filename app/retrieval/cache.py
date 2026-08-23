"""Semantic query cache: skip the full retrieval pipeline for near-duplicate queries.

Production RAG systems see many near-identical queries ("learning rate run 47"
vs "what is the learning rate for run 47").  A semantic cache maps query
embeddings to cached results, saving BM25 + dense + rerank + graph work.

The cache is in-memory with an optional JSON persistence layer.  Eviction
is LRU.  Similarity threshold is configurable — too low misses savings,
too high returns stale or wrong results.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.retrieval.embedders import Embedder, cosine_scores


@dataclass
class CacheEntry:
    query: str
    query_hash: str
    embedding: list[float]
    result: dict[str, Any]
    timestamp: float
    hit_count: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
        }


class SemanticCache:
    """In-memory LRU cache with cosine-similarity lookup."""

    def __init__(
        self,
        embedder: Embedder,
        max_entries: int = 256,
        similarity_threshold: float = 0.92,
        ttl_seconds: float = 3600.0,
        persist_path: str | Path | None = None,
    ):
        self.embedder = embedder
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.persist_path = Path(persist_path) if persist_path else None

        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embeddings: list[np.ndarray] = []
        self._keys: list[str] = []
        self.stats = CacheStats()

        if self.persist_path and self.persist_path.exists():
            self._load()

    def get(self, query: str) -> dict[str, Any] | None:
        """Look up a cached result for a semantically similar query."""
        q_embed = self.embedder.encode([query])[0]
        now = time.time()

        if not self._embeddings:
            self.stats.misses += 1
            return None

        mat = np.stack(self._embeddings)
        sims = cosine_scores(q_embed.reshape(1, -1), mat)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self.similarity_threshold:
            self.stats.misses += 1
            return None

        key = self._keys[best_idx]
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None

        if now - entry.timestamp > self.ttl_seconds:
            self._evict(key)
            self.stats.misses += 1
            return None

        entry.hit_count += 1
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return entry.result

    def put(self, query: str, result: dict[str, Any]) -> None:
        """Store a query result in the cache."""
        q_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        q_embed = self.embedder.encode([query])[0]

        if q_hash in self._entries:
            idx = self._keys.index(q_hash)
            self._embeddings[idx] = q_embed
            self._entries[q_hash].result = result
            self._entries[q_hash].timestamp = time.time()
            self._entries.move_to_end(q_hash)
            return

        while len(self._entries) >= self.max_entries:
            oldest_key = next(iter(self._entries))
            self._evict(oldest_key)

        entry = CacheEntry(
            query=query,
            query_hash=q_hash,
            embedding=q_embed.tolist(),
            result=result,
            timestamp=time.time(),
        )
        self._entries[q_hash] = entry
        self._embeddings.append(q_embed)
        self._keys.append(q_hash)

    def invalidate(self, query: str | None = None) -> int:
        """Invalidate cache entries. If query is None, clear everything."""
        if query is None:
            n = len(self._entries)
            self._entries.clear()
            self._embeddings.clear()
            self._keys.clear()
            return n

        q_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        if q_hash in self._entries:
            self._evict(q_hash)
            return 1
        return 0

    def _evict(self, key: str) -> None:
        if key in self._entries:
            idx = self._keys.index(key)
            del self._entries[key]
            del self._embeddings[idx]
            del self._keys[idx]
            self.stats.evictions += 1

    def save(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return
        data = []
        for entry in self._entries.values():
            data.append({
                "query": entry.query,
                "query_hash": entry.query_hash,
                "embedding": entry.embedding,
                "result": entry.result,
                "timestamp": entry.timestamp,
                "hit_count": entry.hit_count,
            })
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps(data, default=str))

    def _load(self) -> None:
        try:
            data = json.loads(self.persist_path.read_text())
            for item in data:
                entry = CacheEntry(
                    query=item["query"],
                    query_hash=item["query_hash"],
                    embedding=item["embedding"],
                    result=item["result"],
                    timestamp=item["timestamp"],
                    hit_count=item.get("hit_count", 0),
                )
                self._entries[entry.query_hash] = entry
                self._embeddings.append(np.array(entry.embedding))
                self._keys.append(entry.query_hash)
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._entries)
