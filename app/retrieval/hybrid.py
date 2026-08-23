"""In-memory hybrid index: BM25 + dense, fused with RRF, optional rerank/router."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from app.eval.latency import StageTimer
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedders import Embedder, HashEmbedder, cosine_scores
from app.retrieval.fuse import rrf_score_map
from app.retrieval.router import QueryRouter, RouteDecision, RouteType
from app.retrieval.types import Chunk, RetrievalHit

RerankFn = Callable[[str, list[tuple[Chunk, float]]], list[tuple[Chunk, float]]]


@dataclass
class RetrievalResult:
    query: str
    hits: list[RetrievalHit]
    route: RouteDecision
    stages_ms: dict[str, float]
    skipped_rerank: bool = False
    skipped_embed: bool = False

    @property
    def paths(self) -> list[str]:
        seen = []
        for hit in self.hits:
            if hit.chunk.path not in seen:
                seen.append(hit.chunk.path)
        return seen

    @property
    def chunk_ids(self) -> list[str]:
        return [hit.chunk.chunk_id for hit in self.hits]


class InMemoryHybridIndex:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        retrieve_k: int = 50,
        rerank_n: int = 8,
        rrf_k: int = 60,
        router: QueryRouter | None = None,
        rerank_fn: RerankFn | None = None,
        enable_router: bool = True,
        enable_rerank: bool = True,
    ):
        if not chunks:
            raise ValueError("InMemoryHybridIndex requires at least one chunk")
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.retrieve_k = retrieve_k
        self.rerank_n = rerank_n
        self.rrf_k = rrf_k
        self.router = router or QueryRouter()
        self.rerank_fn = rerank_fn
        self.enable_router = enable_router
        self.enable_rerank = enable_rerank
        self.embedder = embedder or HashEmbedder()

        self.bm25 = BM25Index().fit(
            [c.chunk_id for c in self.chunks],
            [c.text for c in self.chunks],
        )
        if hasattr(self.embedder, "fit"):
            self.embedder.fit([c.text for c in self.chunks])  # type: ignore[attr-defined]
        self.doc_mat = self.embedder.encode([c.text for c in self.chunks])
        self._id_to_row = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    def search_bm25(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        k = k or self.retrieve_k
        return [
            (self.by_id[doc_id], score)
            for doc_id, score in self.bm25.search(query, k=k)
            if doc_id in self.by_id
        ]

    def search_dense(self, query: str, k: int | None = None, query_vec: np.ndarray | None = None) -> list[tuple[Chunk, float]]:
        k = k or self.retrieve_k
        qv = query_vec if query_vec is not None else self.embedder.encode([query])[0]
        scores = cosine_scores(qv, self.doc_mat)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def search_path(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        """Exact / substring path lookup for lexical_path routes."""
        k = k or self.retrieve_k
        q = query.lower()
        hits = []
        for chunk in self.chunks:
            path_l = chunk.path.lower()
            name = Path(chunk.path).name.lower()
            score = 0.0
            if name and name in q:
                score += 5.0
            if path_l and path_l in q:
                score += 3.0
            stem = Path(chunk.path).stem.lower()
            if stem and stem in q:
                score += 2.0
            if score > 0:
                hits.append((chunk, score))
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits[:k]

    def retrieve(
        self,
        query: str,
        timer: StageTimer | None = None,
        k: int | None = None,
        n: int | None = None,
    ) -> RetrievalResult:
        retrieve_k = k or self.retrieve_k
        rerank_n = n or self.rerank_n
        local = StageTimer() if timer is None else timer
        stage_ms: dict[str, float] = {}

        with local.stage("route"):
            if self.enable_router:
                decision = self.router.route(query)
            else:
                decision = RouteDecision(RouteType.SEMANTIC, "router off")
        stage_ms["route"] = local.stages["route"].samples_ms[-1]

        skip_embed = self.enable_router and decision.skip_embed()
        skip_rerank = (not self.enable_rerank) or (
            self.enable_router and decision.skip_rerank()
        )

        bm25_map: dict[str, float] = {}
        dense_map: dict[str, float] = {}
        ranked_ids: list[list[str]] = []

        if decision.route == RouteType.LEXICAL_PATH:
            with local.stage("lexical"):
                path_hits = self.search_path(query, k=retrieve_k)
            stage_ms["lexical"] = local.stages["lexical"].samples_ms[-1]
            ranked_ids.append([c.chunk_id for c, _ in path_hits])
            with local.stage("bm25"):
                bm25_hits = self.search_bm25(query, k=retrieve_k)
            stage_ms["bm25"] = local.stages["bm25"].samples_ms[-1]
            ranked_ids.append([c.chunk_id for c, _ in bm25_hits])
            bm25_map = {c.chunk_id: s for c, s in bm25_hits}
            fused = rrf_score_map(ranked_ids, k=self.rrf_k)
        else:
            with local.stage("bm25"):
                bm25_hits = self.search_bm25(query, k=retrieve_k)
            stage_ms["bm25"] = local.stages["bm25"].samples_ms[-1]
            ranked_ids.append([c.chunk_id for c, _ in bm25_hits])
            bm25_map = {c.chunk_id: s for c, s in bm25_hits}

            if not skip_embed:
                with local.stage("embed"):
                    dense_hits = self.search_dense(query, k=retrieve_k)
                stage_ms["embed"] = local.stages["embed"].samples_ms[-1]
                ranked_ids.append([c.chunk_id for c, _ in dense_hits])
                dense_map = {c.chunk_id: s for c, s in dense_hits}

            with local.stage("vector_search"):
                fused = rrf_score_map(ranked_ids, k=self.rrf_k)
            stage_ms["vector_search"] = local.stages["vector_search"].samples_ms[-1]

        fused_sorted = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:retrieve_k]
        pairs: list[tuple[Chunk, float]] = [
            (self.by_id[cid], score) for cid, score in fused_sorted if cid in self.by_id
        ]

        rerank_scores: dict[str, float] = {}
        if not skip_rerank and self.rerank_fn and pairs:
            with local.stage("rerank"):
                reranked = self.rerank_fn(query, pairs)
            stage_ms["rerank"] = local.stages["rerank"].samples_ms[-1]
            pairs = reranked
            rerank_scores = {c.chunk_id: s for c, s in pairs}
        # Keep the fused top-`retrieve_k` for Recall@50. Generation callers slice to rerank_n.

        hits = []
        for rank, (chunk, score) in enumerate(pairs, start=1):
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    bm25=bm25_map.get(chunk.chunk_id),
                    dense=dense_map.get(chunk.chunk_id),
                    rrf=fused.get(chunk.chunk_id),
                    rerank=rerank_scores.get(chunk.chunk_id),
                )
            )
        return RetrievalResult(
            query=query,
            hits=hits,
            route=decision,
            stages_ms=stage_ms,
            skipped_rerank=skip_rerank or self.rerank_fn is None,
            skipped_embed=skip_embed,
        )


def chunk_id_for(path: str, start: int, end: int) -> str:
    digest = hashlib.sha1(f"{path}:{start}:{end}".encode()).hexdigest()[:12]
    return f"{path}::{digest}"
