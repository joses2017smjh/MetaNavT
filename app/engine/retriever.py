"""
Hybrid Retriever with RRF Fusion and Cross-Encoder Reranking

Combines BM25 (sparse) and vector (dense) retrieval using Reciprocal Rank
Fusion, then reranks results with a cross-encoder model.

Retrieve wide (top-50) and cheap, then rerank to top-8. Optional rule-based
query router skips embed+rerank for exact path lookups.
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.eval.latency import StageTimer
from app.graph.staleness import prefer_current
from app.retrieval.fuse import RRF_K, rrf_score_map
from app.retrieval.router import QueryRouter, RouteType
from app.retrieval.types import Chunk, RetrievalHit

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = ("sql", "python")
DROP_METADATA_KEYS = {"_node_content", "_node_type"}  # LlamaIndex's serialised node; large and redundant


@dataclass
class RetrievalOutcome:
    """Everything one retrieval produced, per request (no shared mutable state)."""

    nodes: List[NodeWithScore]
    route: Any
    stages_ms: dict[str, float]
    counts: dict[str, int]
    scores: dict[str, dict[str, Optional[float]]]  # node_id -> bm25 / dense / rrf / rerank
    staleness: dict[str, Any]
    mode: str
    bm25_backend: Optional[str] = None
    skipped_embed: bool = False
    skipped_rerank: bool = False
    timer: StageTimer = field(default_factory=StageTimer)


def node_to_chunk(node: NodeWithScore) -> Chunk:
    """A Chunk view of a LlamaIndex node so app/graph code (staleness) can be reused unchanged."""
    meta = node.node.metadata or {}
    text = node.node.get_content()
    return Chunk(
        chunk_id=node.node.node_id,
        path=meta.get("path") or meta.get("file_path") or meta.get("file_name") or node.node.node_id,
        text=text,
        start_byte=int(meta.get("start_byte") or 0),
        end_byte=int(meta.get("end_byte") or len(text.encode("utf-8"))),
        mtime=float(meta.get("mtime") or 0.0),
        content_hash=str(meta.get("content_hash") or ""),
        metadata=meta,
    )


def reciprocal_rank_fusion(
    results_lists: List[List[NodeWithScore]],
    k: int = RRF_K,
) -> List[NodeWithScore]:
    """Fuse multiple ranked lists using RRF. Returns nodes sorted by fused score.

    A node that appears in several lists keeps the union of its metadata. The
    first occurrence's node object is kept; keys it lacks are copied from the
    later ones. (Before M0 the duplicate with the higher raw score won, so a
    BM25 node without metadata could replace the vector node carrying file_path.)
    """
    id_lists: List[List[str]] = []
    node_map: dict[str, NodeWithScore] = {}
    merged_meta: dict[str, dict] = {}
    for results in results_lists:
        ids = []
        for node_with_score in results:
            node_id = node_with_score.node.node_id
            ids.append(node_id)
            node_map.setdefault(node_id, node_with_score)
            meta = merged_meta.setdefault(node_id, {})
            for key, value in (node_with_score.node.metadata or {}).items():
                meta.setdefault(key, value)
        id_lists.append(ids)

    fused_scores = rrf_score_map(id_lists, k=k)
    fused = []
    for node_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        node = node_map[node_id].node
        extra = {key: value for key, value in merged_meta[node_id].items() if key not in node.metadata}
        if extra:
            node.metadata.update(extra)
        fused.append(NodeWithScore(node=node, score=score))
    return fused


class HybridRetriever(BaseRetriever):
    """Retriever that combines vector search and lexical search, fused with RRF.

    mode="sql"    one Postgres round trip (VectorStoreManager.hybrid_search: pgvector
                  top-k + full-text top-k + RRF in SQL, app/database/sql/hybrid_*.sql);
                  needs embed_fn for the query vector.
    mode="python" the pre-M3 path: LlamaIndex vector retriever + search_bm25 + RRF in
                  Python (reciprocal_rank_fusion below). Kept as the reference
                  implementation and for stores without the SQL path.
    Staleness Tier 1 (app/graph/staleness.prefer_current) runs after fusion when
    `clusters` is given; the reranker runs last.
    """

    def __init__(
        self,
        vector_retriever: Optional[BaseRetriever],
        vector_store_manager,
        similarity_top_k: int = 50,
        bm25_top_k: int = 50,
        reranker=None,
        rerank_top_n: int = 8,
        router: Optional[QueryRouter] = None,
        enable_router: bool = True,
        mode: str = "python",
        embed_fn: Optional[Callable[[str], list]] = None,
        clusters: Optional[dict] = None,
        rrf_k: int = RRF_K,
    ):
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}, got {mode!r}")
        if mode == "sql" and embed_fn is None:
            raise ValueError("mode='sql' needs embed_fn (query -> embedding)")
        if mode == "python" and vector_retriever is None:
            raise ValueError("mode='python' needs a vector_retriever")
        self._vector_retriever = vector_retriever
        self._vsm = vector_store_manager
        self._similarity_top_k = similarity_top_k
        self._bm25_top_k = bm25_top_k
        self._reranker = reranker
        self._rerank_top_n = rerank_top_n
        self._router = router or QueryRouter()
        self._enable_router = enable_router
        self._mode = mode
        self._embed_fn = embed_fn
        self._clusters = clusters or {}
        self._rrf_k = rrf_k
        # Compatibility mirrors of the last outcome (single-threaded callers only).
        self.last_timer: Optional[StageTimer] = None
        self.last_route = None
        self.last_counts: dict[str, int] = {}
        super().__init__()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def staleness_enabled(self) -> bool:
        return bool(self._clusters)

    # ------------------------------------------------------------------ public

    def retrieve_detailed(self, query_str: str, top_n: Optional[int] = None) -> RetrievalOutcome:
        """Route, retrieve, fuse, prefer current versions, rerank; return everything per request."""
        timer = StageTimer()
        t0 = time.perf_counter()
        with timer.stage("route"):
            route = self._router.route(query_str) if self._enable_router else None
        skip_embed = bool(route and route.skip_embed())
        skip_rerank = bool(route and route.skip_rerank())
        n = top_n or self._rerank_top_n

        if self._mode == "sql":
            fused, scores, counts, backend = self._fuse_sql(query_str, timer, skip_embed)
        else:
            fused, scores, counts, backend = self._fuse_python(query_str, timer, skip_embed)
        counts["fused"] = len(fused)

        staleness = {"enabled": self.staleness_enabled, "applied": False, "dropped": 0}
        if self._clusters and fused:
            with timer.stage("staleness"):
                before = len(fused)
                fused = self._prefer_current(query_str, fused)
                staleness.update(applied=True, dropped=before - len(fused))

        if self._reranker and fused and not skip_rerank:
            with timer.stage("rerank"):
                fused = self._rerank(query_str, fused, n=n)
            for node in fused:
                scores.setdefault(node.node.node_id, {})["rerank"] = node.score
        elif fused:
            fused = fused[:n]
        counts["returned"] = len(fused)

        stages_ms = {name: round(stats.samples_ms[-1], 3) for name, stats in timer.stages.items() if stats.samples_ms}
        stages_ms["total"] = round((time.perf_counter() - t0) * 1000.0, 3)
        outcome = RetrievalOutcome(
            nodes=fused,
            route=route,
            stages_ms=stages_ms,
            counts=counts,
            scores=scores,
            staleness=staleness,
            mode=self._mode,
            bm25_backend=backend,
            skipped_embed=skip_embed,
            skipped_rerank=skip_rerank or self._reranker is None,
            timer=timer,
        )
        self.last_timer, self.last_route, self.last_counts = timer, route, counts
        return outcome

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.retrieve_detailed(query_bundle.query_str).nodes

    # ------------------------------------------------------------------ fusion backends

    def _fuse_sql(self, query_str: str, timer: StageTimer, skip_embed: bool):
        qvec = None
        if not skip_embed:
            with timer.stage("embed"):
                qvec = self._embed_fn(query_str)
        with timer.stage("hybrid_sql"):
            rows = self._vsm.hybrid_search(query_str, qvec, k=self._similarity_top_k, rrf_k=self._rrf_k)
        fused: List[NodeWithScore] = []
        scores: dict[str, dict[str, Optional[float]]] = {}
        for r in rows:
            meta = {k: v for k, v in (r.get("metadata") or {}).items() if k not in DROP_METADATA_KEYS}
            node = TextNode(id_=r["node_id"], text=r["text"], metadata=meta)
            fused.append(NodeWithScore(node=node, score=r["rrf_score"]))
            scores[r["node_id"]] = {"bm25": r.get("bm25_score"), "dense": r.get("dense_score"), "rrf": r["rrf_score"], "rerank": None}
        counts = {
            "bm25": sum(1 for r in rows if r.get("bm25_rank") is not None),
            "vector": sum(1 for r in rows if r.get("dense_rank") is not None),
        }
        return fused, scores, counts, getattr(self._vsm, "last_bm25_backend", None)

    def _fuse_python(self, query_str: str, timer: StageTimer, skip_embed: bool):
        with timer.stage("bm25"):
            bm25_nodes = self._bm25_retrieve(query_str)
        vector_results: List[NodeWithScore] = []
        if not skip_embed:
            with timer.stage("embed"):
                vector_results = self._vector_retriever.retrieve(query_str)
        with timer.stage("vector_search"):
            if not bm25_nodes:
                fused = list(vector_results)
            elif not vector_results:
                fused = list(bm25_nodes)
            else:
                fused = reciprocal_rank_fusion([vector_results, bm25_nodes], k=self._rrf_k)
        bm25_scores = {n.node.node_id: n.score for n in bm25_nodes}
        dense_scores = {n.node.node_id: n.score for n in vector_results}
        scores = {
            n.node.node_id: {"bm25": bm25_scores.get(n.node.node_id), "dense": dense_scores.get(n.node.node_id), "rrf": n.score, "rerank": None}
            for n in fused
        }
        counts = {"bm25": len(bm25_nodes), "vector": len(vector_results)}
        return fused, scores, counts, getattr(self._vsm, "last_bm25_backend", None)

    def _prefer_current(self, query_str: str, fused: List[NodeWithScore]) -> List[NodeWithScore]:
        hits = [RetrievalHit(chunk=node_to_chunk(n), score=float(n.score or 0.0), rank=i) for i, n in enumerate(fused, start=1)]
        kept = prefer_current(hits, self._clusters, query_str)
        keep_ids = [h.chunk.chunk_id for h in kept]
        by_id = {n.node.node_id: n for n in fused}
        return [by_id[i] for i in keep_ids if i in by_id]

    def _bm25_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Run BM25 search via the vector store manager."""
        try:
            raw_results = self._vsm.search_bm25(query=query_str, limit=self._bm25_top_k)
            nodes = []
            for r in raw_results:
                node = TextNode(
                    id_=r["node_id"],
                    text=r["text"],
                    metadata=dict(r.get("metadata") or {}),  # file_path etc. from metadata_
                )
                nodes.append(NodeWithScore(node=node, score=float(r["score"])))
            return nodes
        except Exception as e:
            logger.warning(f"BM25 retrieval failed, falling back to vector-only: {e}")
            return []

    def _rerank(self, query_str: str, nodes: List[NodeWithScore], n: Optional[int] = None) -> List[NodeWithScore]:
        """Rerank nodes using the cross-encoder model; keep the top n (default rerank_top_n)."""
        n = n or self._rerank_top_n
        try:
            from app.retrieval.rerank import cross_encoder_scores

            pairs = [(query_str, n.node.get_content()) for n in nodes]
            scores = cross_encoder_scores(self._reranker, pairs)  # predict() or compute_score()

            scored = list(zip(nodes, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for node, score in scored[:n]:
                reranked.append(NodeWithScore(node=node.node, score=float(score)))

            logger.info(f"Reranker selected top {len(reranked)} results")
            return reranked
        except Exception as e:
            logger.warning(f"Reranking failed, returning RRF results: {e}")
            return nodes[:n]


_reranker_model = None


def get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    reranker_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    if reranker_name.lower() == "none":
        return None

    try:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {reranker_name}")
        _reranker_model = CrossEncoder(reranker_name)
        return _reranker_model
    except ImportError:
        logger.warning("sentence-transformers not installed; reranking disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to load reranker model '{reranker_name}': {e}")
        return None


def create_hybrid_retriever(
    index,
    vector_store_manager,
    similarity_top_k: Optional[int] = None,
    bm25_top_k: Optional[int] = None,
    rerank_top_n: Optional[int] = None,
    use_reranker: bool = True,
    mode: Optional[str] = None,
    embed_fn: Optional[Callable[[str], list]] = None,
    clusters: Optional[dict] = None,
) -> HybridRetriever:
    """Factory: RETRIEVE_K / RERANK_TOP_N / ENABLE_ROUTER / RETRIEVAL_MODE from the environment."""
    retrieve_k = int(os.getenv("RETRIEVE_K", "50"))
    default_n = int(os.getenv("RERANK_TOP_N", os.getenv("TOP_K", "8")))
    similarity_top_k = similarity_top_k or retrieve_k
    bm25_top_k = bm25_top_k or retrieve_k
    rerank_top_n = rerank_top_n or default_n
    enable_router = os.getenv("ENABLE_ROUTER", "true").lower() != "false"
    mode = mode or os.getenv("RETRIEVAL_MODE", "sql").strip().lower()

    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k) if index is not None else None
    if mode == "sql" and embed_fn is None:
        from llama_index.core.settings import Settings

        embed_fn = Settings.embed_model.get_query_embedding

    reranker = get_reranker() if use_reranker else None

    return HybridRetriever(
        vector_retriever=vector_retriever,
        vector_store_manager=vector_store_manager,
        similarity_top_k=similarity_top_k,
        bm25_top_k=bm25_top_k,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
        enable_router=enable_router,
        mode=mode,
        embed_fn=embed_fn,
        clusters=clusters,
    )
