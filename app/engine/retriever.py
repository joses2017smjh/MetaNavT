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
from app.observability import span
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
    degraded: List[dict] = field(default_factory=list)  # [{"component": "rerank", "error": "..."}], never silent
    timer: StageTimer = field(default_factory=StageTimer)


class RerankerUnavailable(RuntimeError):
    """A reranker was configured but could not be loaded (and RERANKER_REQUIRED is true)."""


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
        rerank_depth: Optional[int] = None,
        reranker_info: Optional[dict] = None,
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
        self._rerank_depth = rerank_depth  # rerank the fused top-N only; the rest keep RRF order
        self.reranker_info = dict(reranker_info or {"configured": reranker is not None, "loaded": reranker is not None})
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
        with span("retrieve", mode=self._mode), timer.stage("route"), span("route"):
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
            with timer.stage("staleness"), span("staleness"):
                before = len(fused)
                fused = self._prefer_current(query_str, fused)
                staleness.update(applied=True, dropped=before - len(fused))

        degraded: List[dict] = []
        if self._reranker and fused and not skip_rerank:
            with timer.stage("rerank"), span("rerank", depth=self._rerank_depth, n=len(fused)):
                fused, rerank_error = self._rerank(query_str, fused, n=n)
            if rerank_error:
                degraded.append({"component": "rerank", "error": rerank_error})
            else:
                head = self._rerank_depth or len(fused)
                for node in fused[:head]:
                    scores.setdefault(node.node.node_id, {})["rerank"] = node.score
        elif fused:
            fused = fused[:n]
        if self.reranker_info.get("configured") and not self.reranker_info.get("loaded"):
            degraded.append({"component": "rerank", "error": self.reranker_info.get("error") or "configured but not loaded"})
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
            degraded=degraded + self._bm25_degraded,
            timer=timer,
        )
        self.last_timer, self.last_route, self.last_counts = timer, route, counts
        return outcome

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self.retrieve_detailed(query_bundle.query_str).nodes

    # ------------------------------------------------------------------ fusion backends

    _bm25_degraded: List[dict] = []

    def _fuse_sql(self, query_str: str, timer: StageTimer, skip_embed: bool):
        self._bm25_degraded = []
        qvec = None
        if not skip_embed:
            with timer.stage("embed"), span("embed"):
                qvec = self._embed_fn(query_str)
        with timer.stage("hybrid_sql"), span("hybrid_sql", k=self._similarity_top_k):
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
        self._bm25_degraded = []
        with timer.stage("bm25"), span("bm25"):
            bm25_nodes = self._bm25_retrieve(query_str)
        vector_results: List[NodeWithScore] = []
        if not skip_embed:
            with timer.stage("embed"), span("embed"):
                vector_results = self._vector_retriever.retrieve(query_str)
        with timer.stage("vector_search"), span("vector_search"):
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
        except Exception as e:  # noqa: BLE001 - recorded on the outcome, never silent
            logger.warning(f"BM25 retrieval failed; this request is degraded to vector-only: {e}")
            self._bm25_degraded = [{"component": "bm25", "error": f"{type(e).__name__}: {e}"}]
            return []

    def _rerank(self, query_str: str, nodes: List[NodeWithScore], n: Optional[int] = None):
        """Rerank the fused top-`rerank_depth` with the cross-encoder; the rest keep RRF order.

        Returns (nodes[:n], error). A failure is returned, not hidden: the caller
        records it on the outcome as a degraded component and the unreranked
        RRF order is served for that request.
        """
        n = n or self._rerank_top_n
        depth = self._rerank_depth or len(nodes)
        head, tail = nodes[:depth], nodes[depth:]
        try:
            from app.retrieval.rerank import cross_encoder_scores

            scores = cross_encoder_scores(self._reranker, [(query_str, h.node.get_content()) for h in head])
        except Exception as e:  # noqa: BLE001 - surfaced as degraded
            logger.warning(f"Reranking failed for this request; serving RRF order: {e}")
            return nodes[:n], f"{type(e).__name__}: {e}"
        scored = sorted(zip(head, scores), key=lambda x: x[1], reverse=True)
        reranked = [NodeWithScore(node=node.node, score=float(score)) for node, score in scored]
        return (reranked + tail)[:n], None


_reranker_cache: dict[str, tuple] = {}


def reranker_settings() -> dict:
    """The served reranker configuration, from the environment.

    Defaults come from the measured SciFact sweep (bench/results/beir_scifact.json):
    fp16 with max_length 512 is a quality tie with fp32 uncapped at a fraction of
    the latency, and reranking deeper than 20 loses nDCG@10 while costing more.
    """
    try:
        import torch  # type: ignore

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    precision = os.getenv("RERANKER_PRECISION", "fp16" if cuda else "fp32").strip().lower()
    return {
        "model": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        "precision": precision if precision in {"fp16", "fp32"} else "fp32",
        "max_length": int(os.getenv("RERANKER_MAX_LENGTH", "512")),
        "depth": int(os.getenv("RERANK_DEPTH", "20")),
        "required": os.getenv("RERANKER_REQUIRED", "true").strip().lower() != "false",
        "device": os.getenv("RERANKER_DEVICE") or ("cuda" if cuda else "cpu"),
    }


def load_reranker(settings: Optional[dict] = None):
    """Load the configured cross-encoder; return (model_or_None, info).

    RERANKER_MODEL=none means "not configured" (info.configured False). A
    configured model that cannot be loaded raises RerankerUnavailable unless
    RERANKER_REQUIRED=false, in which case the info block says loaded=False with
    the error and every response lists the component as degraded. Nothing
    falls back silently.
    """
    st = settings or reranker_settings()
    name = st["model"]
    info = {
        "configured": bool(name) and name.lower() != "none",
        "loaded": False,
        "model": None if not name or name.lower() == "none" else name,
        "revision": None,
        "device": None,
        "precision": st["precision"],
        "max_length": st["max_length"],
        "depth": st["depth"],
        "error": None,
    }
    if not info["configured"]:
        return None, info
    key = f"{name}|{st['precision']}|{st['max_length']}|{st['device']}"
    if key in _reranker_cache:
        return _reranker_cache[key]
    from app.eval.provenance import hf_revision, model_device
    from app.retrieval.rerank import get_cross_encoder

    error = None
    model = None
    try:
        model = get_cross_encoder(name, precision=st["precision"], max_length=st["max_length"], device=st["device"])
        if model is None:
            error = "not loadable: sentence-transformers missing, or weights not cached and BGE_ALLOW_DOWNLOAD unset"
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        error = f"{type(e).__name__}: {e}"
    info.update(loaded=model is not None, revision=hf_revision(name), device=model_device(model) if model else None, error=error)
    if model is not None:
        inner = getattr(model, "model", None)
        info["dtype"] = str(getattr(inner, "dtype", None)) if inner is not None else None
    elif st["required"]:
        raise RerankerUnavailable(f"RERANKER_MODEL={name} is configured but could not be loaded ({error}); set RERANKER_MODEL=none or RERANKER_REQUIRED=false")
    _reranker_cache[key] = (model, info)
    return model, info


def get_reranker():
    """Compatibility wrapper: the model only."""
    return load_reranker()[0]


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

    if use_reranker:
        reranker, info = load_reranker()
    else:
        reranker, info = None, {"configured": False, "loaded": False, "model": None}

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
        rerank_depth=info.get("depth"),
        reranker_info=info,
    )
