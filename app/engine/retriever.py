"""
Hybrid Retriever with RRF Fusion and Cross-Encoder Reranking

Combines BM25 (sparse) and vector (dense) retrieval using Reciprocal Rank
Fusion, then reranks results with a cross-encoder model.

Retrieve wide (top-50) and cheap, then rerank to top-8. Optional rule-based
query router skips embed+rerank for exact path lookups.
"""

import os
import logging
from typing import List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.eval.latency import StageTimer
from app.retrieval.fuse import RRF_K, rrf_score_map
from app.retrieval.router import QueryRouter, RouteType

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results_lists: List[List[NodeWithScore]],
    k: int = RRF_K,
) -> List[NodeWithScore]:
    """Fuse multiple ranked lists using RRF. Returns nodes sorted by fused score."""
    id_lists: List[List[str]] = []
    node_map: dict[str, NodeWithScore] = {}
    for results in results_lists:
        ids = []
        for node_with_score in results:
            node_id = node_with_score.node.node_id
            ids.append(node_id)
            prev = node_map.get(node_id)
            if prev is None or (node_with_score.score or 0) > (prev.score or 0):
                node_map[node_id] = node_with_score
        id_lists.append(ids)

    fused_scores = rrf_score_map(id_lists, k=k)
    fused = []
    for node_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        original = node_map[node_id]
        fused.append(NodeWithScore(node=original.node, score=score))
    return fused


class HybridRetriever(BaseRetriever):
    """Retriever that combines vector search and BM25, fused with RRF."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        vector_store_manager,
        similarity_top_k: int = 50,
        bm25_top_k: int = 50,
        reranker=None,
        rerank_top_n: int = 8,
        router: Optional[QueryRouter] = None,
        enable_router: bool = True,
    ):
        self._vector_retriever = vector_retriever
        self._vsm = vector_store_manager
        self._similarity_top_k = similarity_top_k
        self._bm25_top_k = bm25_top_k
        self._reranker = reranker
        self._rerank_top_n = rerank_top_n
        self._router = router or QueryRouter()
        self._enable_router = enable_router
        self.last_timer: Optional[StageTimer] = None
        self.last_route = None
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        timer = StageTimer()
        self.last_timer = timer

        with timer.stage("route"):
            route = self._router.route(query_str) if self._enable_router else None
        self.last_route = route

        skip_embed = bool(route and route.skip_embed())
        skip_rerank = bool(route and route.skip_rerank())

        bm25_nodes: List[NodeWithScore] = []
        vector_results: List[NodeWithScore] = []

        with timer.stage("bm25"):
            bm25_nodes = self._bm25_retrieve(query_str)
        logger.info(f"BM25 retrieval returned {len(bm25_nodes)} results")

        if not skip_embed:
            with timer.stage("embed"):
                vector_results = self._vector_retriever.retrieve(query_str)
            logger.info(f"Vector retrieval returned {len(vector_results)} results")

        with timer.stage("vector_search"):
            if not bm25_nodes:
                fused = vector_results
            elif not vector_results:
                fused = bm25_nodes
            else:
                fused = reciprocal_rank_fusion([vector_results, bm25_nodes])

        logger.info(f"RRF fusion produced {len(fused)} unique results")

        if self._reranker and fused and not skip_rerank:
            with timer.stage("rerank"):
                fused = self._rerank(query_str, fused)
        elif fused:
            fused = fused[: self._rerank_top_n]

        return fused

    def _bm25_retrieve(self, query_str: str) -> List[NodeWithScore]:
        """Run BM25 search via the vector store manager."""
        try:
            raw_results = self._vsm.search_bm25(query=query_str, limit=self._bm25_top_k)
            nodes = []
            for r in raw_results:
                node = TextNode(
                    id_=r["node_id"],
                    text=r["text"],
                )
                nodes.append(NodeWithScore(node=node, score=float(r["score"])))
            return nodes
        except Exception as e:
            logger.warning(f"BM25 retrieval failed, falling back to vector-only: {e}")
            return []

    def _rerank(self, query_str: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Rerank nodes using the cross-encoder model."""
        try:
            texts = [n.node.get_content() for n in nodes]
            pairs = [[query_str, text] for text in texts]
            scores = self._reranker.compute_score(pairs)

            if isinstance(scores, (int, float)):
                scores = [scores]

            scored = list(zip(nodes, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for node, score in scored[: self._rerank_top_n]:
                reranked.append(NodeWithScore(node=node.node, score=float(score)))

            logger.info(f"Reranker selected top {len(reranked)} results")
            return reranked
        except Exception as e:
            logger.warning(f"Reranking failed, returning RRF results: {e}")
            return nodes[: self._rerank_top_n]


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
) -> HybridRetriever:
    """Factory function to create a configured HybridRetriever."""
    retrieve_k = int(os.getenv("RETRIEVE_K", "50"))
    default_n = int(os.getenv("RERANK_TOP_N", os.getenv("TOP_K", "8")))
    similarity_top_k = similarity_top_k or retrieve_k
    bm25_top_k = bm25_top_k or retrieve_k
    rerank_top_n = rerank_top_n or default_n
    enable_router = os.getenv("ENABLE_ROUTER", "true").lower() != "false"

    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    reranker = get_reranker() if use_reranker else None

    return HybridRetriever(
        vector_retriever=vector_retriever,
        vector_store_manager=vector_store_manager,
        similarity_top_k=similarity_top_k,
        bm25_top_k=bm25_top_k,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
        enable_router=enable_router,
    )
