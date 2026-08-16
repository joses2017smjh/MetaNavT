"""
Hybrid Retriever with RRF Fusion and Cross-Encoder Reranking

Combines BM25 (sparse) and vector (dense) retrieval using Reciprocal Rank
Fusion, then reranks results with a cross-encoder model.
"""

import os
import logging
from typing import List, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    results_lists: List[List[NodeWithScore]],
    k: int = RRF_K,
) -> List[NodeWithScore]:
    """Fuse multiple ranked lists using RRF. Returns nodes sorted by fused score."""
    fused_scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for results in results_lists:
        for rank, node_with_score in enumerate(results):
            node_id = node_with_score.node.node_id
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + 1.0 / (k + rank + 1)
            if node_id not in node_map or node_with_score.score > node_map[node_id].score:
                node_map[node_id] = node_with_score

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
        similarity_top_k: int = 10,
        bm25_top_k: int = 10,
        reranker=None,
        rerank_top_n: int = 5,
    ):
        self._vector_retriever = vector_retriever
        self._vsm = vector_store_manager
        self._similarity_top_k = similarity_top_k
        self._bm25_top_k = bm25_top_k
        self._reranker = reranker
        self._rerank_top_n = rerank_top_n
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str

        vector_results = self._vector_retriever.retrieve(query_str)
        logger.info(f"Vector retrieval returned {len(vector_results)} results")

        bm25_nodes = self._bm25_retrieve(query_str)
        logger.info(f"BM25 retrieval returned {len(bm25_nodes)} results")

        if not bm25_nodes:
            fused = vector_results
        elif not vector_results:
            fused = bm25_nodes
        else:
            fused = reciprocal_rank_fusion([vector_results, bm25_nodes])

        logger.info(f"RRF fusion produced {len(fused)} unique results")

        if self._reranker and fused:
            fused = self._rerank(query_str, fused)

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
    top_k = int(os.getenv("TOP_K", "5"))
    similarity_top_k = similarity_top_k or top_k * 2
    bm25_top_k = bm25_top_k or top_k * 2
    rerank_top_n = rerank_top_n or top_k

    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    reranker = get_reranker() if use_reranker else None

    return HybridRetriever(
        vector_retriever=vector_retriever,
        vector_store_manager=vector_store_manager,
        similarity_top_k=similarity_top_k,
        bm25_top_k=bm25_top_k,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
    )
