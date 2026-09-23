"""HybridRetriever (the Postgres/API path) must never lose a hit's path.

Two ways it used to: BM25 rows were turned into TextNode(id_, text) with no
metadata, and reciprocal_rank_fusion kept whichever duplicate had the higher
raw score, so a BM25 score above a cosine score replaced the node that carried
file_path. Real llama_index node types, fake vector store + retriever.
"""

from llama_index.core.schema import NodeWithScore, TextNode

from app.engine.retriever import HybridRetriever, reciprocal_rank_fusion


def _hit(node_id: str, text: str, score: float, meta: dict | None = None) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=meta or {}), score=score)


def test_rrf_keeps_metadata_when_the_bm25_duplicate_has_the_higher_raw_score():
    vector = [_hit("a", "learning_rate: 3e-4", 0.42, {"file_path": "configs/run_047.yaml"})]
    bm25 = [_hit("a", "learning_rate: 3e-4", 12.3)]  # BM25 raw score > cosine, no metadata

    fused = reciprocal_rank_fusion([vector, bm25])

    assert [n.node.node_id for n in fused] == ["a"]
    assert fused[0].node.metadata["file_path"] == "configs/run_047.yaml"


def test_rrf_merges_metadata_from_both_lists():
    vector = [_hit("a", "t", 0.9, {"file_path": "/abs/configs/run_047.yaml"})]
    bm25 = [_hit("a", "t", 0.1, {"path": "configs/run_047.yaml"})]

    fused = reciprocal_rank_fusion([vector, bm25])

    assert fused[0].node.metadata == {"file_path": "/abs/configs/run_047.yaml", "path": "configs/run_047.yaml"}


def test_rrf_scores_and_order():
    vector = [_hit("a", "t", 0.9), _hit("b", "t", 0.8)]
    bm25 = [_hit("b", "t", 5.0), _hit("c", "t", 4.0)]

    fused = reciprocal_rank_fusion([vector, bm25], k=60)

    assert [n.node.node_id for n in fused] == ["b", "a", "c"]
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused[1].score - 1 / 61) < 1e-12


class _FakeVSM:
    """search_bm25 rows as VectorStoreManager returns them: one with metadata, one without."""

    def __init__(self):
        self.calls = []

    def search_bm25(self, query: str, limit: int):
        self.calls.append((query, limit))
        return [
            {"node_id": "b", "text": "encoder: dinov2", "score": 3.0, "metadata": {"file_path": "configs/run_046.yaml"}},
            {"node_id": "c", "text": "lr 3e-4", "score": 2.0},  # legacy row shape: no metadata key
        ]


class _FakeVectorRetriever:
    def retrieve(self, query: str):
        return [
            _hit("a", "learning_rate: 3e-4", 0.9, {"file_path": "configs/run_047.yaml"}),
            _hit("c", "lr 3e-4", 0.5, {"file_path": "notes/lr.md"}),
        ]


def _retriever(vsm=None):
    return HybridRetriever(
        vector_retriever=_FakeVectorRetriever(),
        vector_store_manager=vsm or _FakeVSM(),
        similarity_top_k=50,
        bm25_top_k=50,
        reranker=None,
        rerank_top_n=8,
        enable_router=False,
    )


def test_every_hit_has_a_path():
    retriever = _retriever()

    nodes = retriever.retrieve("current learning rate for run 47")

    paths = {n.node.node_id: n.node.metadata.get("file_path") for n in nodes}
    assert paths == {"a": "configs/run_047.yaml", "b": "configs/run_046.yaml", "c": "notes/lr.md"}


def test_counts_and_bm25_limit():
    vsm = _FakeVSM()
    retriever = _retriever(vsm)

    nodes = retriever.retrieve("anything")

    assert vsm.calls == [("anything", 50)]
    assert retriever.last_counts == {"bm25": 2, "vector": 2, "fused": 3, "returned": len(nodes)}
