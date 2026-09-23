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


class _ReversingCrossEncoder:
    """sentence-transformers CrossEncoder shape: predict() scores pairs; here later pairs score higher."""

    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.calls.append(list(pairs))
        return [float(i) for i in range(len(pairs))]


def test_production_rerank_uses_predict_and_truncates_to_rerank_top_n():
    model = _ReversingCrossEncoder()
    retriever = HybridRetriever(
        vector_retriever=_FakeVectorRetriever(),
        vector_store_manager=_FakeVSM(),
        reranker=model,
        rerank_top_n=2,
        enable_router=False,
    )

    nodes = retriever.retrieve("anything")

    assert len(model.calls) == 1 and len(model.calls[0]) == 3  # every fused node was scored
    assert [n.node.node_id for n in nodes] == [n.node.node_id for n in nodes][:2] and len(nodes) == 2
    fused_order = [p[1] for p in model.calls[0]]
    assert nodes[0].node.get_content() == fused_order[-1]  # highest score = last pair
    assert nodes[0].score == 2.0 and nodes[1].score == 1.0
    assert retriever.last_counts["returned"] == 2 and retriever.last_counts["fused"] == 3


# ---------------------------------------------------------------- M3: SQL mode, staleness, per-request outcome


class _FakeSQLVSM:
    """hybrid_search rows as VectorStoreManager returns them (one round trip, RRF in SQL)."""

    def __init__(self):
        self.calls = []
        self.last_bm25_backend = "ts_rank_cd"

    def hybrid_search(self, query, query_embedding=None, k=50, rrf_k=60):
        self.calls.append((query, query_embedding, k, rrf_k))
        rows = [
            ("cur", "learning_rate: 3e-4", {"path": "configs/run_047.yaml", "mtime": 2.0, "start_byte": 0, "end_byte": 19}, 0.9, 1, 0.8, 1, 1 / 61 + 1 / 61),
            ("old", "learning_rate: 1e-4", {"path": "configs/archive/run_047_v1.yaml", "mtime": 1.0, "start_byte": 0, "end_byte": 19}, 0.5, 2, None, None, 1 / 62),
            ("log", "run 47 val_rmse 0.055", {"path": "logs/run_047.out", "mtime": 2.0, "_node_content": "{...}"}, None, None, 0.4, 2, 1 / 62),
        ]
        return [
            {"node_id": n, "text": t, "metadata": m, "bm25_score": bs, "bm25_rank": br, "dense_score": ds, "dense_rank": dr, "rrf_score": rrf}
            for n, t, m, bs, br, ds, dr, rrf in rows
        ]


def _clusters_for_fake_rows():
    from app.graph.staleness import cluster_versions
    from app.retrieval.types import Chunk

    return cluster_versions(
        [
            Chunk(chunk_id="cur", path="configs/run_047.yaml", text="learning_rate: 3e-4", start_byte=0, end_byte=19, mtime=2.0),
            Chunk(chunk_id="old", path="configs/archive/run_047_v1.yaml", text="learning_rate: 1e-4", start_byte=0, end_byte=19, mtime=1.0),
        ]
    )


def test_sql_mode_one_round_trip_with_scores_and_staleness():
    vsm = _FakeSQLVSM()
    embed = lambda q: [0.1, 0.2]  # noqa: E731
    retriever = HybridRetriever(None, vsm, reranker=None, rerank_top_n=8, enable_router=False, mode="sql", embed_fn=embed, clusters=_clusters_for_fake_rows())

    out = retriever.retrieve_detailed("current learning rate for run 47", top_n=10)

    assert vsm.calls == [("current learning rate for run 47", [0.1, 0.2], 50, 60)]
    assert out.mode == "sql" and out.bm25_backend == "ts_rank_cd"
    assert out.counts == {"bm25": 2, "vector": 2, "fused": 3, "returned": 2}
    assert [n.node.node_id for n in out.nodes] == ["cur", "log"]  # superseded archive copy dropped
    assert out.staleness == {"enabled": True, "applied": True, "dropped": 1}
    assert out.scores["cur"] == {"bm25": 0.9, "dense": 0.8, "rrf": 1 / 61 + 1 / 61, "rerank": None}
    assert out.scores["log"]["bm25"] is None and out.scores["log"]["dense"] == 0.4
    assert "_node_content" not in out.nodes[1].node.metadata
    assert set(out.stages_ms) >= {"route", "embed", "hybrid_sql", "staleness", "total"}


def test_sql_mode_comparative_query_keeps_both_versions_and_lexical_route_skips_embed():
    vsm = _FakeSQLVSM()
    calls = []
    retriever = HybridRetriever(None, vsm, reranker=None, enable_router=True, mode="sql", embed_fn=lambda q: calls.append(q) or [0.0], clusters=_clusters_for_fake_rows())

    out = retriever.retrieve_detailed("compare run_047.yaml with configs/archive/run_047_v1.yaml", top_n=10)

    assert out.staleness["dropped"] == 0 and len(out.nodes) == 3
    assert out.route is not None and out.skipped_embed is True and calls == []  # lexical_path: no embedding
    assert vsm.calls[-1][1] is None  # SQL ran lexical-only


def test_python_mode_still_reports_scores_and_top_n():
    retriever = _retriever()

    out = retriever.retrieve_detailed("anything", top_n=2)

    assert out.mode == "python" and len(out.nodes) == 2 and out.counts["returned"] == 2
    a = out.scores["a"]
    assert a["dense"] == 0.9 and a["bm25"] is None and a["rrf"] is not None and a["rerank"] is None
    assert out.staleness == {"enabled": False, "applied": False, "dropped": 0}


def test_mode_validation():
    import pytest

    with pytest.raises(ValueError):
        HybridRetriever(None, _FakeSQLVSM(), mode="sql")  # no embed_fn
    with pytest.raises(ValueError):
        HybridRetriever(None, _FakeSQLVSM(), mode="python")  # no vector retriever
    with pytest.raises(ValueError):
        HybridRetriever(_FakeVectorRetriever(), _FakeSQLVSM(), mode="graphql")
