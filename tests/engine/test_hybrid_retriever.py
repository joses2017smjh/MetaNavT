"""HybridRetriever (the Postgres/API path) must never lose a hit's path.

Two ways it used to: BM25 rows were turned into TextNode(id_, text) with no
metadata, and reciprocal_rank_fusion kept whichever duplicate had the higher
raw score, so a BM25 score above a cosine score replaced the node that carried
file_path. Real llama_index node types, fake vector store + retriever.
"""

import time

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.database.vector_store_manager import SearchRows
from app.engine.retriever import Deadline, HybridRetriever, StageTimeout, reciprocal_rank_fusion


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

    out = retriever.retrieve_detailed("anything")

    assert vsm.calls == [("anything", 50)]
    assert out.counts == {"bm25": 2, "vector": 2, "fused": 3, "returned": len(out.nodes)}


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

    out = retriever.retrieve_detailed("anything")
    nodes = out.nodes

    assert len(model.calls) == 1 and len(model.calls[0]) == 3  # every fused node was scored
    assert [n.node.node_id for n in nodes] == [n.node.node_id for n in nodes][:2] and len(nodes) == 2
    fused_order = [p[1] for p in model.calls[0]]
    assert nodes[0].node.get_content() == fused_order[-1]  # highest score = last pair
    assert nodes[0].score == 2.0 and nodes[1].score == 1.0
    assert out.counts["returned"] == 2 and out.counts["fused"] == 3


# ---------------------------------------------------------------- M3: SQL mode, staleness, per-request outcome


class _FakeSQLVSM:
    """hybrid_search rows as VectorStoreManager returns them (one round trip, RRF in SQL)."""

    def __init__(self):
        self.calls = []
        self.path_queries = []  # the path_query each call sent (None: no path list)

    def hybrid_search(self, query, query_embedding=None, k=50, rrf_k=60, path_query=None):
        self.calls.append((query, query_embedding, k, rrf_k))
        self.path_queries.append(path_query)
        rows = [
            ("cur", "learning_rate: 3e-4", {"path": "configs/run_047.yaml", "mtime": 2.0, "start_byte": 0, "end_byte": 19}, 0.9, 1, 0.8, 1, 1 / 61 + 1 / 61),
            ("old", "learning_rate: 1e-4", {"path": "configs/archive/run_047_v1.yaml", "mtime": 1.0, "start_byte": 0, "end_byte": 19}, 0.5, 2, None, None, 1 / 62),
            ("log", "run 47 val_rmse 0.055", {"path": "logs/run_047.out", "mtime": 2.0, "_node_content": "{...}"}, None, None, 0.4, 2, 1 / 62),
        ]
        return SearchRows([
            {"node_id": n, "text": t, "metadata": m, "bm25_score": bs, "bm25_rank": br, "dense_score": ds, "dense_rank": dr, "rrf_score": rrf}
            for n, t, m, bs, br, ds, dr, rrf in rows
        ], "ts_rank_cd")


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
    assert out.counts == {"bm25": 2, "vector": 2, "path": 0, "fused": 3, "returned": 2}
    assert [n.node.node_id for n in out.nodes] == ["cur", "log"]  # superseded archive copy dropped
    assert out.staleness == {"enabled": True, "applied": True, "dropped": 1}
    assert out.scores["cur"] == {"bm25": 0.9, "dense": 0.8, "path": None, "rrf": 1 / 61 + 1 / 61, "rerank": None}
    assert vsm.path_queries == [None]  # router off: no path list
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
    assert vsm.calls[-1][1] is None  # SQL ran without a query vector
    assert vsm.path_queries[-1] == "compare run_047.yaml with configs/archive/run_047_v1.yaml"  # ... and with the path list


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


# ---------------------------------------------------------------- M3: no silent fallbacks, served reranker config


class _BrokenCrossEncoder:
    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        raise RuntimeError("CUDA out of memory (simulated)")


def test_rerank_failure_is_reported_as_degraded_not_hidden():
    retriever = HybridRetriever(_FakeVectorRetriever(), _FakeVSM(), reranker=_BrokenCrossEncoder(), rerank_top_n=8, enable_router=False,
                                reranker_info={"configured": True, "loaded": True, "model": "x"})

    out = retriever.retrieve_detailed("anything")

    assert len(out.nodes) == 3  # RRF order served for this request
    assert out.degraded == [{"component": "rerank", "error": "RuntimeError: CUDA out of memory (simulated)"}]
    assert all(out.scores[n.node.node_id]["rerank"] is None for n in out.nodes)


def test_rerank_depth_reranks_the_head_and_keeps_the_tail_in_rrf_order():
    model = _ReversingCrossEncoder()
    retriever = HybridRetriever(_FakeVectorRetriever(), _FakeVSM(), reranker=model, rerank_top_n=8, enable_router=False, rerank_depth=2)

    out = retriever.retrieve_detailed("anything", top_n=3)

    assert len(model.calls[0]) == 2  # only the fused top-2 were scored
    fused_order = [p[1] for p in model.calls[0]]
    assert out.nodes[0].node.get_content() == fused_order[1] and out.nodes[1].node.get_content() == fused_order[0]  # head reversed
    assert out.scores[out.nodes[2].node.node_id]["rerank"] is None  # tail untouched
    assert out.degraded == []


def test_configured_but_unloaded_reranker_is_listed_as_degraded_on_every_response():
    retriever = HybridRetriever(_FakeVectorRetriever(), _FakeVSM(), reranker=None, enable_router=False,
                                reranker_info={"configured": True, "loaded": False, "model": "BAAI/bge-reranker-v2-m3", "error": "weights not cached"})

    out = retriever.retrieve_detailed("anything")

    assert out.degraded == [{"component": "rerank", "error": "weights not cached"}]


def test_bm25_failure_in_python_mode_is_degraded_not_silent():
    class _FailingVSM:
        def search_bm25(self, query, limit):
            raise ConnectionError("db gone")

    retriever = HybridRetriever(_FakeVectorRetriever(), _FailingVSM(), reranker=None, enable_router=False)
    out = retriever.retrieve_detailed("anything")
    assert out.degraded[0]["component"] == "bm25" and "db gone" in out.degraded[0]["error"]
    assert len(out.nodes) == 2  # vector-only for this request


def test_load_reranker_never_falls_back_silently(monkeypatch):
    from app.engine import retriever as mod

    mod._reranker_cache.clear()
    monkeypatch.setattr(mod, "reranker_settings", lambda: {"model": "nope/does-not-exist", "precision": "fp32", "max_length": 512, "depth": 20, "required": True, "device": "cpu"})
    monkeypatch.setattr("app.retrieval.rerank.get_cross_encoder", lambda *a, **k: None)
    with pytest.raises(mod.RerankerUnavailable):
        mod.load_reranker()

    mod._reranker_cache.clear()
    monkeypatch.setattr(mod, "reranker_settings", lambda: {"model": "nope/does-not-exist", "precision": "fp32", "max_length": 512, "depth": 20, "required": False, "device": "cpu"})
    model, info = mod.load_reranker()
    assert model is None and info["configured"] and not info["loaded"] and "not loadable" in info["error"]

    mod._reranker_cache.clear()
    monkeypatch.setattr(mod, "reranker_settings", lambda: {"model": "none", "precision": "fp32", "max_length": 512, "depth": 20, "required": True, "device": "cpu"})
    model, info = mod.load_reranker()
    assert model is None and info == {"configured": False, "loaded": False, "model": None, "revision": None, "device": None, "precision": "fp32", "max_length": 512, "depth": 20, "error": None}


# ---------------------------------------------------------------- M11 (A22): one request's time budget


class _TimedSQLVSM(_FakeSQLVSM):
    """hybrid_search that records its keyword arguments, and can stand in for a cancelled statement."""

    def __init__(self, cancelled=False):
        super().__init__()
        self.kwargs = []
        self.cancelled = cancelled

    def hybrid_search(self, query, query_embedding=None, k=50, rrf_k=60, **kwargs):
        self.kwargs.append(kwargs)
        if self.cancelled:
            raise TimeoutError("hybrid statement cancelled by its statement_timeout")
        return super().hybrid_search(query, query_embedding, k, rrf_k)


class _SlowCrossEncoder:
    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        time.sleep(0.4)
        return [0.0] * len(pairs)


def _sql_retriever(vsm, embed_fn=lambda q: [0.1, 0.2]):
    return HybridRetriever(None, vsm, reranker=None, enable_router=False, mode="sql", embed_fn=embed_fn)


def test_the_statement_gets_the_rest_of_the_budget_as_its_statement_timeout():
    vsm = _TimedSQLVSM()
    out = _sql_retriever(vsm).retrieve_detailed("current learning rate for run 47", top_n=2, deadline=Deadline(5.0))
    assert len(out.nodes) == 2
    (kwargs,) = vsm.kwargs
    assert 0 < kwargs["statement_timeout_ms"] <= 5000


def test_without_a_budget_the_statement_has_no_timeout():
    vsm = _TimedSQLVSM()
    _sql_retriever(vsm).retrieve_detailed("q")
    assert vsm.kwargs == [{}]


def test_a_cancelled_statement_is_a_timeout_of_its_stage():
    with pytest.raises(StageTimeout) as caught:
        _sql_retriever(_TimedSQLVSM(cancelled=True)).retrieve_detailed("q", deadline=Deadline(5.0))
    assert caught.value.stage == "hybrid_sql" and caught.value.budget_s == 5.0


def test_an_embedding_past_the_budget_stops_the_request_before_the_statement():
    vsm = _TimedSQLVSM()

    def slow_embed(query):
        time.sleep(0.3)
        return [0.1, 0.2]

    with pytest.raises(StageTimeout) as caught:
        _sql_retriever(vsm, slow_embed).retrieve_detailed("q", deadline=Deadline(0.1))
    assert caught.value.stage == "embed" and vsm.calls == []


def test_a_rerank_past_the_budget_is_named():
    retriever = HybridRetriever(_FakeVectorRetriever(), _FakeVSM(), reranker=_SlowCrossEncoder(), enable_router=False)
    deadline = Deadline(0.2)
    with pytest.raises(StageTimeout) as caught:
        retriever.retrieve_detailed("anything", deadline=deadline)
    assert caught.value.stage == "rerank" and deadline.stage == "rerank"


# ---------------------------------------------------------------- M17 (C28): the lexical label is request-local

LABEL_HIT, LABEL_MISS = "learning rate", "qqqq"  # a query with a lexical hit, and one with none


class _LabelConnection:
    """A connection whose statements answer by query text: the hybrid statement returns a lexical row
    for LABEL_HIT and a dense-only row otherwise; search_bm25's full-text query finds a row for
    LABEL_HIT only (so LABEL_MISS falls through to ILIKE, which finds nothing)."""

    def cursor(self):
        return _LabelCursor()


class _LabelCursor:
    def __init__(self):
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        text = statement if isinstance(statement, str) else repr(statement)
        if "WITH params AS" in text:  # the hybrid statement (the settings statement rides along in front of it)
            hit = "learning" in params["query"]
            self.rows = [("n1", "learning rate 3e-4", {"path": "a.txt"}, 0.5 if hit else None, 1 if hit else None, 0.8, 1, 1 / 61 + (1 / 61 if hit else 0), None, None)]
        elif "ts_rank_cd" in text:  # search_bm25's full-text query: (query, query, limit)
            self.rows = [("n1", "learning rate 3e-4", {"path": "a.txt"}, 0.5)] if params[0] == LABEL_HIT else []
        else:  # ILIKE, the last resort
            self.rows = []

    def fetchall(self):
        return self.rows


def _label_manager(pause_query: str, paused, resume):
    """A real VectorStoreManager (no connection pool): its hybrid_search and search_bm25 run over
    _LabelConnection, and the call for `pause_query` returns only after `resume` is set, so another
    request can run start to finish between that call and the moment its caller reads the label."""
    from contextlib import contextmanager

    from app.database.vector_store_manager import VectorStoreManager

    class _Pausing(VectorStoreManager):
        def hybrid_search(self, query, *args, **kwargs):
            rows = super().hybrid_search(query, *args, **kwargs)
            if query == pause_query:
                paused.set()
                assert resume.wait(10), "the other request never finished"
            return rows

        def search_bm25(self, query, limit=5):
            rows = super().search_bm25(query, limit=limit)
            if query == pause_query:
                paused.set()
                assert resume.wait(10), "the other request never finished"
            return rows

    vsm = _Pausing.__new__(_Pausing)  # no __init__: it would connect
    vsm.schema_name, vsm.table_name, vsm.data_table = "public", "t", "data_t"
    vsm.lexical_backend, vsm._pgvector_version = "ts_rank_cd", (0, 8, 6)

    @contextmanager
    def get_connection():
        yield _LabelConnection()

    vsm.get_connection = get_connection
    return vsm


@pytest.mark.parametrize("mode", ["sql", "python"])
def test_a_concurrent_request_does_not_change_the_lexical_label_of_another(mode):
    """Request A (a lexical hit) is answered; before A's retriever reads its label, request B (no
    lexical hit) runs start to finish on the same shared manager. Each response must carry the
    backend that answered its own query: ts_rank_cd for A, none for B."""
    import threading

    paused, resume = threading.Event(), threading.Event()
    vsm = _label_manager(LABEL_HIT, paused, resume)
    if mode == "sql":
        retriever = HybridRetriever(None, vsm, reranker=None, enable_router=False, mode="sql", embed_fn=lambda q: [0.1, 0.2])
    else:
        retriever = HybridRetriever(_FakeVectorRetriever(), vsm, reranker=None, enable_router=False, mode="python")
    outcomes = {}
    a = threading.Thread(target=lambda: outcomes.__setitem__("a", retriever.retrieve_detailed(LABEL_HIT)))
    a.start()
    assert paused.wait(10), "request A never reached the database"
    outcomes["b"] = retriever.retrieve_detailed(LABEL_MISS)  # B runs while A waits between its search and its label
    resume.set()
    a.join(10)
    assert outcomes["b"].bm25_backend == "none"
    assert outcomes["a"].bm25_backend == "ts_rank_cd", f"request A reports B's label ({mode} mode)"


# ---------------------------------------------------------------- M17 (C29): nothing per request is kept on the retriever

LEXICAL_DOWN, LEXICAL_UP = "lexical leg down", "lexical leg up"


class _UpOrDownVSM:
    """search_bm25 raises for LEXICAL_DOWN and answers every other query."""

    def search_bm25(self, query, limit):
        if query == LEXICAL_DOWN:
            raise ConnectionError("db gone")
        return SearchRows([{"node_id": "b", "text": "encoder: dinov2", "score": 3.0, "metadata": {"path": "configs/run_046.yaml"}}], "ts_rank_cd")


class _PausingVectorRetriever(_FakeVectorRetriever):
    """The dense leg, which Python mode runs after the lexical leg: the call for `pause_query`
    returns only after `resume` is set, so another request can run start to finish meanwhile."""

    def __init__(self, pause_query, paused, resume):
        self.pause_query, self.paused, self.resume = pause_query, paused, resume

    def retrieve(self, query: str):
        if query == self.pause_query:
            self.paused.set()
            assert self.resume.wait(10), "the other request never finished"
        return super().retrieve(query)


@pytest.mark.parametrize("first, second", [(LEXICAL_DOWN, LEXICAL_UP), (LEXICAL_UP, LEXICAL_DOWN)], ids=["down-then-up", "up-then-down"])
def test_a_concurrent_request_does_not_cross_the_degraded_list(first, second):
    """Python fusion mode: request A's lexical leg has run (failed or not) and A waits in its dense leg
    while request B runs start to finish on the same retriever. A response lists the bm25 component
    as degraded exactly when its own lexical leg failed."""
    import threading

    paused, resume = threading.Event(), threading.Event()
    retriever = HybridRetriever(_PausingVectorRetriever(first, paused, resume), _UpOrDownVSM(), reranker=None, enable_router=False, mode="python")
    outcomes = {}
    a = threading.Thread(target=lambda: outcomes.__setitem__("a", retriever.retrieve_detailed(first)))
    a.start()
    assert paused.wait(10), "request A never reached its dense leg"
    outcomes["b"] = retriever.retrieve_detailed(second)  # B runs while A waits between its lexical leg and its outcome
    resume.set()
    a.join(10)
    for name, query in (("a", first), ("b", second)):
        components = [d["component"] for d in outcomes[name].degraded]
        assert components == (["bm25"] if query == LEXICAL_DOWN else []), f"request {name.upper()} ({query!r}) reports {outcomes[name].degraded}"


@pytest.mark.parametrize("mode", ["sql", "python"])
def test_a_request_writes_nothing_on_the_shared_retriever(mode):
    """The API serves every request from one retriever: its fields are the same after requests
    (a lexical_path route, a failed lexical leg in Python mode, a reranked one) as before them."""
    if mode == "sql":
        retriever = HybridRetriever(None, _FakeSQLVSM(), reranker=_ReversingCrossEncoder(), mode="sql", embed_fn=lambda q: [0.1, 0.2],
                                    clusters=_clusters_for_fake_rows(), rerank_depth=2)
    else:
        retriever = HybridRetriever(_FakeVectorRetriever(), _UpOrDownVSM(), reranker=_ReversingCrossEncoder(), mode="python", rerank_depth=2)
    before = dict(vars(retriever))
    for query in ("current learning rate for run 47", "open configs/run_047.yaml", LEXICAL_DOWN):
        retriever.retrieve_detailed(query)
    assert vars(retriever) == before


# ---------------------------------------------------------------- M17 (B14): the path list on the lexical_path route


@pytest.mark.parametrize(
    "query, enable_router, path_query",
    [
        ("open configs/run_047.yaml", True, "open configs/run_047.yaml"),  # lexical_path: the statement gets the path list
        ("current learning rate for run 47", True, None),  # semantic: no path list
        ("which run has the lowest val_rmse", True, None),  # aggregation_query: no path list
        ("open configs/run_047.yaml", False, None),  # router off: every query is semantic, as in memory
    ],
)
def test_only_the_lexical_path_route_sends_the_path_list(query, enable_router, path_query):
    vsm, embedded = _FakeSQLVSM(), []
    retriever = HybridRetriever(None, vsm, reranker=None, enable_router=enable_router, mode="sql", embed_fn=lambda q: embedded.append(q) or [0.1])
    out = retriever.retrieve_detailed(query)
    assert vsm.path_queries == [path_query]
    assert (embedded == []) == (path_query is not None)  # the path route sends no query vector, as before
    assert "path" in out.counts


def test_the_path_score_is_reported_per_hit():
    class _PathVSM(_FakeSQLVSM):
        def hybrid_search(self, query, query_embedding=None, k=50, rrf_k=60, path_query=None):
            rows = super().hybrid_search(query, query_embedding, k, rrf_k, path_query)
            for row in rows:
                named = path_query and row["metadata"]["path"] == "configs/run_047.yaml"
                row["path_score"], row["path_rank"] = (10.0, 1) if named else (None, None)
            return rows

    out = HybridRetriever(None, _PathVSM(), reranker=None, mode="sql", embed_fn=lambda q: [0.1]).retrieve_detailed("open configs/run_047.yaml")
    assert out.counts["path"] == 1
    assert out.scores["cur"]["path"] == 10.0 and out.scores["log"]["path"] is None
