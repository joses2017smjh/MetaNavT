"""API contract with the backend down: the app starts, /health says why, retrieve is 503.

No Postgres in this suite: PG_CONNECTION_STRING points at a closed port so the
lifespan records a startup error instead of a retriever. The happy path and
the parity run need a database and live in CI's docker-smoke job (M0) and the
parity job (M3).
"""

import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

BAD_DB = "postgresql://postgres:password@127.0.0.1:1/metanavit"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("PG_CONNECTION_STRING", BAD_DB)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1")
    monkeypatch.setenv("INDEX_ON_START", "false")
    monkeypatch.delenv("FRONTEND_ENDPOINT", raising=False)
    import main  # noqa: WPS433 - the FastAPI app module

    from app.database import vector_store

    vector_store.get_vector_store_manager.cache_clear()
    with TestClient(main.app) as c:  # runs the lifespan
        yield c


def test_routes_are_mounted(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/health", "/api/retrieve/", "/api/chat", "/api/query/"} <= paths, sorted(paths)


def test_health_reports_the_startup_error(client):
    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unavailable"
    assert body["error"], body


def test_retrieve_is_503_when_the_backend_is_down(client):
    resp = client.post("/api/retrieve/", json={"query": "current learning rate for run 47"})
    assert resp.status_code == 503
    assert "retrieval backend unavailable" in resp.json()["detail"]


def test_retrieve_validates_the_body(client):
    assert client.post("/api/retrieve/", json={}).status_code == 422
    assert client.post("/api/retrieve/", json={"query": 42}).status_code == 422


# ---------------------------------------------------------------- happy path with a fake retrieval state


class _FakeOutcome:
    def __init__(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        from app.retrieval.router import RouteDecision, RouteType

        self.nodes = [
            NodeWithScore(node=TextNode(id_="n1", text="learning_rate: 3e-4", metadata={"path": "configs/run_047.yaml", "start_byte": 0, "end_byte": 19}), score=0.0325),
            NodeWithScore(node=TextNode(id_="n2", text="run 47 log", metadata={"file_path": "/abs/logs/run_047.out"}), score=0.0164),
        ]
        self.route = RouteDecision(RouteType.SEMANTIC, "test")
        self.stages_ms = {"route": 0.1, "hybrid_sql": 9.0, "total": 9.5}
        self.counts = {"bm25": 2, "vector": 2, "fused": 2, "returned": 2}
        self.scores = {"n1": {"bm25": 0.9, "dense": 0.8, "rrf": 0.0325, "rerank": None}}
        self.staleness = {"enabled": True, "applied": True, "dropped": 0}
        self.mode = "sql"
        self.bm25_backend = "ts_rank_cd"
        self.degraded = [{"component": "rerank", "error": "weights not cached"}]


class _FakeRetriever:
    _reranker = None
    mode = "sql"
    depth = 50
    reranker_info = {"configured": True, "loaded": False, "model": "BAAI/bge-reranker-v2-m3", "revision": None, "device": None,
                     "precision": "fp16", "max_length": 512, "depth": 20, "error": "weights not cached"}

    def __init__(self):
        self.calls = []

    def retrieve_detailed(self, query, top_n=None, deadline=None):
        self.calls.append((query, top_n))
        return _FakeOutcome()


class _FakeState:
    def __init__(self):
        self.retriever = _FakeRetriever()
        self.vsm = _FakeVSM()
        self.index = None
        self.indexing = {"indexed": False}


class _FakeVSM:
    hybrid_backend = "ts_rank_cd"
    pgvector_version = (0, 8, 6)

    def count_nodes(self):
        return 61

    def search_settings(self, k):
        return {"hnsw.ef_search": 2 * k, "hnsw.iterative_scan": "strict_order"}


@pytest.fixture
def ready_client(client):
    client.app.state.retrieval = _FakeState()
    yield client
    client.app.state.retrieval = None


def test_happy_path_is_typed_with_paths_byte_ranges_and_stage_scores(ready_client):
    resp = ready_client.post("/api/retrieve/", json={"query": "  current learning rate for run 47 ", "k": 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query"] == "current learning rate for run 47" and body["k"] == 2
    assert body["route"] == "semantic" and body["retrieval_mode"] == "sql" and body["bm25_backend"] == "ts_rank_cd"
    assert body["staleness"] == {"enabled": True, "applied": True, "dropped": 0}
    assert body["latency_ms"]["total"] == 9.5 and body["counts"]["fused"] == 2
    assert [h["path"] for h in body["hits"]] == ["configs/run_047.yaml", "/abs/logs/run_047.out"]
    first = body["hits"][0]
    assert (first["start_byte"], first["end_byte"], first["rank"]) == (0, 19, 1)
    assert first["scores"] == {"bm25": 0.9, "dense": 0.8, "path": None, "rrf": 0.0325, "rerank": None}
    assert body["hits"][1]["scores"] == {"bm25": None, "dense": None, "path": None, "rrf": None, "rerank": None}
    assert ready_client.app.state.retrieval.retriever.calls == [("current learning rate for run 47", 2)]


def test_k_and_query_are_validated(ready_client):
    assert ready_client.post("/api/retrieve/", json={"query": "x", "k": 0}).status_code == 422
    assert ready_client.post("/api/retrieve/", json={"query": "x", "k": 51}).status_code == 422
    assert ready_client.post("/api/retrieve/", json={"query": "   "}).status_code == 422
    assert ready_client.post("/api/retrieve/", json={"query": "y" * 2001}).status_code == 422
    assert ready_client.post("/api/retrieve/", json={"query": "ok"}).status_code == 200


def test_response_and_health_list_degraded_components_and_the_reranker(ready_client):
    body = ready_client.post("/api/retrieve/", json={"query": "q"}).json()
    assert body["degraded"] == [{"component": "rerank", "error": "weights not cached"}]
    assert body["reranker"]["configured"] is True and body["reranker"]["loaded"] is False
    assert body["reranker"]["precision"] == "fp16" and body["reranker"]["depth"] == 20 and body["reranker_loaded"] is False

    health = ready_client.get("/health")
    assert health.status_code == 200
    h = health.json()
    assert h["status"] == "degraded" and h["degraded"][0]["component"] == "rerank"
    assert h["reranker"]["model"] == "BAAI/bge-reranker-v2-m3" and h["retrieval_mode"] == "sql" and h["n_nodes"] == 61
    # the settings each hybrid statement runs with at the retriever's depth (VectorStoreManager.search_settings)
    assert h["dense_search"] == {"depth": 50, "settings": {"hnsw.ef_search": 100, "hnsw.iterative_scan": "strict_order"}, "pgvector": "0.8.6"}


def test_health_refuses_unverified_required_cuda_device(ready_client, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_REQUIRE_DEVICE", "cuda")
    response = ready_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["required_device"] == "cuda"
    assert body["devices"] == {"embedding": None, "reranker": None}
    assert {issue["component"] for issue in body["degraded"]} == {"embedding", "reranker", "rerank"}


# ---------------------------------------------------------------- the request's time budget (RETRIEVE_TIMEOUT_SECONDS)


class _StuckRetriever(_FakeRetriever):
    """Enters a stage and does not come back until the test releases it (a hung model call)."""

    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.release = threading.Event()

    def retrieve_detailed(self, query, top_n=None, deadline=None):
        deadline.stage = self.stage
        self.release.wait(30)
        return _FakeOutcome()


class _OverrunRetriever(_FakeRetriever):
    """A stage that notices the overrun itself (the hybrid statement's statement_timeout, or a stage check)."""

    def retrieve_detailed(self, query, top_n=None, deadline=None):
        from app.engine.retriever import StageTimeout

        raise StageTimeout("hybrid_sql", deadline.budget_s)


def test_a_retrieval_that_overruns_the_budget_is_504_naming_the_stage(ready_client, monkeypatch):
    from app.api.routers.retrieve import TIMEOUT_GRACE_SECONDS

    monkeypatch.setenv("RETRIEVE_TIMEOUT_SECONDS", "0.2")
    stuck = _StuckRetriever("rerank")
    ready_client.app.state.retrieval.retriever = stuck
    started = time.monotonic()
    try:
        resp = ready_client.post("/api/retrieve/", json={"query": "q"})
    finally:
        stuck.release.set()
    elapsed = time.monotonic() - started
    assert resp.status_code == 504, resp.text
    assert "rerank" in resp.json()["detail"] and "0.2 s" in resp.json()["detail"]
    assert elapsed < 0.2 + TIMEOUT_GRACE_SECONDS + 2.0, elapsed  # answered at the budget, not when the stage returned


def test_a_stage_timeout_from_the_retriever_is_504(ready_client, monkeypatch):
    monkeypatch.setenv("RETRIEVE_TIMEOUT_SECONDS", "5")
    ready_client.app.state.retrieval.retriever = _OverrunRetriever()
    resp = ready_client.post("/api/retrieve/", json={"query": "q"})
    assert resp.status_code == 504 and "hybrid_sql" in resp.json()["detail"], resp.text


class _ClientTimeoutRetriever(_FakeRetriever):
    """A client library's own timeout, well inside the budget (socket.timeout is TimeoutError)."""

    def retrieve_detailed(self, query, top_n=None, deadline=None):
        deadline.stage = "embed"
        raise TimeoutError("timed out")


def test_a_timeout_error_inside_the_budget_is_a_backend_failure_not_the_budget(ready_client, monkeypatch):
    """Only the route's own wait, or a stage's StageTimeout, is a 504 that blames the budget."""
    monkeypatch.setenv("RETRIEVE_TIMEOUT_SECONDS", "5")
    ready_client.app.state.retrieval.retriever = _ClientTimeoutRetriever()
    resp = ready_client.post("/api/retrieve/", json={"query": "q"})
    assert resp.status_code == 503 and "timed out" in resp.json()["detail"], resp.text
    assert "budget" not in resp.json()["detail"], resp.text


def test_within_the_budget_the_answer_is_unchanged(ready_client, monkeypatch):
    monkeypatch.setenv("RETRIEVE_TIMEOUT_SECONDS", "5")
    assert ready_client.post("/api/retrieve/", json={"query": "q"}).status_code == 200


@pytest.mark.parametrize("raw, expected", [(None, 30.0), ("2.5", 2.5), ("abc", 30.0), ("0", 30.0), ("-1", 30.0), ("inf", 30.0), ("nan", 30.0)])
def test_the_budget_setting(monkeypatch, raw, expected):
    from app.api.routers.retrieve import retrieve_timeout_seconds

    if raw is None:
        monkeypatch.delenv("RETRIEVE_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("RETRIEVE_TIMEOUT_SECONDS", raw)
    assert retrieve_timeout_seconds() == expected
