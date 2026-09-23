"""Approve / reject plans: nothing moves without an approval, every decision is logged."""

import pytest
from fastapi.testclient import TestClient

from app.api.routers.plans import PlanStore, build_plan_store
from app.mcp.filesystem import FilesystemTools


class _MemoryLog:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail

    def write(self, record):
        if self.fail:
            raise ConnectionError("db down")
        self.rows.append(dict(record))

    def update(self, record):
        self.rows.append(dict(record))

    def recent(self, limit=50):
        return self.rows[-limit:]


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "run_047.yaml").write_text("learning_rate: 3e-4\n")
    return tmp_path


@pytest.fixture
def client(monkeypatch, corpus):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://postgres:password@127.0.0.1:1/metanavit")
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1")
    monkeypatch.setenv("INDEX_ON_START", "false")
    monkeypatch.delenv("FRONTEND_ENDPOINT", raising=False)
    import main

    from app.database import vector_store

    vector_store.get_vector_store_manager.cache_clear()
    with TestClient(main.app) as c:
        c.app.state.plans = PlanStore(FilesystemTools(root=corpus), _MemoryLog())
        yield c
        c.app.state.plans = None


def test_plans_are_503_when_the_store_is_missing(client):
    client.app.state.plans = None
    assert client.get("/api/plans/").status_code == 503


def test_propose_reject_and_approve_with_a_log(client, corpus):
    created = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"})
    assert created.status_code == 201, created.text
    plan = created.json()
    assert plan["status"] == "pending_approval" and (corpus / "configs" / "run_047.yaml").exists()  # nothing moved

    listed = client.get("/api/plans/").json()
    assert [p["plan_id"] for p in listed] == [plan["plan_id"]]

    rejected = client.post(f"/api/plans/{plan['plan_id']}/reject", json={"actor": "jose", "note": "keep it"}).json()
    assert rejected["status"] == "rejected" and rejected["actor"] == "jose"
    assert (corpus / "configs" / "run_047.yaml").exists()  # still nothing moved
    assert client.post(f"/api/plans/{plan['plan_id']}/approve").status_code == 409  # decided plans are final

    second = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}).json()
    approved = client.post(f"/api/plans/{second['plan_id']}/approve", json={"actor": "jose"}).json()
    assert approved["status"] == "applied"
    assert not (corpus / "configs" / "run_047.yaml").exists() and (corpus / "configs" / "archive" / "run_047.yaml").exists()

    log = client.app.state.plans.log.rows
    actions = [(row["plan_id"] == plan["plan_id"], row["action"], row["result"]) for row in log]
    assert (True, "reject", "pending") in actions and (True, "reject", "rejected") in actions
    assert (False, "approve", "pending") in actions and (False, "approve", "applied") in actions
    assert client.get("/api/plans/decisions/recent").json()[-1]["result"] == "applied"


def test_unknown_source_and_path_escape_are_rejected(client):
    assert client.post("/api/plans/", json={"src": "configs/nope.yaml", "dst": "x"}).status_code == 404
    assert client.post("/api/plans/", json={"src": "../../etc/passwd", "dst": "x"}).status_code == 400
    assert client.post("/api/plans/nope/approve").status_code == 404
    assert client.post("/api/plans/", json={"src": "", "dst": "x"}).status_code == 422


def test_action_is_refused_when_the_decision_log_is_down(client, corpus):
    client.app.state.plans = PlanStore(FilesystemTools(root=corpus), _MemoryLog(fail=True))
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}).json()
    resp = client.post(f"/api/plans/{plan['plan_id']}/approve")
    assert resp.status_code == 503 and "decision log unavailable" in resp.json()["detail"]
    assert (corpus / "configs" / "run_047.yaml").exists()  # refused, not moved


def test_build_plan_store_without_a_database_has_no_log(tmp_path):
    store = build_plan_store(None, str(tmp_path))
    assert store.log is None and store.tools.allow_apply is False
