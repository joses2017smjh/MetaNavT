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


def test_propose_reject_and_approve_with_a_log(client, corpus, operator_grant):
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
    token = operator_grant(client.app.state.plans.tools, second["review"])  # the operator signs the exported review
    approved = client.post(f"/api/plans/{second['plan_id']}/approve", json={"actor": "jose", "approval_token": token}).json()
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
    assert "db down" not in resp.text  # the cause is logged, not sent
    assert (corpus / "configs" / "run_047.yaml").exists()  # refused, not moved


def test_build_plan_store_without_a_database_has_no_log(tmp_path):
    store = build_plan_store(None, str(tmp_path))
    assert store.log is None and store.tools.allow_apply is False


def test_actor_or_boolean_cannot_authorize_and_review_is_immutable(client, corpus, operator_grant):
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "archive/config.yaml"}).json()
    review = plan["review"]
    assert review["payload"] == {"src": "configs/run_047.yaml", "dst": "archive/config.yaml"}
    for body in ({"actor": "operator"}, {"actor": "operator", "approved": True}, {"approval_token": "forged"}):
        assert client.post(f"/api/plans/{plan['plan_id']}/approve", json=body).status_code == 403
    token = operator_grant(client.app.state.plans.tools, plan["review"])
    (corpus / "configs/run_047.yaml").write_text("changed after review")
    assert client.get(f"/api/plans/{plan['plan_id']}").json()["review"] == review
    assert client.post(f"/api/plans/{plan['plan_id']}/approve", json={"approval_token": token}).status_code == 403
    assert not (corpus / "archive/config.yaml").exists()


# Text a database error carries: a table, an address, a role. None of it may reach a client.
INTERNALS = ('relation "plan_proposals"', "approval_grants", "db.internal.example", "5432", "metanavit_user")


class _BrokenTable:
    """Plans whose database is down: every call raises an error naming its table and server."""

    def _fail(self, *args, **kwargs):
        raise RuntimeError('relation "plan_proposals" on db.internal.example:5432 (role metanavit_user) is not reachable')

    add = get = list = deciding = _fail


class _BrokenLedger:
    """A verifier whose grant ledger's database is down (the Postgres ledger): consume raises, audit works."""

    def consume(self, token, review, **kwargs):
        raise RuntimeError('could not write approval_grants on db.internal.example:5432 as metanavit_user')

    def audit(self, *args, **kwargs):
        pass


def test_a_database_failure_answers_503_with_a_fixed_text_and_logs_the_cause(client, corpus, caplog):
    client.app.state.plans = PlanStore(FilesystemTools(root=corpus), _MemoryLog(), _BrokenTable())
    answers = [
        client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}),
        client.get("/api/plans/"),
        client.get("/api/plans/some-plan"),
        client.post("/api/plans/some-plan/approve", json={"approval_token": "x"}),
        client.post("/api/plans/some-plan/reject"),
    ]
    assert [a.status_code for a in answers] == [503] * 5
    assert {a.json()["detail"] for a in answers} == {"plan store unavailable"}
    assert not [needle for a in answers for needle in INTERNALS if needle in a.text]
    assert "plan_proposals" in caplog.text  # the server log has the cause
    assert (corpus / "configs" / "run_047.yaml").exists()


def test_an_approval_whose_ledger_cannot_be_written_is_503_and_moves_nothing(client, corpus, operator_grant):
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}).json()
    token = operator_grant(client.app.state.plans.tools, plan["review"])
    client.app.state.plans.tools.approval_verifier = _BrokenLedger()
    resp = client.post(f"/api/plans/{plan['plan_id']}/approve", json={"approval_token": token})
    assert resp.status_code == 503 and resp.json()["detail"] == "the approval could not be recorded; nothing was applied"
    assert not [needle for needle in INTERNALS if needle in resp.text]
    assert (corpus / "configs" / "run_047.yaml").exists()
    assert client.get(f"/api/plans/{plan['plan_id']}").json()["status"] == "pending_approval"
    assert client.app.state.plans.log.rows[-1]["result"] == "failed: RuntimeError"


def test_a_failed_start_tells_the_client_an_approval_error_and_only_the_class_of_anything_else(client):
    client.app.state.plans = None
    client.app.state.startup_error = 'OperationalError: connection to server at "db.internal.example", port 5432 failed'
    resp = client.get("/api/plans/")
    assert resp.status_code == 503 and resp.json()["detail"] == "plans unavailable: OperationalError (the server log has the reason)"
    client.app.state.startup_error = "ApprovalError: configure both the approval public key and ledger paths"
    assert client.get("/api/plans/").json()["detail"] == "plans unavailable: ApprovalError: configure both the approval public key and ledger paths"
    client.app.state.startup_error = None


def test_no_plan_is_kept_in_the_tools_between_requests(client, corpus, operator_grant):
    """Every request reads its plan from the store (the plans table in the API); the gate holds it for one decision."""
    store = client.app.state.plans
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}).json()
    assert store.tools.plans == {}
    token = operator_grant(store.tools, plan["review"])
    assert client.post(f"/api/plans/{plan['plan_id']}/approve", json={"approval_token": token}).json()["status"] == "applied"
    assert store.tools.plans == {} and store.tools._reviews == {}
