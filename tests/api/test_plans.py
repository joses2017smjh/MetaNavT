"""Approve / reject plans: nothing moves without an approval, every decision is logged."""

import base64
import hashlib

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
    for name in ("METANAVIT_API_KEY", "METANAVIT_API_KEYS_FILE", "RATE_LIMIT_PER_MINUTE"):  # open API unless a test sets one
        monkeypatch.delenv(name, raising=False)
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


class _ResultNotLogged(_MemoryLog):
    """A decision log that takes the intent, then cannot take the result: its database went down in between."""

    def update(self, record):
        raise RuntimeError('relation "plan_decisions" on db.internal.example:5432 (role metanavit_user) is not reachable')


def test_a_saved_decision_whose_result_cannot_be_logged_is_answered_with_the_decision(client, corpus, operator_grant, caplog):
    """The plan's status is saved before the decision log gets the attempt's result. When only that last write fails,
    the answer is the decision as GET reports it (it was a bare 500 for a plan already decided), and the cause goes
    to the server log only."""
    store = PlanStore(FilesystemTools(root=corpus), _ResultNotLogged())
    client.app.state.plans = store
    kept = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}).json()
    rejected = client.post(f"/api/plans/{kept['plan_id']}/reject", json={"actor": "jose"})
    assert (rejected.status_code, rejected.json()["status"]) == (200, "rejected")
    moved = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "archive/run_047.yaml"}).json()
    token = operator_grant(store.tools, moved["review"])
    approved = client.post(f"/api/plans/{moved['plan_id']}/approve", json={"approval_token": token})
    assert (approved.status_code, approved.json()["status"]) == (200, "applied")
    assert (corpus / "archive" / "run_047.yaml").exists() and client.get(f"/api/plans/{moved['plan_id']}").json()["status"] == "applied"
    # A 200 carries the review (hashes, device and inode numbers), so the needles are the error's own words.
    assert not [needle for resp in (rejected, approved) for needle in ('relation "plan_decisions"', "db.internal.example", "metanavit_user")
                if needle in resp.text]
    assert [row["result"] for row in store.log.rows] == ["pending", "pending"]  # the intents; neither result reached the log
    assert "plan_decisions" in caplog.text  # the server log has the cause


def test_a_move_whose_audit_event_cannot_be_written_after_it_ran_is_answered_as_applied(client, corpus, operator_grant, caplog):
    """The grant is consumed and the file moved when the audit event after the move fails: the answer, the plan and
    the decision log say applied, and the cause goes to the server log only."""
    store = client.app.state.plans
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}).json()
    token = operator_grant(store.tools, plan["review"])
    verifier = store.tools.approval_verifier
    real_audit = verifier.audit

    def audit(plan_id, action, result, detail=""):
        if result == "applied":  # the event written after the move
            raise RuntimeError("could not write approval_audit on db.internal.example:5432 as metanavit_user")
        return real_audit(plan_id, action, result, detail)

    verifier.audit = audit
    resp = client.post(f"/api/plans/{plan['plan_id']}/approve", json={"approval_token": token})
    assert (resp.status_code, resp.json()["status"]) == (200, "applied")
    assert (corpus / "configs" / "archive" / "run_047.yaml").exists() and store.log.rows[-1]["result"] == "applied"
    assert not [needle for needle in ("approval_audit", "db.internal.example", "metanavit_user") if needle in resp.text]
    assert "approval_audit" in caplog.text


def test_a_failed_start_tells_the_client_an_approval_error_and_only_the_class_of_anything_else(client):
    """...and the id the failed attempt was logged under (main.py stores both; C47)."""
    client.app.state.plans = None
    client.app.state.startup_error = 'OperationalError: connection to server at "db.internal.example", port 5432 failed'
    client.app.state.startup_error_id = "0123456789abcdef"
    resp = client.get("/api/plans/")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "plans unavailable: OperationalError (the server log has the reason, id 0123456789abcdef)"
    client.app.state.startup_error = "ApprovalError: configure both the approval public key and ledger paths"
    assert client.get("/api/plans/").json()["detail"] == "plans unavailable: ApprovalError: configure both the approval public key and ledger paths"
    client.app.state.startup_error = client.app.state.startup_error_id = None


def test_no_plan_is_kept_in_the_tools_between_requests(client, corpus, operator_grant):
    """Every request reads its plan from the store (the plans table in the API); the gate holds it for one decision."""
    store = client.app.state.plans
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}).json()
    assert store.tools.plans == {}
    token = operator_grant(store.tools, plan["review"])
    assert client.post(f"/api/plans/{plan['plan_id']}/approve", json={"approval_token": token}).json()["status"] == "applied"
    assert store.tools.plans == {} and store.tools._reviews == {}


def test_a_refused_proposal_leaves_no_plan_in_the_tools(client, corpus):
    """A move onto an existing file (409) or of a directory (400) is refused by the review propose_move builds after it
    stored the plan; the refused plan is dropped with it, so the process holds none (it held one per refusal)."""
    store = client.app.state.plans
    (corpus / "configs" / "taken.yaml").write_text("existing\n")
    for _ in range(3):
        assert client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/taken.yaml"}).status_code == 409
        assert client.post("/api/plans/", json={"src": "configs", "dst": "elsewhere"}).status_code == 400
    assert store.tools.plans == {} and store.tools._reviews == {}
    assert client.get("/api/plans/").json() == []
    assert (corpus / "configs" / "taken.yaml").read_text() == "existing\n"


def test_an_operating_system_errors_text_never_reaches_the_client(client, corpus, operator_grant, monkeypatch):
    """A refusal this repository words (a check's error, "reviewed source is missing") is sent as it is; an
    operating system error's text names files, here an absolute path, and is not: the client gets a fixed text."""
    import errno

    from app.mcp import filesystem

    store = client.app.state.plans
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}).json()
    secret = str(corpus / "configs" / "secret_name.yaml")
    cases = [(PermissionError(errno.EACCES, "Permission denied", secret), 403, "the file system refused the move"),
             (FileExistsError(errno.EEXIST, "File exists", secret), 409, "the destination already exists"),
             (OSError(errno.ENOSPC, "No space left on device", secret), 500, "move failed")]
    for error, status, detail in cases:
        def failing_move(*args, _error=error, **kwargs):
            raise _error

        monkeypatch.setattr(filesystem, "move_file", failing_move)
        token = operator_grant(store.tools, plan["review"])  # each failed move burns its capability
        resp = client.post(f"/api/plans/{plan['plan_id']}/approve", json={"approval_token": token})
        assert (resp.status_code, resp.json()["detail"]) == (status, detail)
        log = client.get("/api/plans/decisions/recent")  # the decision log's result is served too
        assert log.status_code == 200 and log.json()[-1]["result"].endswith(detail)
        for text in (resp.text, log.text):
            assert "Errno" not in text and "secret_name" not in text and str(corpus) not in text
    monkeypatch.undo()
    (corpus / "configs" / "run_047.yaml").unlink()  # a refusal worded here is sent as it is
    resp = client.post(f"/api/plans/{plan['plan_id']}/approve", json={"approval_token": operator_grant(store.tools, plan["review"])})
    assert (resp.status_code, resp.json()["detail"]) == (500, "move failed: reviewed source is missing")


# ---------------------------------------------------------------- who decided: the key's name, and the grant's actor (A13, C40, D45)

ALICE, BOB, SHARED = "alice-key-4b1d9e-do-not-log", "bob-key-93ac07-do-not-log", "shared-key-5e0c2a-do-not-log"


@pytest.fixture
def named_keys(monkeypatch, tmp_path_factory):
    """METANAVIT_API_KEYS_FILE with alice and bob (hashes only), outside DATA_DIR and the plans' root."""
    path = tmp_path_factory.mktemp("api-keys") / "api_keys"
    path.write_text("".join(f"{name}:{hashlib.sha256(key.encode()).hexdigest()}\n" for name, key in (("alice", ALICE), ("bob", BOB))))
    monkeypatch.setenv("METANAVIT_API_KEYS_FILE", str(path))
    return {"alice": {"X-API-Key": ALICE}, "bob": {"Authorization": f"Bearer {BOB}"}}


def _log_rows(client, headers):
    return [(row["plan_id"], row["action"], row["actor"], row.get("grant_actor"), row["result"])
            for row in client.get("/api/plans/decisions/recent", headers=headers).json()]


def test_with_named_keys_the_decision_actor_is_the_key_s_name_whatever_the_body_says(client, corpus, operator_grant, named_keys):
    """A13 / D45: a decision posted with alice's key and body actor "bob" is recorded as alice, in the plan and in the
    decision log; two keys give two actors. C40: an approval also records the actor the operator signed into the
    capability (operator_grant signs as "test-operator"), never the body's."""
    alice, bob = named_keys["alice"], named_keys["bob"]
    plan = {"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}
    assert client.post("/api/plans/", json=plan).status_code == 401  # keys on: no key, no plan
    kept = client.post("/api/plans/", json=plan, headers=alice).json()
    rejected = client.post(f"/api/plans/{kept['plan_id']}/reject", json={"actor": "bob", "note": "not now"}, headers=alice)
    assert rejected.status_code == 200 and rejected.json()["actor"] == "alice"
    assert client.get(f"/api/plans/{kept['plan_id']}", headers=bob).json()["actor"] == "alice"

    moved = client.post("/api/plans/", json=plan, headers=bob).json()
    token = operator_grant(client.app.state.plans.tools, moved["review"])
    approved = client.post(f"/api/plans/{moved['plan_id']}/approve", json={"actor": "alice", "approval_token": token}, headers=bob)
    assert (approved.status_code, approved.json()["status"], approved.json()["actor"]) == (200, "applied", "bob")

    rows = _log_rows(client, alice)
    assert (kept["plan_id"], "reject", "alice", None, "rejected") in rows
    assert (moved["plan_id"], "approve", "bob", "test-operator", "applied") in rows
    assert {actor for _, _, actor, _, _ in rows} == {"alice", "bob"}  # the body's forged names never reach the log


def test_with_the_single_key_every_holder_is_its_one_identity(client, corpus, monkeypatch):
    monkeypatch.setenv("METANAVIT_API_KEY", SHARED)
    headers = {"X-API-Key": SHARED}
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "archive/run_047.yaml"}, headers=headers).json()
    rejected = client.post(f"/api/plans/{plan['plan_id']}/reject", json={"actor": "jose"}, headers=headers).json()
    assert rejected["actor"] == "shared-key"
    assert _log_rows(client, headers)[-1][2] == "shared-key"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def test_a_capability_that_does_not_verify_leaves_no_grant_actor(client, corpus, operator_grant, named_keys, tmp_path_factory):
    """The grant's actor is read only from a capability the verifier accepted: a token whose body names "mallory", unsigned
    or signed by a key the server does not trust, is refused (403) and the log row carries no grant actor."""
    from app.mcp.approvals import canonical, create_key, review_hash, sign_review

    bob = named_keys["bob"]
    plan = client.post("/api/plans/", json={"src": "configs/run_047.yaml", "dst": "archive/run_047.yaml"}, headers=bob).json()
    operator_grant(client.app.state.plans.tools, plan["review"])  # the server's verifier, with the operator's public key
    body = canonical({"review_sha256": review_hash(plan["review"]), "grant_id": "a" * 48, "issued_at": 1.0, "expires_at": 2.0,
                      "actor": "mallory"})
    other = tmp_path_factory.mktemp("not-the-operator")
    create_key(other / "k", other / "k.pub")
    for token in (_b64(body) + "." + _b64(bytes(64)), sign_review(plan["review"], other / "k", actor="mallory")):
        resp = client.post(f"/api/plans/{plan['plan_id']}/approve", json={"approval_token": token}, headers=bob)
        assert resp.status_code == 403, resp.text
        row = client.app.state.plans.log.rows[-1]
        assert row["result"].startswith("refused") and row.get("grant_actor") is None and row["actor"] == "bob"
    assert "mallory" not in client.get("/api/plans/decisions/recent", headers=bob).text
    assert (corpus / "configs" / "run_047.yaml").exists()
