"""Pending file plans shared through Postgres: two plan stores on one database, as two API replicas are.

A15 (= C24, D33), C25 and C26. Each store is what build_plan_store gives an API process: a FilesystemTools of its
own (its own session) over the one DATA_DIR, the decision log and the plans table (plan_proposals), on a database
manager (connection pool) of its own. Plans used to live in each process's memory: a plan proposed on one replica
was a 404 on the other and a restart lost it. The operator key pair and the grant ledger sit outside DATA_DIR.
Needs PG_TEST_DSN (the CI parity service, or a local instance); skipped otherwise. Every test runs in a schema of
its own, dropped at the end.
"""

import json
import os
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

SOURCE = "configs/run_047.yaml"


def _schema_dsn(schema):
    """PG_TEST_DSN (a URL) with search_path set to `schema`."""
    dsn = os.environ["PG_TEST_DSN"]
    return f"{dsn}{'&' if '?' in dsn else '?'}options=-csearch_path%3D{schema}"


@pytest.fixture
def schema():
    import psycopg2

    name = "plantest_" + uuid.uuid4().hex[:10]
    admin = psycopg2.connect(os.environ["PG_TEST_DSN"])
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {name};")
    try:
        yield name
    finally:
        with admin.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE;")
        admin.close()


@pytest.fixture
def deployment(schema, tmp_path, monkeypatch):
    """A migrated schema, one DATA_DIR, an operator key pair and ledger outside it; returns (open_store, sign, sql)."""
    from urllib.parse import urlparse

    from app.api.routers.plans import build_plan_store
    from app.database.db_base_manager import DatabaseManager
    from app.database.migrate import apply_migrations
    from app.mcp.approvals import create_key, sign_review

    root = tmp_path / "data"
    (root / "configs").mkdir(parents=True)
    (root / SOURCE).write_text("learning_rate: 3e-4\n")
    private = tmp_path / "operator"
    private.mkdir(mode=0o700)
    create_key(private / "operator.key", private / "operator.pub")
    monkeypatch.setenv("PG_CONNECTION_STRING", os.environ["PG_TEST_DSN"])
    monkeypatch.setenv("DB_NAME", urlparse(os.environ["PG_TEST_DSN"]).path.lstrip("/"))
    monkeypatch.delenv("METANAVIT_APPROVAL_KEY_FILE", raising=False)
    monkeypatch.setenv("METANAVIT_APPROVAL_PUBLIC_KEY_FILE", str(private / "operator.pub"))
    monkeypatch.setenv("METANAVIT_APPROVAL_LEDGER_FILE", str(private / "ledger.sqlite"))  # one host: one ledger file
    managers = []

    def open_store(data_dir=root):
        manager = DatabaseManager(conn_string=_schema_dsn(schema))
        managers.append(manager)
        return build_plan_store(manager, str(data_dir))

    def sign(review, **kwargs):
        return sign_review(review, private / "operator.key", actor="operator", **kwargs)

    def sql(statement, params=()):
        import psycopg2

        conn = psycopg2.connect(_schema_dsn(schema))
        try:
            with conn.cursor() as cur:
                cur.execute(statement, params)
                rows = cur.fetchall() if cur.description else None
            conn.commit()
            return rows
        finally:
            conn.close()

    first = DatabaseManager(conn_string=_schema_dsn(schema))
    with first.get_connection() as conn:
        apply_migrations(conn)  # the deploy step
    first.close()
    yield open_store, sign, sql, root
    for manager in managers:
        if not manager.pool.closed:  # a test may have stopped a store already
            manager.close()


def _status(exc_info):
    return exc_info.value.status_code


def test_a_plan_proposed_on_one_store_is_listed_viewed_and_approved_on_the_other(deployment):
    """A15 / C24 / D33: proposed through one FilesystemTools, applied through another. C26: the capability the
    operator signed over the review replica A returned applies on replica B (a review used to carry the session of
    the process that built it, so B rebuilt a different one and refused it as stale). C25: A then reports the result."""
    open_store, sign, sql, root = deployment
    a, b = open_store(), open_store()
    assert a.tools._session != b.tools._session  # two processes' sessions

    created = a.create(SOURCE, "archive/run_047.yaml")
    assert created.status == "pending_approval" and (root / SOURCE).exists()
    assert [p.plan_id for p in b.list()] == [created.plan_id]
    seen = b.view(created.plan_id)
    assert seen.review == created.review and seen.src == SOURCE and seen.status == "pending_approval"

    approved = b.decide(created.plan_id, "approve", "operator", "ok", sign(seen.review))
    assert approved.status == "applied" and approved.actor == "operator" and approved.note == "ok"
    assert not (root / SOURCE).exists() and (root / "archive/run_047.yaml").read_text() == "learning_rate: 3e-4\n"
    assert a.view(created.plan_id).status == "applied"  # C25: the other replica reads the decision from the table
    assert a.tools.plans == {} and b.tools.plans == {}  # nothing kept in a process between requests
    decisions = [(r["action"], r["result"]) for r in a.log.recent(10)]
    assert decisions == [("approve", "applied")]


def test_a_rejection_on_one_store_is_reported_as_rejected_by_the_other(deployment):
    """C25: the decision state lives in the plans table, not in the deciding process's memory."""
    from fastapi import HTTPException

    open_store, sign, sql, root = deployment
    a, b = open_store(), open_store()
    plan = a.create(SOURCE, "archive/run_047.yaml")
    rejected = a.decide(plan.plan_id, "reject", "jose", "keep it")
    assert rejected.status == "rejected"
    seen = b.view(plan.plan_id)
    assert (seen.status, seen.actor, seen.note) == ("rejected", "jose", "keep it") and seen.decided_at is not None
    with pytest.raises(HTTPException) as refused:
        b.decide(plan.plan_id, "approve", "operator", None, sign(seen.review))
    assert _status(refused) == 409 and (root / SOURCE).exists()  # decided plans are final, on every replica


def test_plans_and_their_decisions_survive_a_restart(deployment):
    """C24: a store built again on the same database (a restart) lists every plan with its state and approves."""
    open_store, sign, sql, root = deployment
    a = open_store()
    first = a.create(SOURCE, "archive/one.yaml")
    a.decide(first.plan_id, "reject", "jose", None)
    second = a.create(SOURCE, "archive/two.yaml")
    a.log.vsm.close()  # the process stops

    restarted = open_store()
    listed = {p.plan_id: p.status for p in restarted.list()}
    assert listed == {first.plan_id: "rejected", second.plan_id: "pending_approval"}
    assert [p.plan_id for p in restarted.list()] == [second.plan_id, first.plan_id]  # newest first
    review = restarted.view(second.plan_id).review
    assert restarted.decide(second.plan_id, "approve", "operator", None, sign(review)).status == "applied"
    assert (root / "archive/two.yaml").exists()


def test_a_token_for_one_stores_plan_does_not_apply_the_same_move_proposed_on_the_other(deployment):
    """C26: the same move proposed on two replicas is two reviews (plan id and session differ): a capability for one
    is refused for the other by the binding check, nothing moves and the plan stays pending."""
    from fastapi import HTTPException

    open_store, sign, sql, root = deployment
    a, b = open_store(), open_store()
    on_a = a.create(SOURCE, "archive/run_047.yaml")
    on_b = b.create(SOURCE, "archive/run_047.yaml")
    with pytest.raises(HTTPException) as refused:
        b.decide(on_b.plan_id, "approve", "operator", None, sign(on_a.review))
    assert _status(refused) == 403 and "different reviewed action" in refused.value.detail
    assert (root / SOURCE).exists() and b.view(on_b.plan_id).status == "pending_approval"
    assert a.view(on_a.plan_id).status == "pending_approval"  # and its own token is still unused
    assert b.decide(on_a.plan_id, "approve", "operator", None, sign(on_a.review)).status == "applied"


def test_a_row_edited_after_signing_is_refused_and_nothing_moves(deployment):
    """C26: only the move and the session come from the row, and the review is rebuilt from them and the files as
    this process sees them: an edited destination, session or review refuses the signed capability."""
    from fastapi import HTTPException

    open_store, sign, sql, root = deployment
    a, b = open_store(), open_store()
    plan = a.create(SOURCE, "archive/run_047.yaml")
    token = sign(plan.review)
    edits = [
        ("UPDATE plan_proposals SET dst = 'elsewhere/run_047.yaml' WHERE plan_id = %s", "stale"),
        ("UPDATE plan_proposals SET session = 'another-session' WHERE plan_id = %s", "stale"),
    ]
    original = sql("SELECT dst, session, review, review_sha256 FROM plan_proposals WHERE plan_id = %s", (plan.plan_id,))[0]
    for statement, reason in edits:
        sql(statement, (plan.plan_id,))
        with pytest.raises(HTTPException) as refused:
            b.decide(plan.plan_id, "approve", "operator", None, token)
        assert _status(refused) == 403 and reason in refused.value.detail, refused.value.detail
        sql("UPDATE plan_proposals SET dst = %s, session = %s WHERE plan_id = %s", (original[0], original[1], plan.plan_id))
    # a review rewritten to match the edited destination passes the rebuild, and is a different review: the
    # capability is bound to the one the operator signed
    import hashlib

    from app.mcp.secure_io import fingerprint

    edited = dict(plan.review, payload={"src": SOURCE, "dst": "elsewhere/run_047.yaml"},
                  destinations=[fingerprint(root, "elsewhere/run_047.yaml")])
    text = json.dumps(edited, sort_keys=True, separators=(",", ":"))
    sql("UPDATE plan_proposals SET dst = 'elsewhere/run_047.yaml', review = %s, review_sha256 = %s WHERE plan_id = %s",
        (text, hashlib.sha256(text.encode()).hexdigest(), plan.plan_id))
    with pytest.raises(HTTPException) as refused:
        b.decide(plan.plan_id, "approve", "operator", None, token)
    assert _status(refused) == 403 and "different reviewed action" in refused.value.detail
    assert (root / SOURCE).exists() and not (root / "elsewhere").exists() and not (root / "archive").exists()
    assert b.view(plan.plan_id).status == "pending_approval"


def test_a_store_over_another_root_neither_lists_nor_decides_the_plan(deployment, tmp_path):
    """C26: plans are read by the root identity they were proposed under (the root's path, device and inode)."""
    from fastapi import HTTPException

    open_store, sign, sql, root = deployment
    other_root = tmp_path / "other"
    (other_root / "configs").mkdir(parents=True)
    (other_root / SOURCE).write_text("learning_rate: 3e-4\n")
    a, other = open_store(), open_store(other_root)
    plan = a.create(SOURCE, "archive/run_047.yaml")
    assert other.list() == []
    for call in (lambda: other.view(plan.plan_id),
                 lambda: other.decide(plan.plan_id, "approve", "operator", None, sign(plan.review)),
                 lambda: other.decide(plan.plan_id, "reject", "operator", None)):
        with pytest.raises(HTTPException) as unknown:
            call()
        assert _status(unknown) == 404
    assert (other_root / SOURCE).exists() and (root / SOURCE).exists()
    assert a.view(plan.plan_id).status == "pending_approval"


def _hold_inside_apply(store, entered, release):
    """Make `store`'s next apply wait, holding whatever the decision holds, until `release` is set."""
    real = store.tools.apply_plan

    def held(plan_id, *args, **kwargs):
        entered.set()
        assert release.wait(30)
        return real(plan_id, *args, **kwargs)

    store.tools.apply_plan = held


def test_an_approval_and_a_rejection_racing_on_two_stores_are_decided_one_after_the_other(deployment):
    """A decision holds the plan's row lock until its status is saved: a rejection sent while another replica is
    applying waits, then finds the plan applied (409), so a plan is never reported rejected with its file moved."""
    from fastapi import HTTPException

    open_store, sign, sql, root = deployment
    a, b = open_store(), open_store()
    plan = a.create(SOURCE, "archive/run_047.yaml")
    entered, release, outcome = threading.Event(), threading.Event(), {}
    _hold_inside_apply(a, entered, release)
    approver = threading.Thread(target=lambda: outcome.setdefault("approve", a.decide(plan.plan_id, "approve", "operator", None, sign(plan.review))))
    approver.start()
    assert entered.wait(30)

    def reject():
        try:
            b.decide(plan.plan_id, "reject", "jose", None)
            outcome["reject"] = "rejected"
        except HTTPException as exc:
            outcome["reject"] = exc.status_code

    rejecter = threading.Thread(target=reject)
    rejecter.start()
    time.sleep(1.0)
    assert rejecter.is_alive() and "reject" not in outcome  # waiting for the row
    release.set()
    approver.join(30)
    rejecter.join(30)
    assert outcome["approve"].status == "applied" and outcome["reject"] == 409
    assert not (root / SOURCE).exists() and b.view(plan.plan_id).status == "applied"


def test_two_approvals_with_two_valid_capabilities_move_the_file_once(deployment):
    """Two capabilities for one review (the operator signed twice) presented on two replicas at once: the second
    decision waits for the first, finds the plan applied (409) and never reaches its capability (left unused)."""
    import sqlite3

    from fastapi import HTTPException

    open_store, sign, sql, root = deployment
    a, b = open_store(), open_store()
    plan = a.create(SOURCE, "archive/run_047.yaml")
    first, second = sign(plan.review), sign(plan.review)
    entered, release, outcome = threading.Event(), threading.Event(), {}
    _hold_inside_apply(a, entered, release)
    approver = threading.Thread(target=lambda: outcome.setdefault("a", a.decide(plan.plan_id, "approve", "operator", None, first)))
    approver.start()
    assert entered.wait(30)

    def approve_on_b():
        try:
            outcome["b"] = b.decide(plan.plan_id, "approve", "operator", None, second).status
        except HTTPException as exc:
            outcome["b"] = exc.status_code

    other = threading.Thread(target=approve_on_b)
    other.start()
    time.sleep(1.0)
    assert other.is_alive()
    release.set()
    approver.join(30)
    other.join(30)
    assert outcome["a"].status == "applied" and outcome["b"] == 409
    assert (root / "archive/run_047.yaml").exists()
    with sqlite3.connect(os.environ["METANAVIT_APPROVAL_LEDGER_FILE"]) as db:
        assert db.execute("SELECT count(*) FROM grants").fetchone()[0] == 1  # the second capability was never consumed


def test_each_attempt_keeps_its_own_result_in_the_decision_log(deployment):
    """The decision log records every approve / reject attempt; updating one attempt's result used to rewrite every
    row of that plan and action, so a refusal on one replica read "applied" once another replica applied the plan."""
    from fastapi import HTTPException

    open_store, sign, sql, root = deployment
    a, b = open_store(), open_store()
    plan = a.create(SOURCE, "archive/run_047.yaml")
    with pytest.raises(HTTPException) as refused:
        a.decide(plan.plan_id, "approve", "web-ui", None, None)  # no capability
    assert _status(refused) == 403
    b.decide(plan.plan_id, "approve", "operator", None, sign(plan.review))
    attempts = [(r["action"], r["actor"], r["result"]) for r in a.log.recent(10)]
    assert attempts[0] == ("approve", "operator", "applied")
    assert attempts[1][:2] == ("approve", "web-ui") and attempts[1][2].startswith("refused: ") and len(attempts) == 2


def test_the_decision_log_records_who_decided_and_the_operator_signed_actor_of_an_approval(deployment, tmp_path):
    """C40, on Postgres: the actor column is the one the API passes (the caller's key name when keys are on); an
    approval's row also carries the actor the operator signed into the capability, read back by DecisionLog.recent
    (GET /api/plans/decisions/recent) from the row's payload. A capability that does not verify (here: signed by a
    key the server does not trust, naming "mallory") is refused and leaves no grant actor."""
    from fastapi import HTTPException

    from app.mcp.approvals import create_key, sign_review

    open_store, sign, sql, root = deployment
    store = open_store()
    kept = store.create(SOURCE, "archive/kept.yaml")
    store.decide(kept.plan_id, "reject", "alice", "not now")
    moved = store.create(SOURCE, "archive/run_047.yaml")
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    create_key(stranger / "k", stranger / "k.pub")
    with pytest.raises(HTTPException) as refused:
        store.decide(moved.plan_id, "approve", "bob", None, sign_review(moved.review, stranger / "k", actor="mallory"))
    assert _status(refused) == 403
    store.decide(moved.plan_id, "approve", "bob", None, sign(moved.review))  # deployment's sign(): actor "operator"
    rows = [(r["plan_id"], r["action"], r["actor"], r["grant_actor"], r["result"].split(":")[0]) for r in store.log.recent(10)]
    assert rows == [(moved.plan_id, "approve", "bob", "operator", "applied"),
                    (moved.plan_id, "approve", "bob", None, "refused"),
                    (kept.plan_id, "reject", "alice", None, "rejected")]
    assert sql("SELECT payload->>'grant_actor' FROM plan_decisions WHERE result = 'applied'") == [("operator",)]
    assert (root / "archive" / "run_047.yaml").exists()
