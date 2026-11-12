"""The grant ledger in Postgres (C49, and D33's second half): approval_grants and approval_audit (migration 0005),
which METANAVIT_APPROVAL_LEDGER=postgres gives the API's replicas instead of a SQLite file per host.

A capability is consumed by committing its grant row before the action runs; the grant id is the primary key, so
of two verifiers presenting one capability, in two processes on the database, exactly one insert commits and the
other's unique violation is read as consumed. The 22 attack families of the approval benchmark run here with the
Postgres ledger and must give the rows they give with the SQLite file. Needs PG_TEST_DSN; skipped otherwise. Every
test runs in a schema of its own, dropped at the end.
"""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "configs/run_047.yaml"


def _schema_dsn(schema):
    dsn = os.environ["PG_TEST_DSN"]
    return f"{dsn}{'&' if '?' in dsn else '?'}options=-csearch_path%3D{schema}"


@pytest.fixture
def database(tmp_path, monkeypatch):
    """A migrated schema of its own; returns (open_manager, sql, dsn)."""
    from urllib.parse import urlparse

    import psycopg2

    from app.database.db_base_manager import DatabaseManager
    from app.database.migrate import apply_migrations

    schema = "ledgertest_" + uuid.uuid4().hex[:10]
    admin = psycopg2.connect(os.environ["PG_TEST_DSN"])
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema};")
    monkeypatch.setenv("PG_CONNECTION_STRING", os.environ["PG_TEST_DSN"])
    monkeypatch.setenv("DB_NAME", urlparse(os.environ["PG_TEST_DSN"]).path.lstrip("/"))
    dsn = _schema_dsn(schema)
    managers = []

    def open_manager():
        manager = DatabaseManager(conn_string=dsn)
        managers.append(manager)
        return manager

    def sql(statement, params=()):
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(statement, params)
                rows = cur.fetchall() if cur.description else None
            conn.commit()
            return rows
        finally:
            conn.close()

    with open_manager().get_connection() as conn:
        apply_migrations(conn)  # the deploy step
    try:
        yield open_manager, sql, dsn
    finally:
        for manager in managers:
            if not manager.pool.closed:
                manager.close()
        with admin.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
        admin.close()


@pytest.fixture
def operator(tmp_path):
    from app.mcp.approvals import create_key

    private = tmp_path / "operator"
    private.mkdir(mode=0o700)
    create_key(private / "operator.key", private / "operator.pub")
    return private


def _proposal(tmp_path, verifier):
    from app.mcp.filesystem import FilesystemTools

    root = tmp_path / "data"
    (root / "configs").mkdir(parents=True)
    (root / SOURCE).write_text("learning_rate: 3e-4\n")
    tools = FilesystemTools(root=root, approval_verifier=verifier)
    return tools, tools.propose_move(SOURCE, "archive/run_047.yaml")


def test_every_approval_benchmark_case_gives_the_same_row_with_the_postgres_ledger(database):
    """C49: the 22 attack families (all ten variants each) and the five benign families, once with each case's own
    SQLite file and once with the Postgres ledger: every row is the same, every attack is denied without mutation
    (the five burnt replay_patch variants by the ledger), and the grants table holds exactly the grants the
    authorized applies consumed."""
    from app.eval.trusted_actions import ATTACK_FAMILIES, BENIGN_FAMILIES, action_case
    from app.mcp.postgres_ledger import PostgresLedger

    open_manager, sql, dsn = database
    manager = open_manager()
    benign_variants = 4
    cases = [(f, v) for f in ATTACK_FAMILIES for v in range(10)] + [(f, v) for f in BENIGN_FAMILIES for v in range(benign_variants)]
    rows = []
    for family, variant in cases:
        on_postgres = action_case(family, variant, ledger=lambda: PostgresLedger(manager))
        assert on_postgres == action_case(family, variant), (family, variant)
        assert on_postgres["passed"], on_postgres
        rows.append(on_postgres)
    assert sum(row["refused_by"] == "ledger" for row in rows) == 5
    # One grant per applied action: replay_move and replay_patch apply (or burn) once before the replay, and four
    # benign families apply once per variant (unapproved_reads applies nothing). No grant was consumed twice.
    expected = 10 + 10 + (len(BENIGN_FAMILIES) - 1) * benign_variants
    assert sql("SELECT count(*), count(DISTINCT grant_id) FROM approval_grants;") == [(expected, expected)]


def test_a_capability_consumed_by_one_verifier_is_refused_by_a_second_on_the_database(database, operator, tmp_path):
    """D33: two verifiers (two API processes' connection pools) share the ledger; the audit trail is shared too."""
    from app.mcp.approvals import ApprovalError, ApprovalVerifier, sign_review
    from app.mcp.postgres_ledger import PostgresLedger

    open_manager, sql, dsn = database
    first = ApprovalVerifier(tmp_path / "data", operator / "operator.pub", ledger=PostgresLedger(open_manager()))
    second = ApprovalVerifier(tmp_path / "data", operator / "operator.pub", ledger=PostgresLedger(open_manager()))
    tools, plan = _proposal(tmp_path, first)
    token = sign_review(plan["review"], operator / "operator.key", actor="operator")
    first.consume(token, plan["review"])
    with pytest.raises(ApprovalError) as refused:
        second.consume(token, plan["review"])
    assert refused.value.check == "ledger" and str(refused.value) == "approval already consumed"
    with pytest.raises(ApprovalError) as again:
        first.consume(token, plan["review"])
    assert again.value.check == "ledger"
    assert sql("SELECT count(*) FROM approval_grants;") == [(1,)]
    # the verifier holds the public key and the ledger (a database handle), no private key or shared secret
    assert set(vars(second.database_ledger)) == {"database"}


def test_the_audit_trail_is_written_to_postgres_without_the_token(database, operator, tmp_path):
    from app.mcp.approvals import ApprovalVerifier, sign_review
    from app.mcp.postgres_ledger import PostgresLedger

    open_manager, sql, dsn = database
    verifier = ApprovalVerifier(tmp_path / "data", operator / "operator.pub", ledger=PostgresLedger(open_manager()))
    tools, plan = _proposal(tmp_path, verifier)
    token = sign_review(plan["review"], operator / "operator.key", actor="operator")
    assert tools.apply_plan(plan["plan_id"], approval_token=token)["status"] == "applied"
    events = sql("SELECT plan_id, action, result, detail FROM approval_audit ORDER BY id;")
    assert [(e[0], e[1], e[2]) for e in events] == [(plan["plan_id"], "apply_plan", "authorized"), (plan["plan_id"], "apply_plan", "applied")]
    assert token not in json.dumps(events)


# One verifier in a process of its own: it builds its ledger, says "ready", waits for "go" on stdin, consumes the
# capability and prints what happened. The parent starts two, then lets both go at once.
CHILD = r"""
import json, sys
from pathlib import Path
args = json.loads(sys.argv[1])
from app.mcp.approvals import ApprovalError, ApprovalVerifier
if args["ledger"] == "postgres":
    from app.database.db_base_manager import DatabaseManager
    from app.mcp.postgres_ledger import PostgresLedger
    verifier = ApprovalVerifier(Path(args["root"]), Path(args["public"]), ledger=PostgresLedger(DatabaseManager(conn_string=args["dsn"])))
else:
    verifier = ApprovalVerifier(Path(args["root"]), Path(args["public"]), Path(args["file"]))
print("ready", flush=True)
sys.stdin.readline()
try:
    verifier.consume(args["token"], args["review"])
    print(json.dumps({"outcome": "consumed"}), flush=True)
except ApprovalError as exc:
    print(json.dumps({"outcome": "refused", "check": exc.check}), flush=True)
"""


def race(args: dict, processes: int = 2) -> list[dict]:
    """Start `processes` verifier processes on one ledger, release them together, return what each one did."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    children = [subprocess.Popen([sys.executable, "-c", CHILD, json.dumps(args)], cwd=ROOT, env=env, text=True,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(processes)]
    try:
        for child in children:
            assert child.stdout.readline().strip() == "ready", child.stderr.read()
        for child in children:  # both are waiting on stdin: release them as close together as the pipes allow
            child.stdin.write("go\n")
            child.stdin.flush()
        outcomes = [json.loads(child.stdout.readline()) for child in children]
        for child in children:
            assert child.wait(60) == 0, child.stderr.read()
        return outcomes
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()


def test_two_verifier_processes_on_one_postgres_ledger_consume_a_capability_exactly_once(database, operator, tmp_path):
    """C49: five rounds, a fresh capability each; in every round one process consumes it and the other is refused
    by the ledger check (a unique violation, read as consumed); the table holds one grant per round."""
    from app.mcp.approvals import ApprovalVerifier, sign_review
    from app.mcp.postgres_ledger import PostgresLedger

    open_manager, sql, dsn = database
    verifier = ApprovalVerifier(tmp_path / "data", operator / "operator.pub", ledger=PostgresLedger(open_manager()))
    tools, plan = _proposal(tmp_path, verifier)
    for _ in range(5):
        token = sign_review(plan["review"], operator / "operator.key", actor="operator")
        outcomes = race({"ledger": "postgres", "dsn": dsn, "root": str(tools.root), "public": str(operator / "operator.pub"),
                         "token": token, "review": plan["review"]})
        assert sorted(o["outcome"] for o in outcomes) == ["consumed", "refused"], outcomes
        assert [o["check"] for o in outcomes if o["outcome"] == "refused"] == ["ledger"]
    assert sql("SELECT count(*), count(DISTINCT grant_id) FROM approval_grants;") == [(5, 5)]


def test_api_replicas_configured_for_postgres_share_the_ledger_across_plan_stores(database, operator, tmp_path, monkeypatch):
    """METANAVIT_APPROVAL_LEDGER=postgres: build_plan_store gives each replica a verifier on its own database pool and
    the one table. A capability burnt on replica B (consumed, then the move failed) is refused on replica A by the
    ledger, where a SQLite file per host would have let it apply; a fresh capability then applies on A."""
    from fastapi import HTTPException

    from app.api.routers.plans import build_plan_store
    from app.mcp import filesystem
    from app.mcp.approvals import sign_review
    from app.mcp.postgres_ledger import PostgresLedger

    open_manager, sql, dsn = database
    monkeypatch.delenv("METANAVIT_APPROVAL_KEY_FILE", raising=False)
    monkeypatch.delenv("METANAVIT_APPROVAL_LEDGER_FILE", raising=False)
    monkeypatch.setenv("METANAVIT_APPROVAL_PUBLIC_KEY_FILE", str(operator / "operator.pub"))
    monkeypatch.setenv("METANAVIT_APPROVAL_LEDGER", "postgres")
    root = tmp_path / "data"
    (root / "configs").mkdir(parents=True)
    (root / SOURCE).write_text("learning_rate: 3e-4\n")
    a, b = build_plan_store(open_manager(), str(root)), build_plan_store(open_manager(), str(root))
    assert isinstance(a.tools.approval_verifier.database_ledger, PostgresLedger) and a.tools.approval_verifier.ledger is None
    plan = a.create(SOURCE, "archive/run_047.yaml")
    token = sign_review(plan.review, operator / "operator.key", actor="operator")

    real = filesystem.move_file

    def failing_move(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(filesystem, "move_file", failing_move)
    with pytest.raises(HTTPException) as failed:
        b.decide(plan.plan_id, "approve", "operator", None, token)
    assert failed.value.status_code == 500 and (root / SOURCE).exists()
    monkeypatch.setattr(filesystem, "move_file", real)
    with pytest.raises(HTTPException) as replayed:
        a.decide(plan.plan_id, "approve", "operator", None, token)
    assert replayed.value.status_code == 403 and replayed.value.detail == "approval already consumed"
    assert (root / SOURCE).exists() and a.view(plan.plan_id).status == "pending_approval"
    fresh = sign_review(plan.review, operator / "operator.key", actor="operator")
    assert a.decide(plan.plan_id, "approve", "operator", None, fresh).status == "applied"
    assert sql("SELECT count(*) FROM approval_grants;") == [(2,)]  # the burnt grant and the fresh one
