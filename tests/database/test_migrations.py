"""Schema migrations against a live Postgres: upgrade without losing rows.

Needs PG_TEST_DSN (the CI parity job's service, or a local instance); skipped
otherwise. Every test works in its own throwaway schema (search_path), so it
never touches the application's tables in the same database.
"""

import os
import threading
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

# The DDL the application ran inline at startup before migrations existed
# (app/api/routers/plans.py and app/database/index_manager.py at that commit).
# Kept verbatim: it is "the previous version" that a deployed database has.
LEGACY_DDL = """
CREATE TABLE IF NOT EXISTS plan_decisions (
    id BIGSERIAL PRIMARY KEY,
    plan_id TEXT NOT NULL,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT,
    note TEXT,
    decided_at TIMESTAMPTZ NOT NULL,
    result TEXT NOT NULL,
    payload JSONB
);
CREATE TABLE IF NOT EXISTS indexed_files (
    file_path TEXT NOT NULL,
    process_name TEXT NOT NULL,
    process_version TEXT NOT NULL,
    mtime BIGINT NOT NULL,
    data BYTEA,
    content_hash TEXT,
    PRIMARY KEY (file_path, process_name, process_version)
);
ALTER TABLE indexed_files ADD COLUMN IF NOT EXISTS content_hash TEXT;
CREATE TABLE IF NOT EXISTS directory_processing_results (
    dir_path TEXT NOT NULL,
    process_name TEXT NOT NULL,
    process_version TEXT NOT NULL,
    is_applicable BOOLEAN NOT NULL,
    mtime BIGINT NOT NULL,
    PRIMARY KEY (dir_path, process_name, process_version)
);
"""
APP_TABLES = ("plan_decisions", "indexed_files", "directory_processing_results")
INSERT_DECISION = (
    "INSERT INTO plan_decisions (plan_id, src, dst, action, actor, note, decided_at, result, payload) "
    "VALUES (%s, 'a.txt', 'b.txt', 'reject', 'tester', NULL, now(), 'rejected', '{}'::jsonb);"
)


def _connect(schema):
    import psycopg2

    conn = psycopg2.connect(os.environ["PG_TEST_DSN"], options=f"-c search_path={schema}")
    return conn


@pytest.fixture
def schema():
    import psycopg2

    name = "migtest_" + uuid.uuid4().hex[:10]
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


def _columns(conn, schema):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name, data_type, is_nullable, column_default IS NOT NULL "
            "FROM information_schema.columns WHERE table_schema = %s AND table_name = ANY(%s) "
            "ORDER BY table_name, ordinal_position;",
            (schema, list(APP_TABLES)),
        )
        rows = cur.fetchall()
    conn.commit()
    return rows


def _versions(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations ORDER BY version;")
        rows = [r[0] for r in cur.fetchall()]
    conn.commit()
    return rows


def _indexes(conn, schema):
    with conn.cursor() as cur:
        cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = %s;", (schema,))
        rows = {r[0] for r in cur.fetchall()}
    conn.commit()
    return rows


def test_version_1_then_the_next_version_keeps_the_row(schema):
    from app.database.migrate import apply_migrations, discover

    newest = discover()[-1].version
    assert newest >= 2
    conn = _connect(schema)
    try:
        assert apply_migrations(conn, target=1) == [1]
        assert _versions(conn) == [1]
        plan_id = "plan-" + uuid.uuid4().hex
        with conn.cursor() as cur:
            cur.execute(INSERT_DECISION, (plan_id,))
        conn.commit()
        assert "plan_decisions_plan_action_idx" not in _indexes(conn, schema)

        assert apply_migrations(conn) == list(range(2, newest + 1))  # the upgrade
        assert _versions(conn) == list(range(1, newest + 1))
        with conn.cursor() as cur:
            cur.execute("SELECT plan_id, src, dst, action, actor, result FROM plan_decisions;")
            assert cur.fetchall() == [(plan_id, "a.txt", "b.txt", "reject", "tester", "rejected")]
        conn.commit()
        assert "plan_decisions_plan_action_idx" in _indexes(conn, schema)

        assert apply_migrations(conn) == []  # idempotent on a live database
        assert _versions(conn) == list(range(1, newest + 1))
    finally:
        conn.close()


def test_a_database_from_the_pre_migration_code_is_adopted_with_its_rows(schema):
    from app.database.migrate import apply_migrations, discover

    other = "migtest_" + uuid.uuid4().hex[:10]
    conn = _connect(schema)
    try:
        with conn.cursor() as cur:
            cur.execute(LEGACY_DDL)  # what a deployed database looks like today: tables, no ledger
            cur.execute(INSERT_DECISION, ("legacy-plan",))
            cur.execute(
                "INSERT INTO indexed_files (file_path, process_name, process_version, mtime, content_hash) "
                "VALUES ('/data/a.txt', 'default', '1.0', 1, 'abc');"
            )
        conn.commit()
        legacy_columns = _columns(conn, schema)

        assert apply_migrations(conn) == [m.version for m in discover()]
        with conn.cursor() as cur:
            cur.execute("SELECT plan_id FROM plan_decisions;")
            assert cur.fetchall() == [("legacy-plan",)]
            cur.execute("SELECT file_path, content_hash FROM indexed_files;")
            assert cur.fetchall() == [("/data/a.txt", "abc")]
        conn.commit()
        assert _columns(conn, schema) == legacy_columns  # adopting changed no column

        # Migration 0001 on an empty schema builds the same columns as the legacy DDL did.
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {other};")
        conn.commit()
        fresh = _connect(other)
        try:
            assert apply_migrations(fresh, target=1) == [1]
            fresh_columns = _columns(fresh, other)
        finally:
            fresh.close()
        assert fresh_columns == legacy_columns
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {other} CASCADE;")
        conn.commit()
        conn.close()


def test_an_edited_applied_migration_is_refused_on_a_live_database(schema, tmp_path):
    from app.database.migrate import MigrationError, apply_migrations

    (tmp_path / "0001_a.sql").write_text("CREATE TABLE mig_a (id int);")
    conn = _connect(schema)
    try:
        assert apply_migrations(conn, tmp_path) == [1]
        (tmp_path / "0001_a.sql").write_text("CREATE TABLE mig_a (id bigint);")
        (tmp_path / "0002_b.sql").write_text("CREATE TABLE mig_b (id int);")
        with pytest.raises(MigrationError, match="edited after it was applied"):
            apply_migrations(conn, tmp_path)
        assert _versions(conn) == [1]
    finally:
        conn.close()


def test_a_failing_migration_leaves_no_partial_schema(schema, tmp_path):
    from app.database.migrate import apply_migrations

    (tmp_path / "0001_a.sql").write_text("CREATE TABLE mig_a (id int);")
    (tmp_path / "0002_b.sql").write_text("CREATE TABLE mig_b (id int); SELECT * FROM table_that_does_not_exist;")
    conn = _connect(schema)
    try:
        with pytest.raises(Exception, match="table_that_does_not_exist"):
            apply_migrations(conn, tmp_path)
        assert _versions(conn) == [1]
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('mig_a') IS NOT NULL, to_regclass('mig_b') IS NOT NULL;")
            assert cur.fetchone() == (True, False)  # mig_b was created and rolled back with its file
        conn.commit()
    finally:
        conn.close()


def test_a_second_replica_waits_for_the_advisory_lock(schema):
    """Replica A holds the migration lock; replica B's runner must not apply anything until A lets go."""
    from app.database.migrate import LOCK_KEY, apply_migrations, discover

    holder, runner, observer = _connect(schema), _connect(schema), _connect(schema)
    result: dict = {}
    try:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s);", (LOCK_KEY,))
        holder.commit()

        def run():
            try:
                result["applied"] = apply_migrations(runner)
            except Exception as exc:  # noqa: BLE001 - reported by the assertion below
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=1.5)
        assert thread.is_alive(), f"the runner did not wait for the lock: {result}"
        with observer.cursor() as cur:
            cur.execute("SELECT to_regclass('schema_migrations') IS NULL, to_regclass('plan_decisions') IS NULL;")
            assert cur.fetchone() == (True, True)  # nothing was created while the lock was held elsewhere
        observer.commit()

        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s);", (LOCK_KEY,))
        holder.commit()
        thread.join(timeout=30)
        assert not thread.is_alive() and "error" not in result, result
        assert result["applied"] == [m.version for m in discover()]
        assert _versions(observer) == result["applied"]
    finally:
        for conn in (holder, runner, observer):
            conn.close()


# ---------------------------------------------------------------- the application's entry points


class _SchemaPool:
    """What DecisionLog needs from the vector store manager: get_connection(), here bound to one schema."""

    def __init__(self, schema):
        self.schema = schema

    def get_connection(self):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            conn = _connect(self.schema)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

        return scope()


def test_decision_log_runs_the_migrations_and_works_on_the_migrated_table(schema):
    import time

    from app.api.routers.plans import DecisionLog
    from app.database.migrate import discover

    log = DecisionLog(_SchemaPool(schema))  # startup path: no inline DDL any more
    conn = _connect(schema)
    try:
        assert _versions(conn) == [m.version for m in discover()]
    finally:
        conn.close()
    record = {"plan_id": "p1", "src": "a.txt", "dst": "b.txt", "action": "reject", "actor": "tester", "note": None,
              "decided_at": time.time(), "result": "pending"}
    log.write(record)
    log.update({**record, "result": "rejected"})
    rows = log.recent(10)
    assert [(r["plan_id"], r["action"], r["result"]) for r in rows] == [("p1", "reject", "rejected")]
    DecisionLog(_SchemaPool(schema))  # a restart, or a second replica: nothing to apply, the row stays
    assert len(log.recent(10)) == 1


def test_index_manager_runs_the_migrations(schema, monkeypatch):
    from app.database.index_manager import IndexManager
    from app.database.migrate import discover

    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("DB_NAME", dsn.rsplit("/", 1)[-1].split("?")[0])
    manager = IndexManager(conn_string=f"{dsn}{'&' if '?' in dsn else '?'}options=-csearch_path%3D{schema}")
    try:
        assert manager.hashes_for() == {}  # indexed_files exists and is readable
        with manager.get_connection() as conn:
            assert _versions(conn) == [m.version for m in discover()]
            assert {t for t, *_ in _columns(conn, schema)} == set(APP_TABLES)
    finally:
        manager.close()
