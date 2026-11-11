"""Schema migrations against a live Postgres: upgrade without losing rows; start-up runs no DDL.

Needs PG_TEST_DSN (the CI parity job's service, or a local instance); skipped
otherwise. Most tests work in their own throwaway schema (search_path for the
database sequence, PGVECTOR_SCHEMA for a vector table), so they never touch the
application's tables in the same database; the start-up tests use a vector table
of their own in the default schema and drop it (and its ledger rows) at the end.
"""

import os
import threading
import time
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


def _schema_dsn(schema):
    """PG_TEST_DSN (a URL) with search_path set to `schema`, for code that takes a connection string."""
    dsn = os.environ["PG_TEST_DSN"]
    return f"{dsn}{'&' if '?' in dsn else '?'}options=-csearch_path%3D{schema}"


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


def test_two_deploy_steps_waiting_for_the_lock_both_succeed(schema):
    """Two deploy steps wait for the migration lock; the one that gets it builds indexes CONCURRENTLY, which
    waits for every transaction holding an older snapshot to end. A waiter blocked inside pg_advisory_lock
    holds one, so the two used to deadlock and one step failed ("deadlock detected"). The runner now waits
    by polling with no transaction open: both steps succeed, and each file is applied once."""
    from app.database import migrate

    table = migrate.VectorTable(schema=schema, table="twodeploy", dim=256)
    results: dict = {}

    def deploy(name):
        conn = _connect(schema)
        try:
            results[name] = migrate.migrate_all(conn, table)
        except Exception as exc:  # noqa: BLE001 - reported by the assertion below
            results[name] = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()

    holder = _plain()  # autocommit: holds the lock without holding a snapshot
    threads = [threading.Thread(target=deploy, args=(name,), daemon=True) for name in ("a", "b")]
    try:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s);", (migrate.LOCK_KEY,))
        for thread in threads:
            thread.start()
        time.sleep(1.5)  # longer than deadlock_timeout: both steps are waiting for the lock
        assert all(thread.is_alive() for thread in threads) and not results, results
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s);", (migrate.LOCK_KEY,))
        for thread in threads:
            thread.join(timeout=60)
    finally:
        holder.close()
    assert not any(thread.is_alive() for thread in threads) and len(results) == 2, results
    assert all(isinstance(done, dict) for done in results.values()), results
    everything = {"database": [m.version for m in migrate.discover()],
                  "vector": [m.version for m in migrate.discover(migrate.VECTOR_DIR)]}
    assert sorted(results.values(), key=lambda done: len(done["vector"])) == [{"database": [], "vector": []}, everything]


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


def _exists(conn, relation):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL;", (relation,))
        found = cur.fetchone()[0]
    conn.commit()
    return found


def test_decision_log_refuses_pending_migrations_and_works_on_the_migrated_table(schema):
    import time

    from app.api.routers.plans import DecisionLog
    from app.database.migrate import MigrationsPending, apply_migrations, discover

    with pytest.raises(MigrationsPending, match="0001_baseline.*make migrate"):
        DecisionLog(_SchemaPool(schema))  # the start-up path only reads the ledger
    conn = _connect(schema)
    try:
        assert not _exists(conn, "schema_migrations") and not _exists(conn, "plan_decisions")  # no DDL ran
        assert apply_migrations(conn) == [m.version for m in discover()]  # the deploy step
    finally:
        conn.close()
    log = DecisionLog(_SchemaPool(schema))
    record = {"plan_id": "p1", "src": "a.txt", "dst": "b.txt", "action": "reject", "actor": "tester", "note": None,
              "decided_at": time.time(), "result": "pending"}
    log.write(record)
    log.update({**record, "result": "rejected"})
    rows = log.recent(10)
    assert [(r["plan_id"], r["action"], r["result"]) for r in rows] == [("p1", "reject", "rejected")]
    DecisionLog(_SchemaPool(schema))  # a restart, or a second replica: nothing to apply, the row stays
    assert len(log.recent(10)) == 1


def test_index_manager_refuses_pending_migrations_and_reads_the_migrated_tables(schema, monkeypatch):
    from app.database.index_manager import IndexManager
    from app.database.migrate import MigrationsPending, apply_migrations, discover

    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("DB_NAME", dsn.rsplit("/", 1)[-1].split("?")[0])
    with pytest.raises(MigrationsPending):
        IndexManager(conn_string=_schema_dsn(schema))
    conn = _connect(schema)
    try:
        apply_migrations(conn)
    finally:
        conn.close()
    manager = IndexManager(conn_string=_schema_dsn(schema))
    try:
        assert manager.hashes_for() == {}  # indexed_files exists and is readable
        with manager.get_connection() as conn:
            assert _versions(conn) == [m.version for m in discover()]
            assert {t for t, *_ in _columns(conn, schema)} == set(APP_TABLES)
    finally:
        manager.close()


# ---------------------------------------------------------------- the vector table (migrations/vector/)

# What fe666cb ran inline for a vector table, verbatim but for the placeholders: ensure_text_index,
# ensure_hnsw (bootstrap, after indexing) and ensure_ingest_manifest (every indexing run). The table
# itself came from PGVectorStore's own setup (perform_setup=True), which _store() reproduces.
LEGACY_VECTOR_DDL = """
CREATE INDEX IF NOT EXISTS "{data}_text_tsv_gin" ON "{schema}"."{data}" USING GIN (to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
    ON {schema}.{data}
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE TABLE IF NOT EXISTS "{schema}"."ingest_manifest_{table}" (path TEXT PRIMARY KEY, content_hash TEXT NOT NULL, n_chunks INTEGER NOT NULL, pipeline TEXT NOT NULL DEFAULT '', indexed_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS "{data}_path_idx" ON "{schema}"."{data}" ((metadata_->>'path'));
"""
VECTOR_DOCS = {"a.txt": "learning_rate: 3e-4 encoder: dinov2", "b.txt": "run 47 epoch 12 val_rmse 0.055", "c.txt": "paper draft"}


def _plain():
    import psycopg2

    conn = psycopg2.connect(os.environ["PG_TEST_DSN"])
    conn.autocommit = True  # never idle in a transaction: CREATE INDEX CONCURRENTLY would wait for it
    return conn


def _store(schema, table, setup):
    """PGVectorStore as VectorStoreManager builds it; setup=True is the pre-migration code's own DDL."""
    from llama_index.vector_stores.postgres import PGVectorStore

    dsn = os.environ["PG_TEST_DSN"]
    scheme = dsn.split("://", 1)[0] + "://"
    return PGVectorStore(
        connection_string=dsn.replace(scheme, "postgresql+psycopg2://", 1),
        async_connection_string=dsn.replace(scheme, "postgresql+asyncpg://", 1),
        schema_name=schema, table_name=table, embed_dim=256, perform_setup=setup,
    )


def _nodes():
    from llama_index.core.schema import TextNode

    from app.retrieval.embedders import HashEmbedder

    emb = HashEmbedder(256)
    return [TextNode(id_=f"node-{path}", text=text, embedding=emb.encode([text])[0].tolist(), metadata={"path": path})
            for path, text in VECTOR_DOCS.items()]


def _index_rows(conn, schema, data_table):
    """{index name: (oid, valid, definition)} on one table."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT i.relname, i.oid, x.indisvalid, pg_get_indexdef(i.oid) FROM pg_index x "
            "JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_class t ON t.oid = x.indrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace WHERE n.nspname = %s AND t.relname = %s;",
            (schema, data_table),
        )
        return {name: (oid, valid, definition) for name, oid, valid, definition in cur.fetchall()}


def _rows(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def test_a_version_2_database_holding_manifest_rows_upgrades_to_version_3_and_keeps_them(schema, monkeypatch, capsys):
    """The A04 upgrade: a database as fe666cb left it (schema version 2, a vector table PGVectorStore made,
    rows indexed, a manifest ensure_ingest_manifest wrote, the start-up indexes) goes through the deploy step's
    entry point to version 3 and the vector sequence, and keeps every row; adopted indexes are not rebuilt."""
    from app.database import migrate

    table = "legacy_" + uuid.uuid4().hex[:8]
    data = f"data_{table}"
    conn, plain = _connect(schema), _plain()
    try:
        assert migrate.apply_migrations(conn, target=2) == [1, 2]  # the database fe666cb's start-up left
        _store(schema, table, setup=True).add(_nodes())  # the vector table, made by LlamaIndex's setup
        with plain.cursor() as cur:
            cur.execute(LEGACY_VECTOR_DDL.format(schema=schema, data=data, table=table))
            cur.execute(f'INSERT INTO "{schema}"."ingest_manifest_{table}" (path, content_hash, n_chunks, pipeline) VALUES '
                        "('a.txt', 'h-a', 1, 'chunker-v1|hash-256'), ('b.txt', 'h-b', 1, 'chunker-v1|hash-256'), "
                        "('c.txt', 'h-c', 1, 'chunker-v1|hash-256');")
        rows_before = _rows(plain, f'SELECT id, node_id, metadata_->>\'path\', text FROM "{schema}"."{data}" ORDER BY id;')
        manifest_before = _rows(plain, f'SELECT path, content_hash, n_chunks, pipeline, indexed_at FROM "{schema}"."ingest_manifest_{table}" ORDER BY path;')
        indexes_before = _index_rows(plain, schema, data)
        assert len(rows_before) == len(VECTOR_DOCS) and len(manifest_before) == len(VECTOR_DOCS)
        assert _versions(conn) == [1, 2]

        dsn = os.environ["PG_TEST_DSN"]
        monkeypatch.setenv("PG_CONNECTION_STRING", _schema_dsn(schema))  # the ledgers live in the throwaway schema
        monkeypatch.setenv("DB_NAME", dsn.rsplit("/", 1)[-1].split("?")[0])
        monkeypatch.setenv("PGVECTOR_SCHEMA", schema)
        monkeypatch.setenv("PGVECTOR_TABLE", table)
        monkeypatch.setenv("EMBEDDING_DIM", "256")
        monkeypatch.delenv("LEXICAL_BACKEND", raising=False)
        assert migrate.main(["--check"]) == 1  # pending: 0003 and the whole vector sequence
        assert migrate.main([]) == 0  # python -m app.database.migrate, the deploy step
        assert migrate.main(["--check"]) == 0
        printed = capsys.readouterr().out
        vector_versions = [m.version for m in migrate.discover(migrate.VECTOR_DIR)]
        assert "database: applied [3]" in printed and f"vector ({schema}.{data}): applied {vector_versions}" in printed

        assert _versions(conn) == [1, 2, 3]
        assert _rows(plain, f'SELECT id, node_id, metadata_->>\'path\', text FROM "{schema}"."{data}" ORDER BY id;') == rows_before
        assert _rows(plain, f'SELECT path, content_hash, n_chunks, pipeline, indexed_at FROM "{schema}"."ingest_manifest_{table}" ORDER BY path;') == manifest_before
        ledger = _rows(plain, f'SELECT version, params FROM "{schema}".vector_table_migrations WHERE sequence = \'vector\' AND vector_table = \'{schema}.{data}\' ORDER BY version;')
        assert [v for v, _ in ledger] == [m.version for m in migrate.discover(migrate.VECTOR_DIR)]
        assert all(params["dim"] == 256 for _, params in ledger)  # the embedding dimension, recorded

        indexes_after = _index_rows(plain, schema, data)
        for adopted in (f"{data}_path_idx", f"{table}_idx_1", f"{data}_pkey"):
            assert indexes_after[adopted][0] == indexes_before[adopted][0], adopted  # the same index, not rebuilt
        assert f"{table}_embedding_hnsw" in indexes_before and f"{table}_embedding_hnsw" not in indexes_after
        assert indexes_after[f"{table}_embedding_hnsw_m16_efc64"][1] is True  # built CONCURRENTLY, valid
        assert "m='16', ef_construction='64'" in indexes_after[f"{table}_embedding_hnsw_m16_efc64"][2]
    finally:
        conn.close()
        plain.close()


def test_vector_0001_builds_the_table_pgvectorstore_builds(schema):
    """vector/0001 creates the columns and the LlamaIndex indexes PGVectorStore's own setup gives it, so a
    table either made is the same table when the later files (indexes, the stored text_tsv column) start."""
    from app.database.migrate import VectorTable, apply_vector_migrations, apply_migrations

    conn, plain = _connect(schema), _plain()
    try:
        apply_migrations(conn)
        _store(schema, "shape_llx", setup=True).add(_nodes()[:1])
        apply_vector_migrations(conn, VectorTable(schema=schema, table="shape_mig", dim=256), target=1)  # the table as created

        def shape(table):
            data = f"data_{table}"
            columns = _rows(plain, "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull, "
                                   "replace(pg_get_expr(d.adbin, d.adrelid), %s, 'T') FROM pg_attribute a "
                                   "LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
                                   f"WHERE a.attrelid = '\"{schema}\".\"{data}\"'::regclass AND a.attnum > 0 AND NOT a.attisdropped "
                                   "ORDER BY a.attnum;".replace("%s", f"'{data}'"))
            indexes = {name.replace(table, "T"): definition.replace(f"{schema}.{data}", "S.T").replace(table, "T")
                       for name, (_, _, definition) in _index_rows(plain, schema, data).items()}
            return columns, indexes

        llx_columns, llx_indexes = shape("shape_llx")
        mig_columns, mig_indexes = shape("shape_mig")
        assert mig_columns == llx_columns
        assert all(mig_indexes.get(name) == definition for name, definition in llx_indexes.items()), (llx_indexes, mig_indexes)
        assert ("embedding", "vector(256)") in [(c[0], c[1]) for c in mig_columns]
    finally:
        conn.close()
        plain.close()


def test_the_deploy_step_creates_a_missing_vector_schema(schema):
    """PGVectorStore's own setup created PGVECTOR_SCHEMA when it did not exist; the API now builds the store
    with perform_setup=False, so vector/0001 creates it (and leaves an existing one alone)."""
    from app.database import migrate

    vector_schema = schema + "_v"
    conn, plain = _connect(schema), _plain()
    try:
        assert _rows(plain, f"SELECT to_regnamespace('{vector_schema}') IS NULL;") == [(True,)]
        table = migrate.VectorTable(schema=vector_schema, table="schematest", dim=256)
        assert migrate.migrate_all(conn, table)["vector"] == [m.version for m in migrate.discover(migrate.VECTOR_DIR)]
        assert _rows(plain, f"SELECT to_regclass('\"{vector_schema}\".\"data_schematest\"') IS NOT NULL;") == [(True,)]
        assert migrate.pending_all(conn, table) == []
    finally:
        conn.close()
        with plain.cursor() as cur:
            cur.execute(f'DROP SCHEMA IF EXISTS "{vector_schema}" CASCADE;')
        plain.close()


def test_a_migration_that_changes_hnsw_m_rebuilds_the_index(schema, tmp_path):
    """IF NOT EXISTS on a fixed name never changed an existing index. A parameter change is a new file that
    builds the index under its new name (CONCURRENTLY) and drops the old one; the rows are untouched."""
    import shutil

    from app.database import migrate

    table = migrate.VectorTable(schema=schema, table="hnswtest", dim=256)
    conn, plain = _connect(schema), _plain()
    try:
        migrate.apply_migrations(conn)
        migrate.apply_vector_migrations(conn, table)
        store = _store(schema, table.table, setup=False)
        store.add(_nodes())
        before = _index_rows(plain, schema, table.data_table)
        assert "hnswtest_embedding_hnsw_m16_efc64" in before

        for path in migrate.VECTOR_DIR.glob("*.sql"):
            shutil.copy(path, tmp_path / path.name)
        nxt = len(migrate.discover(migrate.VECTOR_DIR)) + 1
        (tmp_path / f"{nxt:04d}_hnsw_m24_efc64.sql").write_text(
            "-- migrate: no-transaction\n"
            'DROP INDEX CONCURRENTLY IF EXISTS {schema}."{table}_embedding_hnsw_m24_efc64";\n'
            'CREATE INDEX CONCURRENTLY "{table}_embedding_hnsw_m24_efc64" ON {data_table}\n'
            "    USING hnsw (embedding {vector_schema}.vector_cosine_ops) WITH (m = 24, ef_construction = 64);\n"
            'DROP INDEX CONCURRENTLY IF EXISTS {schema}."{table}_embedding_hnsw_m16_efc64";\n'
        )
        assert migrate.apply_vector_migrations(conn, table, directory=tmp_path) == [nxt]
        after = _index_rows(plain, schema, table.data_table)
        assert "hnswtest_embedding_hnsw_m16_efc64" not in after
        oid, valid, definition = after["hnswtest_embedding_hnsw_m24_efc64"]
        assert valid and "m='24', ef_construction='64'" in definition
        assert _rows(plain, f'SELECT count(*) FROM "{schema}"."{table.data_table}";') == [(len(VECTOR_DOCS),)]
        assert migrate.apply_vector_migrations(conn, table, directory=tmp_path) == []  # applied once
    finally:
        conn.close()
        plain.close()


# ---------------------------------------------------------------- start-up: refuses a pending migration, runs no DDL


@pytest.fixture
def startup_env(monkeypatch, tmp_path):
    """Environment for build_retrieval_state over a vector table of its own in the default schema."""
    import shutil
    from pathlib import Path

    from app.database import vector_store

    dsn = os.environ["PG_TEST_DSN"]
    table = "startup_" + uuid.uuid4().hex[:8]
    corpus = tmp_path / "files"
    shutil.copytree(Path(__file__).resolve().parents[2] / "bench" / "corpus" / "files", corpus)
    for name, value in {"PG_CONNECTION_STRING": dsn, "PSYCOPG2_CONNECTION_STRING": dsn, "PGVECTOR_TABLE": table,
                        "EMBEDDING_DIM": "256", "EMBEDDING_PROVIDER": "hash", "MODEL_PROVIDER": "ollama",
                        "RERANKER_MODEL": "none", "INDEX_ON_START": "true", "DATA_DIR": str(corpus),
                        "INGEST_MAX_REJECTED": "0"}.items():
        monkeypatch.setenv(name, value)
    for name in ("PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES"):
        monkeypatch.delenv(name, raising=False)
    from app.settings import init_settings

    init_settings()  # the hash embedder, as main.py's lifespan sets it before build_retrieval_state
    vector_store.get_vector_store_manager.cache_clear()
    yield table, corpus
    from app.database.migrate import VectorTable, drop_vector_table

    vector_store.get_vector_store_manager.cache_clear()
    conn = _plain()
    conn.autocommit = False
    try:
        drop_vector_table(conn, VectorTable(table=table, dim=256))
    finally:
        conn.close()


def test_start_up_refuses_while_a_migration_is_pending(startup_env, monkeypatch, tmp_path):
    from app.database.migrate import MigrationsPending, migrate_all
    from app.database.vector_store import get_vector_store_manager
    from app.engine.bootstrap import build_retrieval_state

    table, _ = startup_env
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "missing"))  # indexing would fail on this; the check comes first
    with pytest.raises(MigrationsPending, match=rf"vector/0001_data_table \(public\.data_{table}\).*make migrate"):
        build_retrieval_state()
    vsm = get_vector_store_manager()
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)  # the deploy step
    vsm.check_migrations()  # nothing pending now
    with pytest.raises(FileNotFoundError):  # past the check, into indexing
        build_retrieval_state()


def test_start_up_runs_no_ddl(startup_env):
    """Deploy step, then build_retrieval_state indexing a copy of the fixture: an event trigger on
    ddl_command_start (every CREATE, ALTER, DROP, ...) records nothing during start-up."""
    from app.database.migrate import migrate_all
    from app.database.vector_store import get_vector_store_manager
    from app.engine.bootstrap import build_retrieval_state

    table, corpus = startup_env
    admin = _plain()
    if not _rows(admin, "SELECT rolsuper FROM pg_roles WHERE rolname = current_user;")[0][0]:
        admin.close()
        pytest.skip("an event trigger needs a superuser")
    suffix = uuid.uuid4().hex[:8]
    log, function, trigger = f"ddl_log_{suffix}", f"ddl_log_fn_{suffix}", f"ddl_log_trg_{suffix}"
    vsm = get_vector_store_manager()
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)
    with admin.cursor() as cur:
        cur.execute(f"CREATE TABLE {log} (tag text NOT NULL, at timestamptz NOT NULL DEFAULT now());")
        cur.execute(f"CREATE FUNCTION {function}() RETURNS event_trigger LANGUAGE plpgsql AS "
                    f"$$ BEGIN INSERT INTO public.{log} (tag) VALUES (tg_tag); END $$;")
        cur.execute(f"CREATE EVENT TRIGGER {trigger} ON ddl_command_start EXECUTE FUNCTION {function}();")
    try:
        state = build_retrieval_state()
        n_files = sum(1 for p in corpus.rglob("*") if p.is_file())
        assert state.indexing["files_indexed"] == n_files and state.vsm.count_nodes() > 0
        assert _rows(admin, f"SELECT tag FROM {log};") == []
    finally:
        with admin.cursor() as cur:
            cur.execute(f"DROP EVENT TRIGGER IF EXISTS {trigger};")
            cur.execute(f"DROP FUNCTION IF EXISTS {function}();")
            cur.execute(f"DROP TABLE IF EXISTS {log};")
        admin.close()


def test_the_paradedb_sequence_is_applied_only_with_the_paradedb_backend(schema):
    """LEXICAL_BACKEND=ts_rank_cd needs no paradedb file; with paradedb configured its files are pending, and
    on an image without pg_search (CI's, this one's) the deploy step fails there and records none of them.
    The files themselves have never run: no image used here has pg_search (OD07)."""
    from app.database import migrate

    plain, conn = _plain(), _connect(schema)
    try:
        table = migrate.VectorTable(schema=schema, table="pdbtest", dim=256)
        assert set(migrate.migrate_all(conn, table)) == {"database", "vector"}
        assert migrate.pending_all(conn, table) == []
        paradedb = migrate.VectorTable(schema=schema, table="pdbtest", dim=256, lexical_backend="paradedb")
        names = [f"paradedb/{m.name} ({paradedb.key})" for m in migrate.discover(migrate.PARADEDB_DIR)]
        assert migrate.pending_all(conn, paradedb) == names
        if not _rows(plain, "SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search';"):
            with pytest.raises(Exception, match="pg_search"):
                migrate.migrate_all(conn, paradedb)
            assert migrate.pending_all(conn, paradedb) == names  # the failed file was rolled back, nothing recorded
    finally:
        conn.close()
        plain.close()
