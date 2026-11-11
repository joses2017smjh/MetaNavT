"""Schema migration runner: ordering, checksums, idempotence. No database.

The runner only needs cursor()/commit()/rollback(), so a small fake stands in
for Postgres here: it keeps a ledger with transactional inserts and a log of
every statement. The live upgrade test is tests/database/test_migrations.py.
"""

import pytest

from app.database import migrate
from app.database.migrate import MigrationError


def _write(directory, *files):
    for name, sql in files:
        (directory / name).write_text(sql)
    return directory


class FakeDb:
    def __init__(self):
        self.ledger: dict[int, str] = {}  # committed rows of schema_migrations: version -> checksum
        self.log: list[str] = []
        self.modes: list[tuple[str, bool]] = []  # (statement, autocommit) for every statement
        self.locked = False


class FakeConn:
    def __init__(self, db):
        self.db = db
        self.uncommitted: dict[int, str] = {}
        self._autocommit = False

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        assert not self.uncommitted, "autocommit switched inside a transaction"  # psycopg2 refuses it too
        self._autocommit = value

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.db.ledger.update(self.uncommitted)
        self.uncommitted = {}
        self.db.log.append("COMMIT")

    def rollback(self):
        self.uncommitted = {}
        self.db.log.append("ROLLBACK")


class FakeCursor:
    def __init__(self, conn):
        self.conn, self.rows = conn, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        db, text = self.conn.db, sql.strip()
        db.log.append(text)
        db.modes.append((text, self.conn.autocommit))
        if text.startswith("SELECT pg_advisory_lock"):
            assert not db.locked, "lock taken twice"
            db.locked = True
        elif text.startswith("SELECT pg_advisory_unlock"):
            db.locked = False
        elif text.startswith("SELECT version, checksum"):
            self.rows = sorted(db.ledger.items())
        elif text.startswith("INSERT INTO schema_migrations"):
            assert db.locked, "ledger written without the advisory lock"
            self.conn.uncommitted[params[0]] = params[2]
        elif "BOOM" in text:
            raise RuntimeError("syntax error at or near BOOM")

    def fetchall(self):
        return self.rows


# ---------------------------------------------------------------- ordering


def test_discover_orders_by_version_and_ignores_other_files(tmp_path):
    _write(tmp_path, ("0002_second.sql", "SELECT 2;"), ("0001_first.sql", "SELECT 1;"), ("0003_third.sql", "SELECT 3;"),
           ("README.md", "not a migration"))
    found = migrate.discover(tmp_path)
    assert [(m.version, m.name) for m in found] == [(1, "0001_first"), (2, "0002_second"), (3, "0003_third")]
    assert found[0].sql == "SELECT 1;"


@pytest.mark.parametrize("files,message", [
    ([("0001_a.sql", "x"), ("0003_c.sql", "x")], "contiguous"),
    ([("0002_b.sql", "x")], "contiguous"),
    ([("0001_a.sql", "x"), ("0001_b.sql", "x")], "two migration files have version 1"),
    ([("1_a.sql", "x")], "file name must look like"),
    ([("0001-Bad-Name.sql", "x")], "file name must look like"),
])
def test_discover_rejects_gaps_duplicates_and_bad_names(tmp_path, files, message):
    _write(tmp_path, *files)
    with pytest.raises(MigrationError, match=message):
        migrate.discover(tmp_path)


def test_the_committed_migrations_are_well_formed():
    found = migrate.discover()
    assert [m.version for m in found] == list(range(1, len(found) + 1)) and len(found) >= 2
    baseline = found[0].sql
    for table in ("plan_decisions", "indexed_files", "directory_processing_results"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in baseline
    assert "schema_migrations" not in "".join(m.sql for m in found)  # the ledger belongs to the runner


# ---------------------------------------------------------------- checksums


def test_checksum_changes_with_content_but_not_with_line_endings():
    assert migrate.checksum("CREATE TABLE a (id int);\n") == migrate.checksum("CREATE TABLE a (id int);\r\n")
    assert migrate.checksum("CREATE TABLE a (id int);\n") != migrate.checksum("CREATE TABLE a (id bigint);\n")
    assert len(migrate.checksum("x")) == 64


def test_pending_returns_unapplied_versions_in_order(tmp_path):
    found = migrate.discover(_write(tmp_path, ("0001_a.sql", "a"), ("0002_b.sql", "b"), ("0003_c.sql", "c")))
    assert [m.version for m in migrate.pending(found, {})] == [1, 2, 3]
    assert [m.version for m in migrate.pending(found, {1: found[0].checksum})] == [2, 3]
    assert [m.version for m in migrate.pending(found, {}, target=1)] == [1]
    assert migrate.pending(found, {m.version: m.checksum for m in found}) == []


def test_pending_refuses_an_applied_file_that_was_edited(tmp_path):
    found = migrate.discover(_write(tmp_path, ("0001_a.sql", "a"), ("0002_b.sql", "b")))
    with pytest.raises(MigrationError, match="0001_a was edited after it was applied"):
        migrate.pending(found, {1: migrate.checksum("something else")})


def test_pending_refuses_a_database_newer_than_the_code_or_with_a_gap(tmp_path):
    found = migrate.discover(_write(tmp_path, ("0001_a.sql", "a"), ("0002_b.sql", "b")))
    with pytest.raises(MigrationError, match="schema version 3"):
        migrate.pending(found, {1: found[0].checksum, 2: found[1].checksum, 3: "f" * 64})
    with pytest.raises(MigrationError, match="gap"):
        migrate.pending(found, {2: found[1].checksum})


# ---------------------------------------------------------------- the runner against the fake


def test_apply_is_ordered_locked_and_idempotent(tmp_path):
    _write(tmp_path, ("0002_b.sql", "CREATE TABLE b ();"), ("0001_a.sql", "CREATE TABLE a ();"))
    db = FakeDb()
    assert migrate.apply_migrations(FakeConn(db), tmp_path) == [1, 2]
    assert sorted(db.ledger) == [1, 2] and not db.locked

    statements = [s for s in db.log if s not in {"COMMIT", "ROLLBACK"}]
    assert statements[0].startswith("SELECT pg_advisory_lock")  # nothing runs before the lock
    assert statements[-1].startswith("SELECT pg_advisory_unlock")
    assert statements.index("CREATE TABLE a ();") < statements.index("CREATE TABLE b ();")

    before = len(db.log)
    assert migrate.apply_migrations(FakeConn(db), tmp_path) == []  # a second replica, or a restart
    assert not [s for s in db.log[before:] if s.startswith("CREATE TABLE a") or s.startswith("CREATE TABLE b")]
    assert sorted(db.ledger) == [1, 2] and not db.locked


def test_apply_stops_at_the_target_then_continues(tmp_path):
    _write(tmp_path, ("0001_a.sql", "CREATE TABLE a ();"), ("0002_b.sql", "CREATE TABLE b ();"))
    db = FakeDb()
    assert migrate.apply_migrations(FakeConn(db), tmp_path, target=1) == [1]
    assert migrate.apply_migrations(FakeConn(db), tmp_path) == [2]


def test_a_failing_migration_is_rolled_back_and_not_recorded(tmp_path):
    _write(tmp_path, ("0001_a.sql", "CREATE TABLE a ();"), ("0002_b.sql", "BOOM;"), ("0003_c.sql", "CREATE TABLE c ();"))
    db = FakeDb()
    with pytest.raises(RuntimeError, match="BOOM"):
        migrate.apply_migrations(FakeConn(db), tmp_path)
    assert sorted(db.ledger) == [1]  # 1 stays, 2 is not recorded, 3 never ran
    assert "CREATE TABLE c ();" not in db.log
    assert not db.locked  # the lock is released on failure


def test_apply_refuses_to_run_when_an_applied_file_changed(tmp_path):
    _write(tmp_path, ("0001_a.sql", "CREATE TABLE a ();"))
    db = FakeDb()
    migrate.apply_migrations(FakeConn(db), tmp_path)
    _write(tmp_path, ("0001_a.sql", "CREATE TABLE a (id int);"), ("0002_b.sql", "CREATE TABLE b ();"))
    with pytest.raises(MigrationError, match="edited after it was applied"):
        migrate.apply_migrations(FakeConn(db), tmp_path)
    assert sorted(db.ledger) == [1] and "CREATE TABLE b ();" not in db.log and not db.locked


# ---------------------------------------------------------------- no-transaction files (CREATE INDEX CONCURRENTLY)


NO_TX = """-- migrate: no-transaction
-- comment; with a semicolon
DROP INDEX CONCURRENTLY IF EXISTS i_new;
CREATE INDEX CONCURRENTLY i_new ON t (a);
"""


def test_a_no_transaction_file_runs_statement_by_statement_in_autocommit(tmp_path):
    _write(tmp_path, ("0001_a.sql", "CREATE TABLE t (a int);"), ("0002_idx.sql", NO_TX))
    found = migrate.discover(tmp_path)
    assert [m.no_transaction for m in found] == [False, True]
    db = FakeDb()
    assert migrate.apply_migrations(FakeConn(db), tmp_path) == [1, 2]
    modes = dict((text.splitlines()[-1], autocommit) for text, autocommit in db.modes)
    assert modes["CREATE TABLE t (a int);"] is False  # an ordinary file: one transaction with its ledger row
    assert modes["DROP INDEX CONCURRENTLY IF EXISTS i_new;"] is True  # one statement per execute, no transaction
    assert modes["CREATE INDEX CONCURRENTLY i_new ON t (a);"] is True
    ledger_rows = [(i, autocommit) for i, (text, autocommit) in enumerate(db.modes) if text.startswith("INSERT INTO schema_migrations")]
    assert len(ledger_rows) == 2 and not any(autocommit for _, autocommit in ledger_rows)  # each row in a transaction
    last_statement = [text for text, _ in db.modes].index("CREATE INDEX CONCURRENTLY i_new ON t (a);")
    assert last_statement < ledger_rows[1][0]  # the file is recorded after its last statement ran
    assert sorted(db.ledger) == [1, 2] and not db.locked


def test_a_failing_no_transaction_file_is_not_recorded_and_leaves_autocommit_off(tmp_path):
    _write(tmp_path, ("0001_idx.sql", "-- migrate: no-transaction\nCREATE INDEX CONCURRENTLY i ON t (a);\nBOOM;\n"))
    db = FakeDb()
    conn = FakeConn(db)
    with pytest.raises(RuntimeError, match="BOOM"):
        migrate.apply_migrations(conn, tmp_path)
    assert db.ledger == {} and not db.locked and conn.autocommit is False


@pytest.mark.parametrize("sql,expected", [
    ("SELECT 1; SELECT 2;", ["SELECT 1;", "SELECT 2;"]),
    ("SELECT 'a;b'; SELECT \"x;y\" FROM t;", ["SELECT 'a;b';", 'SELECT "x;y" FROM t;']),
    ("SELECT 'it''s; here';", ["SELECT 'it''s; here';"]),
    ("DO $$ BEGIN PERFORM 1; PERFORM 2; END $$;\nSELECT 3;", ["DO $$ BEGIN PERFORM 1; PERFORM 2; END $$;", "SELECT 3;"]),
    ("DO $body$ BEGIN RAISE NOTICE '$$;'; END $body$;", ["DO $body$ BEGIN RAISE NOTICE '$$;'; END $body$;"]),
    ("-- a comment; not a statement\nSELECT 1; /* also; not */ SELECT 2;\n-- trailing", ["-- a comment; not a statement\nSELECT 1;", "/* also; not */ SELECT 2;"]),
])
def test_split_statements_splits_only_at_statement_ends(sql, expected):
    assert migrate.split_statements(sql) == expected


# ---------------------------------------------------------------- vector-table templates

TABLE = migrate.VectorTable(schema="public", table="llamaindex_embedding", dim=256)


def test_render_fills_identifiers_and_the_dimension():
    sql = "CREATE TABLE {data_table} (e {vector_schema}.vector({dim})); CREATE INDEX \"{table}_x\" ON {data_table} (a); SELECT {data_table_literal}::regclass, {manifest};"
    assert migrate.render(sql, TABLE, "ext") == (
        'CREATE TABLE "public"."data_llamaindex_embedding" (e "ext".vector(256)); '
        'CREATE INDEX "llamaindex_embedding_x" ON "public"."data_llamaindex_embedding" (a); '
        'SELECT \'"public"."data_llamaindex_embedding"\'::regclass, "public"."ingest_manifest_llamaindex_embedding";'
    )
    assert migrate.render("text_fields='{\"text\": {\"tokenizer\": {}}}'", TABLE) == "text_fields='{\"text\": {\"tokenizer\": {}}}'"  # JSON is not a field


def test_render_refuses_an_unknown_field_and_a_name_postgres_would_truncate():
    with pytest.raises(MigrationError, match="unknown template field"):
        migrate.render("SELECT {nope};", TABLE)
    long_table = migrate.VectorTable(table="t" * 40, dim=8)
    with pytest.raises(MigrationError, match="longer than 63 bytes"):
        migrate.render('CREATE INDEX "{table}_embedding_hnsw_m16_efc64" ON {data_table} (a);', long_table)


@pytest.mark.parametrize("schema,table", [("public", "Bad"), ("public", "x; DROP TABLE y"), ("pub lic", "x"), ("public", "")])
def test_a_table_name_that_is_not_a_plain_lowercase_identifier_is_refused(schema, table):
    with pytest.raises(MigrationError, match="lowercase"):
        migrate.VectorTable(schema=schema, table=table)


def test_the_committed_vector_templates_render_and_keep_concurrently_out_of_transactions():
    found = migrate.discover(migrate.VECTOR_DIR)
    assert [m.version for m in found] == list(range(1, len(found) + 1)) and found
    for m in found:
        rendered = migrate.render(m.sql, TABLE)  # every field known, every name short enough
        for statement in migrate.split_statements(rendered):
            code = " ".join(line for line in statement.splitlines() if not line.lstrip().startswith("--"))
            if "CONCURRENTLY" in code:
                assert m.no_transaction, f"{m.name}: CONCURRENTLY cannot run inside the file's transaction"


def test_vector_index_builds_after_the_tables_are_concurrent():
    for m in migrate.discover(migrate.VECTOR_DIR):
        if "CREATE TABLE" in m.sql:
            continue  # an index made with its (new, empty) table needs no concurrent build
        creates = [s for s in migrate.split_statements(m.sql) if "CREATE INDEX" in s.upper()]
        assert all("CONCURRENTLY" in s for s in creates), m.name


def test_the_committed_paradedb_templates_render_and_never_drop_an_index():
    found = migrate.discover(migrate.PARADEDB_DIR)
    assert [m.version for m in found] == list(range(1, len(found) + 1)) and found
    for m in found:
        rendered = migrate.render(m.sql, TABLE)
        assert "DROP INDEX" not in rendered.upper(), m.name  # the BM25 index is built once, never dropped and rebuilt
        if "CREATE INDEX" in rendered:
            assert "CREATE INDEX IF NOT EXISTS" in rendered, m.name


def test_the_paradedb_sequence_is_needed_only_with_the_paradedb_backend():
    assert [s for s, _ in migrate.sequences_for(TABLE)] == ["vector"]
    paradedb = migrate.VectorTable(table="x", dim=8, lexical_backend="paradedb")
    assert [s for s, _ in migrate.sequences_for(paradedb)] == ["vector", "paradedb"]
    assert migrate.sequences_for(None) == []
