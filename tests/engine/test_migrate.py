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
        self.locked = False


class FakeConn:
    def __init__(self, db):
        self.db = db
        self.uncommitted: dict[int, str] = {}

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
