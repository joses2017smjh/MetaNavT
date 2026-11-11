"""Versioned SQL schema migrations, applied by a deploy step. No dependency beyond psycopg2.

    python -m app.database.migrate            (make migrate)   apply every pending migration
    python -m app.database.migrate --check    list what is pending, exit 1 if anything is; applies nothing

The API does not apply them. Start-up (app/engine/bootstrap.py), the plan decision log and
IndexManager call require_current(), which only reads the ledgers and raises
MigrationsPending, so a process never runs against a schema older than its code. Compose
and the CI parity job run the command above before uvicorn.

Three sequences, each a directory of files named NNNN_description.sql (0001_baseline.sql,
0002_...), versions contiguous from 1, applied in order:

  migrations/           the application's own tables (the plan decision log, the file index)
                        and, from 0003, the ledger of the next two. Ledger: schema_migrations.
  migrations/vector/    one vector table: the data table LlamaIndex's PGVectorStore reads and
                        writes, the ingest manifest, and their indexes. The table's schema, name
                        and embedding dimension are configuration (PGVECTOR_SCHEMA,
                        PGVECTOR_TABLE, EMBEDDING_DIM), so these files are templates (render())
                        applied and recorded per table in vector_table_migrations, with the
                        values they were rendered with.
  migrations/paradedb/  the pg_search extension and the BM25 index, applied and required only
                        when LEXICAL_BACKEND=paradedb, per table like vector/. No image used
                        here has pg_search, so these files have never run (owner decision OD07).

apply_migrations(), apply_vector_migrations() and migrate_all():
  * take a session-level Postgres advisory lock first, so two deploy steps that run together
    apply one after the other: the second waits, then finds nothing pending. It waits by
    polling with no transaction open (_advisory_lock), never inside pg_advisory_lock, which
    would deadlock with the first one's CREATE INDEX CONCURRENTLY;
  * refuse to run (MigrationError) if a file that was already applied has a different
    checksum now (the checksum is of the file, not of a rendered template), if the database
    is at a version this code does not have, or if the files have a gap or a duplicate version;
  * apply each pending file and its ledger row in one transaction: a failure rolls that file
    back completely and leaves the earlier versions applied;
  * except a file whose first line is "-- migrate: no-transaction". CREATE INDEX CONCURRENTLY
    cannot run inside a transaction block, so its statements run one by one in autocommit and
    its ledger row is written after the last one. An interrupted file runs again from its
    first statement, so each of its statements must be safe to run twice;
  * are idempotent: a second call applies nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("uvicorn")

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
VECTOR_DIR = MIGRATIONS_DIR / "vector"
PARADEDB_DIR = MIGRATIONS_DIR / "paradedb"
FILENAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
NO_TRANSACTION = "-- migrate: no-transaction"
COMMAND = "python -m app.database.migrate (make migrate)"
# One fixed key for pg_advisory_lock (a signed 64-bit integer), derived from a name so it
# cannot collide with another application's small integer keys in the same database.
LOCK_KEY = int.from_bytes(hashlib.sha256(b"metanavit.schema_migrations").digest()[:8], "big", signed=True)
LOCK_POLL_SECONDS = 0.2  # how often a waiting deploy step tries the lock again (_advisory_lock)

CREATE_LEDGER = """CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);"""
VECTOR_LEDGER = "vector_table_migrations"  # created by migrations/0003_vector_table_migrations.sql
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
MAX_IDENTIFIER_BYTES = 63  # Postgres truncates a longer name, and two truncated names can collide
TEMPLATE_FIELD = re.compile(r"\{([a-z_]+)\}")


class MigrationError(RuntimeError):
    """The migration files and the database disagree; nothing was applied."""


class MigrationsPending(MigrationError):
    """The database is behind the code: run the deploy step before starting."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str
    no_transaction: bool = False


@dataclass(frozen=True)
class VectorTable:
    """The vector table a configuration names: data_<table> in <schema>, embeddings of dim floats."""

    schema: str = "public"
    table: str = "llamaindex_embedding"
    dim: int = 1024
    lexical_backend: str = "ts_rank_cd"

    def __post_init__(self):
        for value in (self.schema, self.table):
            if not IDENTIFIER.match(value or ""):
                raise MigrationError(f"PGVECTOR_SCHEMA and PGVECTOR_TABLE must be lowercase letters, digits and _: {value!r}")
        if int(self.dim) < 1:
            raise MigrationError(f"EMBEDDING_DIM must be positive, got {self.dim!r}")

    @property
    def data_table(self) -> str:
        return f"data_{self.table}"  # the name PGVectorStore gives it

    @property
    def manifest_table(self) -> str:
        return f"ingest_manifest_{self.table}"

    @property
    def key(self) -> str:
        """How the ledger names this table: schema.data_table."""
        return f"{self.schema}.{self.data_table}"

    @classmethod
    def from_env(cls) -> "VectorTable":
        """The table the API would use: PGVECTOR_SCHEMA, PGVECTOR_TABLE, EMBEDDING_DIM, LEXICAL_BACKEND."""
        from app.database.vector_store_manager import lexical_backend_setting

        return cls(
            schema=os.getenv("PGVECTOR_SCHEMA", "public"),
            table=os.getenv("PGVECTOR_TABLE", "llamaindex_embedding"),
            dim=int(os.getenv("EMBEDDING_DIM", 1024)),  # VectorStoreManager's default as well
            lexical_backend=lexical_backend_setting(),
        )


def checksum(sql: str) -> str:
    """SHA-256 of the file text with line endings normalised (a CRLF checkout must not look like an edit)."""
    return hashlib.sha256(sql.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def discover(directory: Path | str = MIGRATIONS_DIR) -> list[Migration]:
    """Read the migration files, sorted by version. Versions must be 1..n with no gap or duplicate."""
    found: dict[int, Migration] = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix != ".sql":
            continue
        match = FILENAME.match(path.name)
        if not match:
            raise MigrationError(f"migration file name must look like 0001_description.sql: {path.name}")
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"two migration files have version {version}: {found[version].name}, {path.stem}")
        sql = path.read_text(encoding="utf-8")
        no_transaction = sql.lstrip().split("\n", 1)[0].strip() == NO_TRANSACTION
        found[version] = Migration(version=version, name=path.stem, sql=sql, checksum=checksum(sql), no_transaction=no_transaction)
    ordered = [found[v] for v in sorted(found)]
    expected = list(range(1, len(ordered) + 1))
    if [m.version for m in ordered] != expected:
        raise MigrationError(f"migration versions must be contiguous from 1: found {[m.version for m in ordered]}")
    return ordered


def pending(migrations: list[Migration], applied: dict[int, str], target: int | None = None) -> list[Migration]:
    """The migrations still to apply, in order. Pure: `applied` maps version -> recorded checksum.

    Raises MigrationError when the recorded history cannot be a prefix of the files.
    """
    known = {m.version: m for m in migrations}
    for version in sorted(applied):
        if version not in known:
            raise MigrationError(
                f"database is at schema version {version}, which this code does not have "
                f"(newest file: {max(known, default=0)}); deploy the newer code instead of downgrading"
            )
        if applied[version] != known[version].checksum:
            raise MigrationError(
                f"migration {known[version].name} was edited after it was applied "
                f"(recorded checksum {applied[version][:12]}, file {known[version].checksum[:12]}); "
                "add a new migration instead of changing an applied one"
            )
    if sorted(applied) != list(range(1, len(applied) + 1)):
        raise MigrationError(f"applied versions have a gap: {sorted(applied)}")
    limit = max(known, default=0) if target is None else target
    return [m for m in migrations if m.version not in applied and m.version <= limit]


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def render(sql: str, table: VectorTable, vector_schema: str = "public") -> str:
    """A vector-table template with its fields filled in for `table`.

    {schema} {data_table} {manifest}  quoted identifiers ("public"."data_x")
    {data_table_literal}              the data table's quoted name as a string literal (for ::regclass)
    {table}                           the bare PGVECTOR_TABLE, to build index names inside quotes
    {dim}                             EMBEDDING_DIM
    {vector_schema}                   the schema of the vector extension (types, operator classes)
    An unknown field, or a quoted name longer than Postgres keeps, is a MigrationError.
    """
    data_table = f"{quote_ident(table.schema)}.{quote_ident(table.data_table)}"
    values = {
        "schema": quote_ident(table.schema),
        "table": table.table,
        "data_table": data_table,
        "data_table_literal": "'" + data_table.replace("'", "''") + "'",
        "manifest": f"{quote_ident(table.schema)}.{quote_ident(table.manifest_table)}",
        "dim": str(int(table.dim)),
        "vector_schema": quote_ident(vector_schema),
    }

    def field(match: re.Match) -> str:
        if match.group(1) not in values:
            raise MigrationError(f"unknown template field {match.group(0)} (known: {', '.join(sorted(values))})")
        return values[match.group(1)]

    rendered = TEMPLATE_FIELD.sub(field, sql)
    for name in re.findall(r'"([^"]*)"', rendered):
        if len(name.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
            raise MigrationError(f"name longer than {MAX_IDENTIFIER_BYTES} bytes: {name!r} (shorten PGVECTOR_TABLE)")
    return rendered


def split_statements(sql: str) -> list[str]:
    """Split SQL text at the semicolons that end statements, for a no-transaction file.

    Semicolons inside quotes ('...', "..."), dollar quotes ($$...$$, $tag$...$tag$) and
    comments do not split. Comment-only pieces are dropped.
    """
    statements, start, i, n = [], 0, 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in "'\"":
            end = sql.find(ch, i + 1)
            while end != -1 and sql[end + 1 : end + 2] == ch:  # a doubled quote is part of the text
                end = sql.find(ch, end + 2)
            i = n if end == -1 else end + 1
        elif sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
        elif ch == "$" and (tag := re.match(r"\$[A-Za-z_]*\$", sql[i:])):
            end = sql.find(tag.group(0), i + len(tag.group(0)))
            i = n if end == -1 else end + len(tag.group(0))
        elif ch == ";":
            statements.append(sql[start : i + 1])
            start = i = i + 1
        else:
            i += 1
    statements.append(sql[start:])
    kept = []
    for statement in statements:
        code = re.sub(r"--[^\n]*", "", statement).strip()
        if code and code != ";":
            kept.append(statement.strip())
    return kept


def _exists(conn, relation: str) -> bool:
    """Whether a table exists (by search_path, or schema-qualified). Reads only."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL;", (relation,))
        return bool(cur.fetchone()[0])


def applied_versions(conn) -> dict[int, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT version, checksum FROM schema_migrations ORDER BY version;")
        return {int(version): digest for version, digest in cur.fetchall()}


def vector_applied_versions(conn, sequence: str, table: VectorTable) -> dict[int, str]:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT version, checksum FROM {VECTOR_LEDGER} WHERE sequence = %s AND vector_table = %s ORDER BY version;",
            (sequence, table.key),
        )
        return {int(version): digest for version, digest in cur.fetchall()}


def vector_extension_schema(conn) -> str | None:
    """The schema the vector extension is installed in, or None when it is not installed."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'vector';"
        )
        row = cur.fetchone()
    return row[0] if row else None


@contextmanager
def _advisory_lock(conn, poll_s: float = LOCK_POLL_SECONDS):
    """Hold the migration lock (session level: it outlives the transactions below) for the block.

    It waits by trying again every poll_s seconds with no transaction open in between, not by
    blocking in pg_advisory_lock: a session blocked there holds a snapshot, and the holder's
    CREATE INDEX CONCURRENTLY waits for every older snapshot to go away, so the two deadlocked
    and one deploy step failed (tests/database/test_migrations.py starts two together).
    """
    conn.rollback()  # start from a clean transaction state
    waiting = False
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s);", (LOCK_KEY,))
            locked = bool(cur.fetchone()[0])
        conn.commit()  # end the transaction, and its snapshot, before waiting
        if locked:
            break
        if not waiting:
            logger.info("Another process holds the schema migration lock; waiting for it")
            waiting = True
        time.sleep(poll_s)
    try:
        yield
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s);", (LOCK_KEY,))
        conn.commit()


def _run_without_transaction(conn, migration: Migration, sql: str) -> None:
    """Each statement in autocommit; the connection is back in transaction mode afterwards."""
    conn.commit()  # autocommit can only be switched with no transaction open
    conn.autocommit = True
    try:
        for statement in split_statements(sql):
            with conn.cursor() as cur:
                cur.execute(statement)
    except Exception:
        logger.error(f"Schema migration {migration.name} failed; it runs again from its first statement next time")
        raise
    finally:
        conn.autocommit = False


def _apply(conn, migrations, applied, target, record, render_sql=None, label="") -> list[int]:
    """Apply what pending() returns, each file with its ledger row (record(cur, migration))."""
    todo = pending(migrations, applied, target)
    conn.commit()
    done: list[int] = []
    for migration in todo:
        sql = render_sql(migration.sql) if render_sql else migration.sql
        if migration.no_transaction:
            _run_without_transaction(conn, migration, sql)
            with conn.cursor() as cur:
                record(cur, migration)
            conn.commit()
        else:
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    record(cur, migration)
                conn.commit()
            except Exception:
                conn.rollback()  # the file and its ledger row succeed or fail together
                logger.error(f"Schema migration {label}{migration.name} failed and was rolled back")
                raise
        done.append(migration.version)
        logger.info(f"Schema migration applied: {label}{migration.name}")
    return done


def _apply_database(conn, migrations: list[Migration], target: int | None) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(CREATE_LEDGER)
    conn.commit()

    def record(cur, migration: Migration) -> None:
        cur.execute(
            "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s);",
            (migration.version, migration.name, migration.checksum),
        )

    return _apply(conn, migrations, applied_versions(conn), target, record)


def _apply_vector(conn, table: VectorTable, migrations: list[Migration], target: int | None, sequence: str) -> list[int]:
    if not _exists(conn, VECTOR_LEDGER):
        raise MigrationError(f"{VECTOR_LEDGER} does not exist: apply the database migrations first ({COMMAND})")
    applied = vector_applied_versions(conn, sequence, table)
    if applied and not _exists(conn, f"{quote_ident(table.schema)}.{quote_ident(table.data_table)}"):
        raise MigrationError(
            f"{VECTOR_LEDGER} lists {sequence} migrations of {table.key}, but the table does not exist: "
            "delete its rows there (app.database.migrate.drop_vector_table) to build it again"
        )
    vector_schema = vector_extension_schema(conn) or "public"  # 0001 creates the extension there when it is missing
    params = json.dumps({"dim": int(table.dim), "vector_schema": vector_schema})

    def record(cur, migration: Migration) -> None:
        cur.execute(
            f"INSERT INTO {VECTOR_LEDGER} (sequence, vector_table, version, name, checksum, params) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb);",
            (sequence, table.key, migration.version, migration.name, migration.checksum, params),
        )

    return _apply(conn, migrations, applied, target, record, lambda sql: render(sql, table, vector_schema), f"{sequence}/")


def apply_migrations(conn, directory: Path | str = MIGRATIONS_DIR, target: int | None = None) -> list[int]:
    """Apply the database sequence's pending migrations on `conn` (a psycopg2 connection); return their versions.

    `target` stops at that version (used by the upgrade test); the default is the newest file.
    The connection is left with no open transaction and without the advisory lock.
    """
    migrations = discover(directory)
    with _advisory_lock(conn):
        done = _apply_database(conn, migrations, target)
    if not done:
        logger.info(f"Schema migrations: nothing pending ({len(migrations)} files known)")
    return done


def apply_vector_migrations(
    conn, table: VectorTable, directory: Path | str = VECTOR_DIR, target: int | None = None, sequence: str = "vector"
) -> list[int]:
    """Apply one vector table's pending migrations (the database sequence must be at 0003 or later)."""
    migrations = discover(directory)
    with _advisory_lock(conn):
        return _apply_vector(conn, table, migrations, target, sequence)


def sequences_for(table: VectorTable | None) -> list[tuple[str, Path]]:
    """The per-table sequences a configuration needs: vector always, paradedb only when it is the lexical backend."""
    if table is None:
        return []
    return [("vector", VECTOR_DIR)] + ([("paradedb", PARADEDB_DIR)] if table.lexical_backend == "paradedb" else [])


def migrate_all(conn, table: VectorTable | None = None) -> dict[str, list[int]]:
    """The deploy step: the database sequence, then each sequence `table` needs, under one lock."""
    with _advisory_lock(conn):
        done = {"database": _apply_database(conn, discover(MIGRATIONS_DIR), None)}
        for sequence, directory in sequences_for(table):
            done[sequence] = _apply_vector(conn, table, discover(directory), None, sequence)
    return done


def pending_all(conn, table: VectorTable | None = None) -> list[str]:
    """Every pending migration's name, in the order migrate_all applies them. Reads only: no DDL, no lock.

    Raises MigrationError when a ledger disagrees with the files (pending()) or names a table that is gone.
    """
    try:
        applied = applied_versions(conn) if _exists(conn, "schema_migrations") else {}
        names = [m.name for m in pending(discover(MIGRATIONS_DIR), applied)]
        has_ledger = _exists(conn, VECTOR_LEDGER)
        for sequence, directory in sequences_for(table):
            done = vector_applied_versions(conn, sequence, table) if has_ledger else {}
            if done and not _exists(conn, f"{quote_ident(table.schema)}.{quote_ident(table.data_table)}"):
                raise MigrationError(f"{VECTOR_LEDGER} lists {sequence} migrations of {table.key}, but the table does not exist")
            names += [f"{sequence}/{m.name} ({table.key})" for m in pending(discover(directory), done)]
    finally:
        conn.rollback()  # end the read; an idle open transaction would hold up CREATE INDEX CONCURRENTLY
    return names


def require_current(conn, table: VectorTable | None = None) -> None:
    """Raise MigrationsPending unless every migration the code (and `table`) needs is applied. Reads only."""
    names = pending_all(conn, table)
    if names:
        raise MigrationsPending(f"{len(names)} schema migration(s) pending: {', '.join(names)}; run {COMMAND} before starting")


def drop_vector_table(conn, table: VectorTable) -> None:
    """Drop a scratch vector table, its manifest and its ledger rows (tests, make ingest-runs, the synthetic plan)."""
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {quote_ident(table.schema)}.{quote_ident(table.data_table)};")
        cur.execute(f"DROP TABLE IF EXISTS {quote_ident(table.schema)}.{quote_ident(table.manifest_table)};")
    if _exists(conn, VECTOR_LEDGER):
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {VECTOR_LEDGER} WHERE vector_table = %s;", (table.key,))
    conn.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply the pending schema migrations: the deploy step, run before the API starts "
        "(PG_CONNECTION_STRING; the vector table from PGVECTOR_SCHEMA, PGVECTOR_TABLE, EMBEDDING_DIM, LEXICAL_BACKEND)."
    )
    parser.add_argument("--check", action="store_true", help="list the pending migrations and exit 1 if there are any; apply nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv()  # .env as the API reads it (main.py): set variables win over the file
    except ImportError:
        pass
    dsn = os.getenv("PG_CONNECTION_STRING")
    if not dsn:
        print("set PG_CONNECTION_STRING")
        return 2
    from app.database.db_base_manager import DatabaseManager  # creates the database when it is missing

    table = VectorTable.from_env()
    manager = DatabaseManager(conn_string=dsn)
    try:
        with manager.get_connection() as conn:
            if args.check:
                names = pending_all(conn, table)
                print(f"{len(names)} schema migration(s) pending" + "".join(f"\n  {name}" for name in names))
                return 1 if names else 0
            done = migrate_all(conn, table)
    finally:
        manager.close()
    for sequence, versions in done.items():
        where = "" if sequence == "database" else f" ({table.key})"
        print(f"{sequence}{where}: " + (f"applied {versions}" if versions else "nothing pending"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
