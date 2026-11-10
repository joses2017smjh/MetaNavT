"""Versioned SQL schema migrations. No dependency beyond psycopg2.

Files live in app/database/migrations/ and are named NNNN_description.sql
(0001_baseline.sql, 0002_...). Versions are contiguous from 1 and applied in
order. The table schema_migrations records each applied version with the
SHA-256 of its file.

apply_migrations(conn):
  * takes a session-level Postgres advisory lock first, so two replicas that
    start together run the migrations one after the other: the second waits,
    then finds nothing pending;
  * refuses to run (MigrationError) if a file that was already applied has a
    different checksum now, if the database is at a version this code does not
    have, or if the files have a gap or a duplicate version;
  * applies each pending file and its schema_migrations row in one
    transaction: a failure rolls that file back completely and leaves the
    earlier versions applied;
  * is idempotent: a second call applies nothing.

Scope: the tables in migrations/ (the plan decision log and the file index).
The vector table is created by LlamaIndex's PGVectorStore and its indexes by
VectorStoreManager; they are not managed here.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("uvicorn")

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
FILENAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
# One fixed key for pg_advisory_lock (a signed 64-bit integer), derived from a name so it
# cannot collide with another application's small integer keys in the same database.
LOCK_KEY = int.from_bytes(hashlib.sha256(b"metanavit.schema_migrations").digest()[:8], "big", signed=True)

CREATE_LEDGER = """CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);"""


class MigrationError(RuntimeError):
    """The migration files and the database disagree; nothing was applied."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str


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
        found[version] = Migration(version=version, name=path.stem, sql=sql, checksum=checksum(sql))
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


def applied_versions(conn) -> dict[int, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT version, checksum FROM schema_migrations ORDER BY version;")
        return {int(version): digest for version, digest in cur.fetchall()}


def apply_migrations(conn, directory: Path | str = MIGRATIONS_DIR, target: int | None = None) -> list[int]:
    """Apply the pending migrations on `conn` (a psycopg2 connection) and return their versions.

    `target` stops at that version (used by the upgrade test); the default is the newest file.
    The connection is left with no open transaction and without the advisory lock.
    """
    migrations = discover(directory)
    done: list[int] = []
    conn.rollback()  # start from a clean transaction state
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s);", (LOCK_KEY,))
    conn.commit()  # the lock is session-level: it outlives this transaction
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_LEDGER)
        conn.commit()
        todo = pending(migrations, applied_versions(conn), target)
        conn.commit()
        for migration in todo:
            try:
                with conn.cursor() as cur:
                    cur.execute(migration.sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s);",
                        (migration.version, migration.name, migration.checksum),
                    )
                conn.commit()
            except Exception:
                conn.rollback()  # the file and its ledger row succeed or fail together
                logger.error(f"Schema migration {migration.name} failed and was rolled back")
                raise
            done.append(migration.version)
            logger.info(f"Schema migration applied: {migration.name}")
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s);", (LOCK_KEY,))
        conn.commit()
    if not done:
        logger.info(f"Schema migrations: nothing pending ({len(migrations)} files known)")
    return done
