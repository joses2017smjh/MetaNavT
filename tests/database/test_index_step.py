"""The index step against a live Postgres: the command, the lock wait's limit, and an API that starts while it runs.

Needs PG_TEST_DSN (the CI parity service, or a local instance); skipped otherwise. Every test indexes a temporary
copy of fixture v1 (never bench/corpus/files itself) into a vector table of its own, built by the schema migrations
and dropped at the end, with the hash embedder. "Another process" is a real one: a child Python process that takes
the table's ingest lock and holds it until the test closes its stdin.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"
HOLD_LOCK = (
    "import sys, psycopg2\n"
    "conn = psycopg2.connect(sys.argv[1]); conn.autocommit = True\n"
    "conn.cursor().execute('SELECT pg_advisory_lock(%s);', (int(sys.argv[2]),))\n"
    "print('locked', flush=True)\n"
    "sys.stdin.read()\n"
)


def _lock_key(schema: str, data_table: str) -> int:
    name = f"metanavit.ingest:{schema}.{data_table}"
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)


class LockHolder:
    """A separate process holding the table's ingest lock from start() until release()."""

    def __init__(self, dsn: str, key: int):
        self.proc = subprocess.Popen([sys.executable, "-c", HOLD_LOCK, dsn, str(key)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        assert self.proc.stdout.readline().strip() == "locked"

    def release(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)


@pytest.fixture
def table(monkeypatch, tmp_path):
    """A migrated vector table, a fixture copy as DATA_DIR, and the environment the API and the index step read."""
    from urllib.parse import urlparse

    from app.database import vector_store
    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager

    dsn = os.environ["PG_TEST_DSN"]
    corpus = tmp_path / "files"
    shutil.copytree(FIXTURE, corpus)
    env = {"PG_CONNECTION_STRING": dsn, "PSYCOPG2_CONNECTION_STRING": dsn, "DB_NAME": urlparse(dsn).path.lstrip("/"),
           "PGVECTOR_TABLE": "indexstep_" + uuid.uuid4().hex[:8], "EMBEDDING_DIM": "256", "EMBEDDING_PROVIDER": "hash",
           "MODEL_PROVIDER": "ollama", "RERANKER_MODEL": "none", "DATA_DIR": str(corpus), "INGEST_MAX_REJECTED": "0"}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    for name in ("PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES", "INDEX_ON_START",
                 "INGEST_LOCK_TIMEOUT_SECONDS", "STARTUP_RETRY_ATTEMPTS", "FRONTEND_ENDPOINT", "RETRIEVAL_REQUIRE_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    vsm = VectorStoreManager(conn_string=dsn)
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)  # the deploy step
    vector_store.get_vector_store_manager.cache_clear()
    yield vsm, env
    if vector_store.get_vector_store_manager.cache_info().currsize:
        vector_store.get_vector_store_manager().close()
    vector_store.get_vector_store_manager.cache_clear()
    with vsm.get_connection() as conn:
        drop_vector_table(conn, vsm.vector_table)
    vsm.close()


def _index_command(env: dict, *extra_env) -> subprocess.CompletedProcess:
    full = {**os.environ, **env, "PYTHONPATH": str(ROOT), **dict(extra_env)}
    return subprocess.run([sys.executable, "-m", "app.engine.ingest"], cwd=ROOT, env=full, capture_output=True, text=True, timeout=300)


def test_the_index_command_indexes_data_dir_and_a_second_run_embeds_nothing(table):
    vsm, env = table
    n_files = sum(1 for p in Path(env["DATA_DIR"]).rglob("*") if p.is_file())

    first = _index_command(env)
    assert first.returncode == 0, first.stderr
    report = json.loads(first.stdout.strip().splitlines()[-1])
    assert (report["files_seen"], report["files_indexed"], report["files_rejected"]) == (n_files, n_files, 0)
    assert report["chunks_embedded"] == report["n_nodes"] == vsm.count_nodes() > 0
    assert report["pipeline"].endswith("|hash-256")  # the label the API's own start would write

    second = _index_command(env)
    assert second.returncode == 0, second.stderr
    again = json.loads(second.stdout.strip().splitlines()[-1])
    assert (again["chunks_embedded"], again["files_unchanged"], again["indexed"]) == (0, n_files, False)


def test_an_index_run_waits_at_most_the_lock_timeout_and_writes_nothing(table):
    """The wait for the ingest lock had no limit: a run behind a stuck holder never returned."""
    from app.database.vector_store_manager import IngestLockTimeout
    from app.engine.ingest import index_directory
    from app.retrieval.embedders import HashEmbedder

    vsm, env = table
    holder = LockHolder(env["PG_CONNECTION_STRING"], _lock_key(vsm.schema_name, vsm.data_table))
    outcome: dict = {}

    def run():
        started = time.monotonic()
        try:
            outcome["report"] = index_directory(vsm, env["DATA_DIR"], embed_fn=lambda t: HashEmbedder(256).encode(t).tolist(),
                                                embedder="hash-256", lock_timeout_s=1.0)
        except Exception as exc:  # noqa: BLE001 - asserted below
            outcome["error"] = exc
        outcome["seconds"] = time.monotonic() - started

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=30)
    still_waiting = worker.is_alive()
    holder.release()
    worker.join(timeout=60)
    assert not still_waiting, "the run was still waiting for the lock after 30 s"
    assert isinstance(outcome.get("error"), IngestLockTimeout), outcome
    assert 0.9 <= outcome["seconds"] < 15
    assert vsm.count_nodes() == 0 and vsm.ingest_manifest() == {}  # nothing indexed

    holder = LockHolder(env["PG_CONNECTION_STRING"], _lock_key(vsm.schema_name, vsm.data_table))
    try:  # the step itself behind a holder: exit status 1 and one line saying why
        blocked = _index_command(env, ("INGEST_LOCK_TIMEOUT_SECONDS", "1"))
    finally:
        holder.release()
    assert blocked.returncode == 1 and "indexing failed: IngestLockTimeout:" in blocked.stderr
    assert vsm.ingest_lock_key == _lock_key(vsm.schema_name, vsm.data_table)


def _serve(env: dict, until, deadline_s: float) -> dict:
    """Start main.app in a thread (its lifespan's first attempt runs before it serves), poll /health until
    until(response) or the deadline; {"entered": bool, "health": [(status, body)...]}."""
    from fastapi.testclient import TestClient

    import main  # noqa: WPS433 - the FastAPI app module

    seen: dict = {"entered": False, "health": []}
    stop = time.monotonic() + deadline_s

    def run():
        with TestClient(main.app) as client:
            seen["entered"] = True
            while time.monotonic() < stop:
                response = client.get("/health")
                seen["health"].append((response.status_code, response.json()))
                if until(response):
                    break
                time.sleep(0.1)

    seen["thread"] = threading.Thread(target=run, daemon=True)
    seen["thread"].start()
    return seen


def test_the_api_as_compose_runs_it_answers_health_while_another_process_holds_the_ingest_lock(table):
    """Compose's api container indexed before the server started; now the server's start reads the table and the
    manifest without touching the lock, so a process that is indexing (or stuck) does not hold it up. Before, Compose
    set INDEX_ON_START=true and the start waited in pg_advisory_lock with no limit: no /health answer at all."""
    import yaml

    vsm, env = table
    assert _index_command(env).returncode == 0  # the index step ran first
    rows = vsm.count_nodes()
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["api"]["environment"]
    os.environ["INDEX_ON_START"] = str(compose["INDEX_ON_START"])  # as the Compose file sets it (restored by monkeypatch)
    holder = LockHolder(env["PG_CONNECTION_STRING"], _lock_key(vsm.schema_name, vsm.data_table))
    try:
        seen = _serve(env, lambda r: r.status_code == 200, deadline_s=30)
        seen["thread"].join(timeout=45)
        entered = seen["entered"]
    finally:
        holder.release()
        seen["thread"].join(timeout=120)
    assert entered, "the API never started serving while the lock was held"
    status, body = seen["health"][-1]
    assert status == 200 and body["n_nodes"] == rows
    assert body["indexing"]["indexed"] is False and body["indexing"]["manifest"]["chunks"] == rows
    assert body["indexing"]["chunker"] == "app/chunking (structure-aware)"


def test_an_api_that_indexes_on_start_answers_503_while_the_lock_is_held_then_200(table, monkeypatch):
    """INDEX_ON_START=true (a single-process setup) still indexes at start: behind a holder it now gives up after
    INGEST_LOCK_TIMEOUT_SECONDS, /health says why (503), the start is retried, and /health turns 200 once the
    holder is gone. Before, the start waited for the lock with no limit and /health never answered."""
    vsm, env = table
    for name, value in {"INDEX_ON_START": "true", "INGEST_LOCK_TIMEOUT_SECONDS": "1", "STARTUP_RETRY_SECONDS": "0.2",
                        "STARTUP_RETRY_MAX_SECONDS": "0.5"}.items():
        monkeypatch.setenv(name, value)
    holder = LockHolder(env["PG_CONNECTION_STRING"], _lock_key(vsm.schema_name, vsm.data_table))
    try:
        seen = _serve(env, lambda r: r.status_code == 200, deadline_s=60)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not any(s == 503 and "IngestLockTimeout" in str(b.get("error")) for s, b in seen["health"]):
            time.sleep(0.1)
        refused = [(s, b) for s, b in seen["health"] if s == 503]
    finally:
        holder.release()
    seen["thread"].join(timeout=90)
    assert refused and refused[0][1]["error"].startswith("IngestLockTimeout: another process held the ingest lock"), seen["health"][:3]
    assert refused[0][1]["startup"]["retrying"] is True
    status, body = seen["health"][-1]
    assert status == 200, seen["health"][-3:]
    assert body["indexing"]["indexed"] is True and body["indexing"]["chunks_embedded"] == body["n_nodes"] > 0
