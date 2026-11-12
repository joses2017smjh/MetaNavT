"""`python -m app.engine.generate` (scripts/run.sh generate) against a live Postgres: it is the index step.

It used to split DATA_DIR with LlamaIndex's SentenceSplitter into the served table, rows without a path, outside
the manifest, so a run after an API start added a second set of rows. Needs PG_TEST_DSN (the CI parity service,
or a local instance); skipped otherwise. Each table is built by the schema migrations and dropped at the end; the
corpus is a temporary copy of fixture v1; the embedder is the hash adapter.
"""

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"


@pytest.fixture
def served(monkeypatch, tmp_path):
    """make_table() -> a migrated table (PGVECTOR_TABLE set to it); DATA_DIR a fixture copy; the API's environment."""
    from urllib.parse import urlparse

    from app.database import vector_store
    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager

    dsn = os.environ["PG_TEST_DSN"]
    corpus = tmp_path / "files"
    shutil.copytree(FIXTURE, corpus)
    for name, value in {"PG_CONNECTION_STRING": dsn, "PSYCOPG2_CONNECTION_STRING": dsn, "DB_NAME": urlparse(dsn).path.lstrip("/"),
                        "EMBEDDING_DIM": "256", "EMBEDDING_PROVIDER": "hash", "MODEL_PROVIDER": "ollama", "RERANKER_MODEL": "none",
                        "DATA_DIR": str(corpus), "INGEST_MAX_REJECTED": "0"}.items():
        monkeypatch.setenv(name, value)
    for name in ("PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES", "INDEX_ON_START", "INGEST_LOCK_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    managers = []

    def make_table():
        monkeypatch.setenv("PGVECTOR_TABLE", "generatecli_" + uuid.uuid4().hex[:8])
        vsm = VectorStoreManager(conn_string=dsn)
        with vsm.get_connection() as conn:
            migrate_all(conn, vsm.vector_table)
        managers.append(vsm)
        vector_store.get_vector_store_manager.cache_clear()
        return vsm

    yield make_table, corpus
    vector_store.get_vector_store_manager.cache_clear()
    for vsm in managers:
        with vsm.get_connection() as conn:
            drop_vector_table(conn, vsm.vector_table)
        vsm.close()


def _generate() -> dict:
    """Run the CLI in a process of its own with this environment; return its report."""
    done = subprocess.run([sys.executable, "-m", "app.engine.generate"], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _rows(vsm) -> list[tuple]:
    """Every row as (serial id, node_id, text, metadata_ as text, embedding as text), in id order."""
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT id, node_id, text, metadata_::text, embedding::text FROM "{vsm.schema_name}"."{vsm.data_table}" ORDER BY id;')
            return cur.fetchall()


def _start():
    """What the API's lifespan runs: the settings, then build_retrieval_state (INDEX_ON_START=true here)."""
    from app.database import vector_store
    from app.engine.bootstrap import build_retrieval_state
    from app.settings import init_settings

    init_settings()
    vector_store.get_vector_store_manager.cache_clear()
    try:
        return build_retrieval_state()
    finally:
        vector_store.get_vector_store_manager().close()
        vector_store.get_vector_store_manager.cache_clear()


def test_the_cli_after_a_served_start_embeds_nothing(served, monkeypatch):
    make_table, corpus = served
    vsm = make_table()
    monkeypatch.setenv("INDEX_ON_START", "true")
    state = _start()  # a served start indexes DATA_DIR
    n_files = sum(1 for p in corpus.rglob("*") if p.is_file())
    assert state.indexing["files_indexed"] == n_files
    before = _rows(vsm)

    report = _generate()

    assert (report["chunks_embedded"], report["files_unchanged"], report["files_indexed"], report["indexed"]) == (0, n_files, 0, False)
    assert _rows(vsm) == before  # not a row added or rewritten: no second set of chunks


def test_the_command_writes_the_nodes_a_start_writes_and_a_second_run_embeds_nothing(served, monkeypatch):
    """One chunking path: the rows the command writes (ids, texts, metadata, vectors) are the rows an API start
    writes from the same directory, and running the command again embeds nothing."""
    make_table, corpus = served
    by_command = make_table()
    first = _generate()
    by_start = make_table()
    monkeypatch.setenv("INDEX_ON_START", "true")
    state = _start()

    def nodes(vsm):
        return sorted(row[1:] for row in _rows(vsm))  # everything but the serial id

    assert first["chunks_embedded"] == state.indexing["chunks_embedded"] == len(nodes(by_command)) > 0
    assert first["pipeline"] == state.indexing["pipeline"]
    assert nodes(by_command) == nodes(by_start)
    assert all(json.loads(meta).get("path") for _, _, meta, _ in nodes(by_command))  # every row is on the manifest's terms

    monkeypatch.setenv("PGVECTOR_TABLE", by_command.table_name)
    again = _generate()
    assert (again["chunks_embedded"], again["files_unchanged"]) == (0, first["files_seen"])
