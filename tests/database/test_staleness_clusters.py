"""Staleness clusters at start-up against a live Postgres: built from the manifest, without reading chunk text.

Start-up used to select node_id, text and metadata of every row of the data table to build the version clusters
(app/graph/staleness.cluster_versions, unchanged here). The indexer now writes each new or changed file's mtime and
member id on its manifest row (vector/0008), and start-up builds the clusters from one record per file. Needs
PG_TEST_DSN (the CI parity service, or a local instance); skipped otherwise. Each test indexes a temporary copy of
fixture v1 into a vector table of its own, built by the schema migrations and dropped at the end, with the hash
embedder.
"""

import os
import shutil
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

FIXTURE = Path(__file__).resolve().parents[2] / "bench" / "corpus" / "files"
EDITED = "configs/run_047.yaml"


@pytest.fixture
def indexed(monkeypatch, tmp_path):
    """A migrated table holding a fresh index of a fixture copy; the environment an API start reads."""
    from urllib.parse import urlparse

    from app.database import vector_store
    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager
    from app.engine.ingest import index_data_dir
    from app.settings import init_settings

    dsn = os.environ["PG_TEST_DSN"]
    corpus = tmp_path / "files"
    shutil.copytree(FIXTURE, corpus)
    for name, value in {"PG_CONNECTION_STRING": dsn, "PSYCOPG2_CONNECTION_STRING": dsn, "DB_NAME": urlparse(dsn).path.lstrip("/"),
                        "PGVECTOR_TABLE": "staleness_" + uuid.uuid4().hex[:8], "EMBEDDING_DIM": "256", "EMBEDDING_PROVIDER": "hash",
                        "MODEL_PROVIDER": "ollama", "RERANKER_MODEL": "none", "DATA_DIR": str(corpus), "INGEST_MAX_REJECTED": "0"}.items():
        monkeypatch.setenv(name, value)
    for name in ("PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES", "INDEX_ON_START", "ENABLE_STALENESS"):
        monkeypatch.delenv(name, raising=False)
    init_settings()
    vsm = VectorStoreManager(conn_string=dsn)
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)
    index_data_dir(vsm)  # the index step
    vector_store.get_vector_store_manager.cache_clear()
    yield vsm, corpus
    vector_store.get_vector_store_manager.cache_clear()
    with vsm.get_connection() as conn:
        drop_vector_table(conn, vsm.vector_table)
    vsm.close()


def clusters_from_every_chunk(vsm) -> dict:
    """The clusters as start-up built them before: every row's node_id, text and metadata, in the table's order."""
    from app.graph.staleness import cluster_versions
    from app.retrieval.types import Chunk

    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'SELECT node_id, text, metadata_ FROM "{vsm.schema_name}"."{vsm.data_table}";')
            rows = cur.fetchall()
    chunks = []
    for node_id, text, meta in rows:
        meta = meta or {}
        chunks.append(Chunk(chunk_id=node_id, path=meta.get("path") or meta.get("file_path") or node_id, text=text or "",
                            start_byte=int(meta.get("start_byte") or 0), end_byte=int(meta.get("end_byte") or 0),
                            mtime=float(meta.get("mtime") or 0.0), content_hash=str(meta.get("content_hash") or "")))
    return cluster_versions(chunks)


def _by_path(clusters: dict) -> dict:
    """Order-free view: {key: sorted (path, mtime, content_hash, current)} (a rewritten file's rows move in the table)."""
    return {key: sorted((m.path, m.mtime, m.content_hash, m.current) for m in c.members) for key, c in clusters.items()}


def _start():
    """build_retrieval_state as the API's lifespan runs it, with a manager of its own; returns the state."""
    from app.database import vector_store
    from app.engine.bootstrap import build_retrieval_state

    vector_store.get_vector_store_manager.cache_clear()
    try:
        return build_retrieval_state()
    finally:
        vector_store.get_vector_store_manager().close()
        vector_store.get_vector_store_manager.cache_clear()


def test_the_clusters_start_up_builds_from_the_manifest_are_the_ones_every_chunk_gave(indexed):
    from app.engine.bootstrap import build_clusters

    vsm, _ = indexed
    expected = clusters_from_every_chunk(vsm)
    assert expected and any(len(c.members) >= 2 for c in expected.values())
    assert build_clusters(vsm) == expected  # every field: keys, members and their order, member ids, mtimes, current


@pytest.mark.parametrize("index_on_start", ["true", "false"])
def test_a_restart_over_an_unchanged_directory_reads_no_chunk_text_and_serves_the_same_clusters(indexed, monkeypatch, index_on_start):
    """D32: a restart (with or without its own index run) sends no statement that selects the text column, and the
    retriever it builds holds the clusters the chunks give."""
    from app.engine.ingest_scale import count_text_reads

    vsm, _ = indexed
    expected = clusters_from_every_chunk(vsm)
    monkeypatch.setenv("INDEX_ON_START", index_on_start)
    with count_text_reads(vsm.data_table) as reads:
        state = _start()
    assert reads == {"statements": 0, "rows": 0}
    assert state.retriever._clusters == expected and state.retriever.staleness_enabled
    if index_on_start == "true":
        assert state.indexing["chunks_embedded"] == 0  # nothing changed on disk


def test_an_edit_rewrites_only_that_files_record_and_the_clusters_follow(indexed):
    """The records are computed in the indexing run, for new and changed files only."""
    from app.engine.bootstrap import build_clusters
    from app.engine.ingest import index_data_dir

    vsm, corpus = indexed

    def manifest() -> dict:
        with vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT path, mtime, member_node_id, indexed_at FROM "{vsm.schema_name}"."{vsm.manifest_table}";')
                return {row[0]: row[1:] for row in cur.fetchall()}

    before = manifest()
    target = corpus / EDITED
    target.write_text(target.read_text() + "note: edited\n")
    os.utime(target, (before[EDITED][0] + 3600, before[EDITED][0] + 3600))
    report = index_data_dir(vsm)

    after = manifest()
    assert report["files_indexed"] == 1
    assert {p for p in after if after[p] != before[p]} == {EDITED}
    assert after[EDITED][0] == before[EDITED][0] + 3600
    rows = {r["path"]: r for r in vsm.cluster_records()}
    assert after[EDITED][1] == rows[EDITED]["member_node_id"]
    assert _by_path(build_clusters(vsm)) == _by_path(clusters_from_every_chunk(vsm))


def test_a_manifest_from_before_the_staleness_columns_gets_them_from_its_rows(indexed):
    """The upgrade: a manifest written before vector/0008 has neither column. The deploy step adds them and fills
    them from the rows' metadata, so the first start after it already has its clusters."""
    from app.database.migrate import VECTOR_LEDGER, migrate_all, pending_all
    from app.engine.bootstrap import build_clusters

    vsm, _ = indexed
    expected, records = build_clusters(vsm), vsm.cluster_records()
    manifest, table = f'"{vsm.schema_name}"."{vsm.manifest_table}"', vsm.vector_table
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:  # back to the shape vector/0007 left
            cur.execute(f"ALTER TABLE {manifest} DROP COLUMN mtime, DROP COLUMN member_node_id;")
            cur.execute(f"DELETE FROM {VECTOR_LEDGER} WHERE sequence = 'vector' AND vector_table = %s AND version = 8;", (table.key,))
    with vsm.get_connection() as conn:
        assert pending_all(conn, table) == [f"vector/0008_manifest_staleness ({table.key})"]
        assert migrate_all(conn, table) == {"database": [], "vector": [8]}

    assert vsm.cluster_records() == records
    assert build_clusters(vsm) == expected
