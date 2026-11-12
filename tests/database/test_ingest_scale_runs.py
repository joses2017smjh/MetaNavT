"""make ingest-scale against a live Postgres: what each row counts, and the client-side count of chunk text reads.

Needs PG_TEST_DSN (the CI parity service, or a local instance); skipped otherwise. Each measured size uses a
vector table of its own, dropped at the end; the corpus copies live in a temporary directory.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

SET_BY_RECORD = ("PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "PGVECTOR_TABLE", "EMBEDDING_PROVIDER", "EMBEDDING_DIM",
                 "RERANKER_MODEL", "RETRIEVAL_MODE", "DATA_DIR", "INGEST_MAX_REJECTED", "INDEX_ON_START", "ENABLE_STALENESS",
                 "PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES")


@pytest.fixture
def restored_env(monkeypatch):
    """record() sets the environment a start-up reads; monkeypatch puts every variable back afterwards."""
    for name in SET_BY_RECORD:
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.delenv(name)
    yield


def test_each_row_counts_files_rows_and_what_every_step_embedded(restored_env):
    from app.engine.ingest_scale import FIXTURE, record

    n_fixture = sum(1 for p in FIXTURE.rglob("*") if p.is_file())
    rows = record(os.environ["PG_TEST_DSN"], "test", sizes=(1, 2), repeats=1)

    assert [row["size"] for row in rows] == ["1x", "2x"]
    for row in rows:
        assert row["label"] == "test" and row["git_sha"] and row["command"].startswith("python -m")
        assert row["files_seen"] == row["corpus"]["files"] == n_fixture * row["copies"] and row["files_rejected"] == 0
        assert row["index_first"]["files_indexed"] == row["files_seen"]
        assert row["index_first"]["chunks_embedded"] == row["rows"]  # one row per chunk
        assert row["index_unchanged"]["files_unchanged"] == row["files_seen"] and row["index_unchanged"]["chunks_embedded"] == 0
        assert row["index_unchanged"]["text_rows_read"] == 0  # the diff reads hashes and row counts, never text
        assert row["clusters"]["clusters"] > 0
        for start in ("startup_index_on_start", "startup_no_index"):
            assert row[start]["chunks_embedded"] == 0 and len(row[start]["samples"]) == 1
    one, two = rows
    assert two["rows"] == 2 * one["rows"] + 1  # the CSV copy gains one row
    assert two["clusters"]["members"] == two["files_seen"]  # at 2x every file shares its key with its copy


def test_the_text_read_count_sees_a_select_of_the_text_column_and_nothing_else(restored_env):
    from llama_index.core.schema import TextNode

    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager
    from app.engine.ingest_scale import count_text_reads

    dsn = os.environ["PG_TEST_DSN"]
    os.environ.update({"PG_CONNECTION_STRING": dsn, "PGVECTOR_TABLE": "textreads_" + uuid.uuid4().hex[:8], "EMBEDDING_DIM": "3"})
    vsm = VectorStoreManager(conn_string=dsn)
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)
    try:
        vsm.get_vector_store().add([TextNode(id_=f"n{i}", text=f"chunk {i}", embedding=[1.0, 0.0, float(i)], metadata={"path": f"{i}.txt"})
                                    for i in range(3)])
        table = f'"{vsm.schema_name}"."{vsm.data_table}"'
        with count_text_reads(vsm.data_table) as reads:
            reader = VectorStoreManager(conn_string=dsn)  # opened inside the count, like a start-up's manager
            try:
                with reader.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"SELECT node_id, metadata_->>'path' FROM {table};")  # no text
                        cur.execute(f"SELECT count(*) FROM {table} WHERE metadata_->>'path' IS NOT NULL;")
                        assert reads == {"statements": 0, "rows": 0}
                        cur.execute(f"SELECT node_id, text, metadata_ FROM {table};")
                reader.rows_per_path()
            finally:
                reader.close()
        assert reads == {"statements": 1, "rows": 3}
    finally:
        with vsm.get_connection() as conn:
            drop_vector_table(conn, vsm.vector_table)
        vsm.close()
