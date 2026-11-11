"""Start-up against a live Postgres: what it writes to the shared table, and two replicas creating the database.

Needs PG_TEST_DSN (the CI parity job's service, or a local instance); skipped otherwise.
Each test uses a vector table (or a database) of its own and removes it at the end.
"""

import os
import threading
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")


def _plain():
    import psycopg2

    conn = psycopg2.connect(os.environ["PG_TEST_DSN"])
    conn.autocommit = True
    return conn


@pytest.fixture
def migrated(monkeypatch):
    """A migrated vector table holding a few rows, and a manager on it (the deploy step has run)."""
    from llama_index.core.schema import TextNode

    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager
    from app.retrieval.embedders import HashEmbedder

    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PGVECTOR_TABLE", "probetest_" + uuid.uuid4().hex[:8])
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    monkeypatch.delenv("LEXICAL_BACKEND", raising=False)
    vsm = VectorStoreManager(conn_string=dsn)
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)
    emb = HashEmbedder(256)
    texts = {"a": "learning rate 3e-4", "b": "run 47 log", "c": "paper draft"}
    vsm.get_vector_store().add([TextNode(id_=k, text=t, embedding=emb.encode([t])[0].tolist(), metadata={"path": f"{k}.txt"})
                                for k, t in texts.items()])
    yield vsm
    with vsm.get_connection() as conn:
        drop_vector_table(conn, vsm.vector_table)
    vsm.close()


def test_get_vector_store_never_shows_a_probe_row_to_a_concurrent_reader(migrated):
    """Every replica start used to add a zero-vector test_node_* row to the shared table and delete it
    again: a reader in between saw it. A row trigger logs every write; a reader polls meanwhile."""
    from app.database.vector_store_manager import VectorStoreManager

    vsm = migrated
    table, log = f'"{vsm.schema_name}"."{vsm.data_table}"', f"probe_log_{uuid.uuid4().hex[:8]}"
    admin = _plain()
    with admin.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table};")
        rows_before = cur.fetchone()[0]
        cur.execute(f"CREATE TABLE {log} (op text, node_id text);")
        cur.execute(f"CREATE FUNCTION {log}_fn() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
                    f"INSERT INTO {log} VALUES (TG_OP, COALESCE(NEW.node_id, OLD.node_id)); RETURN NULL; END $$;")
        cur.execute(f"CREATE TRIGGER {log}_trg AFTER INSERT OR UPDATE OR DELETE ON {table} FOR EACH ROW EXECUTE FUNCTION {log}_fn();")
    seen, stop = [], threading.Event()

    def read():
        reader = _plain()
        try:
            with reader.cursor() as cur:
                while not stop.is_set():
                    cur.execute(f"SELECT count(*) FROM {table} WHERE node_id LIKE 'test_node_%%';")
                    seen.append(cur.fetchone()[0])
        finally:
            reader.close()

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
        for _ in range(5):  # five replica starts
            replica = VectorStoreManager(conn_string=os.environ["PG_TEST_DSN"])
            try:
                replica.get_vector_store()
            finally:
                replica.close()
        time.sleep(0.2)
    finally:
        stop.set()
        thread.join(timeout=10)
        with admin.cursor() as cur:
            cur.execute(f"SELECT op, node_id FROM {log};")
            writes = cur.fetchall()
            cur.execute(f"SELECT count(*) FROM {table};")
            rows_after = cur.fetchone()[0]
            cur.execute(f"DROP TRIGGER IF EXISTS {log}_trg ON {table};")
            cur.execute(f"DROP FUNCTION IF EXISTS {log}_fn();")
            cur.execute(f"DROP TABLE IF EXISTS {log};")
        admin.close()
    assert seen and max(seen) == 0, f"a reader saw a probe row: {max(seen)}"
    assert writes == [], writes  # nothing was inserted, updated or deleted, not even briefly
    assert rows_after == rows_before


def test_the_probe_refuses_a_missing_table_and_keeps_no_store(monkeypatch):
    """Before the deploy step there is no table: the probe says so (it no longer creates one by writing a
    row), and the manager keeps no store, so the next start-up attempt probes again."""
    from app.database.vector_store_manager import VectorStoreManager

    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PGVECTOR_TABLE", "probemissing_" + uuid.uuid4().hex[:8])
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    vsm = VectorStoreManager(conn_string=dsn)
    try:
        with pytest.raises(RuntimeError, match="does not exist: apply the schema migrations"):
            vsm.get_vector_store()
        assert vsm.vector_store is None
        with vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s) IS NULL;", (f'"{vsm.schema_name}"."{vsm.data_table}"',))
                assert cur.fetchone()[0]  # still no table: the probe wrote nothing
    finally:
        vsm.close()
