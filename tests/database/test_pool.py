"""The connection pool under concurrent requests (C52), against a live Postgres.

/api/retrieve runs the retriever in worker threads (AnyIO allows 40 at once), and each hybrid statement takes a
pooled connection. The pool was a SimpleConnectionPool, which psycopg2 documents as not for use across threads,
fixed in code at 1..10 connections; any psycopg2 pool raises PoolError ("connection pool exhausted") when more
threads ask than it holds. Needs PG_TEST_DSN; skipped otherwise. The table is built by the schema migrations and
dropped at the end; the embedder is the hash adapter.
"""

import os
import threading
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

DOCS = {
    "cfg47": "learning_rate: 3e-4\nencoder: dinov2\nrun_id: 47",
    "cfg46": "learning_rate: 1e-4\nencoder: resnet50\nrun_id: 46",
    "log47": "run 47 epoch 12 val_rmse 0.055 learning rate schedule cosine",
    "draft": "The paper draft describes the fusion model and the current learning rate.",
}


@pytest.fixture
def table(monkeypatch):
    dsn = os.environ["PG_TEST_DSN"]
    name = "pooltest_" + uuid.uuid4().hex[:8]
    for key, value in {"PG_CONNECTION_STRING": dsn, "PSYCOPG2_CONNECTION_STRING": dsn, "PGVECTOR_TABLE": name,
                       "EMBEDDING_DIM": "256"}.items():
        monkeypatch.setenv(key, value)
    for key in ("PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "PG_POOL_MIN", "PG_POOL_MAX", "PG_POOL_WAIT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    from llama_index.core.schema import TextNode

    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager
    from app.retrieval.embedders import HashEmbedder

    builder = VectorStoreManager(conn_string=dsn)
    with builder.get_connection() as conn:
        migrate_all(conn, builder.vector_table)
    emb = HashEmbedder(256)
    builder.get_vector_store().add([
        TextNode(id_=nid, text=text, embedding=emb.encode([text])[0].tolist(), metadata={"path": f"{nid}.txt"})
        for nid, text in DOCS.items()
    ])
    yield dsn, emb
    with builder.get_connection() as conn:
        drop_vector_table(conn, builder.vector_table)
    builder.close()


def test_many_threads_run_hybrid_search_on_a_pool_sized_from_the_environment(table, monkeypatch):
    """24 threads, five hybrid statements each, on a pool of three connections: every call returns the rows one
    call alone returns, none fails, and no more than three connections are ever out at once (the others wait)."""
    from app.database.vector_store_manager import VectorStoreManager

    dsn, emb = table
    monkeypatch.setenv("PG_POOL_MIN", "1")
    monkeypatch.setenv("PG_POOL_MAX", "3")
    vsm = VectorStoreManager(conn_string=dsn)
    try:
        query = "current learning rate for run 47"
        vector = emb.encode([query])[0].tolist()
        reference = vsm.hybrid_search(query, vector, k=10)
        assert len(reference) == len(DOCS)
        out, most, lock = [0], [0], threading.Lock()
        real_get, real_put = vsm.pool.getconn, vsm.pool.putconn

        def getconn(*args, **kwargs):
            conn = real_get(*args, **kwargs)
            with lock:
                out[0] += 1
                most[0] = max(most[0], out[0])
            return conn

        def putconn(*args, **kwargs):
            with lock:
                out[0] -= 1
            return real_put(*args, **kwargs)

        vsm.pool.getconn, vsm.pool.putconn = getconn, putconn
        errors, results = [], []
        start = threading.Barrier(24)

        def worker():
            start.wait()
            for _ in range(5):
                try:
                    results.append(list(vsm.hybrid_search(query, vector, k=10)))
                except Exception as exc:  # noqa: BLE001 - collected and asserted below
                    errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker) for _ in range(24)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        assert errors == []
        assert len(results) == 24 * 5 and all(rows == list(reference) for rows in results)
        assert (vsm.pool.minconn, vsm.pool.maxconn) == (1, 3)  # from the environment
        assert 1 <= most[0] <= 3 and out[0] == 0
    finally:
        vsm.close()


def _count_checkouts(pool):
    """Wrap the pool's getconn/putconn; returns [out now, most out at once]."""
    counts, lock = [0, 0], threading.Lock()
    real_get, real_put = pool.getconn, pool.putconn

    def getconn(*args, **kwargs):
        conn = real_get(*args, **kwargs)
        with lock:
            counts[0] += 1
            counts[1] = max(counts[1], counts[0])
        return conn

    def putconn(*args, **kwargs):
        with lock:
            counts[0] -= 1
        return real_put(*args, **kwargs)

    pool.getconn, pool.putconn = getconn, putconn
    return counts


def test_more_callers_than_connections_wait_for_one_instead_of_failing(table, monkeypatch):
    """16 threads released together, each holding a connection for 0.1 s, on a pool of three. Unsynchronised, the
    old pool let every thread that found no idle connection open a new one, all past its size check before any was
    counted, so 16 connections were out of a pool of 10; a synchronised pool without waiting raises PoolError
    ("connection pool exhausted") for every caller beyond its size. Now the callers wait their turn, and never more
    than PG_POOL_MAX connections are out at once."""
    from app.database.vector_store_manager import VectorStoreManager

    dsn, _ = table
    monkeypatch.setenv("PG_POOL_MAX", "3")
    vsm = VectorStoreManager(conn_string=dsn)
    try:
        counts = _count_checkouts(vsm.pool)
        errors, done = [], []
        start = threading.Barrier(16)

        def worker():
            start.wait()
            try:
                with vsm.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_sleep(0.1), count(*) FROM pg_stat_activity WHERE application_name = '';")
                        done.append(cur.fetchone()[1])
            except Exception as exc:  # noqa: BLE001 - collected and asserted below
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        assert errors == [] and len(done) == 16
        assert counts[1] <= vsm.pool.maxconn and counts[0] == 0  # the pool's own size held
        assert vsm.pool.maxconn == 3  # from the environment
    finally:
        vsm.close()


def test_a_caller_that_waits_longer_than_the_pool_allows_gets_a_fixed_error(table, monkeypatch):
    """PG_POOL_WAIT_SECONDS bounds the wait; the error names the settings, not the database."""
    from app.database.db_base_manager import PoolTimeout
    from app.database.vector_store_manager import VectorStoreManager

    dsn, _ = table
    monkeypatch.setenv("PG_POOL_MAX", "1")
    monkeypatch.setenv("PG_POOL_WAIT_SECONDS", "0.3")
    vsm = VectorStoreManager(conn_string=dsn)
    try:
        failure = []
        with vsm.get_connection():  # the only connection, held
            waiter = threading.Thread(target=lambda: failure.append(_error_of(vsm)))
            waiter.start()
            waiter.join(30)
        assert len(failure) == 1 and isinstance(failure[0], PoolTimeout)
        assert str(failure[0]) == "no database connection became free within 0.3 s (PG_POOL_MAX=1, PG_POOL_WAIT_SECONDS)"
        with vsm.get_connection() as conn:  # released: the next caller gets it
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                assert cur.fetchone() == (1,)
    finally:
        vsm.close()


def _error_of(vsm):
    try:
        with vsm.get_connection():
            return None
    except Exception as exc:  # noqa: BLE001 - returned to the test
        return exc
