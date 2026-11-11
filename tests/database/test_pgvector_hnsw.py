"""HNSW against the exact scan on a live table (app/eval/pgvector_hnsw.py; PG_TEST_DSN, skipped otherwise).

A scratch table of generated rows with hash vectors, created through PGVectorStore (the served schema)
with the HNSW index start-up creates. The sweep must read the plans it claims (a sequential scan for
the exact rows, the HNSW index for the others), an index scan with iterative scan off must stop at
ef_search rows, and a wide ef_search must find the exact neighbours of a table this small.
"""

import json
import os
import random
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

DIM = 64
ROWS = 1500


@pytest.fixture(scope="module")
def table():
    dsn = os.environ["PG_TEST_DSN"]
    saved = {k: os.environ.get(k) for k in ("PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "PGVECTOR_TABLE", "EMBEDDING_DIM")}
    os.environ.update(PG_CONNECTION_STRING=dsn, PSYCOPG2_CONNECTION_STRING=dsn, PGVECTOR_TABLE="hnswtest_" + uuid.uuid4().hex[:8], EMBEDDING_DIM=str(DIM))
    from psycopg2.extras import execute_values

    from app.database.vector_store_manager import VectorStoreManager
    from app.retrieval.embedders import HashEmbedder

    vsm = VectorStoreManager(conn_string=dsn)
    vsm.get_vector_store()
    rng = random.Random(7)
    words = [f"w{i}" for i in range(400)]
    docs = [" ".join(rng.choice(words) for _ in range(rng.randint(5, 30))) for _ in range(ROWS)]
    emb = HashEmbedder(DIM)
    vectors = emb.encode(docs)
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, f'INSERT INTO "{vsm.schema_name}"."{vsm.data_table}" (text, metadata_, node_id, embedding) VALUES %s',
                           [(d, json.dumps({"path": f"g/{i}.txt"}), f"n{i:05d}", vsm.vector_literal(v)) for i, (d, v) in enumerate(zip(docs, vectors))],
                           template="(%s, %s::json, %s, %s::vector)")
    assert vsm.ensure_hnsw() and vsm.ensure_text_index()
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'ANALYZE "{vsm.schema_name}"."{vsm.data_table}";')
    queries = emb.encode([" ".join(rng.choice(words) for _ in range(6)) for _ in range(12)])
    yield vsm, queries
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{vsm.schema_name}"."{vsm.data_table}";')
    vsm.close()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_sweep_reads_the_plans_it_names_and_ef_search_bounds_the_scan(table):
    from app.eval import pgvector_hnsw as hnsw
    from app.eval.stats import resample_indices

    vsm, queries = table
    index = f"{vsm.table_name}_embedding_hnsw"
    out = hnsw.hnsw_sweep(vsm, queries, resample_indices(len(queries), n_boot=100, seed=0), hnsw_index=index, ef_values=(10, 400))
    rows = {r["config"]: r for r in out["rows"]}
    assert "Seq Scan" in rows["exact"]["plan_node"]
    assert rows["hnsw ef_search=10"]["plan_node"] == rows["hnsw ef_search=400"]["plan_node"] == f"Index Scan using {index}"
    assert rows["hnsw ef_search=10"]["rows_returned"]["mean"] <= 10  # iterative scan off: at most ef_search rows
    assert rows["hnsw ef_search=10"]["recall@50"]["mean"] <= 0.2
    assert rows["hnsw ef_search=400"]["rows_returned"]["min"] == hnsw.K
    assert rows["hnsw ef_search=400"]["recall@50"]["mean"] >= 0.9
    served = rows["served dense CTE"]
    assert served["forced"] is False and served["rows_returned"]["min"] == hnsw.K
    assert served["settings"] == vsm.search_settings(hnsw.K)
    assert all(len(e) == hnsw.K for e in out["exact"])


def test_exact_rows_are_the_true_nearest_neighbours(table):
    import numpy as np

    from app.eval import pgvector_hnsw as hnsw

    vsm, queries = table
    session = hnsw.Session(vsm)
    try:
        session.configure({"enable_indexscan": "off", "enable_bitmapscan": "off"})
        rows, _ = session.run(hnsw.exact_sql(session.table), queries[:3], k=5)
        with vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT node_id, embedding::text FROM "{vsm.schema_name}"."{vsm.data_table}";')
                stored = {n: np.array(json.loads(e), dtype=np.float32) for n, e in cur.fetchall()}
    finally:
        session.close()
    for q, got in zip(queries[:3], rows):
        q = np.asarray(q, dtype=np.float64)
        dist = {n: 1 - float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q))) for n, v in stored.items() if np.linalg.norm(v)}
        best = sorted(dist.items(), key=lambda kv: (kv[1], kv[0]))[:5]
        assert [d for _, d in got] == pytest.approx([d for _, d in best], abs=1e-5)
