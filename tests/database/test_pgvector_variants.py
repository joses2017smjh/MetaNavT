"""halfvec and binary + rescore against float32 on one live table (C04; PG_TEST_DSN, skipped otherwise).

VectorStoreManager.ensure_halfvec and ensure_binary add a half-precision and a binary copy of the
embedding column with an HNSW index each; app/eval/pgvector_hnsw.variants measures the three on the
same table after them, with plans checked like the float32 sweep's.
"""

import json
import os
import random
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

DIM = 64
ROWS = 1200


@pytest.fixture(scope="module")
def table():
    dsn = os.environ["PG_TEST_DSN"]
    saved = {k: os.environ.get(k) for k in ("PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "PGVECTOR_TABLE", "EMBEDDING_DIM")}
    os.environ.update(PG_CONNECTION_STRING=dsn, PSYCOPG2_CONNECTION_STRING=dsn, PGVECTOR_TABLE="varianttest_" + uuid.uuid4().hex[:8], EMBEDDING_DIM=str(DIM))
    from psycopg2.extras import execute_values

    from app.database.vector_store_manager import VectorStoreManager
    from app.retrieval.embedders import HashEmbedder

    vsm = VectorStoreManager(conn_string=dsn)
    vsm.get_vector_store()
    rng = random.Random(11)
    words = [f"w{i}" for i in range(300)]
    docs = [" ".join(rng.choice(words) for _ in range(rng.randint(5, 30))) for _ in range(ROWS)]
    emb = HashEmbedder(DIM)
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, f'INSERT INTO "{vsm.schema_name}"."{vsm.data_table}" (text, metadata_, node_id, embedding) VALUES %s',
                           [(d, json.dumps({"path": f"g/{i}.txt"}), f"n{i:05d}", vsm.vector_literal(v)) for i, (d, v) in enumerate(zip(docs, emb.encode(docs)))],
                           template="(%s, %s::json, %s, %s::vector)")
    assert vsm.ensure_hnsw()
    queries = emb.encode([" ".join(rng.choice(words) for _ in range(6)) for _ in range(10)])
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


def test_ensure_halfvec_builds_an_index_that_a_halfvec_query_uses(table):
    from app.eval import pgvector_hnsw as hnsw

    vsm, queries = table
    assert vsm.ensure_halfvec()
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s AND indexname = %s;",
                        (vsm.schema_name, vsm.data_table, f"{vsm.table_name}_embedding_half_hnsw"))
            ddl = cur.fetchone()
            cur.execute(f'SELECT count(*) FROM "{vsm.schema_name}"."{vsm.data_table}" WHERE embedding_half IS NULL;')
            unfilled = cur.fetchone()[0]
    assert ddl and "halfvec_cosine_ops" in ddl[0] and unfilled == 0
    session = hnsw.Session(vsm)
    try:
        session.configure({"enable_seqscan": "off", "hnsw.ef_search": 100})
        plan = session.explain(hnsw.halfvec_sql(session.table), {"qvec": hnsw.literal(queries[0]), "k": 5})
    finally:
        session.close()
    assert hnsw.check_plan(plan, expect_index=f"{vsm.table_name}_embedding_half_hnsw", table=vsm.data_table)
    assert vsm.ensure_halfvec()  # a second call changes nothing and still succeeds


def test_variants_measure_float32_halfvec_and_binary_on_one_table(table):
    from app.eval import pgvector_hnsw as hnsw
    from app.eval.stats import resample_indices

    vsm, queries = table
    idx = resample_indices(len(queries), n_boot=100, seed=0)
    session = hnsw.Session(vsm)
    try:
        session.configure({"enable_indexscan": "off", "enable_bitmapscan": "off"})
        exact, _ = session.run(hnsw.exact_sql(session.table), queries)
    finally:
        session.close()
    out = hnsw.variants(vsm, queries, exact, idx, ef_values=(100, 1000), candidates=(50, 400))
    rows = {r["config"]: r for r in out["rows"]}
    names = out["indexes"]
    assert out["exact_unchanged"] is True
    assert rows["float32 ef_search=100"]["plan_node"] == f"Index Scan using {names['float32']}"
    assert rows["halfvec ef_search=1000"]["plan_node"] == f"Index Scan using {names['halfvec']}"
    assert rows["binary+rescore candidates=400"]["plan_node"] == f"Index Scan using {names['binary']}"
    assert rows["binary+rescore candidates=400"]["ef_search"] == 400  # the candidate list is not cut by ef_search
    assert rows["binary+rescore candidates=400"]["recall@50"]["mean"] >= rows["binary+rescore candidates=50"]["recall@50"]["mean"]
    assert rows["halfvec ef_search=1000"]["recall@50"]["mean"] >= 0.9
    assert set(out["build_seconds"]) == {"ensure_halfvec", "ensure_binary", "vacuum_analyze", "reindex_float32", "reindex_halfvec", "reindex_binary"}
    before, after = out["sizes_before"], out["sizes_after"]
    assert after["avg_column_bytes"]["embedding_half"] < after["avg_column_bytes"]["embedding"]
    assert after["indexes"][names["halfvec"]]["bytes"] < after["indexes"][names["float32"]]["bytes"]
    assert after["table_bytes_with_toast"] > 0 and before["rows"] == after["rows"] == ROWS
