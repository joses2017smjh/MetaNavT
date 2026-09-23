"""SQL reciprocal rank fusion must equal app/retrieval/fuse.py on identical input lists.

Needs a live Postgres with pgvector: set PG_TEST_DSN (e.g. the CI service or
`postgresql://postgres:password@127.0.0.1:55432/metanavit`). Skipped otherwise.
The hash embedder is used, so no model downloads.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

DOCS = {
    "cfg47": "learning_rate: 3e-4\nencoder: dinov2\nrun_id: 47",
    "cfg46": "learning_rate: 1e-4\nencoder: resnet50\nrun_id: 46",
    "log47": "run 47 epoch 12 val_rmse 0.055 learning rate schedule cosine",
    "draft": "The paper draft describes the fusion model and the current learning rate.",
    "readme": "Repository layout: configs, logs, checkpoints, paper.",
    "slurm": "#SBATCH --gres=gpu:1\npython train.py --config configs/run_047.yaml",
}


@pytest.fixture(scope="module")
def vsm():
    dsn = os.environ["PG_TEST_DSN"]
    table = "hybridtest_" + uuid.uuid4().hex[:8]
    os.environ["PG_CONNECTION_STRING"] = dsn
    os.environ["PSYCOPG2_CONNECTION_STRING"] = dsn
    os.environ["PGVECTOR_TABLE"] = table
    os.environ["EMBEDDING_DIM"] = "256"
    from llama_index.core.schema import TextNode

    from app.database.vector_store_manager import VectorStoreManager
    from app.retrieval.embedders import HashEmbedder

    manager = VectorStoreManager(conn_string=dsn)
    store = manager.get_vector_store()
    emb = HashEmbedder(256)
    nodes = [
        TextNode(id_=nid, text=text, embedding=emb.encode([text])[0].tolist(), metadata={"path": f"{nid}.txt"})
        for nid, text in DOCS.items()
    ]
    store.add(nodes)
    manager.ensure_text_index()
    yield manager, emb
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{manager.schema_name}"."{manager.data_table}";')
        conn.commit()


def test_sql_rrf_matches_python_rrf(vsm):
    from app.retrieval.fuse import rrf_score_map

    manager, emb = vsm
    query = "current learning rate for run 47"
    rows = manager.hybrid_search(query, emb.encode([query])[0].tolist(), k=50, rrf_k=60)
    assert rows, "no rows fused"
    assert manager.last_bm25_backend in {"ts_rank_cd", "paradedb"}

    dense_list = [r["node_id"] for r in sorted((r for r in rows if r["dense_rank"]), key=lambda r: r["dense_rank"])]
    bm25_list = [r["node_id"] for r in sorted((r for r in rows if r["bm25_rank"]), key=lambda r: r["bm25_rank"])]
    expected = rrf_score_map([dense_list, bm25_list], k=60)

    assert {r["node_id"] for r in rows} == set(expected)
    for r in rows:
        assert abs(r["rrf_score"] - expected[r["node_id"]]) < 1e-9, r["node_id"]
    python_order = sorted(expected, key=lambda nid: (-expected[nid], nid))
    assert [r["node_id"] for r in rows] == python_order
    assert all(r["metadata"].get("path") for r in rows)  # metadata_ comes back with the row


def test_lexical_only_when_no_embedding(vsm):
    manager, _ = vsm
    rows = manager.hybrid_search("learning_rate dinov2", None, k=10)
    assert rows and all(r["dense_rank"] is None and r["bm25_rank"] is not None for r in rows)
    assert rows[0]["node_id"] == "cfg47"
    assert manager.lexical_query("Current learning-rate for run_047!") == "current or learning or rate or for or run_047"  # stopwords are dropped by Postgres


def test_explain_mentions_both_lists(vsm):
    manager, emb = vsm
    plan = manager.explain_hybrid("learning rate", emb.encode(["learning rate"])[0].tolist(), k=10)
    assert "Planning Time" in plan and "Execution Time" in plan
