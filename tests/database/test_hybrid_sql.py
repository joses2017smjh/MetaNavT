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


# ---------------------------------------------------------------- a tie at the dense list's cut

TIE_DIM = 256  # the module table's width (EMBEDDING_DIM above)


def unit_vector(*weights: tuple[int, float]) -> list[float]:
    v = [0.0] * TIE_DIM
    for i, w in weights:
        v[i] = w
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


QUERY_VECTOR = unit_vector((0, 1.0))
TIE_ROWS = {  # cosine with QUERY_VECTOR: 1.0, 0.894, 0.707 (twice, the same vector), 0.0
    "near-1": unit_vector((0, 1.0)),
    "near-2": unit_vector((0, 1.0), (1, 0.5)),
    "tie-a": unit_vector((0, 1.0), (2, 1.0)),
    "tie-b": unit_vector((0, 1.0), (2, 1.0)),
    "far": unit_vector((3, 1.0)),
}


@pytest.fixture(params=["tie-b", "tie-a"], ids=["larger-id-stored-first", "smaller-id-stored-first"])
def tie_table(request, vsm):
    """A table where the 3rd and 4th nearest rows are the same vector; `param` is stored first of the two."""
    from llama_index.core.schema import TextNode
    from psycopg2.extensions import make_dsn

    from app.database.vector_store_manager import VectorStoreManager

    dsn = os.environ["PG_TEST_DSN"]
    saved = os.environ["PGVECTOR_TABLE"]
    os.environ["PGVECTOR_TABLE"] = "tietest_" + uuid.uuid4().hex[:8]
    try:
        manager = VectorStoreManager(conn_string=dsn)
        forced = VectorStoreManager(conn_string=make_dsn(dsn, options="-c enable_seqscan=off -c jit=off"))
    finally:
        os.environ["PGVECTOR_TABLE"] = saved
    first = request.param
    order = ["near-1", "near-2", first, "tie-a" if first == "tie-b" else "tie-b", "far"]
    manager.get_vector_store().add(
        [TextNode(id_=nid, text=f"document {nid}", embedding=TIE_ROWS[nid], metadata={"path": f"{nid}.txt"}) for nid in order]
    )
    manager.ensure_text_index()
    manager.ensure_hnsw()
    yield manager, forced
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{manager.schema_name}"."{manager.data_table}";')
    forced.close()
    manager.close()


def test_a_tie_at_the_dense_cut_keeps_the_smaller_node_id(tie_table):
    """k=3 cuts between two rows at the same distance: node_id decides, whatever the storage order or the plan."""
    manager, forced = tie_table
    for label, m in (("planner's choice", manager), ("HNSW forced", forced)):
        rows = m.hybrid_search("qqqq", QUERY_VECTOR, k=3)  # no lexical match: the fused list is the dense list
        dense = [r["node_id"] for r in sorted(rows, key=lambda r: r["dense_rank"])]
        assert dense == ["near-1", "near-2", "tie-a"], (label, dense)
        assert [r["dense_rank"] for r in rows] == [1, 2, 3] and all(r["bm25_rank"] is None for r in rows), label
    plan = forced.explain_hybrid("qqqq", QUERY_VECTOR, k=3)
    assert f"Index Scan using {forced.table_name}_embedding_hnsw" in plan, plan  # the tie-break keeps the index usable


# ---------------------------------------------------------------- the lexical list's query text in Postgres


@pytest.mark.parametrize(
    "query, tsquery, document",
    [
        # An accented word stays one term. Document text is lowercase: how Postgres lowercases
        # non-ASCII letters depends on the database's locale, which differs between installs.
        ("Café crème brûlée", "'café' | 'crème' | 'brûlée'", "un café crème, une brûlée"),
        # So does a word whose vowel signs are combining marks (Devanagari).
        ("हिन्दी भाषा", "'हिन्दी' | 'भाषा'", "हिन्दी भाषा का इतिहास"),
        # websearch_to_tsquery's operator word in the query is not passed on as a term.
        ("cats or dogs", "'cat' | 'dog'", "two dogs"),
        ("which ablation is worse, fusion_off_resnet or fusion_on_resnet",  # gold question q095
         "'ablat' | 'wors' | 'fusion' <2> 'resnet' | 'fusion' <2> 'resnet'", "the ablation fusion_off_resnet"),
    ],
)
def test_lexical_query_as_postgres_reads_it(vsm, query, tsquery, document):
    manager, _ = vsm
    text = manager.lexical_query(query)
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT websearch_to_tsquery('english', %s)::text, to_tsvector('english', %s) @@ websearch_to_tsquery('english', %s);",
                (text, document, text),
            )
            got, matches = cur.fetchone()
    assert got == tsquery, (query, text)
    assert matches, (query, document)


# ---------------------------------------------------------------- the lexical backend is configured, never detected


def test_paradedb_runs_only_when_configured_and_installed(vsm, monkeypatch):
    from app.database.vector_store_manager import VectorStoreManager

    manager, _ = vsm
    assert manager.hybrid_backend == "ts_rank_cd"  # LEXICAL_BACKEND unset
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_search';")
            installed = cur.fetchone() is not None
    monkeypatch.setenv("LEXICAL_BACKEND", "paradedb")
    configured = VectorStoreManager(conn_string=os.environ["PG_TEST_DSN"])
    try:
        if installed:
            assert configured.hybrid_backend == "paradedb"
        else:  # the pgvector image CI and the local runs use: a start-up error, not a quiet ts_rank_cd
            with pytest.raises(RuntimeError, match="pg_search"):
                configured.hybrid_backend
    finally:
        configured.close()
