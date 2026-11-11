"""The dense leg of the served SQL must equal the in-memory dense search on every gold question.

With the shared chunker the API and the benchmark index identical chunks with
identical hash embeddings, so the dense list (cosine top-k) has no reason to
differ. The lexical leg differs by design (Postgres ts_rank_cd with ties broken
by row id, against in-memory BM25) and is not compared here; the end-to-end gap
is reported by `make parity` instead.

Needs a live Postgres with pgvector: set PG_TEST_DSN (the CI parity service, or a
local instance). Skipped otherwise. The fixture is indexed through the served
indexer (app/engine/ingest.py::index_directory) into a temporary table, with the
same text and HNSW indexes the API's startup creates, and the dense CTE of
app/database/sql/hybrid_tsrank.sql is run with the retriever's depth (RETRIEVE_K)
and the per-statement settings the served path sends with it, in the same execute
(VectorStoreManager.settings_sql).

Two plans are checked. On the 71-row fixture the planner answers the dense CTE
with a sequential scan, so the list is exact. With the HNSW index forced
(enable_seqscan off) the served statement must still return RETRIEVE_K rows, the
same rows: an HNSW scan returns at most hnsw.ef_search rows, and pgvector's
default (40) is below RETRIEVE_K, so without the served settings every gold
question gets 40 rows.
"""

import json
import os
import shutil
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"
QUESTIONS = ROOT / "bench" / "gold" / "questions.jsonl"
SCORE_TOL = 1e-5  # the vector literal carries 8 significant digits; pgvector stores float4
# Chunks whose scores are equal in exact arithmetic (near-duplicate run logs, or zero overlap
# with the query) come out of the float32 in-memory path a few 1e-8 apart, so "exact ties" are
# scores within this much of each other. Measured: q003 run_040/run_050 logs are both
# 0.101494586043 in float64; q017's two chunks are both 0.0 in float64, -6e-9 and -2e-9 in float32.
TIE_TOL = 1e-6
# Forcing the index: every sequential scan costs more than any index path. The same penalty
# pushes each plan past jit_above_cost, and compiling it would dominate the run, so JIT is off.
FORCE_INDEX_OPTIONS = "-c enable_seqscan=off -c jit=off"


def dense_sql(vsm) -> str:
    """The served statement's dense CTE, verbatim, returning each row's benchmark chunk id."""
    sql = vsm.hybrid_sql("ts_rank_cd")
    head, sep, _ = sql.partition(",\nlexical AS (")
    assert sep, "hybrid_tsrank.sql no longer has a lexical CTE after the dense CTE"
    table = f'"{vsm.schema_name}"."{vsm.data_table}"'
    return (
        head
        + f"\nSELECT t.metadata_->>'chunk_id', d.dense_score, d.dense_rank FROM dense d"
        + f" JOIN {table} t ON t.node_id = d.node_id ORDER BY d.dense_rank;"
    )


def served_dense(manager, sql, text, qvec, k) -> list[tuple[str, float]]:
    """The dense CTE sent as hybrid_search sends its statement (settings first, one execute): [(chunk_id, score)] by rank."""
    settings, settings_params = manager.settings_sql(k)
    params = {"qvec": manager.vector_literal(qvec), "query": manager.lexical_query(text), "k": k, "rrf_k": 60.0}
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(settings + sql, {**params, **settings_params})
            return [(cid, float(score)) for cid, score, _ in cur.fetchall()]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    dsn = os.environ["PG_TEST_DSN"]
    saved = {k: os.environ.get(k) for k in ("PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "PGVECTOR_TABLE", "EMBEDDING_DIM")}
    os.environ.update(
        PG_CONNECTION_STRING=dsn,
        PSYCOPG2_CONNECTION_STRING=dsn,
        PGVECTOR_TABLE="densetest_" + uuid.uuid4().hex[:8],
        EMBEDDING_DIM="256",
    )
    from app.database.vector_store_manager import VectorStoreManager
    from app.engine.ingest import index_directory
    from app.retrieval.embedders import HashEmbedder

    corpus = tmp_path_factory.mktemp("dense") / "files"
    shutil.copytree(FIXTURE, corpus)
    vsm = VectorStoreManager(conn_string=dsn)
    emb = HashEmbedder(256)
    report = index_directory(vsm, str(corpus), embed_fn=lambda texts: emb.encode(texts).tolist(), embedder="hash-256")
    assert report["files_rejected"] == 0
    vsm.ensure_text_index()  # what app/engine/bootstrap.py does after indexing
    vsm.ensure_hnsw()
    yield vsm, corpus
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{vsm.schema_name}"."{vsm.data_table}";')
            cur.execute(f'DROP TABLE IF EXISTS "{vsm.schema_name}"."{vsm.manifest_table}";')
        conn.commit()
    vsm.close()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def memory(served):
    """The in-memory dense search over the same corpus, and the gold questions."""
    from app.eval.index_loader import build_index

    _, corpus = served
    k = int(os.getenv("RETRIEVE_K", "50"))
    index = build_index(corpus, embedder_name="hash", retrieve_k=k, enable_rerank=False, enable_router=False)
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    return index, questions, k


@pytest.fixture(scope="module")
def forced(served):
    """A second manager on the same table whose connections cannot plan a sequential scan."""
    from psycopg2.extensions import make_dsn

    from app.database.vector_store_manager import VectorStoreManager

    manager = VectorStoreManager(conn_string=make_dsn(os.environ["PG_TEST_DSN"], options=FORCE_INDEX_OPTIONS))
    yield manager
    manager.close()


def in_memory(index, text, k):
    """(ranked [(chunk_id, score)], {chunk_id: score} over every chunk) from the in-memory dense search."""
    from app.retrieval.embedders import cosine_scores

    mem_list = [(c.chunk_id, s) for c, s in index.search_dense(text, k=k)]
    all_scores = cosine_scores(index.embedder.encode([text])[0], index.doc_mat)
    return mem_list, {c.chunk_id: float(s) for c, s in zip(index.chunks, all_scores)}


def compare_dense(sql_rows, mem_list, mem_all, k):
    """Problems (empty if equal). sql_rows/mem_list: [(chunk_id, score)] in rank order; mem_all: every chunk's score.

    Scores must agree position by position. Chunks with equal scores (within TIE_TOL)
    form a tie group whose order is arbitrary on both sides (SQL breaks it by row id),
    so a group is compared as a set; a group cut by the depth k must be a subset of
    the full tie group.
    """
    problems = []
    if len(sql_rows) != len(mem_list):
        problems.append(f"length sql={len(sql_rows)} memory={len(mem_list)} (k={k})")
    for i, ((sid, ss), (mid, ms)) in enumerate(zip(sql_rows, mem_list)):
        if abs(ss - ms) > SCORE_TOL:
            problems.append(f"rank {i + 1}: sql {sid} {ss:.6f} vs memory {mid} {ms:.6f}")
            break
    for sid, ss in sql_rows:
        if sid not in mem_all:
            problems.append(f"sql chunk {sid} not in the in-memory index")
        elif abs(mem_all[sid] - ss) > SCORE_TOL:
            problems.append(f"sql chunk {sid} score {ss:.6f}, in memory {mem_all[sid]:.6f}")
    n = min(len(sql_rows), len(mem_list))
    sql_ids, mem_ids = [c for c, _ in sql_rows[:n]], [c for c, _ in mem_list[:n]]
    i = 0
    while i < n:
        group = {cid for cid, s in mem_all.items() if abs(s - mem_list[i][1]) <= TIE_TOL}
        j = i
        while j < n and mem_ids[j] in group:
            j += 1
        if j < n or len(group) == j - i:  # whole tie group inside the top k
            if set(sql_ids[i:j]) != set(mem_ids[i:j]):
                problems.append(f"ranks {i + 1}-{j}: sql {sorted(sql_ids[i:j])} vs memory {sorted(mem_ids[i:j])}")
        elif not set(sql_ids[i:j]) <= group:  # group cut by k: any members may be kept
            problems.append(f"ranks {i + 1}-{j} (tie cut by k): sql {sorted(set(sql_ids[i:j]) - group)} outside the tie group")
        i = j
    return problems


def check_served_statement(manager, text, qvec, sql_rows, k):
    """hybrid_search (the call the served retriever makes) carries the dense list's ranks and scores.

    A row whose chunk is not in the dense list must come back without a dense rank, so a
    dense list cut short inside hybrid_search shows up here even when this module's own
    dense query is complete.
    """
    by_id = {cid: (s, r + 1) for r, (cid, s) in enumerate(sql_rows)}
    for row in manager.hybrid_search(text, qvec, k=k, rrf_k=60):
        cid = row["metadata"].get("chunk_id")
        expected = by_id.get(cid)
        assert row["dense_rank"] == (expected[1] if expected else None), (text, cid, row["dense_rank"], expected)
        if expected:
            assert abs(expected[0] - row["dense_score"]) < 1e-9


def test_dense_leg_matches_in_memory_on_every_gold_question(served, memory):
    from app.engine.embeddings import HashEmbedding

    vsm, _ = served
    index, questions, k = memory
    query_embed = HashEmbedding(256).get_query_embedding  # the served retriever's embed_fn with the hash model
    sql = dense_sql(vsm)

    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            vsm.apply_search_settings(cur, k)
            cur.execute("EXPLAIN " + sql, {"qvec": vsm.vector_literal(query_embed("x")), "query": "x", "k": k, "rrf_k": 60.0})
            plan = "\n".join(r[0] for r in cur.fetchall())
    uses_hnsw = "hnsw" in plan.lower()

    failures, short = [], []
    for q in questions:
        text = q["question"]
        qvec = query_embed(text)
        sql_rows = served_dense(vsm, sql, text, qvec, k)
        mem_list, mem_all = in_memory(index, text, k)
        if len(sql_rows) < len(mem_list):
            short.append((q["id"], len(sql_rows)))
        problems = compare_dense(sql_rows, mem_list, mem_all, k)
        if problems:
            failures.append(f"{q['id']}: " + "; ".join(problems[:3]))
        # The full served statement carries the same dense ranks and scores on the rows it returns.
        check_served_statement(vsm, text, qvec, sql_rows, k)

    summary = (
        f"{len(failures)}/{len(questions)} gold questions differ on the dense leg (k={k}, {len(index.chunks)} chunks). "
        f"Plan uses HNSW: {uses_hnsw}; served settings {vsm.search_settings(k)}; "
        f"{len(short)} questions returned fewer SQL rows than in memory"
        + (f" (fewest {min(n for _, n in short)}; an HNSW scan returns at most ef_search rows)" if short else "")
        + ".\n"
    )
    assert not failures, summary + "\n".join(failures[:10])


def test_forced_hnsw_scan_keeps_the_full_depth_and_the_exact_rows(served, memory, forced):
    """enable_seqscan off: the served statement reads the HNSW index and still returns RETRIEVE_K exact rows."""
    from app.engine.embeddings import HashEmbedding

    vsm, _ = served
    index, questions, k = memory
    query_embed = HashEmbedding(256).get_query_embedding
    sql = dense_sql(forced)

    # The whole served statement (not only its dense CTE: a CTE shared by both lists can hide
    # the index) is planned on the HNSW index, with the settings hybrid_search applies.
    first = questions[0]["question"]
    plan = forced.explain_hybrid(first, query_embed(first), k=k)
    index_scan = f"Index Scan using {forced.table_name}_embedding_hnsw"
    assert index_scan in plan, plan
    # Without a query vector (the router's lexical-only routes) the same plan runs the lexical list alone.
    lexical_only = forced.hybrid_search(first, None, k=k)
    assert lexical_only and all(r["dense_rank"] is None and r["bm25_rank"] is not None for r in lexical_only)

    failures, short = [], []
    for q in questions:
        text = q["question"]
        qvec = query_embed(text)
        sql_rows = served_dense(forced, sql, text, qvec, k)
        if len(sql_rows) != k:
            short.append((q["id"], len(sql_rows)))
        mem_list, mem_all = in_memory(index, text, k)
        problems = compare_dense(sql_rows, mem_list, mem_all, k)
        if problems:
            failures.append(f"{q['id']}: " + "; ".join(problems[:3]))
        check_served_statement(forced, text, qvec, sql_rows, k)

    assert not short, (
        f"{len(short)}/{len(questions)} gold questions got fewer than k={k} dense rows from the HNSW scan "
        f"(fewest {min(n for _, n in short)}); served settings {forced.search_settings(k)}"
    )
    assert not failures, f"{len(failures)}/{len(questions)} gold questions differ from the exact dense list:\n" + "\n".join(failures[:10])


def test_search_settings_stay_inside_the_statement_transaction(served):
    """SET LOCAL semantics: the pooled connection is back to the server default after hybrid_search."""
    from app.engine.embeddings import HashEmbedding

    vsm, _ = served
    settings = vsm.search_settings(50)
    assert settings["hnsw.ef_search"] >= 50
    vsm.hybrid_search("learning rate", HashEmbedding(256).get_query_embedding("learning rate"), k=50)
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_setting('hnsw.ef_search', true);")
            value = cur.fetchone()[0]
    assert value in (None, "", "40"), value  # unset placeholder or pgvector's default, never the statement's value


# ---------------------------------------------------------------- the path list (M17, B14)


def served_paths(manager, text, k) -> list[tuple[str, float]]:
    """The served statement's path list for `text` (its CTEs before the fusion): [(chunk_id, path score)] by path rank."""
    sql = manager.hybrid_sql("ts_rank_cd")
    head, sep, _ = sql.partition(",\nfused AS (")
    assert sep, "hybrid_tsrank.sql no longer has a fused CTE after the paths CTE"
    table = f'"{manager.schema_name}"."{manager.data_table}"'
    statement = head + f"\nSELECT t.metadata_->>'chunk_id', p.path_score FROM paths p JOIN {table} t ON t.node_id = p.node_id ORDER BY p.path_rank;"
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, manager.hybrid_params(text, None, k, 60, path_query=text))
            return [(cid, float(score)) for cid, score in cur.fetchall()]


def score_groups(rows) -> list[tuple[float, set]]:
    """[(score, the chunk ids with that score)], best score first."""
    return [(score, {cid for cid, s in rows if s == score}) for score in sorted({s for _, s in rows}, reverse=True)]


def test_path_list_matches_the_in_memory_path_search_on_every_lexical_path_question(served, memory):
    """On every gold question the router sends down the lexical_path route, the served path list gives each
    chunk the score InMemoryHybridIndex.search_path gives it. Equal scores are ordered by node_id in SQL
    and by corpus order in memory, so each group of equal scores is compared as a set."""
    from app.retrieval.router import QueryRouter, RouteType

    vsm, _ = served
    index, questions, k = memory
    router = QueryRouter()
    routed = [q for q in questions if router.route(q["question"]).route == RouteType.LEXICAL_PATH]
    assert routed and {q["category"] for q in routed} == {"exact_path"}, "the fixture's lexical_path questions are its exact_path ones"
    failures = []
    for q in routed:
        sql_rows = served_paths(vsm, q["question"], k)
        mem_rows = [(c.chunk_id, float(s)) for c, s in index.search_path(q["question"], k=k)]
        assert len(mem_rows) < k, "a group cut by k would need a subset comparison"
        if [s for _, s in sql_rows] != [s for _, s in mem_rows] or score_groups(sql_rows) != score_groups(mem_rows):
            failures.append(f"{q['id']}: sql {score_groups(sql_rows)} vs memory {score_groups(mem_rows)}")
        # The full served statement carries the same path ranks on the rows it returns.
        rank = {cid: i for i, (cid, _) in enumerate(sql_rows, start=1)}
        for row in vsm.hybrid_search(q["question"], None, k=k, path_query=q["question"]):
            assert row["path_rank"] == rank.get(row["metadata"]["chunk_id"]), (q["id"], row["metadata"]["chunk_id"])
    assert not failures, f"{len(failures)}/{len(routed)} lexical_path questions differ:\n" + "\n".join(failures)
