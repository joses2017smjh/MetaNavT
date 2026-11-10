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
app/database/sql/hybrid_tsrank.sql is run with the retriever's depth (RETRIEVE_K).

On the 71-row fixture the planner answers the dense CTE with a sequential scan, so
the list is exact. When the HNSW index is used instead (measured by forcing it with
enable_seqscan=off), every gold question gets 40 rows, not 50: pgvector's default
hnsw.ef_search is 40, below RETRIEVE_K. The failure message reports the plan and
ef_search so a larger table that switches to the index shows up here.
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


def test_dense_leg_matches_in_memory_on_every_gold_question(served):
    from app.engine.embeddings import HashEmbedding
    from app.eval.index_loader import build_index
    from app.retrieval.embedders import cosine_scores

    vsm, corpus = served
    k = int(os.getenv("RETRIEVE_K", "50"))
    index = build_index(corpus, embedder_name="hash", retrieve_k=k, enable_rerank=False, enable_router=False)
    query_embed = HashEmbedding(256).get_query_embedding  # the served retriever's embed_fn with the hash model
    sql = dense_sql(vsm)
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]

    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("EXPLAIN " + sql, {"qvec": vsm.vector_literal(query_embed("x")), "query": "x", "k": k, "rrf_k": 60.0})
            plan = "\n".join(r[0] for r in cur.fetchall())
            cur.execute("SHOW hnsw.ef_search;")
            ef_search = int(cur.fetchone()[0])
    uses_hnsw = "hnsw" in plan.lower()

    failures, short = [], []
    for q in questions:
        text = q["question"]
        qvec = query_embed(text)
        params = {"qvec": vsm.vector_literal(qvec), "query": vsm.lexical_query(text), "k": k, "rrf_k": 60.0}
        with vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                sql_rows = [(cid, float(score)) for cid, score, _ in cur.fetchall()]
        mem_list = [(c.chunk_id, s) for c, s in index.search_dense(text, k=k)]
        all_scores = cosine_scores(index.embedder.encode([text])[0], index.doc_mat)
        mem_all = {c.chunk_id: float(s) for c, s in zip(index.chunks, all_scores)}
        if len(sql_rows) < len(mem_list):
            short.append((q["id"], len(sql_rows)))
        problems = compare_dense(sql_rows, mem_list, mem_all, k)
        if problems:
            failures.append(f"{q['id']}: " + "; ".join(problems[:3]))

        # The full served statement carries the same dense ranks and scores on the rows it returns.
        by_id = {cid: (s, r + 1) for r, (cid, s) in enumerate(sql_rows)}
        for row in vsm.hybrid_search(text, qvec, k=k, rrf_k=60):
            if row["dense_rank"] is not None:
                cid = row["metadata"].get("chunk_id")
                assert cid in by_id and by_id[cid][1] == row["dense_rank"], (q["id"], cid)
                assert abs(by_id[cid][0] - row["dense_score"]) < 1e-9

    summary = (
        f"{len(failures)}/{len(questions)} gold questions differ on the dense leg (k={k}, {len(index.chunks)} chunks). "
        f"Plan uses HNSW: {uses_hnsw}; hnsw.ef_search={ef_search}; "
        f"{len(short)} questions returned fewer SQL rows than in memory"
        + (f" (fewest {min(n for _, n in short)}; an HNSW scan returns at most ef_search rows)" if short else "")
        + ".\n"
    )
    assert not failures, summary + "\n".join(failures[:10])
