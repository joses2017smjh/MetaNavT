"""Served-path scale run (owner's upgrade 5): BEIR FiQA through the served indexer, Postgres and POST /api/retrieve/.

    make bench-served-scale PYTHON=<python with torch> PGVECTOR_SIF=<PostgreSQL 16 + pgvector image> \
        SERVED_SCALE_SCRATCH=<node-local directory> [SERVED_SCALE_STATE=<shared directory>] [BEIR_DATA_DIR=bench/external]
    -> bench/results/served_scale_fiqa.json and bench/results/served_scale_fiqa_trec/ (A05)
       bench/results/pgvector_hnsw.json (C03 and C04: app/eval/pgvector_hnsw.py, on the same table)

For a GPU node: bge-small (the served embedder, LlamaIndex HuggingFaceEmbedding) and bge-reranker-v2-m3 on
CUDA from the HuggingFace cache, Apptainer with a PostgreSQL 16 + pgvector image (app/eval/service.py), and
FiQA under the data directory (app/eval/beir.py downloads it once and pins its sha256).

The run is a sequence of stages. Each writes <state>/<stage>.json, and --resume skips a stage whose file
exists, so a job that dies late keeps what it measured. The corpus and the database live under <scratch>
(node-local disk: 57,638 small files and a Postgres data directory on NFS would measure NFS); after the
index stage the data directory is copied to <state>, and a resumed run on a fresh node restores it.

  files          FiQA as one file per document, <doc_id>.txt, the in-memory rows' text (beir_api.write_corpus)
  index          the deploy step first (python -m app.database.migrate: the vector table, the HNSW index on the
                 embedding and the GIN index on the stored text_tsv, all on the empty table), timed; then the
                 build: app/engine/ingest.index_directory into it with the served embedder and the arguments the
                 index step passes (index_data_dir), the indexes maintained per insert: seconds (and the
                 embedder's share), rows, rejections by reason; ANALYZE; the indexes read from pg_indexes; sizes
                 on disk; index_directory again over the unchanged directory; then the deploy order's index step
                 itself, `python -m app.engine.ingest` in its own process with the API's settings, over the same
                 files: it must find every file unchanged under the build's pipeline (its wall seconds are what
                 a redeploy with no change pays before the API starts)
  startup        the API (uvicorn main:app, INDEX_ON_START unset, both models), seconds from launch to /health
                 ok, and that /health against the build (the manifest the start read, the rows); then what every
                 start reads from the database, timed here in the runner: the migration ledgers, the manifest
                 summary and the version clusters, which bootstrap.build_clusters makes from the manifest, with
                 the rows of chunk text those reads return counted (a start read every chunk back before M20)
  probe          a few hybrid statements and lexical-only statements, timed in this process before the replay
  replay_rerank  every test query through POST /api/retrieve/ with k = 50 and the served defaults (reranker on)
  load           app/eval/http_load on that API: concurrency 1, 4 and 16, k = 8, failures counted; the note says
                 whether the API had served earlier stages of this process or was started for this one
  replay_hybrid  the same replay with RERANKER_MODEL=none: the like-for-like pair of the in-memory hybrid row
  explain        EXPLAIN (ANALYZE, BUFFERS) of the served statement for three queries: which scan each leg uses
  inmemory       app/eval/beir.py's hybrid@bge-small and hybrid+bge-rerank@bge-small/fp16-512/top20 rows,
                 document embeddings from the BEIR cache, on the same queries, scored by the same function; each
                 again with the served embedder's query prompt (read from the served model) before the query on
                 its dense leg, the query form the served path uses; the two sides' query vectors compared
  hnsw           pgvector's HNSW index against the exact scan on the same table and queries, at several
                 hnsw.ef_search values, and the served dense CTE (app/eval/pgvector_hnsw.py)
  variants       last, because it rewrites the table: halfvec and binary + rescore against float32 on it

The results file pairs each served row with the in-memory row of the same query form (served minus
in-memory, paired bootstrap over queries), per route as well, compares it with the bare-query row too,
checks the bare-query rows against the committed in-memory FiQA rows (bench/results/beir_fiqa.json: same
code, corpus and document embeddings), and lists every difference between the two sides. Every number comes
from this run; what the run could not do (a failed request, a rejected document) is counted, not dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = "bench/results/served_scale_fiqa.json"
HNSW_FILE = "bench/results/pgvector_hnsw.json"
TREC_DIR = "bench/results/served_scale_fiqa_trec"
DATASET = "fiqa"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
SERVED_K = 50  # MAX_K of POST /api/retrieve/
LOAD_LEVELS = (1, 4, 16)
# served row -> (the API's extra environment, the in-memory row of app/eval/beir.py it is compared with). The run adds
# that row again with the served query prompt (SAME_FORM_SUFFIX) and pairs the served row with it (paired_rows).
PASSES = {
    "api:hybrid+rerank": ({}, "hybrid+bge-rerank@bge-small/fp16-512/top20"),
    "api:hybrid": ({"RERANKER_MODEL": "none"}, "hybrid@bge-small"),
}
SAME_FORM_SUFFIX = "+served-query-prompt"
REPLAY_STAGES = {"replay_rerank": "api:hybrid+rerank", "replay_hybrid": "api:hybrid"}  # stage -> the served row it replays
QUERY_CHECK_N = 16  # queries whose served and in-memory query vectors are compared in the job
BEIR_FIQA_FILE = "bench/results/beir_fiqa.json"  # make bench-beir-fiqa: the committed in-memory FiQA rows
STAGES = ("files", "index", "startup", "probe", "replay_rerank", "load", "replay_hybrid", "explain", "inmemory", "hnsw", "variants")
SERVED_STAGES = ("files", "index", "startup", "probe", "replay_rerank", "load", "replay_hybrid", "explain", "inmemory")  # served_scale_fiqa.json
HNSW_STAGES = ("files", "index", "hnsw", "variants")  # pgvector_hnsw.json
SERVER_SETTINGS = ("shared_buffers", "maintenance_work_mem", "work_mem", "effective_cache_size", "max_parallel_maintenance_workers",
                   "max_parallel_workers_per_gather", "jit", "synchronous_commit", "random_page_cost")
SOURCES = ("app/eval/served_scale.py", "app/eval/beir_api.py", "app/eval/service.py", "app/eval/beir.py", "app/eval/http_load.py", "app/engine/ingest.py", "app/database/migrate.py",
           "app/eval/ingest.py", "app/engine/bootstrap.py", "app/engine/retriever.py", "app/database/vector_store_manager.py",
           "app/database/sql/hybrid_tsrank.sql", "app/api/routers/retrieve.py", "main.py", "app/eval/pgvector_hnsw.py")
_VECTOR_LITERAL = re.compile(r"'\[[-0-9.,eE+ ]*\]'::(vector|halfvec)")
_SCAN = re.compile(r"(Parallel Seq Scan|Seq Scan|Index Scan|Index Only Scan|Bitmap Heap Scan|Bitmap Index Scan)(?: Backward)?(?: using (\S+))? on (\S+)(?: (\w+))?")


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- data


def verified_dataset(data_dir: Path, dataset: str = DATASET) -> tuple[Path, dict]:
    """The extracted dataset and its identity. An existing extraction is only read (its zip's sha256 checked
    against the pin); a missing one is downloaded by app/eval/beir.download_dataset."""
    from app.eval import beir

    root = data_dir / dataset
    zip_path = data_dir / f"{dataset}.zip"
    pinned = beir.DATASETS[dataset]["sha256"]
    if not (root / "corpus.jsonl").exists():
        beir.download_dataset(data_dir, dataset)
    actual = beir.sha256_of(zip_path) if zip_path.exists() else None
    if actual is not None and actual != pinned:
        raise RuntimeError(f"{dataset}.zip sha256 {actual} != pinned {pinned}")
    files = {name: beir.sha256_of(root / name) for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")}
    return root, {"dataset": f"BEIR/{dataset}", "split": "test", "url": beir.DATASETS[dataset]["url"], "zip_sha256": actual, "expected_zip_sha256": pinned, "files_sha256": files}


def select(corpus: dict, queries: dict, qrels: dict, *, max_queries: int | None = None, max_docs: int | None = None) -> tuple[dict, dict, dict, str]:
    """The full split, or a slice for a rate check: the first queries by id and, with max_docs, every document
    they judge plus the first other documents by id. A slice is never reported as the benchmark."""
    queries = dict(sorted(queries.items()))
    if max_queries:
        queries = dict(list(queries.items())[:max_queries])
    qrels = {qid: qrels[qid] for qid in queries}
    note = "full split" if not max_queries else f"first {len(queries)} queries by id"
    if max_docs:
        judged = sorted({d for docs in qrels.values() for d in docs if d in corpus})
        judged_set = set(judged)
        rest = [d for d in sorted(corpus) if d not in judged_set][: max(0, max_docs - len(judged))]
        corpus = {d: corpus[d] for d in sorted(judged + rest)}
        note += f"; slice of {len(corpus)} documents (every judged one, then the first by id)"
    return corpus, queries, qrels, note


def plan_facts(plan: str, *, table: str, hnsw_index: str, text_index: str) -> dict:
    """Which scan each leg of the served statement used, read from EXPLAIN text.

    The statement aliases the table t_1 in the dense CTE, t_2 in the lexical CTE and t in the final join
    (pgvector's index scan on the dense CTE, the GIN bitmap scan on the lexical CTE, the join by node_id).
    A Bitmap Index Scan line names the index where the other scans name the table.
    """
    scans = [{"node": m.group(1), "index": m.group(2), "relation": m.group(3), "alias": m.group(4)} for m in _SCAN.finditer(plan)]
    on_table = [s for s in scans if s["relation"] == table]
    timing = re.search(r"Execution Time: ([0-9.]+) ms", plan)
    return {
        "scans": scans,
        "dense_uses_hnsw": any(s["node"] == "Index Scan" and s["index"] == hnsw_index for s in on_table),
        "lexical_uses_gin": any(s["node"] == "Bitmap Index Scan" and s["relation"] == text_index for s in scans),  # "Bitmap Index Scan on <index>"
        "final_join_scan": next((s["node"] for s in on_table if s["alias"] == "t"), None),
        "execution_ms": float(timing.group(1)) if timing else None,
    }


def plan_block(explain: dict, index_names: dict, *, table: str) -> dict:
    """The explain stage's plans with their facts read again, using the index names the index stage found in
    pg_indexes: the facts depend only on the plan text and those names, whatever names the stage file carries."""
    names = {"table": table, "hnsw_index": index_names["hnsw"], "text_index": index_names["text_index"]}
    queries = [{"id": q["id"], "facts": plan_facts("\n".join(q["plan"]), **names), "plan": q["plan"]} for q in explain["queries"]]
    return {"queries": queries, "settings": explain.get("settings"), "names": names}


def elide_vectors(plan: str) -> str:
    """The plan with each vector literal shortened to its length (a query vector is 384 numbers)."""
    return _VECTOR_LITERAL.sub(lambda m: f"'[{m.group(0).count(',') + 1} values]'::{m.group(1)}", plan)


def same_form_name(bare: str) -> str:
    """The in-memory row `bare` with the served query prompt before the query on its dense leg."""
    return bare + SAME_FORM_SUFFIX


def paired_rows(bare: str, available) -> tuple[str, str | None]:
    """(the in-memory row a served row is paired with, the bare-query row it is also compared with, or None).

    The pair is the row with the served query form: `bare` with the served prompt when the run has that
    row, else `bare` itself (a served embedder without a prompt embeds the bare query, as `bare` does).
    """
    twin = same_form_name(bare)
    return (twin, bare) if twin in available else (bare, None)


def served_prompts(model) -> dict:
    """What the served embedder puts before a query and before a document, read from the object the API builds too.

    LlamaIndex's HuggingFaceEmbedding hands sentence-transformers one prompt per kind and encodes a query
    with prompt_name="query" (the served retriever embeds with get_query_embedding, app/engine/retriever.py).
    A model without sentence-transformers prompts (the hash embedding) puts nothing before either.
    """
    prompts = getattr(getattr(model, "_model", None), "prompts", None)
    if not isinstance(prompts, dict):
        return {"query": "", "text": "", "source": f"{type(model).__name__}: no sentence-transformers prompts"}
    return {"query": prompts.get("query") or "", "text": prompts.get("text") or "",
            "source": f"{type(model).__name__}._model.prompts (sentence-transformers), the 'query' one used by get_query_embedding"}


def query_vector_check(served, prompted, bare) -> dict:
    """How far the in-memory embedder's query vectors are from the served ones, as directions (rows unit-normalised).

    `served`: get_query_embedding of the served model; `prompted`: the in-memory embedder on prompt + query;
    `bare`: the in-memory embedder on the query alone. The first pair should be the same vectors.
    """
    import numpy as np

    def unit(m):
        m = np.asarray(m, dtype=np.float64)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    s, p, b = unit(served), unit(prompted), unit(bare)

    def compare(other):
        cos = (s * other).sum(axis=1)
        return {"max_abs_difference": float(f"{np.abs(s - other).max():.3g}"), "min_cosine": round(float(cos.min()), 8), "mean_cosine": round(float(cos.mean()), 8)}

    return {"n_queries": int(len(s)), "with_served_prompt": compare(p), "bare_query": compare(b)}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_trec_run(path: Path) -> dict[str, list[str]]:
    """query id -> document ids in rank order, from a TREC run file."""
    ranked: dict[str, list[tuple[int, str]]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            qid, _, did, rank, *_ = line.split()
            ranked.setdefault(qid, []).append((int(rank), did))
    return {qid: [did for _, did in sorted(pairs)] for qid, pairs in ranked.items()}


def committed_reproduction(mem_rows: dict[str, list[dict]], committed: dict | None, root: Path, *, corpus_sha256: str, query_ids: list[str]) -> dict:
    """This job's in-memory rows against the committed rows of the same name (BEIR_FIQA_FILE, make bench-beir-fiqa).

    Same code, corpus and document embeddings: per row, the nDCG@10 each file states, and from the
    committed TREC run how many queries got exactly the same ranking (and the same first 10). A row the
    committed file does not have (the served-prompt rows) is not compared.
    """
    if not committed:
        return {"status": "not_run", "reason": f"{BEIR_FIQA_FILE} is not in this checkout"}
    if committed.get("corpus_sha256") != corpus_sha256 or committed.get("n_queries") != len(query_ids):
        return {"status": "not_comparable", "reason": "a different corpus or query set (a slice)", "file": BEIR_FIQA_FILE}
    by = {r["config"]: r for r in committed.get("results", [])}
    out: dict[str, Any] = {"status": "compared", "file": BEIR_FIQA_FILE, "git_sha": committed.get("git_sha"), "device": (committed.get("device") or {}).get("device_name"), "rows": {}}
    for name, rows in mem_rows.items():
        trec = (by.get(name) or {}).get("trec") or {}
        if not trec.get("run"):
            continue
        run_path = Path(root) / trec["run"]
        ranked = read_trec_run(run_path)
        mine = {r["id"]: r["ranking"] for r in rows}
        mean = round(sum(r["ndcg@10"] for r in rows) / len(rows), 4)
        stated = by[name]["retrieval"]["ndcg@10"]
        out["rows"][name] = {
            "committed_ndcg@10": stated, "this_job_ndcg@10": mean, "ndcg@10_equal": mean == stated,
            "identical_rankings": sum(1 for q in query_ids if ranked.get(q, []) == mine.get(q)),
            "identical_top10": sum(1 for q in query_ids if ranked.get(q, [])[:10] == (mine.get(q) or [])[:10]),
            "n": len(query_ids), "run_file": trec["run"], "run_sha256_ok": file_sha256(run_path) == trec.get("run_sha256"),
        }
    return out


def differences(*, rows: int | None, chunker: str | None, rejected: dict, not_indexed: dict, routes: dict, served_depth: int | None, search_settings: dict,
                in_memory_settings: dict, reranker: dict, query_form: dict) -> list[str]:
    """What separates a served row from its in-memory row, in words, from the run's own records and the code's defaults."""
    import inspect

    from app.chunking.structure import fixed_size_chunks

    window = {name: p.default for name, p in inspect.signature(fixed_size_chunks).parameters.items() if name in {"size", "overlap"}}
    mem_k = in_memory_settings.get("k")
    served = query_form.get("served") or {}
    if served.get("query"):
        query_line = (f"query form: the served embedder puts {served['query']!r} before each query on the dense leg ({served.get('source')}); each served row is "
                      f"paired with the in-memory row named '<row>{SAME_FORM_SUFFIX}', which puts the same text before the query on its dense leg, and also compared "
                      "with the bare-query row of that name, which embeds the query alone; the lexical legs and the rerankers read the bare query on both sides")
    else:
        query_line = f"query form: neither side puts anything before a query ({served.get('source')})"
    return [
        f"documents vs chunks: the in-memory rows rank whole documents; the served index holds {rows} chunks ({chunker}; a .txt file takes the "
        f"fixed window, {window.get('size')} characters with {window.get('overlap')} overlap), and a served document ranks at its best chunk",
        f"depth: the API returns at most {SERVED_K} chunks, so a served ranking holds at most {SERVED_K} documents and its Recall@50 counts within those chunks; "
        f"the in-memory rows fuse {mem_k} + {mem_k} candidates into {mem_k} documents (RRF k {in_memory_settings.get('rrf_k')}), the served statement {served_depth} + {served_depth} into {served_depth} chunks",
        "lexical leg: in-memory Okapi BM25 (Lucene-style English analyzer, k1 "
        f"{in_memory_settings.get('k1')}, b {in_memory_settings.get('b')}) vs Postgres ts_rank_cd over the stored text_tsv (to_tsvector('english', text)) "
        "with the query terms OR-joined",
        f"dense leg: in-memory exact cosine over whole-document vectors (the BEIR cache) vs pgvector over chunk vectors with {search_settings} per statement, "
        "through the HNSW index when the planner picks it (the plan block)",
        query_line,
        f"router: the served path routes every query ({routes}); lexical_path skips the query embedding and the reranker, aggregation_query skips the reranker; "
        "the in-memory rows have no router",
        f"rejected files: the served scanner rejected {sum(rejected.values())} documents ({ {k: v for k, v in rejected.items() if v} }); "
        f"judged documents among them: {len(not_indexed.get('judged', []))} ({not_indexed.get('judged', [])})",
        f"reranker input: the served reranker reads chunk text, the in-memory reranker whole documents; served {reranker.get('model')} "
        f"{reranker.get('precision')}, max_length {reranker.get('max_length')}, depth {reranker.get('depth')}; in-memory "
        f"{in_memory_settings.get('reranker')} {in_memory_settings.get('precision')}, max_length {in_memory_settings.get('max_length')}, depth {in_memory_settings.get('rerank_top')}",
    ]


# ---------------------------------------------------------------- database helpers (a VectorStoreManager)


def query_rows(vsm, statement: str, params: tuple = ()) -> list[tuple]:
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            return cur.fetchall()


def table_exists(vsm) -> bool:
    return bool(query_rows(vsm, "SELECT to_regclass(%s) IS NOT NULL;", (f'"{vsm.schema_name}"."{vsm.data_table}"',))[0][0])


def applied_migrations(vsm) -> dict:
    """The migrations the deploy step left applied to this database and this vector table: version -> checksum."""
    from app.database.migrate import applied_versions, vector_applied_versions

    with vsm.get_connection() as conn:
        return {"database": applied_versions(conn), "vector": vector_applied_versions(conn, "vector", vsm.vector_table)}


def server_facts(vsm) -> dict:
    version = query_rows(vsm, "SELECT version();")[0][0]
    settings = {name: query_rows(vsm, f"SHOW {name};")[0][0] for name in SERVER_SETTINGS}
    pgvector = ".".join(str(p) for p in (vsm.pgvector_version or ())) or None
    return {"postgres": version.split(" on ")[0], "pgvector": pgvector, "settings": settings, "settings_note": "the image's defaults; nothing tuned"}


def index_ddl(vsm) -> dict[str, str]:
    return dict(query_rows(vsm, "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s ORDER BY indexname;", (vsm.schema_name, vsm.data_table)))


def table_sizes(vsm) -> dict:
    """Bytes on disk: the heap, the heap with TOAST, every index, and the average stored size of each column."""
    qualified = f'"{vsm.schema_name}"."{vsm.data_table}"'
    heap, table, indexes, total = query_rows(vsm, "SELECT pg_relation_size(%s::regclass), pg_table_size(%s::regclass), pg_indexes_size(%s::regclass), pg_total_relation_size(%s::regclass);", (qualified,) * 4)[0]
    per_index = query_rows(vsm, "SELECT i.relname, am.amname, pg_relation_size(i.oid) FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_am am ON am.oid = i.relam WHERE x.indrelid = %s::regclass ORDER BY i.relname;", (qualified,))
    columns = query_rows(vsm, "SELECT a.attname FROM pg_attribute a WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum;", (qualified,))
    names = [c[0] for c in columns]
    averages = query_rows(vsm, f"SELECT count(*), {', '.join(f'avg(pg_column_size({json.dumps(n)}))' for n in names)} FROM {qualified};")[0]
    return {
        "rows": int(averages[0]),
        "heap_bytes": int(heap), "table_bytes_with_toast": int(table), "indexes_bytes": int(indexes), "total_bytes": int(total),
        "indexes": {name: {"method": method, "bytes": int(size)} for name, method, size in per_index},
        "avg_column_bytes": {n: (round(float(v), 1) if v is not None else None) for n, v in zip(names, averages[1:])},
    }


def served_indexes(vsm) -> dict:
    """The indexes the schema migrations built that the hybrid statement reads, found in pg_indexes by what they
    index (the HNSW index on the embedding, the GIN index on the stored text_tsv), and every index's definition."""
    from app.eval.pgvector_hnsw import find_index

    ddl = index_ddl(vsm)
    return {"index_names": {"hnsw": find_index(ddl, "hnsw", "embedding"), "text_index": find_index(ddl, "gin", "text_tsv")}, "indexes_ddl": ddl}


def hnsw_parameters(ddl: str | None) -> dict:
    """m and ef_construction from an HNSW index definition (pg_indexes.indexdef)."""
    found = {key: re.search(rf"{key}='?(\d+)", ddl or "") for key in ("m", "ef_construction")}
    return {key: int(m.group(1)) if m else None for key, m in found.items()}


def query_rows_commit(vsm, statement: str) -> None:
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)


def build_index(vsm, corpus_dir: Path, model, *, embed_fn: Callable[[list], list]) -> dict:
    """index_directory over corpus_dir with the arguments app/engine/ingest.index_data_dir passes (the call of the index
    step and of a start with INDEX_ON_START; DATA_DIR is corpus_dir in this run): the embedder's label as the manifest
    records it, INGEST_MAX_REJECTED, the lock wait from the environment. Only embed_fn differs: model's batch call with a timer."""
    from app.engine.ingest import embedder_label, index_directory, max_rejected_from_env

    return index_directory(vsm, str(corpus_dir), embed_fn=embed_fn, embedder=embedder_label(model), max_rejected=max_rejected_from_env())


def index_corpus(vsm, corpus_dir: Path, model) -> dict:
    """The build (build_index) into the migrated table, timed, with the embedder's own share; ANALYZE; the indexes and the
    sizes; the unchanged re-run."""
    spent = {"seconds": 0.0, "calls": 0, "texts": 0}

    def embed_fn(texts):
        started = time.perf_counter()
        vectors = model.get_text_embedding_batch(texts)
        spent["seconds"] += time.perf_counter() - started
        spent["calls"] += 1
        spent["texts"] += len(texts)
        return vectors

    started = time.perf_counter()
    report = build_index(vsm, corpus_dir, model, embed_fn=embed_fn)
    seconds = time.perf_counter() - started
    embedding = dict(spent)
    started = time.perf_counter()
    query_rows_commit(vsm, f'ANALYZE "{vsm.schema_name}"."{vsm.data_table}";')
    analyze_seconds = time.perf_counter() - started
    built = served_indexes(vsm)
    sizes = table_sizes(vsm)
    spent.update(seconds=0.0, calls=0, texts=0)
    started = time.perf_counter()
    rerun = build_index(vsm, corpus_dir, model, embed_fn=embed_fn)
    rerun_seconds = time.perf_counter() - started
    if rerun["chunks_embedded"] or rerun["files_indexed"]:
        raise RuntimeError(f"the unchanged re-run indexed files: {rerun}")
    return {
        "report": report,
        "rows": sizes["rows"],
        "index_directory_seconds": round(seconds, 3),
        "embedding": {"seconds": round(embedding["seconds"], 3), "calls": embedding["calls"], "texts": embedding["texts"]},
        "other_seconds": round(seconds - embedding["seconds"], 3),
        "files_per_second": round(report["files_indexed"] / seconds, 2) if seconds else None,
        "chunks_per_second": round(report["chunks_embedded"] / seconds, 2) if seconds else None,
        "index_maintenance": "per insert: the schema migrations built the HNSW and GIN indexes on the empty table, so index_directory's seconds include keeping them",
        "analyze_seconds": round(analyze_seconds, 3),
        **built,
        "sizes": sizes,
        "unchanged_rerun": {"seconds": round(rerun_seconds, 3), "files_unchanged": rerun["files_unchanged"], "chunks_embedded": rerun["chunks_embedded"], "embed_calls": spent["calls"]},
    }


INDEX_STEP_FIELDS = ("files_seen", "files_indexed", "files_unchanged", "files_removed", "files_rejected", "chunks_embedded", "chunks_deleted", "n_nodes", "pipeline", "lock")


def index_step_unchanged(env, build_report: dict, *, log_path: Path) -> dict:
    """The deploy order's index step (app/eval/service.index_step: `python -m app.engine.ingest` in its own process, with the
    API's settings) over the files the build just indexed. It must accept the build as current: nothing embedded, every
    indexed file unchanged, the same rejections, rows and pipeline (chunker version and embedder label); otherwise the
    API would serve an index that step would not have written, and the run stops here."""
    from app.eval import service

    report, seconds = service.index_step(env, cwd=ROOT, log_path=log_path)
    expected = {"chunks_embedded": 0, "files_indexed": 0, "files_unchanged": build_report["files_indexed"], "files_rejected": build_report["files_rejected"],
                "n_nodes": build_report["n_nodes"], "pipeline": build_report["pipeline"]}
    found = {key: report.get(key) for key in expected}
    if found != expected:
        raise RuntimeError(f"the index step (python -m app.engine.ingest) did not accept the build: {found}, expected {expected}")
    return {"command": "python -m app.engine.ingest", "seconds": round(seconds, 3), "report": {key: report.get(key) for key in INDEX_STEP_FIELDS},
            "note": "its own process with the API's settings, after the timed build, over the unchanged files: process start, settings and embedder load, "
                    "the migration check and the diff against the manifest, as a redeploy with no change runs it before the API"}


def not_indexed(corpus: dict, qrels: dict, indexed_paths: set[str]) -> dict:
    """Documents with no file in the ingest manifest (rejected by the served scanner), and the judged ones among them."""
    from app.eval.beir_api import doc_file_name

    missing = sorted(d for d in corpus if doc_file_name(d) not in indexed_paths)
    judged = sorted({d for docs in qrels.values() for d in docs} & set(missing))
    return {"n": len(missing), "ids": missing, "judged": judged, "queries": sorted(q for q, docs in qrels.items() if set(docs) & set(judged))}


def start_up_reads(vsm) -> dict:
    """What app/engine/bootstrap.build_retrieval_state reads from the database once the deploy step and the index step
    have run, with INDEX_ON_START unset and staleness on (the defaults), each read timed: the migration ledgers
    (check_migrations), the manifest summary /health reports, and the version clusters, which build_clusters makes
    from the manifest (VectorStoreManager.cluster_records, then staleness.cluster_versions) since M20."""
    from app.engine.bootstrap import build_clusters

    out: dict[str, Any] = {}
    started = time.perf_counter()
    vsm.check_migrations()
    out["check_migrations_seconds"] = round(time.perf_counter() - started, 3)
    started = time.perf_counter()
    out["manifest"] = vsm.manifest_summary()
    out["manifest_summary_seconds"] = round(time.perf_counter() - started, 3)
    started = time.perf_counter()
    clusters = build_clusters(vsm)
    out["build_clusters_seconds"] = round(time.perf_counter() - started, 3)
    out["clusters"], out["cluster_members"] = len(clusters), sum(len(c.members) for c in clusters.values())
    return out


def text_reads(vsm, step: Callable[[Any], object]) -> dict:
    """{"statements", "rows"}: the statements selecting the chunk text of vsm's table that step(manager) sends, and the rows
    they return, counted on the client by app/engine/ingest_scale.count_text_reads. The manager is opened inside the
    count, as a new process would open its own: a pooled connection opened before it would go unseen."""
    from app.database.vector_store_manager import VectorStoreManager
    from app.engine.ingest_scale import count_text_reads

    with count_text_reads(vsm.data_table) as counted:
        fresh = VectorStoreManager(conn_string=vsm.conn_string)
        try:
            step(fresh)
        finally:
            fresh.close()
    return dict(counted)


def startup_components(vsm) -> dict:
    """What every API start reads from the database (start_up_reads), timed in the runner on its open pool, and the rows of
    chunk text those reads return, counted in one more untimed pass (text_reads)."""
    out = start_up_reads(vsm)
    counted = text_reads(vsm, start_up_reads)
    return {**out, "rows_in_table": vsm.count_nodes(), "text_statements": counted["statements"], "text_rows_read": counted["rows"],
            "note": "timed in the runner, not inside the API; before M20 a start read every chunk's text back (fetch_all_chunks) to build the clusters"}


def check_startup_health(health: dict, build: dict) -> None:
    """The API's /health against the index stage: the manifest it read at start lists every chunk the build wrote, under
    the build's one pipeline, and the table holds them. A start that served another table or index would fail here."""
    manifest = ((health.get("indexing") or {}).get("manifest_at_start")) or {}
    expected = {"files": build["report"]["files_indexed"], "chunks": build["rows"], "pipelines": [build["report"]["pipeline"]]}
    found = {key: manifest.get(key) for key in expected}
    if found != expected or health.get("n_nodes") != build["rows"]:
        raise RuntimeError(f"/health at start does not match the index stage: manifest {found}, n_nodes {health.get('n_nodes')}; expected {expected}")


def probe(vsm, queries: list[tuple[str, str]], vectors, k: int = SERVED_K) -> dict:
    """The served statement (hybrid_search) and its lexical leg alone, timed per query, before the replay."""
    rows = []
    for (qid, text), vec in zip(queries, vectors):
        started = time.perf_counter()
        hybrid = vsm.hybrid_search(text, list(map(float, vec)), k=k)
        hybrid_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        lexical = vsm.hybrid_search(text, None, k=k)
        lexical_ms = (time.perf_counter() - started) * 1000.0
        rows.append({"id": qid, "hybrid_ms": round(hybrid_ms, 2), "hybrid_rows": len(hybrid), "lexical_only_ms": round(lexical_ms, 2), "lexical_rows": len(lexical)})
    return {"queries": rows, "note": "VectorStoreManager.hybrid_search in the runner (no HTTP, no reranker); lexical-only = no query vector"}


# ---------------------------------------------------------------- the run


def migration_files() -> tuple[str, ...]:
    """The vector table's schema migrations (app/database/migrations/vector/), repository-relative."""
    return tuple(sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "app" / "database" / "migrations" / "vector").glob("*.sql")))


def git_dirty(root: Path) -> bool | None:
    """Whether a tracked file differs from the commit, the files this run writes (its results files and TREC runs) aside."""
    import subprocess

    outputs = [f":(exclude){path}" for path in (RESULTS_FILE, HNSW_FILE, TREC_DIR)]
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", *outputs], cwd=root, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return None


def machine(device: str) -> dict:
    """What a stage ran on, without a host name: the device, the GPU and its memory, the CPU model, the job's allocation.
    Every stage records it, because a resumed run can finish its stages on another node."""
    from app.eval.provenance import device_info

    info = device_info(device)
    out = {"device": info.get("device"), "device_name": info.get("device_name"), "gpu_memory_gb": None, "cpu_model": None,
           "allocation": {"cpus": os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE"), "memory_mb": os.environ.get("SLURM_MEM_PER_NODE")}}
    try:
        out["cpu_model"] = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and device != "cpu":
            out["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    return out


def stage_machines(stages: dict[str, dict]) -> dict:
    """The machine of each recorded stage, and whether they are all the same one."""
    machines = {s: b.get("machine") for s, b in stages.items()}
    return {"stage_machines": machines, "one_machine": len({json.dumps(m, sort_keys=True) for m in machines.values()}) == 1}


def hardware(device: str) -> dict:
    """The assembling process's machine (machine()) with torch, platform and package versions."""
    from app.eval.provenance import device_info

    info = {**device_info(device), **machine(device)}
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("llama-index-core", "llama-index-embeddings-huggingface", "llama-index-vector-stores-postgres", "sentence-transformers", "transformers", "torch", "psycopg2-binary", "psycopg2", "numpy", "fastapi", "uvicorn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    info["packages"] = packages
    info["storage"] = "corpus files and the Postgres data directory on node-local disk (the run's scratch directory)"
    return info


class Run:
    def __init__(self, args: argparse.Namespace):
        from app.eval.beir import load_beir

        from app.eval.provenance import command_line

        self.args = args
        self.command = command_line()  # before this run sets any environment variable of its own
        self.scratch = args.scratch.resolve()
        self.state = (args.state or self.scratch / "state").resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.corpus_dir = self.scratch / "corpus"
        self.data_root, self.identity = verified_dataset(args.data_dir.resolve())
        corpus, queries, qrels = load_beir(self.data_root, "test")
        self.full = {"n_docs": len(corpus), "n_queries": len(queries)}
        self.corpus, self.queries, self.qrels, self.selection = select(corpus, queries, qrels, max_queries=args.max_queries, max_docs=args.max_docs)
        self.hash = args.embedder == "hash"
        self.pg = None
        self.api = None
        self.api_pass: str | None = None
        self.api_served: list[str] = []  # the stages the running API served in this process, in order
        self._vsm = None
        self._embed_model = None
        self.settings = self.served_settings()

    # -- environment
    def served_settings(self) -> dict:
        s = {"EMBEDDING_PROVIDER": "huggingface", "EMBEDDING_MODEL": EMBEDDING_MODEL, "EMBEDDING_DIM": "384", "RERANKER_MODEL": RERANKER_MODEL, "RERANKER_REQUIRED": "true",
             "MODEL_PROVIDER": "ollama", "ENVIRONMENT": "prod", "LOG_FORMAT": "json", "DATA_DIR": str(self.corpus_dir), "BGE_ALLOW_DOWNLOAD": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        if self.args.device == "cuda":
            s.update(EMBEDDING_DEVICE="cuda", RETRIEVAL_REQUIRE_DEVICE="cuda", RERANKER_DEVICE="cuda")
        if self.hash:  # the login-node dry run: no model at all
            s.update(EMBEDDING_PROVIDER="hash", RERANKER_MODEL="none")
        return s

    def stage_file(self, stage: str) -> Path:
        return self.state / f"{stage}.json"

    def done(self, stage: str) -> dict | None:
        path = self.stage_file(stage)
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, stage: str, blob: dict) -> dict:
        from app.eval.provenance import git_sha

        blob = {"stage": stage, "git_sha": git_sha(ROOT), "finished": datetime.now(timezone.utc).isoformat(), "machine": machine(self.args.device), **blob}
        tmp = self.stage_file(stage).with_suffix(".json.partial")
        tmp.write_text(json.dumps(blob, indent=1) + "\n")
        tmp.replace(self.stage_file(stage))
        return blob

    # -- services
    def postgres(self):
        from app.eval.service import TaskPostgres

        if self.pg is None:
            pgroot = self.scratch / "pg"
            snapshot = self.state / "pgdata.tar"
            if not (pgroot / "pgdata").exists() and self.done("index") and snapshot.exists():
                log("restoring the database snapshot taken after the index stage")
                pgroot.mkdir(parents=True, exist_ok=True)
                with tarfile.open(snapshot) as tar:
                    tar.extractall(pgroot, filter="data")
            if not (pgroot / "pgdata").exists() and self.done("index"):
                raise RuntimeError("the index stage is recorded but its database is gone and there is no snapshot: delete <state>/index.json and later stages")
            self.pg = TaskPostgres(self.args.container, pgroot).start()
            os.environ.update({**self.settings, **self.pg.env()})
        return self.pg

    def vsm(self):
        if self._vsm is None:
            self.postgres()
            from app.database.vector_store_manager import VectorStoreManager

            self._vsm = VectorStoreManager()
        return self._vsm

    def embed_model(self):
        if self._embed_model is None:
            os.environ.update(self.settings)
            from llama_index.core.settings import Settings

            from app.settings import init_embed_model

            init_embed_model()
            self._embed_model = Settings.embed_model
        return self._embed_model

    def query_vectors(self):
        import numpy as np

        path = self.state / "query_vectors.npy"
        ids_path = self.state / "query_vectors.ids.json"
        if path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == list(self.queries):
            return np.load(path)
        model = self.embed_model()
        vectors = np.asarray([model.get_query_embedding(text) for text in self.queries.values()], dtype=np.float32)
        np.save(path, vectors)
        ids_path.write_text(json.dumps(list(self.queries)))
        return vectors

    def api_for(self, row: str) -> tuple[Any, dict, float, bool]:
        """The API serving `row`'s configuration: (server, /health, seconds to ready, started now)."""
        from app.eval.service import ApiServer, base_environment

        if self.api is not None and self.api_pass == row:
            return self.api, self.api_health, 0.0, False
        self.stop_api()
        self.postgres()
        extra = dict(PASSES[row][0])
        env = base_environment({**self.settings, **self.pg.env(), **extra})
        self.api = ApiServer(env, cwd=ROOT, log_path=self.scratch / f"api-{row.split(':')[1]}.log").start()
        health, seconds = self.api.wait_ready(timeout=self.args.api_ready_timeout)
        self.api_pass, self.api_health, self.api_served = row, health, []
        log(f"API for {row} ready in {seconds:.1f} s: n_nodes {health.get('n_nodes')}, reranker {health.get('reranker', {}).get('loaded')}")
        return self.api, health, seconds, True

    def stop_api(self) -> None:
        if self.api is not None:
            self.api.stop()
            self.api = None
            self.api_pass = None
            self.api_served = []

    def close(self) -> None:
        self.stop_api()
        if self._vsm is not None:
            self._vsm.close()
            self._vsm = None
        if self.pg is not None:
            self.pg.stop()
            self.pg = None

    # -- stages
    def stage_files(self) -> dict:
        from app.eval.beir_api import DOC_SUFFIX, doc_file_name, write_corpus

        info = write_corpus(self.corpus, self.corpus_dir)
        extra = sorted({p.name for p in self.corpus_dir.iterdir()} - {doc_file_name(d) for d in self.corpus})
        if extra:  # the scanner would index them too; this run deletes nothing, so it stops here
            raise RuntimeError(f"{len(extra)} files in {self.corpus_dir} are not documents of this run (e.g. {extra[:3]}); use a fresh --scratch")
        return {**info, "layout": f"one file per document, <doc_id>{DOC_SUFFIX}, one directory", "document_text": "title + ' ' + text, stripped (app/eval/beir.load_beir, as the in-memory rows read it)"}

    def ensure_files(self) -> None:
        """The corpus on this node (a resumed run on a fresh node writes it again; its digest must match)."""
        recorded = self.done("files")
        if recorded and not self.corpus_dir.exists():
            from app.eval.beir_api import write_corpus

            if write_corpus(self.corpus, self.corpus_dir)["listing_sha256"] != recorded["listing_sha256"]:
                raise RuntimeError("the corpus written now differs from the one the files stage recorded")

    def stage_index(self) -> dict:
        from app.engine.ingest import embedder_label
        from app.eval.service import base_environment, migrate

        model = self.embed_model()
        pg = self.postgres()
        env = base_environment({**self.settings, **pg.env()})  # what the API reads too (api_for)
        # the deploy step, as on a deployment: the vector table and its indexes are schema migrations, applied before
        # anything is indexed (no start-up runs DDL); on this fresh database they build the indexes on the empty table
        migrate_seconds = migrate(env, cwd=ROOT, log_path=self.scratch / "migrate.log")
        vsm = self.vsm()
        if vsm.count_nodes():  # an interrupted index stage leaves rows: index_directory would resume, and the timing would be partial
            raise RuntimeError("the database already holds rows; the index stage measures a build from empty: use a fresh --scratch")
        blob = {"embedder": embedder_label(model), "server": server_facts(vsm), "migrate_seconds": round(migrate_seconds, 3), "migrations": applied_migrations(vsm),
                **index_corpus(vsm, self.corpus_dir, model)}
        # then the index step the deploy order runs before the API, over the same files, before the snapshot and the restart
        blob["index_step"] = index_step_unchanged(env, blob["report"], log_path=self.scratch / "index-step.log")
        blob["not_indexed"] = not_indexed(self.corpus, self.qrels, set(vsm.ingest_manifest()))
        if self.args.snapshot:
            self.snapshot()
        return blob

    def snapshot(self) -> None:
        """Stop Postgres, copy its data directory to <state>, start it again (a resumed run restores it). The copy holds the
        server's password file (app/eval/service.TaskPostgres), so only its owner may read it."""
        self._vsm.close()
        self._vsm = None
        self.pg.stop()
        self.pg = None
        started = time.perf_counter()
        target = self.state / "pgdata.tar"
        partial = target.with_suffix(".tar.partial")
        with open(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as out, tarfile.open(fileobj=out, mode="w") as tar:
            tar.add(self.scratch / "pg" / "pgdata", arcname="pgdata")
        partial.chmod(0o600)  # a file that already existed keeps its mode through O_CREAT
        partial.replace(target)
        log(f"database snapshot {target.stat().st_size / 2**20:.0f} MiB in {time.perf_counter() - started:.0f} s")
        self.vsm()

    def stage_startup(self) -> dict:
        _, health, seconds, _ = self.api_for("api:hybrid+rerank")
        check_startup_health(health, self.done("index"))
        self.api_served.append("startup")
        components = startup_components(self.vsm())
        log(f"start-up components: {components}")
        return {"api": {"seconds_to_health_ok": round(seconds, 3), "row": "api:hybrid+rerank", "index_on_start": "unset (false)", "health": health},
                "components": components,
                "note": "the API started first, against a Postgres that had just been restarted (after the index stage or its snapshot); the components were then timed in the runner"}

    def stage_probe(self) -> dict:
        vectors = self.query_vectors()
        n = min(self.args.probe_queries, len(self.queries))
        result = probe(self.vsm(), list(self.queries.items())[:n], vectors[:n])
        for row in result["queries"]:
            log(f"probe {row}")
        return result

    def stage_replay(self, row: str) -> dict:
        from app.eval.beir_api import post_retrieve, replay

        api, health, seconds, started = self.api_for(row)
        timeout = self.args.request_timeout

        def progress(i: int, r: dict) -> None:
            if (i + 1) % 50 == 0 or i == 0 or not r["ok"]:
                log(f"{row} {i + 1}/{len(self.queries)} {r['id']} status {r['status']} route {r.get('route')} docs {r['n_docs']} e2e {r['e2e_ms']:.0f} ms {r.get('error') or ''}")

        rows = replay(lambda text, k: post_retrieve(api.url, text, k, timeout=timeout), self.queries, self.qrels, k=SERVED_K, progress=progress)
        self.api_served.append(next(stage for stage, served in REPLAY_STAGES.items() if served == row))
        return {"row": row, "api_env": PASSES[row][0], "health": health, "api_started_for_this_stage": started, "api_seconds_to_health_ok": round(seconds, 3) if started else None, "rows": rows}

    def stage_load(self) -> dict:
        from app.eval.http_load import run_load

        api, _, _, started = self.api_for("api:hybrid+rerank")
        before = "the API was started for this stage" if started else f"the API had served {', '.join(self.api_served)} in this process, without a restart"
        queries = list(self.queries.items())
        report = asyncio.run(run_load(api.url, queries, concurrency_levels=self.args.load_levels, requests_per_level=self.args.load_requests, warmup_requests=4, timeout=self.args.request_timeout,
                                      require_reranker=not self.hash, require_device=self.args.device if self.args.device == "cuda" else None))
        report["workload_note"] = f"the first {min(len(queries), self.args.load_requests)} test queries by id, in order, cycled per level; one cold probe and 4 warm-ups excluded; {before}"
        self.api_served.append("load")
        return {"report": report}

    def stage_explain(self) -> dict:
        from app.retrieval.router import QueryRouter

        self.stop_api()
        vsm = self.vsm()
        vectors = self.query_vectors()
        router = QueryRouter()
        index = list(self.queries)
        picks = [qid for qid in index if router.route(self.queries[qid]).route.value == "semantic"][: self.args.explain_queries]
        catalog = self.done("index")["index_names"]  # read from pg_indexes by the index stage
        names = {"table": vsm.data_table, "hnsw_index": catalog["hnsw"], "text_index": catalog["text_index"]}
        out = []
        for qid in picks:
            plan = vsm.explain_hybrid(self.queries[qid], list(map(float, vectors[index.index(qid)])), k=SERVED_K)
            out.append({"id": qid, "facts": plan_facts(plan, **names), "plan": elide_vectors(plan).splitlines()})
        return {"queries": out, "settings": vsm.search_settings(SERVED_K), "names": names}

    def stage_inmemory(self) -> dict:
        """app/eval/beir.py's rows that PASSES pairs with, each twice when the served embedder has a query prompt:
        as main runs them (the bare query) and with that prompt before the query on the dense leg (SAME_FORM_SUFFIX),
        the query form the served path uses. Then the two sides' query vectors are compared on the first queries."""
        from app.eval import beir
        from app.eval.beir_api import score_ranking
        from app.eval.latency import StageTimer

        self.stop_api()
        prompts = served_prompts(self.embed_model())
        served_vectors = self.query_vectors()
        engine = beir.Engine(beir.chunks_from_corpus(self.corpus), cache_dir=self.data_root.parent / "cache", device=self.args.device)
        configs = {c.name: c for c in beir.DEFAULT_CONFIGS}
        rows = {}
        cached = {}
        for _, paired_with in PASSES.values():
            base = configs[paired_with]
            if self.hash:
                base = replace(base, embedder="hash", reranker="overlap" if base.reranker else None, precision="fp32", max_length=None)
            for cfg in [base] + ([replace(base, name=same_form_name(base.name), query_instruction=prompts["query"])] if prompts["query"] else []):
                if cfg.embedder not in cached:  # read the BEIR cache's document embeddings when it holds them; never add a file to it
                    engine.embedder(cfg.embedder)
                    rev = engine.provenance[cfg.embedder].get("revision") or "norev"
                    expected = engine.cache_dir / f"{engine.corpus_sha256}.{beir._slug(cfg.embedder)}.{rev}.{len(engine.chunks)}.npy"
                    cached[cfg.embedder] = expected.exists()
                    if not expected.exists():
                        engine.cache_dir = None
                build = engine.build(cfg)
                beir.retrieve(engine, cfg, next(iter(self.queries.values())), StageTimer())  # warm-up, excluded
                timer = StageTimer()
                per_query = []
                for qid, text in self.queries.items():
                    ranking = beir.retrieve(engine, cfg, text, timer)
                    per_query.append({"id": qid, "ranking": [d for d in ranking if d != qid], **score_ranking(ranking, self.qrels[qid], qid)})
                rows[cfg.name] = {"config": cfg.name, "settings": {k: v for k, v in vars(cfg).items()}, "build_ms": build, "latency": timer.summary(),
                                  "models": {"embedder": engine.provenance.get(cfg.embedder), **({"reranker": engine.provenance.get(engine.reranker_key(cfg.reranker, cfg.precision, cfg.max_length))} if cfg.reranker else {})},
                                  "rows": per_query}
                log(f"in-memory {cfg.name}: nDCG@10 {sum(r['ndcg@10'] for r in per_query) / len(per_query):.4f}")
        if self.hash:
            check = {"status": "not_run", "reason": "hash dry run: no model on either side"}
        else:
            embedder = engine.embedder(configs[PASSES["api:hybrid"][1]].embedder)
            texts = list(self.queries.values())[:QUERY_CHECK_N]
            check = {"status": "compared", **query_vector_check(served_vectors[: len(texts)], embedder.encode([prompts["query"] + t for t in texts]), embedder.encode(texts))}
            log(f"query vectors, served vs in-memory: {check}")
        form = {name: {"dense_leg": f"{r['settings'].get('query_instruction') or ''}<query>", "lexical_leg": "<query>", "reranker": "<query>"} for name, r in rows.items()}
        return {"corpus_sha256": engine.corpus_sha256, "document_embeddings_from_cache": cached,
                "query_form": {"served": {**prompts, "dense_leg": f"{prompts['query']}<query>", "lexical_leg": "<query>", "reranker": "<query>"}, "rows": form},
                "query_vectors_check": check, "rows": rows}

    def stage_hnsw(self) -> dict:
        from app.eval import pgvector_hnsw
        from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, resample_indices

        self.stop_api()
        vectors = self.query_vectors()
        index = self.done("index")["index_names"]["hnsw"]
        out = pgvector_hnsw.hnsw_sweep(self.vsm(), vectors, resample_indices(len(vectors), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED), hnsw_index=index)
        for row in out["rows"]:
            log(f"{row['config']}: recall@50 {row['recall@50']['mean']}, p50 {row['latency'].get('p50_ms')} ms, plan {row.get('plan_node')}")
        return {**out, "dim": int(vectors.shape[1])}

    def stage_variants(self) -> dict:
        from app.eval import pgvector_hnsw
        from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, resample_indices

        self.stop_api()
        vectors = self.query_vectors()
        exact = self.done("hnsw")["exact"]
        index = self.done("index")
        name = index["index_names"]["hnsw"]
        out = pgvector_hnsw.variants(self.vsm(), vectors, exact, resample_indices(len(vectors), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED), float32_index=name,
                                     **hnsw_parameters(index["indexes_ddl"][name]))
        log(f"variants: build {out['build_seconds']}, exact unchanged {out['exact_unchanged']}")
        for row in out["rows"]:
            log(f"{row['config']}: recall@50 {row['recall@50']['mean']}, p50 {row['latency'].get('p50_ms')} ms, plan {row.get('plan_node')}")
        return out

    def run(self) -> dict:
        try:
            for stage in self.args.stages:
                if stage != "files":
                    self.ensure_files()
                if self.args.resume and self.done(stage):
                    log(f"{stage}: recorded, skipped")
                    continue
                log(f"{stage}: start")
                started = time.perf_counter()
                method = getattr(self, f"stage_{stage}", None)
                blob = method() if method else self.stage_replay(REPLAY_STAGES[stage])
                blob["stage_seconds"] = round(time.perf_counter() - started, 3)
                self.save(stage, blob)
                log(f"{stage}: done in {blob['stage_seconds']:.0f} s")
        finally:
            self.close()
        written = {}
        for name, stages, build in (("served", SERVED_STAGES, assemble), ("hnsw", HNSW_STAGES, assemble_hnsw)):
            missing = [s for s in stages if not self.done(s)]
            if missing:
                log(f"{name}: not recorded yet: {', '.join(missing)}; no results file")
            else:
                written[name] = build(self)
        return written


# ---------------------------------------------------------------- the results file


def trec_outputs(directory: Path, rankings: dict[str, dict[str, list[str]]], qrels: dict, per_query: dict[str, list[dict]]) -> dict:
    """Run files at the scored depth, one per row, and the cross-check with pytrec_eval when it is installed."""
    from app.eval.beir import _slug, write_trec_outputs

    class Named:
        def __init__(self, name):
            self.name = name

    files = {}
    for name, ranking in rankings.items():
        files[name] = write_trec_outputs(directory, Named(name), {q: r[:SERVED_K] for q, r in ranking.items()}, qrels)
    try:
        import pytrec_eval
    except ImportError:
        return {"files": files, "cross_check": {"status": "not_run", "reason": "pytrec_eval not installed"}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10,50"})
    worst = 0.0
    for name, ranking in rankings.items():
        reference = evaluator.evaluate({q: {d: float(len(r) - i) for i, d in enumerate(r[:SERVED_K])} for q, r in ranking.items() if r})
        for row in per_query[name]:
            ref = reference.get(row["id"], {"ndcg_cut_10": 0.0, "recall_10": 0.0, "recall_50": 0.0})  # an empty ranking is absent from the run
            for mine, theirs in (("ndcg@10", "ndcg_cut_10"), ("recall@10", "recall_10"), ("recall@50", "recall_50")):
                difference = abs(row[mine] - ref[theirs])
                worst = max(worst, difference)
                if difference > 1e-9:
                    raise RuntimeError(f"{name} {row['id']} {mine}: {row[mine]} vs pytrec_eval {ref[theirs]}")
    try:
        from importlib.metadata import version

        implementation = f"pytrec_eval/trec_eval (pytrec-eval-terrier {version('pytrec-eval-terrier')})"
    except Exception:
        implementation = "pytrec_eval/trec_eval"
    return {"files": files, "cross_check": {"status": "passed", "implementation": implementation, "metrics": ["ndcg@10", "recall@10", "recall@50"], "max_absolute_difference": worst, "slugs": {n: _slug(n) for n in rankings}}}


def assemble(run: Run) -> dict:
    from app.eval import beir_api
    from app.eval.provenance import git_sha, scrub
    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, resample_indices, settings as bootstrap_settings

    stages = {s: run.done(s) for s in SERVED_STAGES}  # not hnsw / variants: they have their own file (assemble_hnsw)
    missing = [s for s, blob in stages.items() if blob is None]
    if missing:
        raise RuntimeError(f"stages not recorded yet: {missing}")
    idx = resample_indices(len(run.queries), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    build = stages["index"]
    blob: dict[str, Any] = {
        # v2: the schema migrations build the table and its indexes before index_directory. v3 (M20): the deploy order's index
        # step runs over the built files (build.index_step), and the start-up reads the manifest, not every chunk (startup.components)
        "schema": "served-scale-v3",
        "measurement": "BEIR FiQA indexed by the served indexer into Postgres + pgvector and scored through POST /api/retrieve/, paired with the in-memory BEIR rows on the same queries in the same job",
        **run.identity,
        "n_docs": len(run.corpus), "n_queries": len(run.queries), "full_split": run.full, "query_selection": run.selection,
        "corpus_files": stages["files"],
        "build": {k: v for k, v in build.items() if k not in {"stage", "finished", "not_indexed"}},
        "startup": stages["startup"],
    }
    blob["not_indexed"] = build["not_indexed"]
    memory = stages["inmemory"]
    mem_rows = {name: block["rows"] for name, block in memory["rows"].items()}
    served, in_memory, rankings, per_query = {}, {}, {}, {}
    for name, block in memory["rows"].items():
        in_memory[name] = {k: v for k, v in block.items() if k != "rows"} | beir_api.summarize(block["rows"], idx)
        rankings[f"memory:{name}"] = {r["id"]: r["ranking"] for r in block["rows"]}
        per_query[f"memory:{name}"] = block["rows"]
    prompt_effect = {}
    for row, (_, bare) in PASSES.items():
        stage = "replay_rerank" if row == "api:hybrid+rerank" else "replay_hybrid"
        rows = stages[stage]["rows"]
        paired_with, bare_row = paired_rows(bare, mem_rows)
        served[row] = {"api_env": stages[stage]["api_env"], **beir_api.summarize(rows, idx), "health": stages[stage]["health"]}
        served[row]["paired_with"] = paired_with
        served[row]["delta_vs_in_memory"] = beir_api.paired(rows, mem_rows[paired_with], idx)
        served[row]["delta_vs_in_memory_by_route"] = beir_api.by_route(rows, mem_rows[paired_with])
        if bare_row:
            served[row]["bare_query_row"] = bare_row
            served[row]["delta_vs_bare_query_row"] = beir_api.paired(rows, mem_rows[bare_row], idx)
            prompt_effect[bare_row] = {"row": paired_with, "delta": beir_api.paired(mem_rows[paired_with], mem_rows[bare_row], idx)}
        served[row]["per_query"] = beir_api.public_rows(rows)
        rankings[row] = {r["id"]: [d for d in r["ranking"] if d != r["id"]] for r in rows}
        per_query[row] = rows
    blob["served"] = served
    blob["in_memory"] = in_memory
    blob["query_form"] = {**memory["query_form"], "vectors_check": memory["query_vectors_check"],
                          "in_memory_prompt_effect": prompt_effect,  # the in-memory row with the served prompt minus the bare-query row, paired
                          "note": "the served embedder's prompt is read from the model object the run builds with the API's settings; '<query>' is the query text"}
    committed = json.loads((ROOT / BEIR_FIQA_FILE).read_text()) if (ROOT / BEIR_FIQA_FILE).exists() else None
    blob["in_memory_reproduction"] = committed_reproduction(mem_rows, committed, ROOT, corpus_sha256=memory["corpus_sha256"], query_ids=list(run.queries))
    blob["load"] = stages["load"]["report"]
    blob["probe"] = stages["probe"]
    blob["plan"] = plan_block(stages["explain"], build["index_names"], table=stages["explain"]["names"]["table"])
    hybrid_health = stages["replay_hybrid"]["health"]
    rerank_health = stages["replay_rerank"]["health"]
    blob["differences"] = differences(
        rows=build.get("rows"), chunker=build["report"].get("chunker"), rejected=build["report"].get("rejected_by_reason") or {}, not_indexed=build["not_indexed"],
        routes=served["api:hybrid"].get("routes") or {}, served_depth=(hybrid_health.get("dense_search") or {}).get("depth"), search_settings=stages["explain"]["settings"],
        in_memory_settings=memory["rows"][PASSES["api:hybrid+rerank"][1]]["settings"], reranker=rerank_health.get("reranker") or {}, query_form=memory["query_form"])
    trec_dir = run.args.trec_dir if run.args.trec_dir.is_absolute() else ROOT / run.args.trec_dir
    blob["trec"] = trec_outputs(trec_dir, rankings, {q: dict(d) for q, d in run.qrels.items()}, per_query)
    blob["bootstrap"] = bootstrap_settings(len(run.queries), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    blob["hardware"] = hardware(run.args.device)
    blob.update(stage_machines(stages))
    blob["stage_git_sha"] = {s: b.get("git_sha") for s, b in stages.items()}
    blob["stage_seconds"] = {s: b.get("stage_seconds") for s, b in stages.items()}
    blob["git_sha"] = git_sha(ROOT)
    blob["git_dirty"] = git_dirty(ROOT)
    blob["timestamp"] = datetime.now(timezone.utc).isoformat()
    blob["command"] = run.command
    blob["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES + migration_files()}
    blob["embedder_note"] = "hash embedder, no reranker: a dry run of the machinery, not a measurement" if run.hash else None
    return scrub(blob)


def assemble_hnsw(run: Run) -> dict:
    """bench/results/pgvector_hnsw.json: the HNSW index against the exact scan on the run's table (C03)."""
    from app.eval.provenance import git_sha, scrub
    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, settings as bootstrap_settings

    index, sweep, variants = run.done("index"), run.done("hnsw"), run.done("variants")
    name = index["index_names"]["hnsw"]
    ddl = index["indexes_ddl"][name]
    param = hnsw_parameters(ddl)
    blob = {
        "schema": "pgvector-hnsw-v2",  # v2: the served index as the migrations and index_directory build it; C04's DDL is this module's
        "measurement": "pgvector's HNSW index against the exact scan, then halfvec and binary + rescore against float32, on the table the served indexer built from FiQA with bge-small, queried with the served query embedding of every test query",
        "dataset": run.identity["dataset"], "split": run.identity["split"], "files_sha256": run.identity["files_sha256"], "query_selection": run.selection,
        "table": {"rows": index["rows"], "dim": sweep["dim"], "embedder": index["embedder"], "vectors": "the chunk vectors app/engine/ingest.index_directory wrote (the same run as " + RESULTS_FILE + ")",
                  "bytes": index["sizes"]["table_bytes_with_toast"], "avg_embedding_bytes": index["sizes"]["avg_column_bytes"].get("embedding")},
        "queries": {"n": len(run.queries), "embedding": "the served query embedding (LlamaIndex HuggingFaceEmbedding with its query prompt for bge-small)"},
        "index": {"name": name, "ddl": ddl, **param, "bytes": index["sizes"]["indexes"][name]["bytes"],
                  "built_by": "the schema migrations (app/database/migrations/vector/) on the empty table, then index_directory's inserts, row by row: the served build, which the rows below read",
                  "reindex_seconds": variants["build_seconds"].get("reindex_float32"),
                  "reindex_note": "a bulk rebuild (REINDEX) of the same index on the final table, after the variants step's VACUUM; the variants rows read the rebuilt indexes"},
        "server": index["server"],
        "k": sweep["k"], "tie_eps": sweep["tie_eps"],
        "recall_definition": "a returned row counts when its float32 distance is at most the exact k-th distance plus tie_eps; Recall@10 reads the first 10 rows of the same k-row list",
        "latency_definition": "client wall time of each statement (execute and fetch) on one connection, after an untimed pass of the exact scan and of the widest index scan",
        "rows": sweep["rows"],
        "variants": {
            "what": "a halfvec and a bit copy of the embedding column with their HNSW indexes (app/retrieval/pgvector_tune.py HALFVEC_SQL, BINARY_SQL; the measurement's own DDL on this "
                    "task-local table, not a schema migration) after the rows above were measured; VACUUM (ANALYZE), then each HNSW index rebuilt (REINDEX); "
                    "then float32 and halfvec at the same ef_search, and binary candidates rescored by float32 cosine (hnsw.ef_search set to the candidate depth, at most 1000)",
            **{k: v for k, v in variants.items() if k not in {"stage", "finished", "git_sha", "stage_seconds"}},
            "served": "the served statement reads only the float32 column (app/database/sql/hybrid_tsrank.sql)",
        },
        "bootstrap": bootstrap_settings(len(run.queries), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED),
        "hardware": hardware(run.args.device),
        **stage_machines({s: run.done(s) for s in HNSW_STAGES}),
        "stage_git_sha": {s: run.done(s).get("git_sha") for s in HNSW_STAGES},
        "git_sha": git_sha(ROOT), "git_dirty": git_dirty(ROOT), "timestamp": datetime.now(timezone.utc).isoformat(), "command": run.command,
        "source_sha256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ("app/eval/pgvector_hnsw.py", "app/eval/served_scale.py", "app/database/vector_store_manager.py",
                                                                                           "app/database/sql/hybrid_tsrank.sql", "app/retrieval/pgvector_tune.py") + migration_files()},
    }
    return scrub(blob)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "bench" / "external")
    p.add_argument("--container", type=Path, required=True, help="Apptainer image with PostgreSQL 16 and pgvector")
    p.add_argument("--scratch", type=Path, required=True, help="node-local directory for the corpus files and the database")
    p.add_argument("--state", type=Path, default=None, help="stage results and the database snapshot (default <scratch>/state); put it on shared storage to resume on another node")
    p.add_argument("--out", type=Path, default=ROOT / RESULTS_FILE)
    p.add_argument("--hnsw-out", type=Path, default=ROOT / HNSW_FILE)
    p.add_argument("--trec-dir", type=Path, default=Path(TREC_DIR))
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--embedder", choices=["served", "hash"], default="served", help="hash: a dry run of the machinery with no model (not a measurement)")
    p.add_argument("--stages", default=",".join(STAGES), help="comma-separated, in order")
    p.add_argument("--resume", action="store_true", help="skip stages whose results are already in <state>")
    p.add_argument("--no-snapshot", dest="snapshot", action="store_false", help="do not copy the database to <state> after indexing")
    p.add_argument("--max-queries", type=int, default=None, help="a slice for a rate check, never the benchmark")
    p.add_argument("--max-docs", type=int, default=None, help="a slice for a rate check, never the benchmark")
    p.add_argument("--load-requests", type=int, default=64, help="requests per concurrency level")
    p.add_argument("--load-levels", default=",".join(str(c) for c in LOAD_LEVELS), help="concurrency levels of the load stage")
    p.add_argument("--request-timeout", type=float, default=120.0, help="client timeout per request (the API's own budget is RETRIEVE_TIMEOUT_SECONDS)")
    p.add_argument("--api-ready-timeout", type=float, default=1800.0)
    p.add_argument("--probe-queries", type=int, default=5)
    p.add_argument("--explain-queries", type=int, default=3)
    args = p.parse_args(argv)
    args.stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    args.load_levels = tuple(int(c) for c in args.load_levels.split(","))
    unknown = [s for s in args.stages if s not in STAGES]
    if unknown:
        p.error(f"unknown stages {unknown}; known: {', '.join(STAGES)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run = Run(args)
    written = run.run()
    for name, target in (("served", args.out), ("hnsw", args.hnsw_out)):
        if name in written:
            out = target if target.is_absolute() else ROOT / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(written[name], indent=1) + "\n")
            log(f"wrote {out}")
    if "served" in written:
        blob = written["served"]
        for row, s in blob["served"].items():
            d = s["delta_vs_in_memory"]["ndcg@10"]
            log(f"{row}: nDCG@10 {s['retrieval']['ndcg@10']:.4f} vs {s['paired_with']} {blob['in_memory'][s['paired_with']]['retrieval']['ndcg@10']:.4f}: "
                f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]; failed {s['requests']['failed']}")
    return 0 if len(written) == 2 else 3


if __name__ == "__main__":
    sys.exit(main())
