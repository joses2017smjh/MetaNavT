"""Served-path scale run (owner's upgrade 5): BEIR FiQA through the served indexer, Postgres and POST /api/retrieve/.

    make bench-served-scale PYTHON=<python with torch> PGVECTOR_SIF=<PostgreSQL 16 + pgvector image> \
        SERVED_SCALE_SCRATCH=<node-local directory> [SERVED_SCALE_STATE=<shared directory>] [BEIR_DATA_DIR=bench/external]
    -> bench/results/served_scale_fiqa.json and bench/results/served_scale_fiqa_trec/ (A05)
       bench/results/pgvector_hnsw.json (C03: app/eval/pgvector_hnsw.py, on the same table)

For a GPU node: bge-small (the served embedder, LlamaIndex HuggingFaceEmbedding) and bge-reranker-v2-m3 on
CUDA from the HuggingFace cache, Apptainer with a PostgreSQL 16 + pgvector image (app/eval/service.py), and
FiQA under the data directory (app/eval/beir.py downloads it once and pins its sha256).

The run is a sequence of stages. Each writes <state>/<stage>.json, and --resume skips a stage whose file
exists, so a job that dies late keeps what it measured. The corpus and the database live under <scratch>
(node-local disk: 57,638 small files and a Postgres data directory on NFS would measure NFS); after the
index stage the data directory is copied to <state>, and a resumed run on a fresh node restores it.

  files          FiQA as one file per document, <doc_id>.txt, the in-memory rows' text (beir_api.write_corpus)
  index          app/engine/ingest.index_directory into the task-local Postgres with the served embedder:
                 seconds (and the embedder's share), rows, rejections by reason; then the two indexes the API's
                 start-up would create (GIN on to_tsvector, HNSW on the embedding), each timed and checked in
                 pg_indexes, and ANALYZE, so the start-up measured below does not build them; sizes on disk;
                 index_directory again over the unchanged directory (what INDEX_ON_START=true costs per start)
  startup        what start-up reads: every chunk back (fetch_all_chunks) and the version clusters
                 (bootstrap.build_clusters), timed here; then the API (uvicorn main:app, INDEX_ON_START unset,
                 both models), seconds from launch to /health ok
  probe          a few hybrid statements and lexical-only statements, timed in this process before the replay
  replay_rerank  every test query through POST /api/retrieve/ with k = 50 and the served defaults (reranker on)
  load           app/eval/http_load on that API: concurrency 1, 4 and 16, k = 8, failures counted
  replay_hybrid  the same replay with RERANKER_MODEL=none: the like-for-like pair of the in-memory hybrid row
  explain        EXPLAIN (ANALYZE, BUFFERS) of the served statement for three queries: which scan each leg uses
  inmemory       app/eval/beir.py's hybrid@bge-small and hybrid+bge-rerank@bge-small/fp16-512/top20 rows,
                 document embeddings from the BEIR cache, on the same queries, scored by the same function
  hnsw           pgvector's HNSW index against the exact scan on the same table and queries, at several
                 hnsw.ef_search values, and the served dense CTE (app/eval/pgvector_hnsw.py)

The results file pairs each served row with its in-memory row (served minus in-memory, paired bootstrap
over queries), per route as well, and lists every difference between the two sides. Every number comes
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
# served row -> (the API's extra environment, the in-memory row it is paired with)
PASSES = {
    "api:hybrid+rerank": ({}, "hybrid+bge-rerank@bge-small/fp16-512/top20"),
    "api:hybrid": ({"RERANKER_MODEL": "none"}, "hybrid@bge-small"),
}
STAGES = ("files", "index", "startup", "probe", "replay_rerank", "load", "replay_hybrid", "explain", "inmemory", "hnsw")
SERVED_STAGES = ("files", "index", "startup", "probe", "replay_rerank", "load", "replay_hybrid", "explain", "inmemory")  # served_scale_fiqa.json
HNSW_STAGES = ("files", "index", "hnsw")  # pgvector_hnsw.json
SERVER_SETTINGS = ("shared_buffers", "maintenance_work_mem", "work_mem", "effective_cache_size", "max_parallel_maintenance_workers",
                   "max_parallel_workers_per_gather", "jit", "synchronous_commit", "random_page_cost")
SOURCES = ("app/eval/served_scale.py", "app/eval/beir_api.py", "app/eval/service.py", "app/eval/beir.py", "app/eval/http_load.py", "app/engine/ingest.py",
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


def elide_vectors(plan: str) -> str:
    """The plan with each vector literal shortened to its length (a query vector is 384 numbers)."""
    return _VECTOR_LITERAL.sub(lambda m: f"'[{m.group(0).count(',') + 1} values]'::{m.group(1)}", plan)


def differences(*, rows: int | None, chunker: str | None, rejected: dict, not_indexed: dict, routes: dict, served_depth: int | None, search_settings: dict,
                in_memory_settings: dict, reranker: dict) -> list[str]:
    """What separates a served row from its in-memory row, in words, from the run's own records and the code's defaults."""
    import inspect

    from app.chunking.structure import fixed_size_chunks

    window = {name: p.default for name, p in inspect.signature(fixed_size_chunks).parameters.items() if name in {"size", "overlap"}}
    mem_k = in_memory_settings.get("k")
    return [
        f"documents vs chunks: the in-memory rows rank whole documents; the served index holds {rows} chunks ({chunker}; a .txt file takes the "
        f"fixed window, {window.get('size')} characters with {window.get('overlap')} overlap), and a served document ranks at its best chunk",
        f"depth: the API returns at most {SERVED_K} chunks, so a served ranking holds at most {SERVED_K} documents and its Recall@50 counts within those chunks; "
        f"the in-memory rows fuse {mem_k} + {mem_k} candidates into {mem_k} documents (RRF k {in_memory_settings.get('rrf_k')}), the served statement {served_depth} + {served_depth} into {served_depth} chunks",
        "lexical leg: in-memory Okapi BM25 (Lucene-style English analyzer, k1 "
        f"{in_memory_settings.get('k1')}, b {in_memory_settings.get('b')}) vs Postgres ts_rank_cd over to_tsvector('english') with the query terms OR-joined",
        f"dense leg: in-memory exact cosine over whole-document vectors (the BEIR cache) vs pgvector over chunk vectors with {search_settings} per statement, "
        "through the HNSW index when the planner picks it (the plan block)",
        "query embedding: the served embedder (LlamaIndex HuggingFaceEmbedding) prefixes each query with 'Represent this question for searching relevant passages: ' "
        "(its default prompt for this model); the in-memory rows embed the bare query",
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


def build_indexes(vsm) -> dict:
    """The two indexes app/engine/bootstrap.py creates at start-up, each timed, then checked in pg_indexes; ANALYZE."""
    out: dict[str, Any] = {}
    for name, ensure in (("text_index", vsm.ensure_text_index), ("hnsw", vsm.ensure_hnsw)):
        started = time.perf_counter()
        ok = ensure()
        out[f"{name}_seconds"] = round(time.perf_counter() - started, 3)
        if not ok:  # the helpers log a failure and return False; a measurement must not carry on without the index
            raise RuntimeError(f"{ensure.__name__} failed; see the log")
    ddl = index_ddl(vsm)
    expected = {"hnsw": f"{vsm.table_name}_embedding_hnsw", "text_index": f"{vsm.data_table}_text_tsv_gin"}
    missing = [n for n in expected.values() if n not in ddl]
    if missing:
        raise RuntimeError(f"indexes missing after start-up's ensure_* calls: {missing}")
    out["index_names"] = expected
    started = time.perf_counter()
    query_rows_commit(vsm, f'ANALYZE "{vsm.schema_name}"."{vsm.data_table}";')
    out["analyze_seconds"] = round(time.perf_counter() - started, 3)
    out["indexes_ddl"] = ddl
    return out


def query_rows_commit(vsm, statement: str) -> None:
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)


def index_corpus(vsm, corpus_dir: Path, embed_batch: Callable[[list], list], embedder: str) -> dict:
    """index_directory over corpus_dir, timed, with the embedder's own share; the unchanged re-run; the indexes."""
    from app.engine.ingest import index_directory

    spent = {"seconds": 0.0, "calls": 0, "texts": 0}

    def embed_fn(texts):
        started = time.perf_counter()
        vectors = embed_batch(texts)
        spent["seconds"] += time.perf_counter() - started
        spent["calls"] += 1
        spent["texts"] += len(texts)
        return vectors

    started = time.perf_counter()
    report = index_directory(vsm, str(corpus_dir), embed_fn=embed_fn, embedder=embedder)
    seconds = time.perf_counter() - started
    embedding = dict(spent)
    built = build_indexes(vsm)
    sizes = table_sizes(vsm)
    spent.update(seconds=0.0, calls=0, texts=0)
    started = time.perf_counter()
    rerun = index_directory(vsm, str(corpus_dir), embed_fn=embed_fn, embedder=embedder)
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
        **built,
        "sizes": sizes,
        "unchanged_rerun": {"seconds": round(rerun_seconds, 3), "files_unchanged": rerun["files_unchanged"], "chunks_embedded": rerun["chunks_embedded"], "embed_calls": spent["calls"]},
    }


def not_indexed(corpus: dict, qrels: dict, indexed_paths: set[str]) -> dict:
    """Documents with no file in the ingest manifest (rejected by the served scanner), and the judged ones among them."""
    from app.eval.beir_api import doc_file_name

    missing = sorted(d for d in corpus if doc_file_name(d) not in indexed_paths)
    judged = sorted({d for docs in qrels.values() for d in docs} & set(missing))
    return {"n": len(missing), "ids": missing, "judged": judged, "queries": sorted(q for q, docs in qrels.items() if set(docs) & set(judged))}


def startup_components(vsm) -> dict:
    """What every API start reads with staleness on (the default): all chunks back, then the version clusters."""
    from app.engine.bootstrap import build_clusters

    started = time.perf_counter()
    rows = vsm.fetch_all_chunks()
    fetch = time.perf_counter() - started
    text_bytes = sum(len((r["text"] or "").encode("utf-8")) for r in rows)
    del rows
    started = time.perf_counter()
    clusters = build_clusters(vsm)
    total = time.perf_counter() - started
    return {"fetch_all_chunks_seconds": round(fetch, 3), "build_clusters_seconds": round(total, 3), "rows_read": vsm.count_nodes(), "text_bytes_read": text_bytes,
            "clusters": len(clusters), "note": "build_clusters = fetch_all_chunks again + cluster_versions; both timed in the runner, not inside the API"}


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


def git_dirty(root: Path) -> bool | None:
    import subprocess

    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return None


def hardware(device: str) -> dict:
    from app.eval.provenance import device_info

    info = device_info(device)
    try:
        info["cpu_model"] = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu_model"] = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and device != "cpu":
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    info["allocation"] = {"cpus": os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE"), "memory_mb": os.environ.get("SLURM_MEM_PER_NODE")}
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

        blob = {"stage": stage, "git_sha": git_sha(ROOT), "finished": datetime.now(timezone.utc).isoformat(), **blob}
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
        self.api_pass, self.api_health = row, health
        log(f"API for {row} ready in {seconds:.1f} s: n_nodes {health.get('n_nodes')}, reranker {health.get('reranker', {}).get('loaded')}")
        return self.api, health, seconds, True

    def stop_api(self) -> None:
        if self.api is not None:
            self.api.stop()
            self.api = None
            self.api_pass = None

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
        model = self.embed_model()
        vsm = self.vsm()
        if table_exists(vsm) and vsm.count_nodes():  # an interrupted index stage leaves rows: index_directory would resume, and the timing would be partial
            raise RuntimeError("the database already holds rows; the index stage measures a build from empty: use a fresh --scratch")
        label = str(getattr(model, "model_name", "") or type(model).__name__)  # what app/engine/bootstrap.py passes
        blob = {"embedder": label, "server": server_facts(vsm), **index_corpus(vsm, self.corpus_dir, model.get_text_embedding_batch, label)}
        blob["not_indexed"] = not_indexed(self.corpus, self.qrels, set(vsm.ingest_manifest()))
        if self.args.snapshot:
            self.snapshot()
        return blob

    def snapshot(self) -> None:
        """Stop Postgres, copy its data directory to <state>, start it again (a resumed run restores it)."""
        self._vsm.close()
        self._vsm = None
        self.pg.stop()
        self.pg = None
        started = time.perf_counter()
        target = self.state / "pgdata.tar"
        with tarfile.open(target.with_suffix(".tar.partial"), "w") as tar:
            tar.add(self.scratch / "pg" / "pgdata", arcname="pgdata")
        target.with_suffix(".tar.partial").replace(target)
        log(f"database snapshot {target.stat().st_size / 2**20:.0f} MiB in {time.perf_counter() - started:.0f} s")
        self.vsm()

    def stage_startup(self) -> dict:
        _, health, seconds, _ = self.api_for("api:hybrid+rerank")
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
        return {"row": row, "api_env": PASSES[row][0], "health": health, "api_started_for_this_stage": started, "api_seconds_to_health_ok": round(seconds, 3) if started else None, "rows": rows}

    def stage_load(self) -> dict:
        from app.eval.http_load import run_load

        api, _, _, _ = self.api_for("api:hybrid+rerank")
        queries = list(self.queries.items())
        report = asyncio.run(run_load(api.url, queries, concurrency_levels=self.args.load_levels, requests_per_level=self.args.load_requests, warmup_requests=4, timeout=self.args.request_timeout,
                                      require_reranker=not self.hash, require_device=self.args.device if self.args.device == "cuda" else None))
        report["workload_note"] = f"the first {min(len(queries), self.args.load_requests)} test queries by id, in order, cycled per level; one cold probe and 4 warm-ups excluded; the service had just served the replay"
        return {"report": report}

    def stage_explain(self) -> dict:
        from app.retrieval.router import QueryRouter

        self.stop_api()
        vsm = self.vsm()
        vectors = self.query_vectors()
        router = QueryRouter()
        index = list(self.queries)
        picks = [qid for qid in index if router.route(self.queries[qid]).route.value == "semantic"][: self.args.explain_queries]
        names = {"table": vsm.data_table, "hnsw_index": f"{vsm.table_name}_embedding_hnsw", "text_index": f"{vsm.data_table}_text_tsv_gin"}
        out = []
        for qid in picks:
            plan = vsm.explain_hybrid(self.queries[qid], list(map(float, vectors[index.index(qid)])), k=SERVED_K)
            out.append({"id": qid, "facts": plan_facts(plan, **names), "plan": elide_vectors(plan).splitlines()})
        return {"queries": out, "settings": vsm.search_settings(SERVED_K), "names": names}

    def stage_inmemory(self) -> dict:
        from app.eval import beir
        from app.eval.beir_api import score_ranking
        from app.eval.latency import StageTimer

        self.stop_api()
        engine = beir.Engine(beir.chunks_from_corpus(self.corpus), cache_dir=self.data_root.parent / "cache", device=self.args.device)
        configs = {c.name: c for c in beir.DEFAULT_CONFIGS}
        rows = {}
        cached = {}
        for _, paired_with in PASSES.values():
            cfg = configs[paired_with]
            if self.hash:
                cfg = replace(cfg, embedder="hash", reranker="overlap" if cfg.reranker else None, precision="fp32", max_length=None)
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
            rows[paired_with] = {"config": cfg.name, "settings": {k: v for k, v in vars(cfg).items()}, "build_ms": build, "latency": timer.summary(),
                                 "models": {"embedder": engine.provenance.get(cfg.embedder), **({"reranker": engine.provenance.get(engine.reranker_key(cfg.reranker, cfg.precision, cfg.max_length))} if cfg.reranker else {})},
                                 "rows": per_query}
            log(f"in-memory {paired_with}: nDCG@10 {sum(r['ndcg@10'] for r in per_query) / len(per_query):.4f}")
        return {"corpus_sha256": engine.corpus_sha256, "document_embeddings_from_cache": cached, "rows": rows}

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
                blob = method() if method else self.stage_replay("api:hybrid+rerank" if stage == "replay_rerank" else "api:hybrid")
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

    stages = {s: run.done(s) for s in STAGES}
    missing = [s for s, blob in stages.items() if blob is None]
    if missing:
        raise RuntimeError(f"stages not recorded yet: {missing}")
    idx = resample_indices(len(run.queries), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    build = stages["index"]
    blob: dict[str, Any] = {
        "schema": "served-scale-v1",
        "measurement": "BEIR FiQA indexed by the served indexer into Postgres + pgvector and scored through POST /api/retrieve/, paired with the in-memory BEIR rows on the same queries in the same job",
        **run.identity,
        "n_docs": len(run.corpus), "n_queries": len(run.queries), "full_split": run.full, "query_selection": run.selection,
        "corpus_files": stages["files"],
        "build": {k: v for k, v in build.items() if k not in {"stage", "finished", "not_indexed"}},
        "startup": stages["startup"],
    }
    blob["not_indexed"] = build["not_indexed"]
    served, in_memory, rankings, per_query = {}, {}, {}, {}
    for row, (_, paired_with) in PASSES.items():
        stage = "replay_rerank" if row == "api:hybrid+rerank" else "replay_hybrid"
        rows = stages[stage]["rows"]
        served[row] = {"api_env": stages[stage]["api_env"], **beir_api.summarize(rows, idx), "health": stages[stage]["health"]}
        mem_rows = stages["inmemory"]["rows"][paired_with]
        in_memory[paired_with] = {k: v for k, v in mem_rows.items() if k != "rows"} | beir_api.summarize(mem_rows["rows"], idx)
        served[row]["paired_with"] = paired_with
        served[row]["delta_vs_in_memory"] = beir_api.paired(rows, mem_rows["rows"], idx)
        served[row]["delta_vs_in_memory_by_route"] = beir_api.by_route(rows, mem_rows["rows"])
        served[row]["per_query"] = beir_api.public_rows(rows)
        rankings[row] = {r["id"]: [d for d in r["ranking"] if d != r["id"]] for r in rows}
        rankings[f"memory:{paired_with}"] = {r["id"]: r["ranking"] for r in mem_rows["rows"]}
        per_query[row], per_query[f"memory:{paired_with}"] = rows, mem_rows["rows"]
    blob["served"] = served
    blob["in_memory"] = in_memory
    blob["load"] = stages["load"]["report"]
    blob["probe"] = stages["probe"]
    blob["plan"] = stages["explain"]
    hybrid_health = stages["replay_hybrid"]["health"]
    rerank_health = stages["replay_rerank"]["health"]
    blob["differences"] = differences(
        rows=build.get("rows"), chunker=build["report"].get("chunker"), rejected=build["report"].get("rejected_by_reason") or {}, not_indexed=build["not_indexed"],
        routes=served["api:hybrid"].get("routes") or {}, served_depth=(hybrid_health.get("dense_search") or {}).get("depth"), search_settings=stages["explain"]["settings"],
        in_memory_settings=stages["inmemory"]["rows"][PASSES["api:hybrid+rerank"][1]]["settings"], reranker=rerank_health.get("reranker") or {})
    trec_dir = run.args.trec_dir if run.args.trec_dir.is_absolute() else ROOT / run.args.trec_dir
    blob["trec"] = trec_outputs(trec_dir, rankings, {q: dict(d) for q, d in run.qrels.items()}, per_query)
    blob["bootstrap"] = bootstrap_settings(len(run.queries), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    blob["hardware"] = hardware(run.args.device)
    blob["stage_git_sha"] = {s: b.get("git_sha") for s, b in stages.items()}
    blob["stage_seconds"] = {s: b.get("stage_seconds") for s, b in stages.items()}
    blob["git_sha"] = git_sha(ROOT)
    blob["git_dirty"] = git_dirty(ROOT)
    blob["timestamp"] = datetime.now(timezone.utc).isoformat()
    blob["command"] = run.command
    blob["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    blob["embedder_note"] = "hash embedder, no reranker: a dry run of the machinery, not a measurement" if run.hash else None
    return scrub(blob)


def assemble_hnsw(run: Run) -> dict:
    """bench/results/pgvector_hnsw.json: the HNSW index against the exact scan on the run's table (C03)."""
    from app.eval.provenance import git_sha, scrub
    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, settings as bootstrap_settings

    index, sweep = run.done("index"), run.done("hnsw")
    name = index["index_names"]["hnsw"]
    ddl = index["indexes_ddl"][name]
    param = {key: re.search(rf"{key}='?(\d+)", ddl) for key in ("m", "ef_construction")}
    blob = {
        "schema": "pgvector-hnsw-v1",
        "measurement": "pgvector's HNSW index against the exact scan, on the table the served indexer built from FiQA with bge-small, queried with the served query embedding of every test query",
        "dataset": run.identity["dataset"], "split": run.identity["split"], "files_sha256": run.identity["files_sha256"], "query_selection": run.selection,
        "table": {"rows": index["rows"], "dim": sweep["dim"], "embedder": index["embedder"], "vectors": "the chunk vectors app/engine/ingest.index_directory wrote (the same run as " + RESULTS_FILE + ")",
                  "bytes": index["sizes"]["table_bytes_with_toast"], "avg_embedding_bytes": index["sizes"]["avg_column_bytes"].get("embedding")},
        "queries": {"n": len(run.queries), "embedding": "the served query embedding (LlamaIndex HuggingFaceEmbedding with its query prompt for bge-small)"},
        "index": {"name": name, "ddl": ddl, "m": int(param["m"].group(1)) if param["m"] else None, "ef_construction": int(param["ef_construction"].group(1)) if param["ef_construction"] else None,
                  "build_seconds": index["hnsw_seconds"], "bytes": index["sizes"]["indexes"][name]["bytes"], "built_by": "VectorStoreManager.ensure_hnsw (what the API's start-up runs), after every row was inserted"},
        "server": index["server"],
        "k": sweep["k"], "tie_eps": sweep["tie_eps"],
        "recall_definition": "a returned row counts when its float32 distance is at most the exact k-th distance plus tie_eps; Recall@10 reads the first 10 rows of the same k-row list",
        "latency_definition": "client wall time of each statement (execute and fetch) on one connection, after an untimed pass of the exact scan and of the widest index scan",
        "rows": sweep["rows"],
        "bootstrap": bootstrap_settings(len(run.queries), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED),
        "hardware": hardware(run.args.device),
        "stage_git_sha": {s: run.done(s).get("git_sha") for s in HNSW_STAGES},
        "git_sha": git_sha(ROOT), "git_dirty": git_dirty(ROOT), "timestamp": datetime.now(timezone.utc).isoformat(), "command": run.command,
        "source_sha256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ("app/eval/pgvector_hnsw.py", "app/eval/served_scale.py", "app/database/vector_store_manager.py", "app/database/sql/hybrid_tsrank.sql")},
    }
    return scrub(blob)


def main(argv: list[str] | None = None) -> int:
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
