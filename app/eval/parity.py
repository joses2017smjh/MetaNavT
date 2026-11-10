"""Parity: replay the gold questions through POST /api/retrieve/ and score them like the bench.

    python -m app.eval.harness --backend api --url http://localhost:8000 \
        --out bench/results/parity.json

Two rows come out of one run:
  in_memory  the bench harness on fixture v1 with the same switches the API
             serves in CI (hybrid RRF, router on, staleness Tier 1 on, no
             reranker, hash embeddings) -> app/retrieval/hybrid.py
  api        the same 136 questions through the running API (Postgres +
             pgvector, SQL hybrid) -> app/engine/retriever.py in mode "sql"
Both are scored with the same path-level metrics and the same paired bootstrap.
The run reports the nDCG@10 gap (API minus in-memory) with its paired 95% interval;
the gap is a measurement, not a pass/fail gate. The exact check is the dense leg
(tests/database/test_dense_parity.py). The run fails (exit 1) only when the API is
broken: a request error, a question without an answer, or a question with zero hits.

The same in both rows: the chunks. The API indexes DATA_DIR with the bench's
chunker (app/eval/ingest.file_chunks, called by app/eval/index_loader.py and
app/engine/ingest.py), so the dense leg embeds identical texts with identical
hash embeddings. Results files recorded before that change (bench/results/parity.json)
have an API row chunked with LlamaIndex SentenceSplitter(512, 50).

Why the rows still differ (both are reported, neither is "the" number):
- lexical leg: in-memory Okapi BM25 vs Postgres ts_rank_cd over
  websearch_to_tsquery (OR-joined terms); different tokenizer, no IDF
  saturation in ts_rank_cd
- ties: the SQL orders equal scores by node_id, the in-memory fusion by
  first-seen order
- exact-path route: the in-memory index searches paths
  (InMemoryHybridIndex.search_path); the API runs its lexical list alone
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from app.eval.corpus import load_manifest
from app.eval.gold import load_gold
from app.eval.harness import BenchConfig, attach_confidence, project_root, run_config
from app.eval.latency import StageTimer
from app.eval.metrics import aggregate_retrieval
from app.eval.provenance import command_line, device_info, git_sha

IN_MEMORY = BenchConfig(
    name="in_memory:hybrid+router+staleness",
    mode="hybrid",
    embedder="hash",
    enable_rerank=False,
    enable_router=True,
    staleness_tier1=True,
    log_triples=False,
    e2e=False,
)
API_ROW = "api:postgres"


def post_retrieve(url: str, query: str, k: int = 50, timeout: float = 60.0) -> dict:
    body = json.dumps({"query": query, "k": k}).encode()
    req = request.Request(url.rstrip("/") + "/api/retrieve/", data=body, headers={"content-type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_health(url: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last: Any = None
    while time.time() < deadline:
        try:
            with request.urlopen(url.rstrip("/") + "/health", timeout=10) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode())
        except error.HTTPError as exc:
            last = exc.read().decode()[:200]
        except (error.URLError, ConnectionError, TimeoutError) as exc:
            last = str(exc)
        time.sleep(3)
    raise SystemExit(f"API at {url} not healthy after {timeout:.0f}s; last: {last}")


def replay_api(url: str, gold, n_files: int, k: int = 50) -> dict[str, Any]:
    """Score the API on the gold set; per-query metrics kept for the bootstrap."""
    timer = StageTimer()
    per_query = []
    stage_sums: dict[str, float] = {}
    sample: dict | None = None
    zero_hit: list[str] = []
    t0 = time.perf_counter()
    for q in gold:
        t = time.perf_counter()
        body = post_retrieve(url, q.question, k=k)
        timer.record("e2e", (time.perf_counter() - t) * 1000.0)
        for name, ms in (body.get("latency_ms") or {}).items():
            stage_sums[name] = stage_sums.get(name, 0.0) + float(ms)
            timer.record(f"server:{name}", float(ms))
        paths = list(dict.fromkeys(h["path"] for h in body.get("hits", []) if h.get("path")))
        if not body.get("hits"):
            zero_hit.append(q.id)
        per_query.append({"id": q.id, "retrieved": paths, "relevant": q.relevant_ids(), "category": q.category})
        if sample is None:
            sample = {k2: body.get(k2) for k2 in ("route", "retrieval_mode", "bm25_backend", "embedding_provider", "reranker_loaded", "staleness")}
    wall_ms = (time.perf_counter() - t0) * 1000.0
    retrieval = aggregate_retrieval(per_query, n_files=n_files)
    return {
        "config": API_ROW,
        "settings": {"url": url, "k": k, **(sample or {})},
        "retrieval": retrieval.as_dict(),
        "latency": timer.summary(),
        "wall_ms": round(wall_ms, 2),
        "n_queries": len(per_query),
        "zero_hit_queries": zero_hit,
        "_per_query_scores": retrieval.per_query,
    }


def api_problems(api: dict, n_gold: int) -> list[str]:
    """Why the API row cannot be trusted (empty when it can). The nDCG@10 gap is not one of them."""
    problems = []
    if api["n_queries"] != n_gold:
        problems.append(f"{n_gold - api['n_queries']} of {n_gold} questions have no answer")
    if api.get("zero_hit_queries"):
        problems.append(f"{len(api['zero_hit_queries'])} questions returned zero hits: {', '.join(api['zero_hit_queries'][:10])}")
    return problems


def run(url: str, *, root: Path | None = None, k: int = 50) -> dict[str, Any]:
    root = root or project_root()
    files_root = root / "bench" / "corpus" / "files"
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    health = wait_for_health(url)

    mem = run_config(IN_MEMORY, gold, files_root, n_files=manifest["n_files"])
    mem.pop("_triples", None)
    api = replay_api(url, gold, manifest["n_files"], k=k)
    scores = {mem["config"]: mem.pop("_per_query_scores"), api["config"]: api.pop("_per_query_scores")}
    results = [mem, api]
    boot = attach_confidence(results, scores, n_queries=len(gold))

    gap = round(api["retrieval"]["ndcg@10"] - mem["retrieval"]["ndcg@10"], 4)
    problems = api_problems(api, len(gold))
    gap_report = {
        "metric": "ndcg@10",
        "in_memory": mem["retrieval"]["ndcg@10"],
        "api": api["retrieval"]["ndcg@10"],
        "gap": gap,
        "paired_delta_api_minus_in_memory": (api.get("delta_vs_previous") or {}).get("ndcg@10"),
    }
    return {
        "git_sha": git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info("cpu"),
        "corpus_sha256": manifest["aggregate_sha256"],
        "n_gold": len(gold),
        "n_files": manifest["n_files"],
        "api_health": health,
        "bootstrap": boot,
        "gap": gap_report,
        "api_ok": not problems,
        "api_problems": problems,
        "results": results,
    }


def markdown_table(blob: dict) -> str:
    def ci(row, m):
        b = (row.get("ci") or {}).get(m)
        return f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]" if b else f"{row['retrieval'][m]:.3f}"

    lines = [
        "| row | nDCG@10 [95% CI] | Recall@10 [95% CI] | Recall@50 | lexical leg | p50 / p95 ms per query |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in blob["results"]:
        lat = row["latency"]
        e2e = lat.get("e2e") or lat.get("vector_search") or lat.get("bm25") or {}
        lex = row["settings"].get("bm25_backend") or "in-memory Okapi BM25"
        lines.append(f"| {row['config']} | {ci(row, 'ndcg@10')} | {ci(row, 'recall@10')} | {row['retrieval']['recall@50']:.3f} | {lex} | {e2e.get('p50_ms', '—')} / {e2e.get('p95_ms', '—')} |")
    g = blob.get("gap") or blob["gate"]  # files recorded before the gate was dropped carry "gate"
    d = g.get("paired_delta_api_minus_in_memory") or {}
    lines.append("")
    lines.append(
        f"nDCG@10 gap api - in_memory = {g['gap']:+.4f}; "
        f"paired 95% CI {d.get('lo', '?'):+.3f} .. {d.get('hi', '?'):+.3f} (reported, not gated)"
    )
    if blob.get("api_problems"):
        lines.append("API problems: " + "; ".join(blob["api_problems"]))
    return "\n".join(lines)
