"""Parity: replay the gold questions through POST /api/retrieve/ and score them like the bench.

    python -m app.eval.harness --backend api --url http://localhost:8000 \
        --out bench/results/parity.json [--parity-config NAME]

Two rows come out of one run:
  in_memory  the bench harness on fixture v1 with a named configuration
             (IN_MEMORY_CONFIGS) that matches what the API serves:
               in_memory:hybrid+router+staleness (the default, CI's): hybrid
                 RRF, router on, staleness Tier 1 on, no reranker, hash
                 embeddings
               in_memory:hybrid+bge-rerank+router+staleness@bge-small/fp16-512/top20/staleness-first
                 (M17, A19): bge-small and bge-reranker-v2-m3 with the served
                 reranker settings (RERANK_DEPTH 20, RERANKER_MAX_LENGTH 512,
                 fp16 on CUDA) and Tier 1 before the reranker, as served; for an
                 API started with those models and EMBEDDING_QUERY_INSTRUCTION=none
                 (the bare query, as the benchmark embeds it)
             -> app/retrieval/hybrid.py
  api        the same 136 questions through the running API (Postgres +
             pgvector, SQL hybrid) -> app/engine/retriever.py in mode "sql"
The header's `settings` puts what each side ran with next to each other (from the
in-memory configuration and from /health and the responses) and `settings_differ`
lists the ones that differ.
Both are scored with the same path-level metrics and the same paired bootstrap.
The run reports the nDCG@10 gap (API minus in-memory) with its paired 95% interval,
and the paired gap per gold category (on the API row, delta_vs_parent_by_category_all;
on fixture v1 the exact_path questions are the router's lexical_path route); the gap
is a measurement, not a pass/fail gate. The exact check is the dense leg
(tests/database/test_dense_parity.py). The run fails (exit 1) when the API is
broken: a request error, a question without an answer, or a question with zero hits.
Before it contacts the API it refuses (RuntimeError) a bench/corpus/files that differs
from bench/corpus/MANIFEST.json (app/eval/corpus.verify_manifest), as the bench does.

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
  first-seen order (the path list on the lexical_path route included: the SQL
  scores paths as InMemoryHybridIndex.search_path does, and orders equal
  scores by node_id where the in-memory list keeps corpus order)
Results files recorded before M17 have an API row whose lexical_path route ran
the lexical list alone (no path list).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from app.eval.corpus import load_manifest, mtime_mismatches, verify_manifest
from app.eval.gold import load_gold
from app.eval.harness import RERANK_DEPTH_CONFIGS, BenchConfig, attach_confidence, project_root, run_config
from app.eval.index_loader import mtime_provenance
from app.eval.latency import StageTimer
from app.eval.metrics import aggregate_retrieval
from app.eval.provenance import command_line, device_info, git_sha, scrub, sha256_file

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
# The model-backed row (M17, A19): rerank_depth.json's last row, the served reranker settings with Tier 1 first.
IN_MEMORY_NEURAL = replace(RERANK_DEPTH_CONFIGS[-1], name="in_memory:" + RERANK_DEPTH_CONFIGS[-1].name, parent=None)
IN_MEMORY_CONFIGS = {cfg.name: cfg for cfg in (IN_MEMORY, IN_MEMORY_NEURAL)}
API_ROW = "api:postgres"


def in_memory_config(name: str) -> BenchConfig:
    try:
        return IN_MEMORY_CONFIGS[name]
    except KeyError:
        raise ValueError(f"unknown in-memory configuration {name!r}; one of: {', '.join(IN_MEMORY_CONFIGS)}") from None


def _uses_models(cfg: BenchConfig) -> bool:
    return cfg.embedder not in {"hash", "tfidf"} or (cfg.enable_rerank and cfg.reranker not in {"overlap", "none"})


def side_by_side(cfg: BenchConfig, mem: dict, api: dict, health: dict) -> dict[str, dict]:
    """What each side ran with: the in-memory configuration (and the models its row loaded) against what the API
    reported in /health and in its responses. A setting a side does not run is None (no reranker: no depth)."""
    embedder = (mem.get("models") or {}).get("embedder") or {}
    reranker = health.get("reranker") or {}
    mem_rerank = cfg.enable_rerank and bool(mem.get("reranker_loaded")) and cfg.reranker not in {"overlap", "none"}
    api_rerank = bool(reranker.get("loaded"))
    staleness = (api.get("settings") or {}).get("staleness") or {}
    routes = api.get("routes") or {}

    def order(on: bool, reranked: bool, before: bool) -> str:
        return "off" if not on else "on, no reranker" if not reranked else "before the rerank" if before else "after the rerank"

    pairs = {
        "embedding_model": (embedder.get("model") if embedder.get("neural") else f"{cfg.embedder}-{embedder.get('dim')}",
                            health.get("embedding_model")),
        # The benchmark's embedders put nothing in front of a query.
        "query_instruction": ("", health.get("embedding_query_instruction")),
        "retrieve_k": (cfg.retrieve_k, (health.get("dense_search") or {}).get("depth")),
        "router": (cfg.enable_router, bool(routes) and set(routes) != {"none"}),
        "reranker": (cfg.reranker if mem_rerank else None, reranker.get("model") if api_rerank else None),
        "rerank_depth": (cfg.rerank_depth if mem_rerank else None, reranker.get("depth") if api_rerank else None),
        "rerank_precision": ((cfg.rerank_precision or "fp32") if mem_rerank else None, reranker.get("precision") if api_rerank else None),
        "rerank_max_length": (cfg.rerank_max_length if mem_rerank else None, reranker.get("max_length") if api_rerank else None),
        # The served retriever always runs Tier 1 on the fused list before the reranker.
        "staleness": (order(cfg.staleness_tier1, mem_rerank, cfg.staleness_before_rerank), order(bool(staleness.get("enabled")), api_rerank, True)),
    }
    return {key: {"in_memory": a, "api": b} for key, (a, b) in pairs.items()}


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
    routes: dict[str, int] = {}
    t0 = time.perf_counter()
    for q in gold:
        t = time.perf_counter()
        body = post_retrieve(url, q.question, k=k)
        timer.record("e2e", (time.perf_counter() - t) * 1000.0)
        route = body.get("route") or "none"  # the router's prediction for this question (none: router off)
        routes[route] = routes.get(route, 0) + 1
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
        "routes": routes,
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


def run(url: str, *, root: Path | None = None, k: int = 50, in_memory: str = IN_MEMORY.name) -> dict[str, Any]:
    cfg = in_memory_config(in_memory)
    root = root or project_root()
    files_root = root / "bench" / "corpus" / "files"
    gold_path = root / "bench" / "gold" / "questions.jsonl"
    gold = load_gold(gold_path)
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    # Both rows read bench/corpus/files (the in-memory index here, the API through DATA_DIR) and the header
    # copies the MANIFEST's aggregate, so a copy that differs from the MANIFEST is refused before anything runs.
    drift = verify_manifest(files_root, manifest)
    if drift:
        raise RuntimeError("corpus drift:\n" + "\n".join(drift))
    health = wait_for_health(url)

    mem = run_config(cfg, gold, files_root, n_files=manifest["n_files"], mtimes="manifest")
    mem.pop("_triples", None)
    api = replay_api(url, gold, manifest["n_files"], k=k)
    scores = {mem["config"]: mem.pop("_per_query_scores"), api["config"]: api.pop("_per_query_scores")}
    results = [mem, api]
    # all_categories: the API row also gets the paired gap per gold category (delta_vs_parent_by_category_all).
    boot = attach_confidence(results, scores, n_queries=len(gold), categories=[q.category for q in gold], all_categories=True)

    gap = round(api["retrieval"]["ndcg@10"] - mem["retrieval"]["ndcg@10"], 4)
    problems = api_problems(api, len(gold))
    gap_report = {
        "metric": "ndcg@10",
        "in_memory": mem["retrieval"]["ndcg@10"],
        "api": api["retrieval"]["ndcg@10"],
        "gap": gap,
        "paired_delta_api_minus_in_memory": (api.get("delta_vs_previous") or {}).get("ndcg@10"),
    }
    settings = side_by_side(cfg, mem, api, health)
    return {
        "git_sha": git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info("auto" if _uses_models(cfg) else "cpu"),
        "in_memory_config": cfg.name,
        "settings": settings,
        "settings_differ": [key for key, sides in settings.items() if sides["in_memory"] != sides["api"]],
        "corpus_sha256": manifest["aggregate_sha256"],
        "gold_sha256": sha256_file(gold_path),
        "n_gold": len(gold),
        "n_files": manifest["n_files"],
        # The in-memory row reads the MANIFEST's mtimes; the API (app/engine/ingest.py) reads the disk's, so the
        # run says whether bench/corpus/files on disk had them (make fixture-mtimes), as in CI.
        "mtimes": {"in_memory": mtime_provenance(files_root, "manifest", base=root),
                   "api": {"source": "disk", "dir": "bench/corpus/files", "equal_to_manifest": not mtime_mismatches(files_root, manifest)}},
        "api_health": scrub(health),  # what /health reported, with paths, user and host names made publishable
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
