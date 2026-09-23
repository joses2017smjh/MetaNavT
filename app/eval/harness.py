"""One-command retrieval bench over the frozen corpus + gold set."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.corrective import evaluate_retrieval, determine_correction
from app.agent.decompose import heuristic_decompose, should_decompose
from app.agent.deep_research import multi_query_retrieve
from app.agent.retrieval_loop import extractive_answer
from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.index_loader import build_index
from app.eval.jury import KAPPA_GATE, default_jury, kappa_vs_gold
from app.eval.latency import StageTimer
from app.eval.metrics import aggregate_retrieval
from app.eval.provenance import command_line, device_info, model_device, model_provenance
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, mean_ci, paired_delta_ci, resample_indices, settings as bootstrap_settings
from app.eval.ragas_metrics import (
    answer_relevancy,
    aggregate_e2e,
    context_precision,
    context_recall,
    faithfulness,
)
from app.graph.file_graph import build_file_graph, expand_with_graph
from app.graph.hipporag import apply_hipporag, triples_from_chunks
from app.graph.staleness import cluster_versions, prefer_current
from app.retrieval.distill import triples_from_hits, write_triples
from app.retrieval.hybrid import InMemoryHybridIndex, RetrievalHit
from app.retrieval.hyde import hyde_embed


def git_sha(cwd: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class BenchConfig:
    name: str = "hybrid+rerank+router"
    embedder: str = "hash"
    retrieve_k: int = 50
    rerank_n: int = 8
    enable_router: bool = True
    enable_rerank: bool = True
    reranker: str = "overlap"
    chunk_strategy: str = "auto"
    mode: str = "hybrid"  # hybrid | bm25 | dense
    graph_hops: int = 0
    staleness_tier1: bool = True
    e2e: bool = True
    log_triples: bool = True
    jury: bool = False
    hipporag: bool = False
    multi_query: bool = False
    hyde: bool = False
    decompose: bool = False
    corrective: bool = False


def _paths_from_hits(hits: list[RetrievalHit]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        if hit.chunk.path not in seen:
            seen.append(hit.chunk.path)
    return seen


def _retrieve(
    index: InMemoryHybridIndex,
    query: str,
    cfg: BenchConfig,
    timer: StageTimer,
    clusters,
    graph,
    *,
    category: str = "",
    hippo_triples=None,
) -> tuple[list[str], object]:
    if cfg.decompose and should_decompose(query):
        subs = heuristic_decompose(query)
        if len(subs) > 1:
            all_hits: list[RetrievalHit] = []
            for sub_q in subs:
                sub_result = index.retrieve(sub_q, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
                all_hits.extend(sub_result.hits)
            seen_ids: dict[str, RetrievalHit] = {}
            for h in all_hits:
                if h.chunk.chunk_id not in seen_ids or h.score > seen_ids[h.chunk.chunk_id].score:
                    seen_ids[h.chunk.chunk_id] = h
            hits = sorted(seen_ids.values(), key=lambda h: h.score, reverse=True)
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            result.hits = hits
        else:
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            hits = result.hits
    elif cfg.mode == "bm25":
        with timer.stage("bm25"):
            pairs = index.search_bm25(query, k=cfg.retrieve_k)
        hits = [
            RetrievalHit(chunk=c, score=s, rank=i)
            for i, (c, s) in enumerate(pairs, start=1)
        ]
        result = None
    elif cfg.mode == "dense":
        with timer.stage("embed"):
            if cfg.hyde:
                hyde_vec = hyde_embed(query, index.embedder)
                pairs = index.search_dense(query, k=cfg.retrieve_k, query_vec=hyde_vec)
            else:
                pairs = index.search_dense(query, k=cfg.retrieve_k)
        hits = [
            RetrievalHit(chunk=c, score=s, rank=i)
            for i, (c, s) in enumerate(pairs, start=1)
        ]
        result = None
    elif cfg.multi_query:
        with timer.stage("multi_query"):
            result = multi_query_retrieve(index, query, n=3, k=cfg.retrieve_k)
        hits = result.hits
    else:
        if cfg.hyde:
            with timer.stage("hyde"):
                hyde_vec = hyde_embed(query, index.embedder)
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            hyde_dense = index.search_dense(query, k=cfg.retrieve_k, query_vec=hyde_vec)
            hyde_ids = [c.chunk_id for c, _ in hyde_dense]
            from app.retrieval.fuse import rrf_score_map
            orig_ids = [h.chunk.chunk_id for h in result.hits]
            boosted = rrf_score_map([orig_ids, hyde_ids], k=60)
            for h in result.hits:
                if h.chunk.chunk_id in boosted:
                    h.score = boosted[h.chunk.chunk_id]
            result.hits.sort(key=lambda h: h.score, reverse=True)
            hits = result.hits
        else:
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            hits = result.hits

    if cfg.corrective and hits:
        pairs_for_eval = [(h.chunk, h.score) for h in hits[:cfg.rerank_n]]
        evaluated = evaluate_retrieval(query, pairs_for_eval)
        action = determine_correction(query, evaluated)
        if action.strategy == "rewrite" and action.rewritten_query:
            retry = index.retrieve(action.rewritten_query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            seen = {h.chunk.chunk_id for h in hits}
            for rh in retry.hits:
                if rh.chunk.chunk_id not in seen:
                    hits.append(rh)
                    seen.add(rh.chunk.chunk_id)
        elif action.strategy == "decompose" and action.sub_queries:
            for sub_q in action.sub_queries:
                sub_r = index.retrieve(sub_q, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
                seen = {h.chunk.chunk_id for h in hits}
                for rh in sub_r.hits:
                    if rh.chunk.chunk_id not in seen:
                        hits.append(rh)
                        seen.add(rh.chunk.chunk_id)

    if cfg.staleness_tier1 and clusters:
        hits = prefer_current(hits, clusters, query)

    if cfg.hipporag and hippo_triples:
        with timer.stage("hipporag"):
            hits = apply_hipporag(
                query, hits, hippo_triples, category=category, router=index.router
            )
        if result is not None:
            result.hits = hits

    paths = _paths_from_hits(hits)
    if cfg.graph_hops and graph is not None:
        paths = expand_with_graph(paths, graph, hops=cfg.graph_hops)
    return paths, result if result is not None else hits


def _jury_complete():
    """Optional Ollama backend. Probe once; stay heuristic if the daemon is down."""
    flag = os.environ.get("JUDGE", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return None
    from app.eval.judge import ollama_complete

    probe = ollama_complete('Reply JSON {"score": 1, "label": "correct"}')
    if not probe:
        return None
    return lambda prompt: ollama_complete(prompt) or ""


BGE_SMALL = "st:BAAI/bge-small-en-v1.5"
BGE_BASE = "st:BAAI/bge-base-en-v1.5"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"

# Real models on fixture v1 (make bench-neural). bm25_only is repeated so the
# paired deltas have the same reference as the hash table; e2e heuristics and
# triple logging are off because they add nothing to the retrieval question.
NEURAL_CONFIGS = [
    BenchConfig(name="bm25_only", mode="bm25", enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False, e2e=False),
    BenchConfig(name="dense_only@bge-small", mode="dense", embedder=BGE_SMALL, enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False, e2e=False),
    BenchConfig(name="hybrid@bge-small", mode="hybrid", embedder=BGE_SMALL, enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False, e2e=False),
    BenchConfig(name="hybrid+bge-rerank@hash", mode="hybrid", embedder="hash", reranker=BGE_RERANKER, enable_rerank=True, enable_router=False, staleness_tier1=False, log_triples=False, e2e=False),
    BenchConfig(name="hybrid+bge-rerank@bge-small", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, enable_rerank=True, enable_router=False, staleness_tier1=False, log_triples=False, e2e=False),
    BenchConfig(name="hybrid+bge-rerank+router+staleness@bge-small", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, enable_rerank=True, enable_router=True, staleness_tier1=True, log_triples=False, e2e=False),
    BenchConfig(name="dense_only@bge-base", mode="dense", embedder=BGE_BASE, enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False, e2e=False),
    BenchConfig(name="hybrid@bge-base", mode="hybrid", embedder=BGE_BASE, enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False, e2e=False),
]


def run_config(
    cfg: BenchConfig,
    gold: list[GoldQuestion],
    files_root: Path,
    n_files: int | None = None,
) -> dict[str, Any]:
    index = build_index(
        files_root,
        embedder_name=cfg.embedder,
        retrieve_k=cfg.retrieve_k,
        rerank_n=cfg.rerank_n,
        enable_router=cfg.enable_router,
        enable_rerank=cfg.enable_rerank,
        reranker=cfg.reranker,
        chunk_strategy=cfg.chunk_strategy,
    )
    clusters = cluster_versions(index.chunks) if cfg.staleness_tier1 else {}
    graph = None
    if cfg.graph_hops:
        files = []
        seen = set()
        for chunk in index.chunks:
            if chunk.path in seen:
                continue
            seen.add(chunk.path)
            files.append((chunk.path, chunk.text, chunk.mtime))
        graph = build_file_graph(files)
    hippo_triples = triples_from_chunks(index.chunks) if cfg.hipporag else None
    jury = default_jury(_jury_complete()) if cfg.jury else None

    timer = StageTimer()
    per_query = []
    e2e_rows = []
    triples = []
    routes: dict[str, int] = {}
    max_mtime = max((c.mtime for c in index.chunks), default=1.0)
    jury_sf_answers: list[str] = []
    jury_sf_golds: list[str] = []
    jury_sf_labels: list[str] = []

    t0 = time.perf_counter()
    for q in gold:
        paths, payload = _retrieve(
            index,
            q.question,
            cfg,
            timer,
            clusters,
            graph,
            category=q.category,
            hippo_triples=hippo_triples,
        )
        per_query.append(
            {
                "id": q.id,
                "retrieved": paths,
                "relevant": q.relevant_ids(),
                "category": q.category,
            }
        )
        result = payload if hasattr(payload, "hits") else None
        if result is not None:
            routes[result.route.route.value] = routes.get(result.route.route.value, 0) + 1
            if cfg.log_triples:
                triples.extend(triples_from_hits(q.question, result.hits, max_mtime=max_mtime))
        if cfg.e2e:
            hits = result.hits[:8] if result is not None else payload[:8]
            contexts = [h.chunk.text for h in hits]
            answer = extractive_answer(q.question, hits)
            row = {
                "id": q.id,
                "faithfulness": faithfulness(answer, contexts),
                "context_precision": context_precision(q.relevant_paths, paths, k=10),
                "context_recall": context_recall(q.relevant_paths, paths, k=50),
                "answer_relevancy": answer_relevancy(q.question, answer),
            }
            if jury is not None:
                verdict = jury.vote(q.question, answer, contexts, q.answer)
                row["jury_label"] = verdict.label
                row["jury_score"] = verdict.score
                if q.category == "simple_factual":
                    jury_sf_answers.append(answer)
                    jury_sf_golds.append(q.answer)
                    jury_sf_labels.append(verdict.label)
            e2e_rows.append(row)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    retrieval = aggregate_retrieval(per_query, n_files=n_files)
    payload: dict[str, Any] = {
        "config": cfg.name,
        "settings": cfg.__dict__,
        "retrieval": retrieval.as_dict(),
        "latency": timer.summary(),
        "wall_ms": round(wall_ms, 2),
        "routes": routes,
        "n_chunks": len(index.chunks),
    }
    if cfg.e2e:
        payload["e2e"] = aggregate_e2e(e2e_rows).as_dict()
    payload["_per_query_scores"] = retrieval.per_query  # gold order; used for the bootstrap, then dropped
    payload["reranker_name"] = getattr(index, "reranker_name", cfg.reranker)
    payload["reranker_loaded"] = bool(getattr(index, "reranker_loaded", False))
    rerank_model = getattr(getattr(index, "rerank_fn", None), "model", None)
    payload["models"] = {
        "embedder": {
            **model_provenance(cfg.embedder, "embedder"),
            "dim": int(getattr(index.embedder, "dim", 0)),
            "device": model_device(getattr(index.embedder, "_model", None)),
            "query_instruction": None,  # bge-v1.5 recommends a query prefix; not used, same text both sides
        },
        "reranker": {
            **model_provenance(cfg.reranker, "reranker"),
            "loaded": payload["reranker_loaded"],
            "device": model_device(rerank_model) if payload["reranker_loaded"] else None,
        },
    }
    fallback = getattr(index, "reranker_fallback", None)
    if fallback:
        payload["reranker_fallback"] = fallback
    if cfg.multi_query:
        payload["multi_query"] = True
    if cfg.hipporag:
        payload["hipporag"] = True
    if jury is not None and jury_sf_labels:
        report = kappa_vs_gold(jury_sf_answers, jury_sf_golds, jury_sf_labels)
        payload["jury"] = {
            "mean_score": round(
                sum(r.get("jury_score", 0.0) for r in e2e_rows) / max(len(e2e_rows), 1), 4
            ),
            "kappa_simple_factual": report.as_dict(),
            "readme_ok": report.kappa >= KAPPA_GATE,
            "members": [name for name, _ in jury.members],
        }
        if not report.gated:
            payload["jury"]["note"] = (
                f"kappa {report.kappa:.3f} < {KAPPA_GATE}; do not print jury scores on the README"
            )
    if triples:
        payload["n_triples"] = len(triples)
        payload["_triples"] = triples
    return payload


DEFAULT_CONFIGS = [
    BenchConfig(name="dense_only", mode="dense", enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False),
    BenchConfig(name="bm25_only", mode="bm25", enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False),
    BenchConfig(name="hybrid", mode="hybrid", enable_rerank=False, enable_router=False, staleness_tier1=False),
    BenchConfig(name="hybrid+rerank", mode="hybrid", enable_rerank=True, enable_router=False, staleness_tier1=False),
    BenchConfig(name="hybrid+rerank+router", mode="hybrid", enable_rerank=True, enable_router=True, staleness_tier1=False),
    BenchConfig(
        name="hybrid+rerank+router+staleness",
        mode="hybrid",
        enable_rerank=True,
        enable_router=True,
        staleness_tier1=True,
    ),
]


FRONTIER_CONFIGS = [
    BenchConfig(
        name="hybrid+bge-rerank",
        mode="hybrid",
        enable_rerank=True,
        enable_router=False,
        staleness_tier1=False,
        reranker="BAAI/bge-reranker-v2-m3",
        log_triples=True,
    ),
    BenchConfig(
        name="hybrid+rankgpt",
        mode="hybrid",
        enable_rerank=True,
        enable_router=False,
        staleness_tier1=False,
        reranker="rankgpt",
        log_triples=True,
    ),
    BenchConfig(
        name="hybrid+multiquery",
        mode="hybrid",
        enable_rerank=False,
        enable_router=False,
        staleness_tier1=False,
        multi_query=True,
        log_triples=False,
    ),
    BenchConfig(
        name="hybrid+hipporag",
        mode="hybrid",
        enable_rerank=False,
        enable_router=True,
        staleness_tier1=False,
        hipporag=True,
        log_triples=False,
    ),
    BenchConfig(
        name="hybrid+hyde",
        mode="hybrid",
        enable_rerank=True,
        enable_router=True,
        staleness_tier1=False,
        hyde=True,
        log_triples=True,
    ),
    BenchConfig(
        name="hybrid+decompose",
        mode="hybrid",
        enable_rerank=True,
        enable_router=True,
        staleness_tier1=False,
        decompose=True,
        log_triples=True,
    ),
    BenchConfig(
        name="hybrid+corrective",
        mode="hybrid",
        enable_rerank=True,
        enable_router=True,
        staleness_tier1=False,
        corrective=True,
        log_triples=True,
    ),
    BenchConfig(
        name="hybrid+hyde+corrective",
        mode="hybrid",
        enable_rerank=True,
        enable_router=True,
        staleness_tier1=False,
        hyde=True,
        corrective=True,
        log_triples=True,
    ),
]


def run_bench(
    *,
    root: Path | None = None,
    configs: list[BenchConfig] | None = None,
    e2e: bool | None = None,
    out: Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Score every config on fixture v1.

    Without `out`, results go to bench/results/<sha>.json and latest.json (the
    default bench). With `out`, only that file is written, so neural runs never
    touch the CI baseline.
    """
    root = root or project_root()
    files_root = root / "bench" / "corpus" / "files"
    gold_path = root / "bench" / "gold" / "questions.jsonl"
    manifest_path = root / "bench" / "corpus" / "MANIFEST.json"

    gold = load_gold(gold_path)
    manifest = load_manifest(manifest_path)
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))

    configs = configs or DEFAULT_CONFIGS
    if e2e is False:
        for c in configs:
            c.e2e = False

    results = []
    all_triples = []
    scores_by_config: dict[str, list[dict[str, float]]] = {}
    for cfg in configs:
        row = run_config(cfg, gold, files_root, n_files=manifest["n_files"])
        triples = row.pop("_triples", [])
        all_triples.extend(triples)
        scores_by_config[cfg.name] = row.pop("_per_query_scores", [])
        results.append(row)
    boot = attach_confidence(results, scores_by_config, n_queries=len(gold))

    sha = git_sha(root)
    blob = {
        "git_sha": sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info(device),
        "corpus_sha256": manifest["aggregate_sha256"],
        "n_gold": len(gold),
        "n_files": manifest["n_files"],
        "bootstrap": boot,
        "results": results,
    }
    out_dir = root / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out is not None:
        out = out if out.is_absolute() else root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        blob["output"] = str(out)
        return blob
    out_path = out_dir / f"{sha}.json"
    out_path.write_text(json.dumps(blob, indent=2) + "\n")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(blob, indent=2) + "\n")
    if all_triples:
        write_triples(out_dir / f"{sha}.triples.jsonl", all_triples)
    blob["output"] = str(out_path)
    return blob


CI_METRICS = ("ndcg@10", "recall@10", "recall@50")
REFERENCE_CONFIG = "bm25_only"


def attach_confidence(
    results: list[dict[str, Any]],
    scores_by_config: dict[str, list[dict[str, float]]],
    *,
    n_queries: int,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict:
    """Add per-config 95% CIs and paired deltas (vs bm25_only and vs the previous row) in place.

    One index matrix is shared by every config, so deltas are paired. A missing
    reference (frontier runs have no bm25_only; the first row has no previous)
    yields None rather than an error.
    """
    if n_queries <= 0 or not results:
        return bootstrap_settings(n_queries, n_boot=n_boot, seed=seed)
    idx = resample_indices(n_queries, n_boot=n_boot, seed=seed)

    def column(name: str, metric: str) -> list[float] | None:
        rows = scores_by_config.get(name)
        if not rows or len(rows) != n_queries:
            return None
        return [r[metric] for r in rows]

    def deltas(name: str, other: str | None) -> dict | None:
        if other is None or other == name:
            return None
        out: dict[str, Any] = {"reference": other}
        for metric in CI_METRICS:
            a, b = column(name, metric), column(other, metric)
            if a is None or b is None:
                return None
            d, lo, hi = paired_delta_ci(a, b, idx)
            out[metric] = {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        return out

    names = [row["config"] for row in results]
    reference = REFERENCE_CONFIG if REFERENCE_CONFIG in scores_by_config else None
    for i, row in enumerate(results):
        name = row["config"]
        ci: dict[str, Any] = {}
        for metric in CI_METRICS:
            values = column(name, metric)
            if values is None:
                continue
            mean, lo, hi = mean_ci(values, idx)
            ci[metric] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        row["ci"] = ci
        row["delta_vs_bm25_only"] = deltas(name, reference)
        row["delta_vs_previous"] = deltas(name, names[i - 1] if i > 0 else None)
    return bootstrap_settings(n_queries, n_boot=n_boot, seed=seed)


def _fmt_ci(block: dict | None, key: str = "mean") -> str:
    if not block:
        return "—"
    sign = "+" if key == "delta" and block[key] >= 0 else ""
    return f"{sign}{block[key]:.3f} [{block['lo']:+.3f}, {block['hi']:+.3f}]" if key == "delta" else f"{block[key]:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


def markdown_table(blob: dict) -> str:
    lines = [
        "| config | nDCG@10 [95% CI] | Recall@10 [95% CI] | Recall@50 (random list, same length) | unique files in top 50 | MRR@10 | Δ nDCG@10 vs bm25_only [95% CI] | p95 search ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["results"]:
        lat = row.get("latency") or {}
        search = lat.get("vector_search") or lat.get("bm25") or {}
        p95 = search.get("p95_ms", "")
        r = row["retrieval"]
        ci = row.get("ci") or {}
        extra = ""
        if row.get("reranker_fallback"):
            extra = f" (fallback={row['reranker_fallback']})"
        rand = r.get("random_recall@50")
        rand_txt = f"{r['recall@50']:.3f} ({rand:.3f})" if rand is not None else f"{r['recall@50']:.3f}"
        delta = (row.get("delta_vs_bm25_only") or {}).get("ndcg@10")
        lines.append(
            f"| {row['config']}{extra} | {_fmt_ci(ci.get('ndcg@10'))} | {_fmt_ci(ci.get('recall@10'))} | {rand_txt} | "
            f"{r.get('mean_unique_paths@50', 0):.1f} | {r['mrr@10']:.3f} | {_fmt_ci(delta, 'delta')} | {p95} |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(description="Frozen-corpus retrieval bench")
    p.add_argument(
        "--jury",
        action="store_true",
        help="extra Phase 6 jury columns (heuristic; LLM if JUDGE=1 and Ollama is up)",
    )
    p.add_argument(
        "--frontier",
        action="store_true",
        help="Phase 7–9 extra configs only (bge, RankGPT, multi-query, HippoRAG)",
    )
    p.add_argument(
        "--neural",
        action="store_true",
        help="real embedders (bge-small, bge-base) and the real bge-reranker-v2-m3 on fixture v1 (needs .[ml])",
    )
    p.add_argument("--out", type=Path, default=None, help="write only this file (latest.json and the sha file are left alone)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="recorded in the results header; cpu forces CPU for torch models")
    p.add_argument("--only", default=None, help="comma-separated config names to run (subset of the selected list)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    jury = args.jury or os.environ.get("JUDGE", "").strip().lower() in {"1", "true", "yes"}
    if args.neural:
        configs = list(NEURAL_CONFIGS)
    elif args.frontier:
        configs = list(FRONTIER_CONFIGS)
    else:
        configs = list(DEFAULT_CONFIGS)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        configs = [c for c in configs if c.name in wanted]
    if jury:
        configs = [replace(c, jury=True) for c in configs]
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    blob = run_bench(configs=configs, out=args.out, device=args.device)
    print(markdown_table(blob))
    print(f"wrote {blob['output']}")


if __name__ == "__main__":
    main()
