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

from app.agent.deep_research import multi_query_retrieve
from app.agent.retrieval_loop import extractive_answer
from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.index_loader import build_index
from app.eval.jury import KAPPA_GATE, default_jury, kappa_vs_gold
from app.eval.latency import StageTimer
from app.eval.metrics import aggregate_retrieval
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
    if cfg.mode == "bm25":
        with timer.stage("bm25"):
            pairs = index.search_bm25(query, k=cfg.retrieve_k)
        hits = [
            RetrievalHit(chunk=c, score=s, rank=i)
            for i, (c, s) in enumerate(pairs, start=1)
        ]
        result = None
    elif cfg.mode == "dense":
        with timer.stage("embed"):
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
        result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
        hits = result.hits

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


def run_config(
    cfg: BenchConfig,
    gold: list[GoldQuestion],
    files_root: Path,
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
    retrieval = aggregate_retrieval(per_query)
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
    payload["reranker_name"] = getattr(index, "reranker_name", cfg.reranker)
    payload["reranker_loaded"] = bool(getattr(index, "reranker_loaded", False))
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
]


def run_bench(
    *,
    root: Path | None = None,
    configs: list[BenchConfig] | None = None,
    e2e: bool | None = None,
) -> dict[str, Any]:
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
    for cfg in configs:
        row = run_config(cfg, gold, files_root)
        triples = row.pop("_triples", [])
        all_triples.extend(triples)
        results.append(row)

    sha = git_sha(root)
    blob = {
        "git_sha": sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": manifest["aggregate_sha256"],
        "n_gold": len(gold),
        "n_files": manifest["n_files"],
        "results": results,
    }
    out_dir = root / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sha}.json"
    out_path.write_text(json.dumps(blob, indent=2) + "\n")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(blob, indent=2) + "\n")
    if all_triples:
        write_triples(out_dir / f"{sha}.triples.jsonl", all_triples)
    blob["output"] = str(out_path)
    return blob


def markdown_table(blob: dict) -> str:
    lines = [
        "| config | Recall@50 | nDCG@10 | MRR@10 | p95 search ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in blob["results"]:
        lat = row.get("latency") or {}
        search = lat.get("vector_search") or lat.get("bm25") or {}
        p95 = search.get("p95_ms", "")
        r = row["retrieval"]
        extra = ""
        if row.get("reranker_fallback"):
            extra = f" (fallback={row['reranker_fallback']})"
        lines.append(
            f"| {row['config']}{extra} | {r['recall@50']:.3f} | {r['ndcg@10']:.3f} | {r['mrr@10']:.3f} | {p95} |"
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    jury = args.jury or os.environ.get("JUDGE", "").strip().lower() in {"1", "true", "yes"}
    configs = list(FRONTIER_CONFIGS) if args.frontier else list(DEFAULT_CONFIGS)
    if jury:
        configs = [replace(c, jury=True) for c in configs]
    blob = run_bench(configs=configs)
    print(markdown_table(blob))
    print(f"wrote {blob['output']}")


if __name__ == "__main__":
    main()
