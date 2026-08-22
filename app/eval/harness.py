"""One-command retrieval bench over the frozen corpus + gold set."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.retrieval_loop import extractive_answer
from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.index_loader import build_index
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
    else:
        result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
        hits = result.hits

    if cfg.staleness_tier1 and clusters:
        hits = prefer_current(hits, clusters, query)

    paths = _paths_from_hits(hits)
    if cfg.graph_hops and graph is not None:
        paths = expand_with_graph(paths, graph, hops=cfg.graph_hops)
    return paths, result if result is not None else hits


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

    timer = StageTimer()
    per_query = []
    e2e_rows = []
    triples = []
    routes: dict[str, int] = {}
    max_mtime = max((c.mtime for c in index.chunks), default=1.0)

    t0 = time.perf_counter()
    for q in gold:
        paths, payload = _retrieve(index, q.question, cfg, timer, clusters, graph)
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
            e2e_rows.append(
                {
                    "id": q.id,
                    "faithfulness": faithfulness(answer, contexts),
                    "context_precision": context_precision(q.relevant_paths, paths, k=10),
                    "context_recall": context_recall(q.relevant_paths, paths, k=50),
                    "answer_relevancy": answer_relevancy(q.question, answer),
                }
            )
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
        lines.append(
            f"| {row['config']} | {r['recall@50']:.3f} | {r['ndcg@10']:.3f} | {r['mrr@10']:.3f} | {p95} |"
        )
    return "\n".join(lines)


def main() -> None:
    blob = run_bench()
    print(markdown_table(blob))
    print(f"wrote {blob['output']}")


if __name__ == "__main__":
    main()
