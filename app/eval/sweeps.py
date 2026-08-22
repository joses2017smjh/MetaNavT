"""Sweeps that sit off the default `make bench` path: hops, storage, HNSW, GraphRAG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.eval.harness import BenchConfig, git_sha, project_root, run_config
from app.eval.gold import load_gold
from app.eval.index_loader import build_index
from app.graph.graphrag import build_graphrag, answer_global
from app.retrieval.pgvector_tune import compare_storage, sweep_ef_search


def run_sweeps(root: Path | None = None) -> dict:
    root = root or project_root()
    files_root = root / "bench" / "corpus" / "files"
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    index = build_index(files_root, embedder_name="hash", enable_rerank=False, enable_router=False)
    docs = index.doc_mat
    # queries: gold questions encoded with the same embedder
    queries = index.embedder.encode([q.question for q in gold[:32]])

    hnsw = [p.as_dict() for p in sweep_ef_search(docs, queries, k=10, m=8, efs=(10, 20, 40, 80))]
    storage = [p.as_dict() for p in compare_storage(docs, queries, k=10)]

    hop_rows = []
    for hops in (0, 1, 2):
        cfg = BenchConfig(
            name=f"hybrid+hops={hops}",
            mode="hybrid",
            enable_rerank=False,
            enable_router=False,
            staleness_tier1=True,
            graph_hops=hops,
            e2e=False,
            log_triples=False,
        )
        hop_rows.append(run_config(cfg, gold, files_root))

    communities = build_graphrag(index.chunks)
    global_q = "what is in this corpus"
    blob = {
        "git_sha": git_sha(root),
        "hnsw": hnsw,
        "storage": storage,
        "graph_hops": [
            {
                "config": r["config"],
                "retrieval": r["retrieval"],
                "wall_ms": r["wall_ms"],
            }
            for r in hop_rows
        ],
        "graphrag": {
            "n_communities": len(communities),
            "global_answer": answer_global(global_q, communities),
            "communities": [
                {"id": c.community_id, "n": len(c.members), "summary": c.summary}
                for c in communities[:8]
            ],
        },
        "n_chunks": len(index.chunks),
        "embed_dim": int(docs.shape[1]),
    }
    out = root / "bench" / "results" / "sweeps.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    blob["output"] = str(out)
    return blob


def main() -> None:
    blob = run_sweeps()
    print(json.dumps({k: blob[k] for k in blob if k != "output"}, indent=2)[:4000])
    print("wrote", blob["output"])


if __name__ == "__main__":
    main()
