"""Export evidence-backed hard-case traces for HTML, GIF, and README demos.

The curated manifest names gold IDs. This module runs the real offline stack
and writes one source of truth:

  doc/demo/traces.json
  doc/demo/traces.js

No LLM, GPU, network, or corpus mutation is used.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from app.agent.deep_research import DeepResearchAgent, multi_query_retrieve
from app.artifacts.pipeline import ArtifactAgent
from app.artifacts.sandbox import run_sandboxed
from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import git_sha, project_root
from app.eval.index_loader import build_index
from app.eval.metrics import recall_at_k
from app.graph.conflicts import (
    detect_semantic_conflicts,
    detect_structural_conflicts,
)
from app.graph.graphrag import answer_global, build_graphrag
from app.graph.hipporag import apply_hipporag, triples_from_chunks
from app.graph.staleness import cluster_versions, prefer_current
from app.mcp.filesystem import ApprovalRequired, FilesystemTools
from app.retrieval.types import RetrievalHit


def _paths(hits: Sequence[RetrievalHit], limit: int | None = None) -> list[str]:
    out: list[str] = []
    for hit in hits:
        if hit.chunk.path not in out:
            out.append(hit.chunk.path)
        if limit is not None and len(out) >= limit:
            break
    return out


def _rank(paths: Sequence[str], relevant: Sequence[str]) -> int | None:
    wanted = set(relevant)
    for rank, path in enumerate(paths, start=1):
        if path in wanted:
            return rank
    return None


def _hits(
    hits: Sequence[RetrievalHit],
    relevant: Sequence[str],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    relevant_set = set(relevant)
    out = []
    seen = set()
    for hit in hits:
        path = hit.chunk.path
        if path in seen:
            continue
        seen.add(path)
        status = (
            "gold"
            if path in relevant_set
            else "superseded"
            if "archive/" in path or "draft_v1" in path
            else "candidate"
        )
        out.append(
            {
                "path": path,
                "rank": len(out) + 1,
                "status": status,
                "score": round(float(hit.score), 6),
                "rrf": round(float(hit.rrf), 6) if hit.rrf is not None else None,
            }
        )
        if len(out) >= limit:
            break
    return out


def _checks(
    method_paths: Sequence[str],
    gold_paths: Sequence[str],
    *,
    extra: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    coverage = recall_at_k(method_paths, gold_paths, 50)
    rows = [
        {
            "label": "gold path coverage @50",
            "pass": coverage == 1.0,
            "value": f"{len(set(method_paths[:50]) & set(gold_paths))}/{len(set(gold_paths))}",
        }
    ]
    rows.extend(extra)
    return rows


def _gold_case(
    index,
    question: GoldQuestion,
    spec: dict[str, Any],
    clusters,
    triples,
) -> dict[str, Any]:
    result = index.retrieve(question.question, k=50, n=8)
    baseline = list(result.hits)
    current = prefer_current(list(baseline), clusters, question.question)
    multi = prefer_current(
        multi_query_retrieve(index, question.question, n=3, k=50).hits,
        clusters,
        question.question,
    )
    graph = apply_hipporag(
        question.question,
        current,
        triples,
        category=question.category,
        router=index.router,
    )
    feature = spec["feature"]
    if feature in {"graph", "aggregation"}:
        method_hits = graph
        method_label = "typed PPR rank fusion"
    elif feature == "deep-research":
        method_hits = multi
        method_label = "three-query RRF + staleness"
    elif feature == "staleness":
        method_hits = current
        method_label = "current-version filter"
    else:
        method_hits = baseline
        method_label = result.route.route.value

    base_paths = _paths(baseline)
    method_paths = _paths(method_hits)
    dropped = [path for path in base_paths if path not in set(_paths(current))]
    conflict_pool = list(baseline[:20])
    conflicts = detect_structural_conflicts(conflict_pool) + detect_semantic_conflicts(
        conflict_pool
    )
    extra_checks: list[dict[str, Any]] = []
    if feature == "staleness":
        extra_checks.append(
            {
                "label": "superseded evidence removed",
                "pass": any("archive/" in path or "draft_v1" in path for path in dropped),
                "value": ", ".join(dropped[:3]) or "none",
            }
        )
        extra_checks.append(
            {
                "label": "conflict surfaced before filtering",
                "pass": bool(conflicts),
                "value": f"{len(conflicts)} conflict(s)",
            }
        )
    if feature == "router":
        extra_checks.extend(
            [
                {
                    "label": "embedding skipped",
                    "pass": result.skipped_embed,
                    "value": str(result.skipped_embed).lower(),
                },
                {
                    "label": "reranker skipped",
                    "pass": result.skipped_rerank,
                    "value": str(result.skipped_rerank).lower(),
                },
            ]
        )

    agent_trace = None
    if feature == "deep-research":
        answer = DeepResearchAgent(index, n_queries=3, token_budget=2048).run(
            question.question
        )
        citation_paths = [citation.path for citation in answer.citations]
        agent_trace = {
            "failed": answer.failed,
            "fail_reason": answer.fail_reason,
            "events": [event.step for event in answer.events],
            "citations": citation_paths[:8],
            "current_gold_cited": any(path in question.relevant_paths for path in citation_paths),
        }
        extra_checks.append(
            {
                "label": "scratchpad cites a gold path",
                "pass": agent_trace["current_gold_cited"],
                "value": ", ".join(citation_paths[:3]) or "none",
            }
        )

    checks = _checks(method_paths, question.relevant_paths, extra=extra_checks)
    passed = all(row["pass"] for row in checks)
    return {
        "id": spec["id"],
        "gold_id": question.id,
        "title": spec["title"],
        "feature": feature,
        "method": spec["method"],
        "question": question.question,
        "category": question.category,
        "gold_answer": question.answer,
        "gold_paths": question.relevant_paths,
        "route": result.route.route.value,
        "control": {
            "label": "single-query hybrid",
            "gold_rank": _rank(base_paths, question.relevant_paths),
            "recall@50": round(recall_at_k(base_paths, question.relevant_paths, 50), 4),
            "hits": _hits(baseline, question.relevant_paths),
        },
        "method_result": {
            "label": method_label,
            "gold_rank": _rank(method_paths, question.relevant_paths),
            "recall@50": round(recall_at_k(method_paths, question.relevant_paths, 50), 4),
            "hits": _hits(method_hits, question.relevant_paths),
            "dropped": dropped[:8],
        },
        "conflicts": [conflict.as_dict() for conflict in conflicts],
        "agent": agent_trace,
        "checks": checks,
        "status": "pass" if passed else "mixed",
        "answer_source": "frozen gold set",
    }


def _hop_control(spec: dict[str, Any], question: GoldQuestion, sweeps: dict) -> dict:
    rows = []
    for row in sweeps.get("graph_hops", []):
        retrieval = row["retrieval"]
        rows.append(
            {
                "config": row["config"],
                "recall@50": retrieval["recall@50"],
                "ndcg@10": retrieval["ndcg@10"],
                "wall_ms": row["wall_ms"],
            }
        )
    no_hops = next((row for row in rows if row["config"].endswith("=0")), None)
    one_hop = next((row for row in rows if row["config"].endswith("=1")), None)
    check = bool(no_hops and one_hop and no_hops["ndcg@10"] > one_hop["ndcg@10"])
    return {
        "id": spec["id"],
        "gold_id": question.id,
        "title": spec["title"],
        "feature": spec["feature"],
        "method": spec["method"],
        "question": question.question,
        "category": question.category,
        "gold_answer": question.answer,
        "gold_paths": question.relevant_paths,
        "rows": rows,
        "checks": [
            {
                "label": "hops=0 ranks better than hops=1",
                "pass": check,
                "value": (
                    f"{no_hops['ndcg@10']:.3f} > {one_hop['ndcg@10']:.3f}"
                    if no_hops and one_hop
                    else "missing sweep"
                ),
            }
        ],
        "status": "pass" if check else "mixed",
        "answer_source": "gold + committed sweep",
    }


def _artifact_cases(index) -> list[dict[str, Any]]:
    prop = ArtifactAgent(index).produce("reproduce run 47 from the paper")
    citations = [row["path"] for row in prop.spec.citations]
    current_evidence = {
        "configs/run_047.yaml",
        "paper/draft_v2.md",
        "src/fusion.py",
    }
    artifact_checks = [
        {
            "label": "current config, paper, and source cited",
            "pass": current_evidence <= set(citations),
            "value": ", ".join(citations),
        },
        {
            "label": "archived evidence excluded",
            "pass": not any("archive/" in path or "draft_v1" in path for path in citations),
            "value": "current-only" if citations else "no citations",
        },
        {
            "label": "generated code executed",
            "pass": prop.exec_result.ok,
            "value": prop.exec_result.stdout.strip() or prop.exec_result.error,
        },
        {
            "label": "claim-support audit",
            "pass": not prop.extra.get("unsupported_claims"),
            "value": str(prop.extra.get("unsupported_claims") or "zero unsupported claims"),
        },
    ]
    with tempfile.TemporaryDirectory(prefix="metanavit-demo-") as tmp:
        tools = FilesystemTools(root=Path(tmp), index=index)
        tools.artifacts[prop.plan_id] = prop
        blocked = False
        try:
            tools.apply_artifact(prop.plan_id, approved=False)
        except ApprovalRequired:
            blocked = True
        applied = tools.apply_artifact(prop.plan_id, approved=True)
        wrote = (Path(tmp) / prop.spec.file_path).is_file()

    denied = run_sandboxed("import os\nos.system('echo should-not-run')")
    artifact = {
        "id": "artifact-run47",
        "gold_id": None,
        "title": "Paper2Code with current evidence",
        "feature": "artifact",
        "method": "plan → analyze → generate → execute → approve",
        "question": "reproduce run 47 from the paper",
        "category": "synthetic (not in 136-gold retrieval set)",
        "gold_answer": "executable reproduction with current citations",
        "gold_paths": sorted(current_evidence),
        "spec": prop.spec.as_dict(),
        "code_preview": "\n".join(prop.code.splitlines()[:18]),
        "execution": prop.exec_result.as_dict(),
        "checks": artifact_checks,
        "status": "pass" if all(row["pass"] for row in artifact_checks) else "mixed",
        "answer_source": "deterministic artifact tests",
    }
    safety_checks = [
        {
            "label": "unapproved write blocked",
            "pass": blocked,
            "value": "ApprovalRequired" if blocked else "not blocked",
        },
        {
            "label": "approved temp write succeeds",
            "pass": applied["status"] == "applied" and wrote,
            "value": applied["path"],
        },
        {
            "label": "unsafe import rejected",
            "pass": not denied.ok and "banned" in denied.error,
            "value": denied.error,
        },
    ]
    safety = {
        "id": "hitl-sandbox",
        "gold_id": None,
        "title": "Sandbox and approval gate",
        "feature": "safety",
        "method": "AST gate + propose/apply split",
        "question": "write an artifact, but do not mutate the tree without approval",
        "category": "synthetic (unit-tested, not retrieval gold)",
        "gold_answer": "blocked, then applied only in a temporary directory",
        "gold_paths": [],
        "checks": safety_checks,
        "status": "pass" if all(row["pass"] for row in safety_checks) else "mixed",
        "answer_source": "deterministic MCP/sandbox execution",
    }
    return [artifact, safety]


def build_demo(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    files_root = root / "bench" / "corpus" / "files"
    manifest = json.loads((root / "bench" / "demo" / "manifest.json").read_text())
    corpus_manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    drift = verify_manifest(files_root, corpus_manifest)
    if drift:
        raise RuntimeError("corpus drift:\n" + "\n".join(drift))

    gold = {q.id: q for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    index = build_index(
        files_root,
        embedder_name="hash",
        retrieve_k=50,
        rerank_n=8,
        enable_router=True,
        enable_rerank=True,
        reranker="overlap",
    )
    clusters = cluster_versions(index.chunks)
    triples = triples_from_chunks(index.chunks)
    sweeps = json.loads((root / "bench" / "results" / "sweeps.json").read_text())

    cases = [
        _gold_case(index, gold[spec["id"]], spec, clusters, triples)
        for spec in manifest["gold_cases"]
    ]
    cases.extend(
        _hop_control(spec, gold[spec["id"]], sweeps)
        for spec in manifest["controls"]
    )
    cases.extend(_artifact_cases(index))
    visualization_path = root / "doc" / "demo" / "visualization.json"
    if visualization_path.exists():
        cases.append(json.loads(visualization_path.read_text()))
    communities = build_graphrag(index.chunks, min_community=2)
    return {
        "meta": {
            "git_sha": git_sha(root),
            "corpus_sha256": corpus_manifest["aggregate_sha256"],
            "n_gold": len(gold),
            "n_files": corpus_manifest["n_files"],
            "config": manifest["config"],
            "generator": "python -m app.eval.demo_export",
            "llm_free": True,
        },
        "methodology": {
            "principles": [
                "curated IDs point to the frozen gold set",
                "control and method run over the same top-50 candidate budget",
                "PPR fuses ranks and never adds unseen candidates",
                "synthetic artifact/safety cases are labeled outside retrieval gold",
                "known losers remain visible",
            ],
            "global_graphrag": answer_global(
                "what runs, encoders, and relations are in this corpus",
                communities,
            ),
        },
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_gold_cases": sum(case["gold_id"] is not None for case in cases),
            "n_synthetic_cases": sum(case["gold_id"] is None for case in cases),
            "passed": sum(case["status"] == "pass" for case in cases),
            "mixed": sum(case["status"] != "pass" for case in cases),
        },
    }


def write_demo(root: Path | None = None) -> tuple[Path, Path]:
    root = root or project_root()
    payload = build_demo(root)
    out = root / "doc" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "traces.json"
    js_path = out / "traces.js"
    encoded = json.dumps(payload, indent=2, sort_keys=False)
    json_path.write_text(encoded + "\n")
    js_path.write_text("window.METANAVIT_DEMO = " + encoded + ";\n")
    return json_path, js_path


def main() -> None:
    for path in write_demo():
        print("wrote", path, path.stat().st_size)


if __name__ == "__main__":
    main()
