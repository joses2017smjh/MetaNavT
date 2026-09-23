"""M4: LLM answer evaluation with a validity gate (make bench-jury).

    python -m app.eval.llm_eval --out bench/results/jury.json \
        [--generator qwen2.5:7b] [--judges qwen2.5:7b,llama3.1:8b] [--limit N] \
        [--embedder st:BAAI/bge-small-en-v1.5 --reranker BAAI/bge-reranker-v2-m3]

For every gold question, on fixture v1:
1. Retrieve with the chosen configuration (default: the full pipeline with real
   models, the best row of bench-neural) and keep the top 8 chunks.
2. Generate a cited answer with a local model through Ollama
   (app/agent/generate.py). An answer with no valid citation is a failure
   ("uncited"), counted and excluded from judging; a NOT IN SOURCES reply is an
   abstention.
3. Deterministic checks, no model involved:
   - cited_in_retrieved: every cited path is in the retrieved set
   - values_in_cited_bytes: every number / config value in the answer appears
     inside the cited chunks (app/agent/citation_verify.py, cited_only=True)
   - exact_match: the gold answer string is in the answer (simple_factual only)
4. Jury: one LlmJudge per judge model (pointwise faithfulness / relevancy /
   groundedness -> label), plus the heuristic judge kept as a labelled column;
   pairwise AB/BA position swap per judge between the LLM answer and the
   extractive answer, so position bias is measured (position_gap).
5. Cohen's kappa between each judge's labels and the exact-match labels on
   simple_factual (n = 80). Judge scores may be published only when kappa >=
   KAPPA_GATE (0.6); the results file says so either way (readme_ok).

Heuristic columns are labelled "heuristic" and stay out of README tables.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.agent.citation_verify import extract_claims, verify_claims
from app.agent.generate import OllamaGenerator, cited_answer, ollama_available
from app.agent.retrieval_loop import extractive_answer
from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import BGE_RERANKER, BGE_SMALL, BenchConfig, _retrieve, project_root
from app.eval.index_loader import build_index
from app.eval.judge import HeuristicJudge, LlmJudge, exact_match_label, label_to_score, ollama_complete, pairwise_ab_ba
from app.eval.jury import KAPPA_GATE, DEFAULT_JUDGE_MODELS, cohens_kappa
from app.eval.latency import StageTimer
from app.eval.provenance import command_line, device_info, git_sha, model_provenance
from app.graph.staleness import cluster_versions
from app.retrieval.types import RetrievalHit

DEFAULT_CONFIG = BenchConfig(
    name="hybrid+bge-rerank+router+staleness@bge-small",
    mode="hybrid",
    embedder=BGE_SMALL,
    reranker=BGE_RERANKER,
    enable_rerank=True,
    enable_router=True,
    staleness_tier1=True,
    log_triples=False,
    e2e=False,
)
TOP_N = 8


def deterministic_checks(question: GoldQuestion, text: str, hits: Sequence[RetrievalHit]) -> dict[str, Any]:
    info = cited_answer(text, hits)
    chunks = [h.chunk for h in hits]
    claims = extract_claims(info["answer"])
    strict = verify_claims(claims, chunks, citations=info["citations"], cited_only=True)
    loose = verify_claims(extract_claims(info["answer"]), chunks)
    checks = {
        "uncited": info["uncited"],
        "abstained": info["abstained"],
        "cited_in_retrieved": bool(info["citations"]) and not info["unknown_citations"],
        "n_citations": len(info["citations"]),
        "unknown_citations": info["unknown_citations"],
        "n_values": sum(1 for c in claims if c.value),
        "values_in_cited_bytes": not strict.hallucinated_values,
        "values_not_in_cited_bytes": strict.hallucinated_values,
        "values_in_any_evidence": not loose.hallucinated_values,
        "verification_ratio_cited": round(strict.verification_ratio, 4),
    }
    if question.category == "simple_factual":
        checks["exact_match_label"] = exact_match_label(info["answer"], question.answer)
        checks["exact_match"] = checks["exact_match_label"] == "correct"
    return checks


def judge_pointwise(judge, question: GoldQuestion, answer: str, contexts: Sequence[str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    verdict = judge.judge_answer(question.question, answer, contexts, question.answer)
    return {**verdict.as_dict(), "seconds": round(time.perf_counter() - t0, 3)}


def run(
    *,
    root: Path | None = None,
    config: BenchConfig = DEFAULT_CONFIG,
    generator_model: str = "qwen2.5:7b",
    judge_models: Sequence[str] = DEFAULT_JUDGE_MODELS,
    limit: int | None = None,
    base_url: str | None = None,
    complete_fn=None,
) -> dict[str, Any]:
    """Score generated answers. `complete_fn` (prompt -> text) replaces Ollama for tests."""
    root = root or project_root()
    files_root = root / "bench" / "corpus" / "files"
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    if verify_manifest(files_root, manifest):
        raise RuntimeError("corpus drift")
    if limit:
        gold = gold[:limit]

    index = build_index(
        files_root,
        embedder_name=config.embedder,
        retrieve_k=config.retrieve_k,
        rerank_n=config.rerank_n,
        enable_router=config.enable_router,
        enable_rerank=config.enable_rerank,
        reranker=config.reranker,
        chunk_strategy=config.chunk_strategy,
    )
    clusters = cluster_versions(index.chunks) if config.staleness_tier1 else {}

    if complete_fn is not None:
        generator = OllamaGenerator(model=generator_model, base_url=base_url)
        generator.complete = complete_fn  # type: ignore[method-assign]
        judges = [(name, LlmJudge(complete=complete_fn, name=name)) for name in judge_models]
        pair_fns = {name: complete_fn for name in judge_models}
    else:
        available = ollama_available(base_url)
        missing = [m for m in {generator_model, *judge_models} if m not in available]
        if missing:
            raise SystemExit(f"Ollama models not available: {missing}; have {available}")
        generator = OllamaGenerator(model=generator_model, base_url=base_url)
        judges = [(name, LlmJudge(model=name, name=name)) for name in judge_models]
        pair_fns = {name: (lambda prompt, _m=name: ollama_complete(prompt, model=_m) or "") for name in judge_models}
    heuristic = HeuristicJudge()

    rows: list[dict[str, Any]] = []
    timer = StageTimer()
    t_start = time.perf_counter()
    # Pass 1: retrieve + generate every answer (one model resident), then deterministic checks.
    work: list[tuple[GoldQuestion, dict[str, Any], list[str], str, str]] = []
    for q in gold:
        paths, payload = _retrieve(index, q.question, config, StageTimer(), clusters, None, category=q.category)
        hits = list(payload.hits[:TOP_N]) if hasattr(payload, "hits") else list(payload[:TOP_N])
        contexts = [h.chunk.text for h in hits]
        with timer.stage("generate"):
            text = generator(q.question, hits)
        gen_meta = generator.last_meta.__dict__ if generator.last_meta else {}
        checks = deterministic_checks(q, text, hits)
        row: dict[str, Any] = {
            "id": q.id,
            "category": q.category,
            "question": q.question,
            "gold": q.answer,
            "retrieved_paths": paths[:TOP_N],
            "answer": text,
            "generation": gen_meta,
            "checks": checks,
            "judges": {},
            "pairwise_llm_vs_extractive": {},
        }
        rows.append(row)
        if checks["uncited"]:
            row["failed"] = "uncited_answer"
            continue
        judged_text = cited_answer(text, hits)["answer"]
        row["judges"]["heuristic"] = judge_pointwise(heuristic, q, judged_text, contexts)
        work.append((q, row, contexts, judged_text, extractive_answer(q.question, hits)))
    # Pass 2: one judge model at a time over every answer (pointwise, then AB/BA), so a
    # model is loaded once rather than swapped per question.
    for name, judge in judges:
        for q, row, contexts, judged_text, extractive in work:
            with timer.stage(f"judge:{name}"):
                row["judges"][name] = judge_pointwise(judge, q, judged_text, contexts)
        for q, row, contexts, judged_text, extractive in work:
            with timer.stage(f"pairwise:{name}"):
                row["pairwise_llm_vs_extractive"][name] = pairwise_ab_ba(pair_fns[name], q.question, judged_text, extractive)

    wall_s = round(time.perf_counter() - t_start, 1)
    summary = summarize(rows, judge_models)
    return {
        "git_sha": git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info("auto"),
        "corpus_sha256": manifest["aggregate_sha256"],
        "n_gold": len(gold),
        "config": config.__dict__,
        "models": {
            "generator": model_provenance(generator_model, "generator") | {"backend": "ollama", "temperature": 0.0, "seed": 0},
            "judges": [model_provenance(m, "judge") | {"backend": "ollama"} for m in judge_models],
            "embedder": model_provenance(config.embedder, "embedder"),
            "reranker": model_provenance(config.reranker, "reranker") | {"loaded": bool(getattr(index, "reranker_loaded", False))},
        },
        "kappa_gate": KAPPA_GATE,
        "latency": timer.summary(),
        "wall_s": wall_s,
        "summary": summary,
        "rows": rows,
    }


def summarize(rows: list[dict[str, Any]], judge_models: Sequence[str]) -> dict[str, Any]:
    n = len(rows)
    judged = [r for r in rows if not r.get("failed")]
    sf = [r for r in judged if r["category"] == "simple_factual"]

    def rate(items, key):
        vals = [bool(r["checks"].get(key)) for r in items if key in r["checks"]]
        return {"rate": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}

    checks = {
        "uncited_answers": {"count": sum(1 for r in rows if r.get("failed") == "uncited_answer"), "n": n},
        "abstained": {"count": sum(1 for r in rows if r["checks"].get("abstained")), "n": n},
        "cited_in_retrieved": rate(judged, "cited_in_retrieved"),
        "values_in_cited_bytes": rate([r for r in judged if r["checks"].get("n_values")], "values_in_cited_bytes"),
        "values_in_any_evidence": rate([r for r in judged if r["checks"].get("n_values")], "values_in_any_evidence"),
        "exact_match_simple_factual": rate(sf, "exact_match"),
    }
    gold_labels = [r["checks"]["exact_match_label"] for r in sf]
    kappa: dict[str, Any] = {}
    members = ["heuristic", *judge_models]
    for name in members:
        labels = [r["judges"][name]["label"] for r in sf if name in r["judges"]]
        k = cohens_kappa(gold_labels[: len(labels)], labels)
        scores = [label_to_score(r["judges"][name]["label"]) for r in judged if name in r["judges"]]
        kappa[name] = {
            "kappa_vs_exact_match_simple_factual": round(k, 4),
            "n": len(labels),
            "readme_ok": k >= KAPPA_GATE,
            "mean_score_all": round(sum(scores) / len(scores), 4) if scores else None,
            "label_counts": {lab: labels.count(lab) for lab in ("correct", "partial", "wrong")},
            "heuristic": name == "heuristic",
        }
    # jury majority over the LLM judges only (heuristic stays a labelled column)
    maj = []
    for r in sf:
        labs = [r["judges"][m]["label"] for m in judge_models if m in r["judges"]]
        if not labs:
            continue
        counts = {lab: labs.count(lab) for lab in set(labs)}
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        maj.append("partial" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0])
    k_j = cohens_kappa(gold_labels[: len(maj)], maj)
    kappa["jury_majority"] = {"members": list(judge_models), "kappa_vs_exact_match_simple_factual": round(k_j, 4), "n": len(maj), "readme_ok": k_j >= KAPPA_GATE}
    agreement = None
    if len(judge_models) >= 2:
        a, b = judge_models[0], judge_models[1]
        pairs = [(r["judges"][a]["label"], r["judges"][b]["label"]) for r in judged if a in r["judges"] and b in r["judges"]]
        if pairs:
            agreement = {"judges": [a, b], "kappa": round(cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs]), 4), "n": len(pairs)}
    pairwise = {}
    for name in judge_models:
        ps = [r["pairwise_llm_vs_extractive"][name] for r in judged if name in r["pairwise_llm_vs_extractive"]]
        if ps:
            pairwise[name] = {
                "p_llm_better_mean": round(sum(p["p_a"] for p in ps) / len(ps), 4),
                "position_gap_mean": round(sum(p["position_gap"] for p in ps) / len(ps), 4),
                "position_flips": sum(1 for p in ps if p["p_a_ab"] != p["p_a_ba"]),
                "n": len(ps),
            }
    publishable = [m for m in judge_models if kappa[m]["readme_ok"]]
    return {
        "n": n,
        "n_judged": len(judged),
        "n_simple_factual": len(sf),
        "checks": checks,
        "kappa": kappa,
        "judge_agreement": agreement,
        "pairwise_llm_vs_extractive": pairwise,
        "publishable_judges": publishable,
        "verdict": (
            f"kappa >= {KAPPA_GATE} for {', '.join(publishable)}: their scores may be published"
            if publishable
            else f"no judge reached kappa >= {KAPPA_GATE} vs exact-match labels on simple_factual; judge scores stay off the README"
        ),
    }


def markdown_table(blob: dict) -> str:
    s = blob["summary"]
    lines = [
        f"LLM answers on fixture v1 ({blob['n_gold']} questions), generator {blob['models']['generator']['model']}, "
        f"judges {', '.join(j['model'] for j in blob['models']['judges'])}; wall {blob['wall_s']} s.",
        "",
        "| deterministic check | rate | n |",
        "|---|---:|---:|",
    ]
    c = s["checks"]
    lines.append(f"| uncited answers (failed loud) | {c['uncited_answers']['count']} | {c['uncited_answers']['n']} |")
    lines.append(f"| abstained (NOT IN SOURCES) | {c['abstained']['count']} | {c['abstained']['n']} |")
    for key in ("cited_in_retrieved", "values_in_cited_bytes", "values_in_any_evidence", "exact_match_simple_factual"):
        lines.append(f"| {key} | {c[key]['rate']} | {c[key]['n']} |")
    lines += ["", "| judge | kappa vs exact-match (simple_factual) | n | readme_ok | mean score (all) | labels c/p/w |", "|---|---:|---:|:---:|---:|---|"]
    for name, k in s["kappa"].items():
        if name == "jury_majority":
            lines.append(f"| jury majority ({', '.join(k['members'])}) | {k['kappa_vs_exact_match_simple_factual']} | {k['n']} | {k['readme_ok']} | — | — |")
        else:
            lc = k["label_counts"]
            lines.append(f"| {name}{' (heuristic)' if k['heuristic'] else ''} | {k['kappa_vs_exact_match_simple_factual']} | {k['n']} | {k['readme_ok']} | {k['mean_score_all']} | {lc['correct']}/{lc['partial']}/{lc['wrong']} |")
    if s.get("judge_agreement"):
        a = s["judge_agreement"]
        lines.append(f"\nJudge-judge agreement ({a['judges'][0]} vs {a['judges'][1]}): kappa {a['kappa']}, n {a['n']}.")
    for name, p in s["pairwise_llm_vs_extractive"].items():
        lines.append(f"Pairwise AB/BA, {name}: P(LLM answer better than extractive) {p['p_llm_better_mean']}, position gap {p['position_gap_mean']}, flips {p['position_flips']}/{p['n']}.")
    lines.append(f"\nVerdict: {s['verdict']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=root / "bench" / "results" / "jury.json")
    p.add_argument("--generator", default=os.environ.get("GENERATOR_MODEL", "qwen2.5:7b"))
    p.add_argument("--judges", default=os.environ.get("JUDGE_MODELS", ",".join(DEFAULT_JUDGE_MODELS)))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--embedder", default=DEFAULT_CONFIG.embedder)
    p.add_argument("--reranker", default=DEFAULT_CONFIG.reranker)
    p.add_argument("--no-router", action="store_true")
    p.add_argument("--no-staleness", action="store_true")
    args = p.parse_args(argv)
    config = replace(
        DEFAULT_CONFIG,
        embedder=args.embedder,
        reranker=args.reranker,
        enable_rerank=args.reranker.lower() not in {"none", ""},
        enable_router=not args.no_router,
        staleness_tier1=not args.no_staleness,
        name=f"{'hybrid'}{'+' + ('bge-rerank' if 'bge' in args.reranker else args.reranker) if args.reranker.lower() != 'none' else ''}"
        f"{'' if args.no_router else '+router'}{'' if args.no_staleness else '+staleness'}@{args.embedder.split(':')[-1].split('/')[-1]}",
    )
    judges = [m.strip() for m in args.judges.split(",") if m.strip()]
    blob = run(root=root, config=config, generator_model=args.generator, judge_models=judges, limit=args.limit)
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    print(markdown_table(blob))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
