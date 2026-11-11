"""LLM answer evaluation with a validity gate (M4; reference and gate revised in M29).

    python -m app.eval.llm_eval --out bench/results/jury.json \
        [--generator qwen2.5:7b] [--judges qwen2.5:7b,llama3.1:8b] [--limit N] \
        [--embedder st:BAAI/bge-small-en-v1.5 --reranker BAAI/bge-reranker-v2-m3]

For every gold question, on fixture v1:
1. Retrieve with the chosen configuration (default: the full pipeline with real
   models, the best row of bench-neural) and keep the top 8 chunks. A run
   refuses to start when the configured reranker falls back.
2. Generate a cited answer with a local model through Ollama
   (app/agent/generate.py). An answer with no valid citation is a failure
   ("uncited"), counted and excluded from judging; a NOT IN SOURCES reply is an
   abstention.
3. Deterministic checks, no model involved:
   - cited_in_retrieved: every cited path is in the retrieved set (None for an
     answer with no citation, e.g. an abstention, so it is left out of the rate)
   - values_in_cited_bytes: every number / config value in the answer appears
     inside the cited chunks (app/agent/citation_verify.py, cited_only=True;
     string containment, case-insensitive, numeric formats not normalised)
   - exact_match: the gold answer is in the answer and not part of a longer
     number or word, or the answer states the same number in another format
     (simple_factual only; exact_match_label(bounded=True))
   - abstention_only: the answer is NOT IN SOURCES and nothing else
4. Jury, one judge model at a time:
   - correctness verdict (the gated label): question, gold answer and answer,
     no retrieved context (judge.correctness_prompt). The abstention rubric
     labels a pure abstention to a question with a gold answer wrong without
     asking the judge.
   - composite label, kept as a labelled column from M4: pointwise
     faithfulness / relevancy / groundedness -> label. Never gated: a wrong
     value copied faithfully from a retrieved chunk scores about 1, 1, 0, which
     reads as partial.
   - pairwise AB/BA position swap between the LLM answer and the extractive
     answer, so position bias is measured (position_gap).
   The heuristic judge is kept as a labelled composite column. Every reply is
   stored; empty and unreadable replies are counted per judge.
5. Cohen's kappa between each judge's correctness verdicts and the exact-match
   labels on the simple_factual answers it labelled (rule-labelled abstentions
   are left out and counted), paired per question, with a bootstrap interval.
   A judge's correctness score may be published only when kappa >= KAPPA_GATE
   (0.6); the results file says so either way (readme_ok). If the reference is
   a single class (every answer matched, or none did) kappa is 0 by
   construction for any judge with variance and 1 for a judge that never
   varies, so the gate is marked not evaluable and stays closed.

Heuristic columns are labelled "heuristic" and stay out of README tables.

The header pins every Ollama model by the manifest digest the server's
/api/tags reports for its tag (`revision`), with the server version; a model
run refuses to start when a tag has no digest.

    python -m app.eval.llm_eval --rescore bench/results/jury.json

recomputes the deterministic checks and the summary from the stored answers
and judge labels (no model call), after re-running retrieval and checking that
every question's retrieved paths equal the stored ones; the file records both
the sha that generated the rows and the sha that rescored them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.agent.citation_verify import extract_claims, verify_claims
from app.agent.generate import NOT_IN_SOURCES, OllamaGenerator, cited_answer
from app.agent.retrieval_loop import extractive_answer
from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import BGE_RERANKER, BGE_SMALL, BenchConfig, _retrieve, bench_settings, project_root
from app.eval.index_loader import build_index
from app.eval.judge import LABELS, HeuristicJudge, LlmJudge, correctness_prompt, exact_match_label, label_to_score, ollama_complete, pairwise_ab_ba
from app.eval.jury import KAPPA_GATE, DEFAULT_JUDGE_MODELS, cohens_kappa
from app.eval.latency import StageTimer
from app.eval.provenance import (
    command_line,
    device_info,
    git_sha,
    model_provenance,
    ollama_model_provenance,
    ollama_tag_entry,
    ollama_tags,
    ollama_version,
    scrub,
)
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, RNG_NAME, resample_indices
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
# The label the gate reads, and the rule that labels a pure abstention without asking a judge.
GATED_LABEL = "correctness verdict (app/eval/judge.py correctness_prompt: question, gold, answer; no retrieved context)"
REFERENCE_LABELS = ("exact_match_label, bounded (the gold may not sit inside a longer number or word), on the "
                    "simple_factual answers the judges labelled")
ABSTENTION_RULE = (f"an answer that is {NOT_IN_SOURCES} and nothing else, to a question with a gold answer, is wrong "
                   "by rule: no judge is asked, and the row is left out of kappa")


def abstention_only(answer: str) -> bool:
    """The answer (citations stripped) says NOT IN SOURCES and nothing else."""
    return NOT_IN_SOURCES in answer and not re.search(r"[A-Za-z0-9]", answer.replace(NOT_IN_SOURCES, " "))


def rule_label(checks: dict[str, Any], gold: str) -> str | None:
    """The abstention rubric: "wrong" for a pure abstention to a question that has a gold answer, else None (ask)."""
    return "wrong" if checks.get("abstention_only") and str(gold or "").strip() else None


def deterministic_checks(question: GoldQuestion, text: str, hits: Sequence[RetrievalHit]) -> dict[str, Any]:
    info = cited_answer(text, hits)
    chunks = [h.chunk for h in hits]
    claims = extract_claims(info["answer"])
    strict = verify_claims(claims, chunks, citations=info["citations"], cited_only=True)
    loose = verify_claims(extract_claims(info["answer"]), chunks)
    checks = {
        "uncited": info["uncited"],
        "abstained": info["abstained"],
        "abstention_only": info["abstained"] and abstention_only(info["answer"]),
        "cited_in_retrieved": (not info["unknown_citations"]) if info["citations"] else None,
        "n_citations": len(info["citations"]),
        "unknown_citations": info["unknown_citations"],
        "n_values": sum(1 for c in claims if c.value),
        "values_in_cited_bytes": not strict.hallucinated_values,
        "values_not_in_cited_bytes": strict.hallucinated_values,
        "values_in_any_evidence": not loose.hallucinated_values,
        "verification_ratio_cited": round(strict.verification_ratio, 4),
    }
    if question.category == "simple_factual":
        checks["exact_match_label"] = exact_match_label(info["answer"], question.answer, bounded=True)
        checks["exact_match"] = checks["exact_match_label"] == "correct"
    return checks


def judge_pointwise(judge, question: GoldQuestion, answer: str, contexts: Sequence[str]) -> dict[str, Any]:
    t0 = time.perf_counter()
    verdict = judge.judge_answer(question.question, answer, contexts, question.answer)
    out = {**verdict.as_dict(), "seconds": round(time.perf_counter() - t0, 3)}
    if isinstance(judge, LlmJudge):
        out["raw"] = verdict.raw  # the three replies, so an empty or unreadable one is visible
    return out


def judge_correctness(judge: LlmJudge, question: GoldQuestion, answer: str) -> dict[str, Any]:
    """The gated label from the judge (run() applies the abstention rule first, without a model call)."""
    t0 = time.perf_counter()
    verdict = judge.judge_correctness(question.question, answer, question.answer)
    return {**verdict, "by": "judge", "seconds": round(time.perf_counter() - t0, 3)}


def _recording(complete):
    """`complete` that also keeps every reply, in call order."""
    replies: list[str] = []

    def call(prompt: str) -> str:
        reply = complete(prompt) or ""
        replies.append(reply)
        return reply

    return call, replies


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
    if getattr(index, "reranker_fallback", None):  # answers would be generated from a different retrieval than the config names
        raise SystemExit(f"the configured reranker {config.reranker} did not load (fallback {index.reranker_fallback}); refusing to record answers")
    clusters = cluster_versions(index.chunks) if config.staleness_tier1 else {}

    ollama_models = list(dict.fromkeys([generator_model, *judge_models]))
    if complete_fn is not None:
        generator = OllamaGenerator(model=generator_model, base_url=base_url)
        generator.complete = complete_fn  # type: ignore[method-assign]
        judges = [(name, LlmJudge(complete=complete_fn, name=name)) for name in judge_models]
        pair_fns = {name: complete_fn for name in judge_models}
        tags = ollama_tags(base_url) if base_url else {}  # a test may point base_url at a fake /api/tags
    else:
        tags = ollama_tags(base_url)
        missing = [m for m in ollama_models if not ollama_tag_entry(m, tags)]
        if missing:
            raise SystemExit(f"Ollama models not available: {missing}; have {sorted(tags)}")
        undigested = [m for m in ollama_models if not ollama_tag_entry(m, tags).get("digest")]
        if undigested:  # the results would name these weights by tag only
            raise SystemExit(f"Ollama lists no digest for {undigested}; refusing to record weights by tag alone")
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
        paths, payload = _retrieve(index, q.question, config, StageTimer(), clusters, None)
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
            "correctness": {},
            "pairwise_llm_vs_extractive": {},
        }
        rows.append(row)
        if checks["uncited"]:
            row["failed"] = "uncited_answer"
            continue
        judged_text = cited_answer(text, hits)["answer"]
        row["judges"]["heuristic"] = judge_pointwise(heuristic, q, judged_text, contexts)
        work.append((q, row, contexts, judged_text, extractive_answer(q.question, hits)))
    # Pass 2: one judge model at a time over every answer (pointwise, correctness, then AB/BA), so a
    # model is loaded once rather than swapped per question.
    for name, judge in judges:
        for q, row, contexts, judged_text, extractive in work:
            with timer.stage(f"judge:{name}"):
                row["judges"][name] = judge_pointwise(judge, q, judged_text, contexts)
        for q, row, contexts, judged_text, extractive in work:
            ruled = rule_label(row["checks"], q.answer)
            if ruled:  # the abstention rubric: no model call
                row["correctness"][name] = {"label": ruled, "by": "rule"}
                continue
            with timer.stage(f"correctness:{name}"):
                row["correctness"][name] = judge_correctness(judge, q, judged_text)
        for q, row, contexts, judged_text, extractive in work:
            with timer.stage(f"pairwise:{name}"):
                complete, replies = _recording(pair_fns[name])
                row["pairwise_llm_vs_extractive"][name] = pairwise_ab_ba(complete, q.question, judged_text, extractive) | {"raw": replies}

    wall_s = round(time.perf_counter() - t_start, 1)
    summary = summarize(rows, judge_models)
    return {
        "git_sha": git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info("auto"),
        "corpus_sha256": manifest["aggregate_sha256"],
        "n_gold": len(gold),
        "config": bench_settings(config),
        "models": {
            "generator": ollama_model_provenance(generator_model, "generator", tags) | {"temperature": 0.0, "seed": 0},
            "judges": [ollama_model_provenance(m, "judge", tags) for m in judge_models],
            "embedder": model_provenance(config.embedder, "embedder"),
            "reranker": model_provenance(config.reranker, "reranker") | {"loaded": bool(getattr(index, "reranker_loaded", False))},
        },
        "ollama": {"version": ollama_version(base_url) if tags else None},
        "kappa_gate": KAPPA_GATE,
        "protocol": {
            "gated_label": GATED_LABEL,
            "correctness_prompt": correctness_prompt("{question}", "{answer}", "{gold}"),
            "reference": REFERENCE_LABELS,
            "abstention_rule": ABSTENTION_RULE,
        },
        "latency": timer.summary(),
        "wall_s": wall_s,
        "summary": summary,
        "rows": rows,
    }


def _counts(labels: Sequence[str]) -> dict[str, int]:
    return {lab: list(labels).count(lab) for lab in LABELS}


def _confusion(pairs: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Rows: the reference label; columns: the judge's label."""
    return {ref: {lab: sum(1 for a, b in pairs if a == ref and b == lab) for lab in LABELS} for ref in LABELS}


def _majority(labels: Sequence[str]) -> str:
    top = Counter(labels).most_common()
    return "partial" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]


def kappa_interval(pairs: Sequence[tuple[str, str]], n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict[str, float] | None:
    """Percentile bootstrap interval of Cohen's kappa over the questions (information: the gate reads the point value)."""
    if len(pairs) < 2:
        return None
    ref, lab = [a for a, _ in pairs], [b for _, b in pairs]
    ks = [cohens_kappa([ref[i] for i in row], [lab[i] for i in row]) for row in resample_indices(len(pairs), n_boot, seed)]
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return {"lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def _ruled(row: dict[str, Any]) -> bool:
    return rule_label(row["checks"], row.get("gold", "")) is not None


def summarize(rows: list[dict[str, Any]], judge_models: Sequence[str]) -> dict[str, Any]:
    n = len(rows)
    judged = [r for r in rows if not r.get("failed")]
    sf = [r for r in judged if r["category"] == "simple_factual"]

    def rate(items, key):
        vals = [bool(r["checks"][key]) for r in items if r["checks"].get(key) is not None]
        return {"rate": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}

    checks = {
        "uncited_answers": {"count": sum(1 for r in rows if r.get("failed") == "uncited_answer"), "n": n},
        "abstained": {"count": sum(1 for r in rows if r["checks"].get("abstained")), "n": n},
        "abstention_only": {"count": sum(1 for r in rows if r["checks"].get("abstention_only")), "n": n},
        "cited_in_retrieved": rate(judged, "cited_in_retrieved"),
        "values_in_cited_bytes": rate([r for r in judged if r["checks"].get("n_values")], "values_in_cited_bytes"),
        "values_in_any_evidence": rate([r for r in judged if r["checks"].get("n_values")], "values_in_any_evidence"),
        "exact_match_simple_factual": rate(sf, "exact_match"),
    }
    # The reference: exact-match labels of the simple-factual answers a judge labelled. A pure abstention is labelled
    # wrong by rule for every judge, so it would agree with the reference by construction: it is left out and counted.
    ref_rows = [r for r in sf if not _ruled(r)]
    ref_labels = [r["checks"]["exact_match_label"] for r in ref_rows]
    ref_counts = _counts(ref_labels)
    # A one-class reference makes kappa meaningless: 0 for any judge with variance, 1 for a
    # judge that never varies (cohens_kappa's pe == 1 branch). The gate cannot be evaluated.
    single_class = bool(ref_labels) and len(set(ref_labels)) == 1
    reference = {"labels": REFERENCE_LABELS, "n": len(ref_labels), "label_counts": ref_counts, "single_class": single_class,
                 "gate_evaluable": bool(ref_labels) and not single_class, "left_out_abstention_only": len(sf) - len(ref_rows),
                 "label_counts_with_abstentions": _counts([r["checks"]["exact_match_label"] for r in sf])}
    kappa: dict[str, Any] = {}
    for name in judge_models:
        pairs = [(r["checks"]["exact_match_label"], r["correctness"][name]["label"]) for r in ref_rows if name in r.get("correctness", {})]
        k = cohens_kappa([a for a, _ in pairs], [b for _, b in pairs])
        every = [r["correctness"][name]["label"] for r in judged if name in r.get("correctness", {})]
        kappa[name] = {
            "kappa_vs_exact_match_simple_factual": round(k, 4),
            "interval": kappa_interval(pairs),
            "n": len(pairs),
            "readme_ok": reference["gate_evaluable"] and bool(pairs) and k >= KAPPA_GATE,
            "label_counts": _counts([b for _, b in pairs]),
            "confusion_reference_by_judge": _confusion(pairs),
            "mean_correctness_all": round(sum(label_to_score(lab) for lab in every) / len(every), 4) if every else None,
            "n_all": len(every),
            "heuristic": False,
        }
    maj = []
    for r in ref_rows:
        labs = [r["correctness"][m]["label"] for m in judge_models if m in r.get("correctness", {})]
        if labs:
            maj.append((r["checks"]["exact_match_label"], _majority(labs)))
    k_j = cohens_kappa([a for a, _ in maj], [b for _, b in maj])
    kappa["jury_majority"] = {"members": list(judge_models), "kappa_vs_exact_match_simple_factual": round(k_j, 4),
                              "interval": kappa_interval(maj), "n": len(maj),
                              "readme_ok": reference["gate_evaluable"] and bool(maj) and k_j >= KAPPA_GATE}
    # The M4 label (mean of faithfulness, relevancy and groundedness), kept as a labelled column: never gated or
    # published, because a wrong value copied faithfully from a retrieved chunk scores about 1, 1, 0 -> partial.
    composite: dict[str, Any] = {}
    for name in ["heuristic", *judge_models]:
        pairs = [(r["checks"]["exact_match_label"], r["judges"][name]["label"]) for r in ref_rows if name in r["judges"]]
        scores = [label_to_score(r["judges"][name]["label"]) for r in judged if name in r["judges"]]
        composite[name] = {
            "kappa_vs_exact_match_simple_factual": round(cohens_kappa([a for a, _ in pairs], [b for _, b in pairs]), 4),
            "n": len(pairs),
            "label_counts": _counts([b for _, b in pairs]),
            "mean_score_all": round(sum(scores) / len(scores), 4) if scores else None,
            "abstentions_labelled_correct": [r["id"] for r in judged if r["checks"].get("abstention_only")
                                             and (r["judges"].get(name) or {}).get("label") == "correct"],
            "heuristic": name == "heuristic",
        }
    ruled = [r for r in judged if _ruled(r)]
    abstentions = {"rule": ABSTENTION_RULE, "label": "wrong", "n": len(ruled), "ids": [r["id"] for r in ruled]}
    agreement = None
    if len(judge_models) >= 2:
        a, b = judge_models[0], judge_models[1]
        asked = [r for r in judged if (r.get("correctness", {}).get(a) or {}).get("by") == "judge"
                 and (r.get("correctness", {}).get(b) or {}).get("by") == "judge"]
        if asked:
            la, lb = [r["correctness"][a]["label"] for r in asked], [r["correctness"][b]["label"] for r in asked]
            agreement = {"judges": [a, b], "labels": "correctness verdicts", "kappa": round(cohens_kappa(la, lb), 4), "n": len(asked)}
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
    replies = {}
    for name in judge_models:
        point = [txt for r in judged for txt in ((r["judges"].get(name) or {}).get("raw") or {}).values()]
        asked = [r["correctness"][name] for r in judged if (r.get("correctness", {}).get(name) or {}).get("by") == "judge"]
        pair = [txt for r in judged for txt in ((r["pairwise_llm_vs_extractive"].get(name) or {}).get("raw") or [])]
        replies[name] = {
            "pointwise": {"n": len(point), "empty": sum(1 for txt in point if not (txt or "").strip())},
            "correctness": {"n": len(asked), "empty": sum(1 for v in asked if not (v.get("raw") or "").strip()),
                            "unparsed": sum(1 for v in asked if not v.get("parsed"))},
            "pairwise": {"n": len(pair), "empty": sum(1 for txt in pair if not (txt or "").strip())},
        }
    publishable = [m for m in judge_models if kappa[m]["readme_ok"]]
    if not reference["gate_evaluable"]:
        counts = ", ".join(f"{v} {k}" for k, v in ref_counts.items())
        verdict = (f"gate not evaluable: the exact-match reference on the judge-labelled simple_factual answers is "
                   f"{'one class' if single_class else 'empty'} ({counts} of {len(ref_labels)}), so Cohen's kappa is 0 by construction "
                   f"for any judge with variance and 1 for a judge that never varies; judge scores stay off the README")
    elif publishable:
        verdict = f"kappa >= {KAPPA_GATE} for {', '.join(publishable)}: their correctness scores may be published"
    else:
        verdict = f"no judge reached kappa >= {KAPPA_GATE} vs exact-match labels on simple_factual; judge scores stay off the README"
    return {
        "n": n,
        "n_judged": len(judged),
        "n_simple_factual": len(sf),
        "checks": checks,
        "gated_label": GATED_LABEL,
        "reference": reference,
        "kappa": kappa,
        "interval": {"method": "percentile bootstrap over the reference questions", "rng": RNG_NAME, "seed": DEFAULT_SEED,
                     "n_boot": DEFAULT_N_BOOT, "level": 0.95, "gate_reads": "the point estimate"},
        "composite": composite,
        "abstention_rule": abstentions,
        "replies": replies,
        "judge_agreement": agreement,
        "pairwise_llm_vs_extractive": pairwise,
        "publishable_judges": publishable,
        "verdict": verdict,
    }


def rescore(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Recompute the deterministic checks and the summary of an existing jury file.

    Generation, judge labels and pairwise verdicts are kept as stored. Retrieval is
    re-run with the stored configuration and must return the same paths for every
    question, and the reranker must load as it did; otherwise nothing is written.
    """
    root = root or project_root()
    blob = json.loads(path.read_text())
    files_root = root / "bench" / "corpus" / "files"
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    if verify_manifest(files_root, manifest):
        raise RuntimeError("corpus drift")
    if manifest["aggregate_sha256"] != blob["corpus_sha256"]:
        raise SystemExit("the file was produced on a different corpus; re-run make bench-jury")
    gold = {q.id: q for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    config = BenchConfig(**blob["config"])
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
    loaded = bool(getattr(index, "reranker_loaded", False))
    if loaded != bool(blob["models"]["reranker"].get("loaded")):
        raise SystemExit(f"reranker loaded={loaded} now, {blob['models']['reranker'].get('loaded')} in the file; re-run make bench-jury")
    clusters = cluster_versions(index.chunks) if config.staleness_tier1 else {}
    mismatched: list[str] = []
    new_checks: dict[str, dict[str, Any]] = {}
    for row in blob["rows"]:
        q = gold[row["id"]]
        paths, payload = _retrieve(index, q.question, config, StageTimer(), clusters, None)
        hits = list(payload.hits[:TOP_N]) if hasattr(payload, "hits") else list(payload[:TOP_N])
        if paths[:TOP_N] != row["retrieved_paths"]:
            mismatched.append(row["id"])
            continue
        checks = deterministic_checks(q, row["answer"], hits)
        old = row["checks"]
        if checks["uncited"] != old["uncited"] or checks["abstained"] != old["abstained"]:
            raise SystemExit(f"{row['id']}: the judged set would change (uncited/abstained); re-run make bench-jury")
        if "abstention_only" in old and checks["abstention_only"] != old["abstention_only"]:  # files from M29 on
            raise SystemExit(f"{row['id']}: the rows the abstention rule labels would change; re-run the answer evaluation")
        new_checks[row["id"]] = checks
    if mismatched:
        raise SystemExit(f"retrieval differs from the stored paths for {mismatched}; re-run make bench-jury")
    for row in blob["rows"]:
        row["checks"] = new_checks[row["id"]]
    judge_models = [j["model"] for j in blob["models"]["judges"]]
    blob["summary"] = summarize(blob["rows"], judge_models)
    prior = blob.get("rescore") or {}
    blob["rescore"] = {
        "rows_git_sha": prior.get("rows_git_sha") or blob["git_sha"],
        "rows_command": scrub(prior.get("rows_command") or blob["command"]),  # carried over, published again
        "rows_timestamp": prior.get("rows_timestamp") or blob["timestamp"],
        "checks_git_sha": git_sha(root),
        "command": command_line(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "what": "deterministic checks and summary recomputed from the stored answers and judge labels; generation, judge labels and pairwise verdicts untouched",
    }
    blob["git_sha"] = blob["rescore"]["checks_git_sha"]
    blob["command"] = blob["rescore"]["command"]
    if isinstance(blob.get("device"), dict):  # the generating run's machine, written out again: no host name, no paths
        blob["device"] = scrub({k: v for k, v in blob["device"].items() if k != "hostname"})
    return blob


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
    lines.append(f"| abstention only (labelled wrong by rule, no judge) | {c['abstention_only']['count']} | {c['abstention_only']['n']} |")
    ref = s["reference"]
    lines += ["", f"Reference: {ref['labels']}; n {ref['n']}, labels c/p/w {ref['label_counts']['correct']}/{ref['label_counts']['partial']}/"
              f"{ref['label_counts']['wrong']}, {ref['left_out_abstention_only']} abstentions left out.", "",
              "| judge (correctness verdict) | kappa vs exact-match (simple_factual) | interval | n | readme_ok | mean correctness (all) | labels c/p/w |",
              "|---|---:|---|---:|:---:|---:|---|"]
    for name, k in s["kappa"].items():
        ci = k["interval"] or {}
        if name == "jury_majority":
            lines.append(f"| jury majority ({', '.join(k['members'])}) | {k['kappa_vs_exact_match_simple_factual']} | [{ci.get('lo')}, {ci.get('hi')}] | {k['n']} | {k['readme_ok']} | — | — |")
        else:
            lc = k["label_counts"]
            lines.append(f"| {name} | {k['kappa_vs_exact_match_simple_factual']} | [{ci.get('lo')}, {ci.get('hi')}] | {k['n']} | {k['readme_ok']} | {k['mean_correctness_all']} | {lc['correct']}/{lc['partial']}/{lc['wrong']} |")
    lines += ["", "| judge (composite label, never gated) | kappa vs exact-match | n | mean score (all) | abstentions labelled correct |", "|---|---:|---:|---:|---|"]
    for name, k in s["composite"].items():
        lines.append(f"| {name}{' (heuristic)' if k['heuristic'] else ''} | {k['kappa_vs_exact_match_simple_factual']} | {k['n']} | {k['mean_score_all']} | {', '.join(k['abstentions_labelled_correct']) or '-'} |")
    for name, r in s["replies"].items():
        lines.append(f"Replies, {name}: pointwise empty {r['pointwise']['empty']}/{r['pointwise']['n']}, correctness empty {r['correctness']['empty']} "
                     f"unparsed {r['correctness']['unparsed']}/{r['correctness']['n']}, pairwise empty {r['pairwise']['empty']}/{r['pairwise']['n']}.")
    if s.get("judge_agreement"):
        a = s["judge_agreement"]
        lines.append(f"\nJudge-judge agreement ({a['judges'][0]} vs {a['judges'][1]}): kappa {a['kappa']}, n {a['n']}.")
    for name, p in s["pairwise_llm_vs_extractive"].items():
        lines.append(f"Pairwise AB/BA, {name}: P(LLM answer better than extractive) {p['p_llm_better_mean']}, position gap {p['position_gap_mean']}, flips {p['position_flips']}/{p['n']}.")
    if not ref.get("gate_evaluable", True):
        lines.append(f"\nReference is one class or empty ({ref['label_counts']}): the kappa gate is not evaluable on this run.")
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
    p.add_argument("--rescore", type=Path, default=None, metavar="FILE",
                   help="recompute the deterministic checks and summary of FILE from its stored answers (no model call) and write it back")
    args = p.parse_args(argv)
    if args.rescore is not None:
        target = args.rescore if args.rescore.is_absolute() else root / args.rescore
        blob = rescore(target, root)
        target.write_text(json.dumps(blob, indent=2) + "\n")
        print(markdown_table(blob))
        print(f"rescored {target} (rows from {blob['rescore']['rows_git_sha']}, checks at {blob['rescore']['checks_git_sha']})")
        return 0
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
