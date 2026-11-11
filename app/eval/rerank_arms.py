"""Two reranker prototypes as arms against the cross-encoder (make bench-rerank-arms).

Both rerank the same candidates: the fused top-k of hybrid retrieval over
bge-small on fixture v1, the list the cross-encoder itself reranks.

  distilled student  app/retrieval/distill.py. The cross-encoder scores every
                     candidate of the TRAIN questions; the student is fitted on
                     those scores with the features it will see when it serves
                     (no dense or BM25 score, the fused rank, never the teacher's
                     rank) and is scored on the TEST questions only. Its parent is
                     the cross-encoder it imitates, on the same test questions.
  listwise rerank    app/retrieval/rankgpt.py with a local model through Ollama:
                     one prompt per question over the top `window` candidates.
                     Every raw completion is stored with how it was read, so the
                     row can be rescored. A reply that is not a full permutation
                     of the window keeps the fused order; those questions are
                     counted and named, apart from calls that returned nothing.

Each arm gets a paired bootstrap delta with an interval against the
cross-encoder and against the un-reranked hybrid list, plus per-question rerank
latency. Nothing here falls back silently: a missing cross-encoder is an error,
the student's backend is recorded, and a failed LLM call is counted.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import BGE_RERANKER, BGE_SMALL, CI_METRICS, git_sha, project_root
from app.eval.index_loader import build_index, mtime_provenance
from app.eval.metrics import aggregate_retrieval
from app.eval.provenance import command_line, device_info, model_device, model_provenance
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, mean_ci, paired_delta_ci, resample_indices, settings as bootstrap_settings
from app.retrieval.distill import DistilledReranker, triples_from_pairs
from app.retrieval.rankgpt import PARSE_STATUSES, PASSAGE_CHARS, RankGPTReranker, listwise_prompt
from app.retrieval.rerank import CrossEncoderReranker

HYBRID, TEACHER, STUDENT = "hybrid@bge-small", "hybrid+bge-rerank@bge-small", "hybrid+distilled-student@bge-small"
OLLAMA_OPTIONS = {"temperature": 0.0, "seed": 0, "num_ctx": 8192, "num_predict": 200}


def split_questions(gold: list[GoldQuestion]) -> tuple[list[int], list[int]]:
    """Alternate train / test inside each category (gold order), so both halves cover every category."""
    seen: dict[str, int] = {}
    train, test = [], []
    for i, q in enumerate(gold):
        k = seen.get(q.category, 0)
        seen[q.category] = k + 1
        (train if k % 2 == 0 else test).append(i)
    return train, test


def _paths(pairs) -> list[str]:
    out: list[str] = []
    for chunk, _ in pairs:
        if chunk.path not in out:
            out.append(chunk.path)
    return out


def _ms(samples: list[float]) -> dict[str, float]:
    return {"p50_ms": round(float(np.percentile(samples, 50)), 3), "p95_ms": round(float(np.percentile(samples, 95)), 3), "n": len(samples)}


def ollama_complete(prompt: str, model: str, base_url: str) -> str | None:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m", "options": OLLAMA_OPTIONS}).encode()
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=float(os.environ.get("OLLAMA_TIMEOUT", "300"))) as resp:
            return json.loads(resp.read().decode()).get("response")
    except Exception:
        return None


def ollama_model_info(model: str, base_url: str) -> dict[str, Any]:
    """The server version and the digest of the weights Ollama served for `model` (None when it does not say)."""
    def get(route: str) -> dict:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + route, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return {}

    tags = [m for m in get("/api/tags").get("models", []) if model in (m.get("name"), m.get("model"))]
    return {"ollama_version": get("/api/version").get("version"), "digest": tags[0].get("digest") if tags else None}


def listwise_arm(gold: list[GoldQuestion], candidates: list[list[tuple[Any, float]]], complete: Callable[[str], str | None], *,
                 window: int) -> tuple[list[list[str]], list[float], dict[str, Any]]:
    """One listwise permutation per question over the top `window` candidates.

    Returns the ranked paths and the rerank latency (ms) per question, and a block that stores
    every raw completion with its call time and how it was read (app/retrieval/rankgpt.py
    read_permutation). A question keeps the fused order when its call returned nothing
    (n_failed_calls) or its reply is not a full permutation (n_parse_failures); both are named
    by question id, and a failed call is not also counted as a parse failure."""
    calls: list[tuple[str | None, float]] = []

    def recorded(prompt: str) -> str:
        t0 = time.perf_counter()
        text = complete(prompt)
        calls.append((text, (time.perf_counter() - t0) * 1000.0))
        return text or ""

    reranker = RankGPTReranker(complete=recorded, window=window)
    orders: list[list[str]] = []
    latency: list[float] = []
    entries: list[dict[str, Any]] = []
    for q, pairs in zip(gold, candidates):
        before = len(calls)
        t0 = time.perf_counter()
        ranked = reranker(q.question, pairs)
        latency.append((time.perf_counter() - t0) * 1000.0)
        orders.append(_paths(ranked))
        for text, ms in calls[before:]:
            entries.append({"id": q.id, "ms": round(ms, 1), "parse": reranker.last_parse.status, "completion": text})
    counts = {status: sum(1 for e in entries if e["parse"] == status) for status in PARSE_STATUSES}
    block = {
        "n_calls": len(entries),
        "n_failed_calls": counts["no_completion"],
        "n_parse_failures": sum(counts.values()) - counts["permutation"] - counts["no_completion"],
        "parse": counts,
        "parse_failure_ids": [e["id"] for e in entries if e["parse"] not in ("permutation", "no_completion")],
        "fused_order_kept_ids": [e["id"] for e in entries if e["parse"] != "permutation"],
        "completions": entries,
    }
    return orders, latency, block


def score_rows(orders: dict[str, list[list[str]]], gold: list[GoldQuestion], ids: list[int], latency: dict[str, list[float]],
               n_files: int, deltas: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], dict]:
    """Rows for the arms in `orders` over the questions `ids`, with paired deltas (arm, other) on those questions."""
    idx = resample_indices(len(ids), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    per_query: dict[str, list[dict[str, float]]] = {}
    rows = []
    for name, lists in orders.items():
        scored = aggregate_retrieval(
            [{"id": gold[i].id, "retrieved": lists[i], "relevant": gold[i].relevant_ids(), "category": gold[i].category} for i in ids], n_files=n_files)
        per_query[name] = scored.per_query
        ci = {}
        for metric in CI_METRICS:
            mean, lo, hi = mean_ci([r[metric] for r in scored.per_query], idx)
            ci[metric] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        row: dict[str, Any] = {"config": name, "n": len(ids), "retrieval": scored.as_dict(), "ci": ci}
        if latency.get(name):
            row["rerank_latency"] = _ms([latency[name][i] for i in ids])
        rows.append(row)
    by = {r["config"]: r for r in rows}
    for name, other in deltas:
        block: dict[str, Any] = {"reference": other}
        for metric in CI_METRICS:
            d, lo, hi = paired_delta_ci([r[metric] for r in per_query[name]], [r[metric] for r in per_query[other]], idx)
            block[metric] = {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        by[name].setdefault("deltas", []).append(block)
    return rows, bootstrap_settings(len(ids))


def run(root: Path | None = None, *, listwise_model: str | None = None, ollama_url: str = "http://127.0.0.1:11434", window: int = 20,
        retrieve_k: int = 50, device: str = "auto", complete: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    root = root or project_root()
    files_root = root / "bench" / "corpus" / "files"
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    # The student's recency feature reads chunk mtimes: the MANIFEST's, not the checkout's.
    index = build_index(files_root, embedder_name=BGE_SMALL, retrieve_k=retrieve_k, enable_router=False, enable_rerank=False, mtimes="manifest")
    teacher = CrossEncoderReranker(model_name=BGE_RERANKER)
    if teacher.model is None:
        raise SystemExit(f"{BGE_RERANKER} did not load; this benchmark has no overlap fallback for the teacher")

    all_ids = list(range(len(gold)))
    train_ids, test_ids = split_questions(gold)
    candidates = []
    orders: dict[str, list[list[str]]] = {HYBRID: [], TEACHER: []}
    latency: dict[str, list[float]] = {TEACHER: []}
    teacher_scores: list[dict[str, float]] = []
    if gold:
        teacher(gold[0].question, [(h.chunk, h.score) for h in index.retrieve(gold[0].question, k=retrieve_k).hits])  # warm-up
    for q in gold:
        pairs = [(h.chunk, h.score) for h in index.retrieve(q.question, k=retrieve_k).hits]  # fused order, RRF score
        candidates.append(pairs)
        orders[HYBRID].append(_paths(pairs))
        t0 = time.perf_counter()
        ranked = teacher(q.question, pairs)
        latency[TEACHER].append((time.perf_counter() - t0) * 1000.0)
        orders[TEACHER].append(_paths(ranked))
        teacher_scores.append({chunk.chunk_id: float(score) for chunk, score in ranked})

    # Student: fitted on the train questions' teacher scores, scored on the test questions.
    triples = [t for i in train_ids for t in triples_from_pairs(gold[i].question, candidates[i], teacher_scores[i])]
    student = DistilledReranker().fit(triples)
    orders[STUDENT] = []
    latency[STUDENT] = []
    for q, pairs in zip(gold, candidates):
        t0 = time.perf_counter()
        ranked = student(q.question, pairs)
        latency[STUDENT].append((time.perf_counter() - t0) * 1000.0)
        orders[STUDENT].append(_paths(ranked))
    test_rows, boot_test = score_rows({k: orders[k] for k in (HYBRID, TEACHER, STUDENT)}, gold, test_ids, latency, manifest["n_files"],
                                      [(TEACHER, HYBRID), (STUDENT, TEACHER), (STUDENT, HYBRID)])
    by_test = {r["config"]: r for r in test_rows}
    by_test[STUDENT]["student"] = {
        "backend": student.backend,
        "fallback": None if student.backend == "lightgbm" else "linear least squares (LightGBM not installed)",
        "n_train_questions": len(train_ids),
        "n_train_triples": len(triples),
        "features": list(student.feature_names),
        "teacher": BGE_RERANKER,
    }

    # Listwise: one LLM permutation per question over the top `window` candidates.
    listwise: dict[str, Any] = {"evaluated": False, "reason": "no --listwise-model given"}
    all_orders = {HYBRID: orders[HYBRID], TEACHER: orders[TEACHER]}
    all_deltas = [(TEACHER, HYBRID)]
    if listwise_model or complete:
        name = f"hybrid+listwise({listwise_model or 'injected'})@bge-small"
        fn = complete or (lambda prompt: ollama_complete(prompt, listwise_model, ollama_url))
        served = ollama_model_info(listwise_model, ollama_url) if complete is None else {}
        all_orders[name], latency[name], block = listwise_arm(gold, candidates, fn, window=window)
        if block["n_failed_calls"] == block["n_calls"]:
            raise SystemExit(f"no completion from {listwise_model} at {ollama_url}; is `ollama serve` running?")
        all_deltas += [(name, HYBRID), (name, TEACHER)]
        first = [f"{chunk.path}\n{(chunk.text or '')[:PASSAGE_CHARS]}" for chunk, _ in candidates[0][:window]] if candidates else []
        listwise = {"evaluated": True, "config": name, "model": listwise_model or "injected", **served, "window": window,
                    "options": OLLAMA_OPTIONS, "passage_chars": PASSAGE_CHARS,
                    "fallback": "a question whose call returned nothing or whose reply is not a full permutation keeps the fused order",
                    "example_prompt": {"id": gold[0].id, "prompt": listwise_prompt(gold[0].question, first)} if first else None, **block}
    all_rows, boot_all = score_rows(all_orders, gold, all_ids, latency, manifest["n_files"], all_deltas)

    return {
        "git_sha": git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info(device),
        "corpus_sha256": manifest["aggregate_sha256"],
        "n_gold": len(gold),
        "n_files": manifest["n_files"],
        "mtimes": mtime_provenance(files_root, "manifest", base=root),
        "n_chunks": len(index.chunks),
        "candidates": {"config": HYBRID, "retrieve_k": retrieve_k, "note": "every arm reranks this fused list; nothing is added to it"},
        "models": {
            "embedder": {**model_provenance(BGE_SMALL, "embedder"), "device": model_device(getattr(index.embedder, "_model", None))},
            "reranker": {**model_provenance(BGE_RERANKER, "reranker"), "loaded": True, "device": model_device(teacher.model)},
        },
        "split": {"rule": "alternate train / test inside each category, gold order", "train_ids": [gold[i].id for i in train_ids],
                  "test_ids": [gold[i].id for i in test_ids]},
        "test_split": {"bootstrap": boot_test, "results": test_rows},
        "all_questions": {"bootstrap": boot_all, "results": all_rows},
        "listwise": listwise,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("bench/results/rerank_arms.json"))
    p.add_argument("--listwise-model", default=None, help="Ollama model for the listwise arm (omit to record it as not evaluated)")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)
    blob = run(listwise_model=args.listwise_model, ollama_url=args.ollama_url, window=args.window, device=args.device)
    out = args.out if args.out.is_absolute() else project_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    for scope in ("test_split", "all_questions"):
        for row in blob[scope]["results"]:
            print(scope, row["config"], row["n"], row["ci"]["ndcg@10"], row.get("rerank_latency"), row.get("deltas"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
