"""RRF k and the dense-versus-BM25 list weight, chosen on half the gold questions (make bench-rrf-sweep).

Hybrid retrieval fuses its BM25 and dense lists with k = 60 and equal weights
(app/retrieval/fuse.py). This sweep asks whether another setting ranks better with a
real embedder (bge-small: the hybrid@bge-small row), without letting the same
questions pick a setting and score it:

  tune      every point of the grid (RRF_KS x DENSE_WEIGHTS; BM25's weight stays 1.0)
            is scored on the train half of app.eval.rerank_arms.split_questions, the
            split the distilled students use. The highest tune-half mean nDCG@10 picks
            one setting (choose(): ties go to the point nearest the default). These
            questions chose the setting, so their numbers are in-sample, not evidence.
  held out  the chosen setting and the default (k = 60, equal weights) are scored on
            the other half, with paired bootstrap deltas (chosen minus default).

The default changes only if the held-out nDCG@10 interval excludes zero in favour of
the chosen setting, and then by the owner, not by this command: it would move every
hybrid row of bench/results/main.json and neural.json.

Each question's BM25 and dense lists are retrieved once (the top retrieve_k chunks of
each: the calls InMemoryHybridIndex.retrieve makes with the router and the reranker
off) and fused for every grid point as retrieve() fuses them: [BM25, dense] through
rrf_score_map, a stable sort by score, the top retrieve_k chunks, their files in
order. At the default setting over all questions those are the lists behind the
committed hybrid row; `reproduces` records whether they score exactly as that row
and the BM25 and dense rows do.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import BGE_BASE, BGE_SMALL, git_sha, project_root
from app.eval.index_loader import build_index, mtime_provenance
from app.eval.metrics import RetrievalScores, aggregate_retrieval
from app.eval.provenance import command_line, device_info, model_device, model_provenance
from app.eval.report import verdict
from app.eval.rerank_arms import score_rows, split_questions
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, paired_delta_ci, resample_indices, settings as bootstrap_settings
from app.retrieval.fuse import RRF_K, rrf_score_map
from app.retrieval.hybrid import InMemoryHybridIndex

RRF_KS = (1, 10, 30, 60, 100, 200)
DENSE_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)  # the dense list's weight; BM25's is 1.0 (only the ratio orders the list)
DEFAULT_SETTING = (RRF_K, 1.0)
SELECTION = "the highest mean nDCG@10 on the tune half (means equal to 10 decimals tie)"
TIE_BREAK = ("the point nearest the default: the dense weight nearest 1 (smallest |log2 w|), then the k nearest 60 "
             "(smallest |ln(k / 60)|), then the smaller k, then the smaller weight; the default wins every tie it is in")
RULE = ("the default changes only if the held-out nDCG@10 interval (chosen minus default) excludes zero in favour of the "
        "chosen setting, and then by the owner: every hybrid row of bench/results/main.json and neural.json would be "
        "re-recorded. This command never changes it.")
# The committed rows the default setting's lists stand for, per embedder: (results file, {leg: row name}).
ANCHORS = {
    BGE_SMALL: ("bench/results/neural.json", {"bm25": "bm25_only", "dense": "dense_only@bge-small", "hybrid": "hybrid@bge-small"}),
    BGE_BASE: ("bench/results/neural.json", {"bm25": "bm25_only", "dense": "dense_only@bge-base", "hybrid": "hybrid@bge-base"}),
    "hash": ("bench/results/main.json", {"bm25": "bm25_only", "dense": "dense_only", "hybrid": "hybrid"}),
}
TAGS = {BGE_SMALL: "bge-small", BGE_BASE: "bge-base", "hash": "hash"}


def embedder_tag(embedder: str) -> str:
    return TAGS.get(embedder, embedder.split("/")[-1])


def setting_name(tag: str, k: int, dense_weight: float) -> str:
    return f"hybrid@{tag}/k={k}/dense-weight={dense_weight:g}"


def leg_lists(index: InMemoryHybridIndex, questions: Sequence[str], retrieve_k: int) -> list[tuple[list[str], list[str]]]:
    """Per question, the chunk ids of BM25's and dense's top retrieve_k: the two lists retrieve() fuses with the router off."""
    return [([c.chunk_id for c, _ in index.search_bm25(q, k=retrieve_k)], [c.chunk_id for c, _ in index.search_dense(q, k=retrieve_k)])
            for q in questions]


def fused_ranking(bm25_ids: list[str], dense_ids: list[str], *, k: int = RRF_K, dense_weight: float = 1.0) -> list[tuple[str, float]]:
    """Every fused chunk id with its RRF score, best first, before any cut: [BM25, dense] through rrf_score_map and the
    stable sort by score that InMemoryHybridIndex.retrieve uses (an exact tie keeps the order the ids were first seen)."""
    fused = rrf_score_map([bm25_ids, dense_ids], k=k, weights=(1.0, dense_weight))
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def files_of(chunk_ids: Sequence[str], index: InMemoryHybridIndex) -> list[str]:
    """The files of a ranked chunk list in the order of their first chunk: the list every scorer ranks."""
    out: list[str] = []
    for cid in chunk_ids:
        path = index.by_id[cid].path
        if path not in out:
            out.append(path)
    return out


def hybrid_files(index: InMemoryHybridIndex, legs: list[tuple[list[str], list[str]]], *, k: int, dense_weight: float,
                 retrieve_k: int) -> list[list[str]]:
    """Per question, the files of the fused top retrieve_k chunks at one setting."""
    return [files_of([cid for cid, _ in fused_ranking(b, d, k=k, dense_weight=dense_weight)[:retrieve_k]], index) for b, d in legs]


def scored(gold: list[GoldQuestion], lists: list[list[str]], ids: Sequence[int], n_files: int) -> RetrievalScores:
    return aggregate_retrieval([{"id": gold[i].id, "retrieved": lists[i], "relevant": gold[i].relevant_ids(), "category": gold[i].category}
                                for i in ids], n_files=n_files)


def choose(points: Sequence[tuple[int, float, float]]) -> tuple[int, float]:
    """(k, dense weight) of the winner among (k, dense weight, tune-half mean nDCG@10): SELECTION, then TIE_BREAK."""
    best = max(round(mean, 10) for _, _, mean in points)
    tied = [(k, w) for k, w, mean in points if round(mean, 10) == best]
    return min(tied, key=lambda kw: (abs(math.log2(kw[1])), abs(math.log(kw[0] / RRF_K)), kw[0], kw[1]))


def reproduces(root: Path, embedder: str, retrieval: dict[str, dict]) -> dict[str, Any] | None:
    """Whether the retrieval blocks scored here ({leg: block}, all questions, the default setting) equal the committed rows
    they stand for (ANCHORS). None for an embedder without committed rows."""
    if embedder not in ANCHORS:
        return None
    rel, names = ANCHORS[embedder]
    path = root / rel
    if not path.exists():
        return {"file": rel, "found": False, "all": False}
    blob = json.loads(path.read_text())
    by = {r["config"]: r for r in blob.get("results", [])}
    rows = {names[leg]: (by.get(names[leg]) or {}).get("retrieval") == block for leg, block in retrieval.items()}
    return {"file": rel, "git_sha": blob.get("git_sha"), "compared": "every field of each row's retrieval block (rounded means, by category)",
            "rows": rows, "all": all(rows.values())}


def _metrics(s: RetrievalScores) -> dict[str, float]:
    r = s.as_dict()
    return {m: r[m] for m in ("ndcg@10", "recall@10", "recall@50", "mrr@10")}


def run(root: Path | None = None, *, embedder: str = BGE_SMALL, retrieve_k: int = 50, device: str = "auto") -> dict[str, Any]:
    root = root or project_root()
    files_root = root / "bench" / "corpus" / "files"
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    n_files = manifest["n_files"]
    index = build_index(files_root, embedder_name=embedder, retrieve_k=retrieve_k, enable_router=False, enable_rerank=False, mtimes="manifest")
    legs = leg_lists(index, [q.question for q in gold], retrieve_k)
    tag = embedder_tag(embedder)
    lists = {(k, w): hybrid_files(index, legs, k=k, dense_weight=w, retrieve_k=retrieve_k) for k in RRF_KS for w in DENSE_WEIGHTS}
    default = lists[DEFAULT_SETTING]
    default_name = setting_name(tag, *DEFAULT_SETTING)
    tune_ids, test_ids = split_questions(gold)

    # Tune half: every grid point, with its paired nDCG@10 delta against the default on the same questions (in-sample).
    idx = resample_indices(len(tune_ids), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    base = [r["ndcg@10"] for r in scored(gold, default, tune_ids, n_files).per_query]
    points, means = [], []
    for (k, w), per_q in lists.items():
        s = scored(gold, per_q, tune_ids, n_files)
        d, lo, hi = paired_delta_ci([r["ndcg@10"] for r in s.per_query], base, idx)
        points.append({"config": setting_name(tag, k, w), "k": k, "dense_weight": w, **_metrics(s),
                       "ndcg@10_vs_default": {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)},
                       "n_top10_differs_from_default": sum(1 for i in tune_ids if per_q[i][:10] != default[i][:10])})
        means.append((k, w, s.ndcg_10))
    chosen = choose(means)
    chosen_name = setting_name(tag, *chosen)
    is_default = chosen == DEFAULT_SETTING

    # Held-out half: the chosen setting against the default, paired (rerank_arms.score_rows, the students' test-split scoring).
    orders = {default_name: default} if is_default else {default_name: default, chosen_name: lists[chosen]}
    held_rows, held_boot = score_rows(orders, gold, test_ids, {}, n_files, [] if is_default else [(chosen_name, default_name)])
    delta = None if is_default else next(r for r in held_rows if r["config"] == chosen_name)["deltas"][0]
    in_favour = delta is not None and delta["ndcg@10"]["lo"] > 0
    decision = {
        "default": {"k": DEFAULT_SETTING[0], "weights": {"bm25": 1.0, "dense": DEFAULT_SETTING[1]}},
        "chosen": {"k": chosen[0], "weights": {"bm25": 1.0, "dense": chosen[1]}},
        "chosen_is_default": is_default,
        "held_out_ndcg@10": delta["ndcg@10"] if delta else None,
        "verdict": ("the tune half chose the default; there is no other setting to compare on the held-out half" if is_default
                    else f"held-out nDCG@10, chosen minus default: {verdict(delta['ndcg@10'])}"),
        "excludes_zero_in_favour_of_chosen": in_favour,
        "owner_decision_needed": in_favour,
        "default_changed": False,
        "rule": RULE,
    }

    every = range(len(gold))
    blocks = {"bm25": scored(gold, [files_of(b, index) for b, _ in legs], every, n_files).as_dict(),
              "dense": scored(gold, [files_of(d, index) for _, d in legs], every, n_files).as_dict(),
              "hybrid": scored(gold, default, every, n_files).as_dict()}
    return {
        "git_sha": git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info(device),
        "corpus_sha256": manifest["aggregate_sha256"],
        "n_gold": len(gold),
        "n_files": n_files,
        "mtimes": mtime_provenance(files_root, "manifest", base=root),
        "n_chunks": len(index.chunks),
        "models": {"embedder": {**model_provenance(embedder, "embedder"), "dim": int(getattr(index.embedder, "dim", 0)),
                                "device": model_device(getattr(index.embedder, "_model", None))}},
        "retrieval": {"config": f"hybrid@{tag}", "retrieve_k": retrieve_k, "router": False, "reranker": None, "staleness_tier1": False,
                      "fusion": "[BM25, dense] top retrieve_k chunks each, app.retrieval.fuse.rrf_score_map, stable sort by score, "
                                "top retrieve_k chunks, scored by their files"},
        "grid": {"k": list(RRF_KS), "dense_weight": list(DENSE_WEIGHTS), "bm25_weight": 1.0, "selection": SELECTION, "tie_break": TIE_BREAK},
        "split": {"rule": "alternate train / test inside each category, gold order (app.eval.rerank_arms.split_questions)",
                  "tune": "train", "held_out": "test", "tune_ids": [gold[i].id for i in tune_ids], "held_out_ids": [gold[i].id for i in test_ids]},
        "tune": {"n": len(tune_ids), "note": "in-sample: these questions chose the setting, so their numbers are not evidence",
                 "bootstrap": bootstrap_settings(len(tune_ids)), "default": default_name, "chosen": chosen_name, "points": points},
        "held_out": {"bootstrap": held_boot, "results": held_rows},
        "decision": decision,
        "reproduces": reproduces(root, embedder, blocks),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("bench/results/rrf_sweep.json"))
    p.add_argument("--embedder", default=BGE_SMALL, help="st:<model> or hash (hash: a CPU dry run against main.json, not a result)")
    p.add_argument("--retrieve-k", type=int, default=50)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    blob = run(embedder=args.embedder, retrieve_k=args.retrieve_k, device=args.device)
    out = args.out if args.out.is_absolute() else project_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    for row in blob["held_out"]["results"]:
        print("held_out", row["config"], row["n"], row["ci"]["ndcg@10"], row.get("deltas"))
    print("decision", json.dumps(blob["decision"]))
    print("reproduces", json.dumps(blob["reproduces"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
