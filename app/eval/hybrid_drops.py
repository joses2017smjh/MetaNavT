"""Which gold files hybrid drops at Recall@50 that dense alone keeps (make bench-hybrid-drops).

bench/results/neural.json records a lower Recall@50 for hybrid@bge-base than for its
parent dense_only@bge-base. Both score the files of their top retrieve_k chunks.
Hybrid fuses BM25's and dense's top retrieve_k chunks (InMemoryHybridIndex.retrieve:
k = 60, equal weights) and keeps the top retrieve_k by RRF score, so a gold file whose
chunks dense ranks low, and BM25 not at all, can fall below that cut while chunks that
BM25 ranks take its place.

For every question whose two lists differ in their gold files (every question whose
Recall@50 differs is one of them; a drop and a gain on one question can also cancel),
each gold file in one list and not the other (`dropped`: dense only; `gained`: hybrid
only), with
  dense   its rank in dense_only's file list and the rank of its best chunk among
          dense's chunks (null: none of its chunks is in dense's top retrieve_k)
  bm25    the same in BM25's lists
  fused   the rank and RRF score of its best chunk in the whole fused list, before the
          cut, whether that chunk was kept, and whether its score equals the score of the
          last chunk kept (`cut`: rank retrieve_k), so the stable order decided it.
`summary` checks that every question whose Recall@50 differs between the two rows is
listed, so the listed files account for the whole delta, and `reproduces` records
whether these lists score exactly as the committed rows.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import BGE_BASE, git_sha, project_root
from app.eval.index_loader import build_index, mtime_provenance
from app.eval.metrics import recall_at_k
from app.eval.provenance import command_line, device_info, git_state, model_device, model_provenance, run_provenance
from app.eval.rrf_sweep import embedder_tag, fused_ranking, leg_lists, reproduces, scored
from app.retrieval.fuse import RRF_K

RECALL_K = 50  # the metric's cutoff (Recall@50), on the file lists


def _files(chunk_ids: Sequence[str], path_of: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    for cid in chunk_ids:
        if path_of[cid] not in out:
            out.append(path_of[cid])
    return out


def _ranks(path: str, chunk_ids: Sequence[str], files: Sequence[str], path_of: Mapping[str, str]) -> dict[str, int | None]:
    chunk = next((r for r, cid in enumerate(chunk_ids, start=1) if path_of[cid] == path), None)
    return {"file_rank": files.index(path) + 1 if path in files else None, "chunk_rank": chunk}


def diff_question(q: GoldQuestion, bm25_ids: Sequence[str], dense_ids: Sequence[str], path_of: Mapping[str, str], *,
                  retrieve_k: int, k: int = RRF_K) -> dict[str, Any] | None:
    """The gold files of one question in dense_only's list and not hybrid's (dropped), and the reverse (gained); None when
    the two lists hold the same gold files. hybrid fuses [BM25, dense] at k with equal weights and keeps retrieve_k chunks."""
    ranking = fused_ranking(list(bm25_ids), list(dense_ids), k=k)
    kept = [cid for cid, _ in ranking[:retrieve_k]]
    dense_files, bm25_files, hybrid_files = _files(dense_ids, path_of), _files(bm25_ids, path_of), _files(kept, path_of)
    relevant = q.relevant_ids()
    in_dense, in_hybrid = set(dense_files[:RECALL_K]), set(hybrid_files[:RECALL_K])
    dropped = [p for p in relevant if p in in_dense and p not in in_hybrid]
    gained = [p for p in relevant if p in in_hybrid and p not in in_dense]
    if not dropped and not gained:
        return None
    cut = ranking[retrieve_k - 1][1] if len(ranking) >= retrieve_k else None

    def entry(path: str) -> dict[str, Any]:
        rank, score = next((r, s) for r, (cid, s) in enumerate(ranking, start=1) if path_of[cid] == path)
        return {"path": path, "dense": _ranks(path, dense_ids, dense_files, path_of), "bm25": _ranks(path, bm25_ids, bm25_files, path_of),
                "fused": {"chunk_rank": rank, "rrf": round(score, 8), "kept": rank <= retrieve_k, "tied_with_cut": cut is not None and score == cut}}

    return {
        "id": q.id,
        "category": q.category,
        "question": q.question,
        "relevant": relevant,
        "recall@50": {"dense_only": round(recall_at_k(dense_files, relevant, RECALL_K), 4), "hybrid": round(recall_at_k(hybrid_files, relevant, RECALL_K), 4)},
        "cut": {"rank": retrieve_k, "rrf": round(cut, 8) if cut is not None else None, "n_fused": len(ranking)},
        "dropped": [entry(p) for p in dropped],
        "gained": [entry(p) for p in gained],
    }


def run(root: Path | None = None, *, embedder: str = BGE_BASE, retrieve_k: int = 50, device: str = "auto") -> dict[str, Any]:
    root = root or project_root()
    started = git_state(root)  # before anything is written (run_provenance)
    files_root = root / "bench" / "corpus" / "files"
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    n_files = manifest["n_files"]
    index = build_index(files_root, embedder_name=embedder, retrieve_k=retrieve_k, enable_router=False, enable_rerank=False, mtimes="manifest")
    legs = leg_lists(index, [q.question for q in gold], retrieve_k)
    path_of = {cid: chunk.path for cid, chunk in index.by_id.items()}
    tag = embedder_tag(embedder)

    questions = [e for e in (diff_question(q, b, d, path_of, retrieve_k=retrieve_k) for q, (b, d) in zip(gold, legs)) if e]
    lists = {"bm25": [_files(b, path_of) for b, _ in legs], "dense": [_files(d, path_of) for _, d in legs],
             "hybrid": [_files([cid for cid, _ in fused_ranking(b, d)[:retrieve_k]], path_of) for b, d in legs]}
    every = range(len(gold))
    scores = {leg: scored(gold, per_q, every, n_files) for leg, per_q in lists.items()}
    moved = {q.id for q, h, d in zip(gold, scores["hybrid"].per_query, scores["dense"].per_query) if h["recall@50"] != d["recall@50"]}
    dropped = [(e, d) for e in questions for d in e["dropped"]]
    summary = {
        "n_questions": len(gold),
        "recall@50": {"dense_only": round(scores["dense"].recall_50, 4), "hybrid": round(scores["hybrid"].recall_50, 4),
                      "delta": round(scores["hybrid"].recall_50 - scores["dense"].recall_50, 4),
                      "note": "means over all questions; delta is hybrid minus dense_only, and only the listed questions move it"},
        "every_question_that_moves_is_listed": moved <= {e["id"] for e in questions},
        "n_questions_affected": len(questions),
        "n_questions_with_drops": sum(1 for e in questions if e["dropped"]),
        "n_dropped": len(dropped),
        "n_gained": sum(len(e["gained"]) for e in questions),
        "dropped_with_no_chunk_in_bm25_list": sum(1 for _, d in dropped if d["bm25"]["chunk_rank"] is None),
        "dropped_tied_with_cut": sum(1 for _, d in dropped if d["fused"]["tied_with_cut"]),
        "dropped_by_category": dict(sorted(Counter(e["category"] for e, _ in dropped).items())),
    }
    return {
        "git_sha": git_sha(root),
        **run_provenance(root=root, started=started),
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
        "rows": {"dense_only": f"dense_only@{tag}", "hybrid": f"hybrid@{tag}"},
        "retrieve_k": retrieve_k,
        "fusion": {"k": RRF_K, "weights": {"bm25": 1.0, "dense": 1.0}, "lists": "[BM25, dense], each its top retrieve_k chunks; "
                   "hybrid keeps the top retrieve_k fused chunks (stable sort by RRF score)"},
        "summary": summary,
        "questions": questions,
        "reproduces": reproduces(root, embedder, {leg: s.as_dict() for leg, s in scores.items()}),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("bench/results/hybrid_drops_bge_base.json"))
    p.add_argument("--embedder", default=BGE_BASE, help="st:<model> or hash (hash: a CPU dry run against main.json, not a result)")
    p.add_argument("--retrieve-k", type=int, default=50)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    blob = run(embedder=args.embedder, retrieve_k=args.retrieve_k, device=args.device)
    out = args.out if args.out.is_absolute() else project_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    print("summary", json.dumps(blob["summary"]))
    for e in blob["questions"]:
        print(e["id"], e["category"], e["recall@50"], [d["path"] for d in e["dropped"]], [g["path"] for g in e["gained"]])
    print("reproduces", json.dumps(blob["reproduces"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
