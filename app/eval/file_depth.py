"""Equal file depth for the real-model hybrid and its legs (make bench-file-depth).

Recall@k scores the first k entries of a FILE list (app/eval/metrics.py). Every list the harness scores is cut at
retrieve_k = 50 CHUNKS before its files are taken (InMemoryHybridIndex.retrieve for the fused list, the BM25 and
dense legs likewise), and fixture v1 has 71 chunks in 61 files, so a list whose chunks share files holds fewer than
50 files and lists compared at Recall@50 can hold different numbers of files.
bench/results/hybrid_drops_bge_base.json shows what that costs hybrid@bge-base: 7 gold files that dense ranks 48th to
50th and BM25 never returns land at fused chunk ranks 51 to 53, so hybrid loses Recall@50 against dense alone.

FILE_DEPTH_RULE, declared before any run (M37), gives every compared list the same file depth. The rows recorded
under it are added beside the committed ones; nothing here changes the default path: InMemoryHybridIndex, the served
SQL and every committed row keep the 50-chunk cut.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from app.eval import per_question
from app.eval.corpus import load_manifest, verify_manifest
from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import BGE_BASE, BGE_SMALL, CI_METRICS, git_sha, project_root
from app.eval.index_loader import build_index, mtime_provenance
from app.eval.provenance import command_line, device_info, git_state, model_device, model_provenance, run_provenance
from app.eval.rerank_arms import score_rows
from app.eval.rrf_sweep import embedder_tag, files_of, fused_ranking, leg_lists, reproduces, scored
from app.retrieval.fuse import RRF_K
from app.retrieval.hybrid import InMemoryHybridIndex

FILE_DEPTH_RULE = ("file-depth cut: every list (the BM25 leg, the dense leg and the fused list) keeps chunks in rank "
                   "order until it holds retrieve_k distinct files (or its candidates run out); metrics are unchanged.")
CUT_AS_CODED = ("a list keeps its chunks in rank order and stops right after the chunk that brings it to retrieve_k "
                "distinct files, so a later chunk of a file it already holds is not kept; a leg's candidates are every "
                "chunk it returns when asked for all chunks of the index (BM25 returns only chunks that share a term "
                "with the question); the fused list is the two cut legs through rrf_score_map and the stable sort by "
                "score InMemoryHybridIndex.retrieve uses, cut again; files are taken in the order of their first chunk")


def file_depth_cut(chunk_ids: Sequence[str], path_of: Callable[[str], str], retrieve_k: int) -> list[str]:
    """The chunks of a ranked chunk list, in rank order, up to and including the one that brings the list to
    retrieve_k distinct files (FILE_DEPTH_RULE); every chunk when the list holds fewer files than that."""
    kept: list[str] = []
    files: set[str] = set()
    for cid in chunk_ids:
        if len(files) >= retrieve_k:
            break
        kept.append(cid)
        files.add(path_of(cid))
    return kept


EMBEDDERS = (BGE_SMALL, BGE_BASE)  # the real embedders bench/results/neural.json has rows for
DROPS_FILE = "bench/results/hybrid_drops_bge_base.json"  # the gold files hybrid@bge-base drops at the chunk cut


def legs_in_full(index: InMemoryHybridIndex, questions: Sequence[str]) -> list[tuple[list[str], list[str]]]:
    """Per question, the chunk ids of BM25's and dense's whole rankings (every chunk each returns when asked for all of
    the index's chunks), best first: the candidates FILE_DEPTH_RULE cuts."""
    return leg_lists(index, questions, len(index.chunks))


def depth_lists(index: InMemoryHybridIndex, bm25_ids: list[str], dense_ids: list[str], retrieve_k: int) -> dict[str, list[str]]:
    """One question's three chunk lists under FILE_DEPTH_RULE: each leg cut, then the fused list of the cut legs cut."""
    path_of = lambda cid: index.by_id[cid].path  # noqa: E731
    bm25 = file_depth_cut(bm25_ids, path_of, retrieve_k)
    dense = file_depth_cut(dense_ids, path_of, retrieve_k)
    fused = file_depth_cut([cid for cid, _ in fused_ranking(bm25, dense)], path_of, retrieve_k)
    return {"bm25": bm25, "dense": dense, "hybrid": fused}


def chunk_lists(bm25_ids: list[str], dense_ids: list[str], retrieve_k: int) -> dict[str, list[str]]:
    """One question's three chunk lists under the committed cut: the top retrieve_k chunks of each leg and of their
    fusion (rrf_sweep's default setting, which reproduces the committed rows)."""
    bm25, dense = bm25_ids[:retrieve_k], dense_ids[:retrieve_k]
    return {"bm25": bm25, "dense": dense, "hybrid": [cid for cid, _ in fused_ranking(bm25, dense)[:retrieve_k]]}


def _depth(lists: list[list[str]], index: InMemoryHybridIndex, retrieve_k: int) -> dict[str, Any]:
    files = [len(files_of(chunks, index)) for chunks in lists]
    return {"mean_chunks": round(sum(map(len, lists)) / len(lists), 2), "mean_files": round(sum(files) / len(files), 2),
            "min_files": min(files), "questions_below_retrieve_k_files": sum(1 for f in files if f < retrieve_k)}


def recovered(root: Path, tag: str, gold: list[GoldQuestion], dense: list[list[str]], hybrid: list[list[str]],
              hybrid_depth: list[list[str]]) -> dict[str, Any]:
    """The gold files the chunk-cut hybrid list lacks while the chunk-cut dense list has them (the files hybrid drops at
    Recall@50), and which of them the file-depth hybrid list holds, with their file rank there. For bge-base the
    dropped set is also compared with the one bench/results/hybrid_drops_bge_base.json lists."""
    dropped = []
    for i, q in enumerate(gold):
        for path in sorted(q.relevant_ids()):
            if path in dense[i] and path not in hybrid[i]:
                rank = hybrid_depth[i].index(path) + 1 if path in hybrid_depth[i] else None
                dropped.append({"id": q.id, "path": path, "recovered": rank is not None, "file_rank_under_file_depth": rank})
    out: dict[str, Any] = {"n_dropped": len(dropped), "n_recovered": sum(d["recovered"] for d in dropped), "files": dropped}
    if tag == "bge-base" and (root / DROPS_FILE).exists():
        listed = {(q["id"], d["path"]) for q in json.loads((root / DROPS_FILE).read_text())["questions"] for d in q["dropped"]}
        out["matches"] = {"file": DROPS_FILE, "same_files": listed == {(d["id"], d["path"]) for d in dropped}}
    return out


def run(root: Path | None = None, *, embedders: Sequence[str] = EMBEDDERS, retrieve_k: int = 50, device: str = "auto") -> dict[str, Any]:
    root = root or project_root()
    started = git_state(root)  # before anything is written (run_provenance)
    files_root = root / "bench" / "corpus" / "files"
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    n_files, every, questions = manifest["n_files"], list(range(len(gold))), [q.question for q in gold]
    orders: dict[str, list[list[str]]] = {}
    deltas: list[tuple[str, str]] = []
    per_embedder: dict[str, Any] = {}
    models: dict[str, Any] = {}
    n_chunks = None
    for embedder in embedders:
        index = build_index(files_root, embedder_name=embedder, retrieve_k=retrieve_k, enable_router=False, enable_rerank=False,
                            mtimes="manifest")
        n_chunks = len(index.chunks)
        tag = embedder_tag(embedder)
        full = legs_in_full(index, questions)
        cut = [chunk_lists(b, d, retrieve_k) for b, d in full]
        depth = [depth_lists(index, b, d, retrieve_k) for b, d in full]
        # the committed cut's legs are the first retrieve_k chunks of the whole rankings (both sort by a total order)
        prefix = leg_lists(index, questions, retrieve_k) == [(c["bm25"], c["dense"]) for c in cut]
        lists = {kind: {leg: [files_of(c[leg], index) for c in chunks] for leg in ("bm25", "dense", "hybrid")}
                 for kind, chunks in (("chunk", cut), ("depth", depth))}
        names = {"bm25": "bm25_only", "dense": f"dense_only@{tag}", "hybrid": f"hybrid@{tag}"}
        for leg in ("bm25", "dense", "hybrid"):
            if leg == "bm25" and names[leg] in orders:  # BM25 does not depend on the embedder: one pair of rows
                continue
            orders[names[leg]] = lists["chunk"][leg]
            orders[f"{names[leg]}/file-depth"] = lists["depth"][leg]
            deltas.append((f"{names[leg]}/file-depth", names[leg]))
        deltas += [(f"{names['hybrid']}/file-depth", f"{names['dense']}/file-depth"), (names["hybrid"], names["dense"])]
        blocks = {leg: scored(gold, lists["chunk"][leg], every, n_files).as_dict() for leg in ("bm25", "dense", "hybrid")}
        per_embedder[tag] = {
            "embedder": embedder,
            "rows": {"chunk_cut": names, "file_depth": {leg: f"{n}/file-depth" for leg, n in names.items()}},
            "chunk_legs_are_prefixes_of_the_whole_rankings": prefix,
            "depth": {f"{names[leg]}{suffix}": _depth([c[leg] for c in chunks], index, retrieve_k)
                      for leg in ("bm25", "dense", "hybrid") for suffix, chunks in (("", cut), ("/file-depth", depth))},
            "dropped_by_the_chunk_cut": recovered(root, tag, gold, lists["chunk"]["dense"], lists["chunk"]["hybrid"],
                                                  lists["depth"]["hybrid"]),
            "reproduces": reproduces(root, embedder, blocks),
        }
        models[tag] = {**model_provenance(embedder, "embedder"), "dim": int(getattr(index.embedder, "dim", 0)),
                       "device": model_device(getattr(index.embedder, "_model", None))}
    rows, boot = score_rows(orders, gold, every, {}, n_files, deltas)
    per_query = {name: scored(gold, lists, every, n_files).per_query for name, lists in orders.items()}
    blob = {
        "git_sha": git_sha(root),
        **run_provenance(root=root, started=started),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info(device),
        "corpus_sha256": manifest["aggregate_sha256"],
        "n_gold": len(gold),
        "n_files": n_files,
        "mtimes": mtime_provenance(files_root, "manifest", base=root),
        "n_chunks": n_chunks,
        "models": {"embedders": models},
        "rule": FILE_DEPTH_RULE,
        "as_coded": CUT_AS_CODED,
        "retrieval": {"retrieve_k": retrieve_k, "router": False, "reranker": None, "staleness_tier1": False,
                      "fusion": {"k": RRF_K, "weights": {"bm25": 1.0, "dense": 1.0}}},
        "default_path": ("unchanged: InMemoryHybridIndex, the served SQL and every committed row keep the top retrieve_k "
                         "chunks; the file-depth rows are added beside them"),
        "bootstrap": boot,
        "embedders": per_embedder,
        "results": rows,
        "reproduces_all": all((e["reproduces"] or {}).get("all") for e in per_embedder.values()),
    }
    blob["_sidecar"] = per_question.build(blob, per_query, [q.id for q in gold], [q.category for q in gold], CI_METRICS)
    return blob


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("bench/results/file_depth.json"))
    p.add_argument("--embedders", nargs="+", default=list(EMBEDDERS), help="st:<model> ... (hash: a CPU check, not a result)")
    p.add_argument("--retrieve-k", type=int, default=50)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    blob = run(embedders=args.embedders, retrieve_k=args.retrieve_k, device=args.device)
    sidecar = blob.pop("_sidecar")
    out = args.out if args.out.is_absolute() else project_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    per_question.write(per_question.sidecar_path(out), sidecar)
    for row in blob["results"]:
        print(row["config"], row["ci"]["ndcg@10"]["mean"], row["ci"]["recall@50"]["mean"],
              [(d["reference"], d["recall@50"]) for d in row.get("deltas", [])])
    for tag, e in blob["embedders"].items():
        print(tag, "reproduces", json.dumps(e["reproduces"]), "dropped", e["dropped_by_the_chunk_cut"]["n_dropped"],
              "recovered", e["dropped_by_the_chunk_cut"]["n_recovered"], json.dumps(e["dropped_by_the_chunk_cut"].get("matches")))
    print(f"wrote {out}")
    return 0 if blob["reproduces_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
