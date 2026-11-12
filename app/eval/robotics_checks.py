"""Two checks outside fixture v1 on the frozen robotics snapshot: chunking and multi-query.

The snapshot (bench/robotics/2026-10-07, app/eval/robotics.py) is generated
file lookup: each question is generated from a top-level JSON field of one
file and names its experiment by the words of that file's path, and no label
is human-reviewed. A score here is source-file retrieval at path level, never
answer correctness.

    python -m app.eval.robotics_checks chunking   --out bench/results/robotics_chunking.json
    python -m app.eval.robotics_checks multiquery --out bench/results/robotics_multiquery.json

chunking    the structure-aware chunker (strategy "auto") against a fixed
            character window for every file ("fixed"), the change the
            fixture's hybrid+fixed-chunks row makes, on files longer than one
            window.
multiquery  the same "auto" index with app.agent.deep_research.multi_query_retrieve
            (the fixture's hybrid+multiquery row) against the index without it,
            on questions its keyword rewrite rules were not written for.

DECLARATION (below) was committed before the test split was scored, and
nothing in it was tuned on either split; tests/eval/test_robotics_checks.py
pins it. Both arms of a check share everything but the one change, and both
legs rank every chunk, so each arm is scored on the same number of distinct
paths however many chunks a file has.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

from app.agent.deep_research import expand_queries, multi_query_retrieve
from app.chunking.structure import fixed_size_chunks
from app.eval.index_loader import build_index
from app.eval.ingest import CHUNKER_VERSION
from app.eval.metrics import score_query
from app.eval.provenance import REPO_ROOT, command_line, device_info, git_sha, sanitize_path
from app.eval.robotics import group_bootstrap, sha256, tracked_blob_mismatches, verify

SCHEMA = "robotics-checks-v1"
DEFAULT_SNAPSHOT = "bench/robotics/2026-10-07"
SPLITS = ("dev", "test")
N_PATHS = 50  # the path list: the first N_PATHS distinct paths of the fused chunk ranking
PRIMARY_METRIC = "ndcg@10"
METRICS = ("ndcg@10", "mrr@10", "recall@5", "recall@10", "recall@50")
EMBEDDER = "hash"
MQ_QUERIES = 3
_WINDOW = inspect.signature(fixed_size_chunks).parameters
FIXED_WINDOW = {"size": _WINDOW["size"].default, "overlap": _WINDOW["overlap"].default}
# check -> (parent, arm); a paired delta is the arm minus its parent, the sign of the fixture's frontier rows.
CHECKS = {"chunking": ("auto", "fixed"), "multiquery": ("hybrid", "hybrid+multiquery")}

DECLARATION = {
    "declared": "committed before the test split was scored; no setting was tuned on either split",
    "inspected_before_declaring": "file sizes, chunk counts per strategy and the questions on which expand_queries "
                                  "returns more than one query, on both splits; no retrieval score on either split",
    "corpus": "every file of the snapshot through app.eval.index_loader.build_index (app.eval.ingest scan_directory "
              "and file_chunks), the files that differ from their tracked blob included as they are (the header "
              "lists them); a run stops unless every manifest file is indexed and none is rejected",
    "index": "in memory: BM25 (default analyzer, k1 1.5, b 0.75) over each chunk's path context "
             "(app.retrieval.context.path_context) and text; the hash embedder (character 3- and 4-grams, "
             "256 dimensions, no model) over the chunk text; RRF k 60; no router, no reranker",
    "depth": f"both legs rank every chunk; the fused ranking is cut to its first {N_PATHS} distinct paths",
    "metric": "nDCG@10 of the path list against the question's one relevant path (primary); MRR@10, Recall@5, "
              "Recall@10 and Recall@50 reported",
    "intervals": "dev and test separately; app.eval.robotics.group_bootstrap: whole experiment families resampled "
                 "with replacement, 2000 resamples, seed 20261007, 95% percentile interval; an arm minus its parent "
                 "is paired within each resample",
    "verdict": "from the paired interval of the primary metric: above zero, covers zero or below zero; "
               "'identical on every question' when no question's primary score changes; for multiquery, 'rules did "
               "not fire' on a split where no rewrite fired",
    "chunking": {"parent": "auto", "arm": "fixed",
                 "auto": "app.chunking chunk_text strategy auto (structure-aware; the chunker version is recorded "
                         "with the run)",
                 "fixed": f"app.chunking fixed_size_chunks for every file: {FIXED_WINDOW['size']} characters, "
                          f"{FIXED_WINDOW['overlap']} characters of overlap"},
    "multiquery": {"parent": "hybrid: the chunking check's auto index", "arm": "hybrid+multiquery",
                   "rewrite": f"app.agent.deep_research.multi_query_retrieve, n={MQ_QUERIES}: expand_queries "
                              "keyword heuristic (no LLM); the fused list of each query RRF-fused again (k 60)",
                   "fired": "a question on which expand_queries returns more than one distinct query (lower case), "
                            "as the harness counts it"},
    "limits": "generated file lookup with no human-reviewed label: a question names its experiment by the words of "
              "its path, and the default analyzer keeps '_' inside tokens, so its field words rarely match the "
              "file's field names. An interval that covers zero is not evidence that the arms find answers equally "
              "well.",
}

# The source files that decide a number; a run refuses to finish if one changed while it ran.
CODE_PATHS = (
    "app/eval/robotics_checks.py", "app/eval/robotics.py", "app/eval/index_loader.py", "app/eval/ingest.py",
    "app/eval/metrics.py", "app/chunking/structure.py", "app/retrieval/hybrid.py", "app/retrieval/bm25.py",
    "app/retrieval/embedders.py", "app/retrieval/fuse.py", "app/retrieval/context.py", "app/agent/deep_research.py",
)


def _code_sha256() -> dict[str, str]:
    return {p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in CODE_PATHS}


def _working_tree_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        return None


def build_snapshot_index(snapshot: Path, strategy: str, manifest: dict):
    """The declared in-memory index over every snapshot file, chunked with `strategy`."""
    index = build_index(snapshot / "corpus", embedder_name=EMBEDDER, enable_router=False, enable_rerank=False,
                        chunk_strategy=strategy, include_path_context=True, mtimes="disk")
    report = index.ingest_report
    indexed = {c.path for c in index.chunks}
    if report["files_rejected"] or indexed != {r["path"] for r in manifest["files"]}:
        raise ValueError(f"the {strategy} index does not hold exactly the snapshot's files: "
                         f"{report['files_rejected']} rejected, {len(indexed)} of {manifest['n_files']} indexed")
    return index


def ingest_block(index, strategy: str) -> dict:
    block = {k: index.ingest_report[k] for k in ("schema", "files_seen", "files_indexed", "files_rejected", "chunks_embedded")}
    return {**block, "strategy": strategy, "chunker_version": CHUNKER_VERSION if strategy == "auto" else None,
            "window": FIXED_WINDOW if strategy == "fixed" else None}


def path_list(hits) -> tuple[list[str], int | None]:
    """The first N_PATHS distinct paths of a fused chunk ranking, and the rank of the chunk that brought the tenth."""
    paths: list[str] = []
    seen: set[str] = set()
    tenth = None
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.path in seen:
            continue
        seen.add(hit.chunk.path)
        paths.append(hit.chunk.path)
        if len(paths) == 10:
            tenth = rank
        if len(paths) == N_PATHS:
            break
    return paths, tenth


def score(paths: list[str], question: dict, n_files: int, tenth: int | None) -> dict:
    relevant = question["relevant_paths"]
    row = {m: v for m, v in score_query(paths, relevant, n_files=n_files).items() if m in METRICS}
    row["relevant_rank"] = next((i for i, p in enumerate(paths, start=1) if p in relevant), None)
    row["top10"] = paths[:10]
    row["chunks_to_10_paths"] = tenth
    return row


def verdict(delta: dict, differences: list[float]) -> str:
    if not any(differences):
        return "identical on every question"
    low, high = delta["ci95"]
    return "above zero" if low > 0 else "below zero" if high < 0 else "covers zero"


def summarize(rows: list[dict], parent: str, arm: str, splits: tuple[str, ...]) -> dict:
    summary = {}
    for split in splits:
        subset = [r for r in rows if r["split"] == split]
        block: dict = {"n_questions": len(subset), "n_families": len({r["family"] for r in subset})}
        for name in (parent, arm):
            block[name] = {m: group_bootstrap([{"family": r["family"], m: r["arms"][name][m]} for r in subset], m) for m in METRICS}
            tenths = [r["arms"][name]["chunks_to_10_paths"] for r in subset if r["arms"][name]["chunks_to_10_paths"] is not None]
            block[name]["mean_chunks_to_10_paths"] = sum(tenths) / len(tenths) if tenths else None
        block["paired_delta"] = {
            m: {**group_bootstrap([{"family": r["family"], "arm": r["arms"][arm][m], "parent": r["arms"][parent][m]} for r in subset],
                                  "arm", paired_field="parent"), "of": arm, "minus": parent}
            for m in METRICS
        }
        block["questions_whose_relevant_file_is_longer_than_one_window"] = sum(r["relevant_file_longer_than_one_window"] for r in subset)
        differences = [r["arms"][arm][PRIMARY_METRIC] - r["arms"][parent][PRIMARY_METRIC] for r in subset]
        block["questions_with_a_different_score"] = sum(1 for d in differences if d)
        block["questions_whose_top10_differs"] = sum(1 for r in subset if r["arms"][arm]["top10"] != r["arms"][parent]["top10"])
        block["verdict"] = verdict(block["paired_delta"][PRIMARY_METRIC], differences)
        summary[split] = block
    return summary


def _question_rows(snapshot: Path, questions: list[dict], splits: tuple[str, ...]) -> list[dict]:
    rows = []
    for q in questions:
        if q["split"] not in splits:
            continue
        relevant = q["relevant_paths"][0]
        chars = len((snapshot / "corpus" / relevant).read_bytes().decode("utf-8"))
        rows.append({"id": q["id"], "family": q["family"], "split": q["split"], "relevant_path": relevant,
                     "relevant_file_chars": chars, "relevant_file_longer_than_one_window": chars > FIXED_WINDOW["size"],
                     "arms": {}})
    return rows


def run_check(check: str, snapshot: Path, splits: tuple[str, ...] = SPLITS) -> dict:
    if check not in CHECKS:
        raise ValueError(f"unknown check {check!r}; expected one of {sorted(CHECKS)}")
    if not splits or set(splits) - set(SPLITS):
        raise ValueError(f"splits must be a non-empty subset of {SPLITS}, got {splits}")
    snapshot = Path(snapshot)
    manifest, questions = verify(snapshot)
    manifest_sha = sha256((snapshot / "manifest.json").read_bytes())
    code = _code_sha256()
    parent, arm = CHECKS[check]
    n_files = manifest["n_files"]
    by_id = {q["id"]: q for q in questions}
    if check == "chunking":
        indexes = {name: build_snapshot_index(snapshot, name, manifest) for name in (parent, arm)}
        ingest = {name: ingest_block(index, name) for name, index in indexes.items()}
    else:
        index = build_snapshot_index(snapshot, "auto", manifest)
        ingest = {"auto": ingest_block(index, "auto")}
    rows = _question_rows(snapshot, questions, splits)
    for row in rows:
        q = by_id[row["id"]]
        if check == "chunking":
            for name, idx in indexes.items():
                paths, tenth = path_list(idx.retrieve(q["question"], k=len(idx.chunks)).hits)
                row["arms"][name] = score(paths, q, n_files, tenth)
        else:
            depth = len(index.chunks)
            paths, tenth = path_list(index.retrieve(q["question"], k=depth).hits)
            row["arms"][parent] = score(paths, q, n_files, tenth)
            queries = expand_queries(q["question"], n=MQ_QUERIES)
            fired = len({x.lower() for x in queries}) > 1
            mq_paths, mq_tenth = path_list(multi_query_retrieve(index, q["question"], n=MQ_QUERIES, k=depth).hits)
            if not fired and mq_paths != paths:
                raise RuntimeError(f"{row['id']}: no rewrite fired but the multi-query list differs from its parent's")
            row["arms"][arm] = score(mq_paths, q, n_files, mq_tenth)
            row["fired"] = fired
            row["queries"] = queries if fired else None
    summary = summarize(rows, parent, arm, splits)
    if check == "multiquery":
        for split in splits:
            fired = [r["id"] for r in rows if r["split"] == split and r["fired"]]
            summary[split]["fired"] = len(fired)
            summary[split]["fired_ids"] = fired
            if not fired:
                summary[split]["verdict"] = "rules did not fire"
    # Refuse a mixed run: the snapshot and the source that decides a number must be what they were at the start.
    verify(snapshot)
    if sha256((snapshot / "manifest.json").read_bytes()) != manifest_sha:
        raise ValueError("manifest changed during the run")
    if _code_sha256() != code:
        raise ValueError("source changed during the run")
    return {
        "schema": SCHEMA,
        "check": check,
        "git_sha": git_sha(REPO_ROOT),
        "working_tree_dirty": _working_tree_dirty(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info("cpu"),
        "code_sha256": code,
        "snapshot": {
            "path": sanitize_path(str(snapshot.resolve())),
            "aggregate_sha256": manifest["aggregate_sha256"],
            "questions_sha256": manifest["questions_sha256"],
            "manifest_sha256": manifest_sha,
            "source_commit": manifest["source_commit"],
            "n_files": n_files,
            "n_questions": len(questions),
            "n_families": manifest["n_families"],
            "labels": manifest["labels"],
            "tracked_blob_mismatches": tracked_blob_mismatches(manifest),
        },
        "interpretation": "generated file lookup (questions generated from top-level JSON fields, named by path "
                          "words); 0 human-reviewed labels; path-level source-file retrieval, not answer correctness",
        "stand_ins": "hash embedder (no model); no reranker",
        "declaration": DECLARATION,
        "parent": parent,
        "arm": arm,
        "splits": list(splits),
        "ingest": ingest,
        "summary": summary,
        "per_query": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", choices=sorted(CHECKS))
    parser.add_argument("--snapshot", type=Path, default=Path(DEFAULT_SNAPSHOT))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", default=",".join(SPLITS), help="comma-separated subset of dev,test")
    args = parser.parse_args()
    snapshot = args.snapshot if args.snapshot.is_absolute() else REPO_ROOT / args.snapshot
    result = run_check(args.check, snapshot, tuple(s for s in args.splits.split(",") if s))
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    brief = {split: {"verdict": block["verdict"], f"paired_delta_{PRIMARY_METRIC}": block["paired_delta"][PRIMARY_METRIC],
                     **({"fired": block["fired"]} if "fired" in block else {})} for split, block in result["summary"].items()}
    print(json.dumps({"check": args.check, "wrote": sanitize_path(str(out)), "summary": brief}, indent=2))


if __name__ == "__main__":
    main()
