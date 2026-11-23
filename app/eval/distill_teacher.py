"""The teacher's scores for the synthetic queries (M43): the candidates a served reranker would see, scored by the
teacher, with no label anywhere.

    python -m app.eval.distill_teacher score --data-dir <BEIR dir> --work <dir> [--shard 0/4]
    python -m app.eval.distill_teacher merge --work <dir> [--out-dir bench/distill]

For every synthetic query of bench/distill/queries.jsonl.gz (app/eval/distill_queries.py):

  candidates  the fused top CANDIDATES_TOP of hybrid@bge-small over the query's own corpus, built as the benchmark rows
              build it: BM25 and bge-small-en-v1.5 fused by RRF (the fixture through InMemoryHybridIndex, SciFact and
              FiQA through the BEIR runner's engine with its Anserini analyzer and its cached document embeddings).
              These are the lists a reranker receives when it serves. The query's source document is appended when it
              is not among them, so every list holds the document the query was written from.
  scores      the teacher's raw logit for each (query, candidate text) pair: mxbai-rerank-large-v2 in fp32 at
              max_length 512, as M41 measured it, loaded through the same get_cross_encoder and checked with the same
              load-and-score check before it scores anything.

`score` takes one shard of the queries (every n-th, by id) and appends a line per query, skipping the ones already
written, so a job that ran out of time is submitted again. `merge` checks that the shards cover every query exactly
once and writes bench/distill/teacher_scores.jsonl.gz with its header (teacher_scores.meta.json).

This step opens each BEIR corpus.jsonl and the fixture's files; never queries.jsonl, never the qrels, never a gold
question. A list carries no relevance label: only where the source document ranked, and what the teacher said.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.eval.distill_queries import QUERIES_FILE, read_jsonl, write_jsonl_gz
from app.eval.provenance import command_line, device_info, git_sha, git_state, model_device, model_provenance, run_provenance
from app.eval.small_rerankers import SHAPE, TEACHER_MODEL

CANDIDATES_TOP = SHAPE["rerank_top"]  # the served depth: the fused top 20
FIRST_STAGE = "hybrid@bge-small"
BGE_SMALL = "st:BAAI/bge-small-en-v1.5"
FIXTURE_RETRIEVE_K = 50  # the fixture rows fuse the top 50 chunks of each leg (neural.json); the BEIR rows the top 100 documents
TEACHER_BATCH = 64  # pairs per forward pass; predict() sorts a block's pairs by length first
BLOCK = 400  # lists scored per predict() call, then written: what an interrupted shard loses at most
SCORES_FILE, SCORES_META = "teacher_scores.jsonl.gz", "teacher_scores.meta.json"


class FixtureFirstStage:
    """hybrid@bge-small over fixture v1's chunks, as neural.json's row builds it (router and reranker off)."""

    def __init__(self, root: Path):
        from app.eval.index_loader import build_index

        self.index = build_index(root / "bench" / "corpus" / "files", embedder_name=BGE_SMALL, retrieve_k=FIXTURE_RETRIEVE_K,
                                 enable_router=False, enable_rerank=False, mtimes="manifest")
        self.texts = {c.chunk_id: c.text for c in self.index.chunks}
        self.provenance = {"index": "InMemoryHybridIndex, router off, reranker off", "retrieve_k": FIXTURE_RETRIEVE_K, "unit": "chunk",
                           "n_units": len(self.texts)}

    def top(self, query: str, n: int) -> list[str]:
        return [h.chunk.chunk_id for h in self.index.retrieve(query, k=FIXTURE_RETRIEVE_K).hits[:n]]


class BeirFirstStage:
    """hybrid@bge-small over a BEIR corpus, as the BEIR runner's row of that name builds it."""

    def __init__(self, data_dir: Path, name: str, device: str = "auto"):
        from app.eval import beir

        self.texts = beir.load_corpus(Path(data_dir) / name)
        self.engine = beir.Engine(beir.chunks_from_corpus(self.texts), cache_dir=Path(data_dir) / "cache", device=device)
        self.cfg = next(c for c in beir.DEFAULT_CONFIGS if c.name == FIRST_STAGE)
        self.engine.build(self.cfg)
        self._beir = beir
        self.provenance = {"index": "app.eval.beir.Engine", "analyzer": self.cfg.analyzer, "k1": self.cfg.k1, "b": self.cfg.b, "k": self.cfg.k,
                           "rrf_k": self.cfg.rrf_k, "unit": "document", "n_units": len(self.texts), "corpus_sha256": self.engine.corpus_sha256,
                           "doc_embeddings": self.engine.provenance[self.cfg.embedder].get("doc_embeddings")}

    def top(self, query: str, n: int) -> list[str]:
        from app.eval.latency import StageTimer

        return self._beir.retrieve(self.engine, self.cfg, query, StageTimer())[:n]


def first_stage(corpus: str, root: Path, data_dir: Path, device: str = "auto"):
    return FixtureFirstStage(root) if corpus == "fixture" else BeirFirstStage(data_dir, corpus, device)


def corpus_texts(corpus: str, root: Path, data_dir: Path) -> dict[str, str]:
    """{candidate id: text} of one corpus without building an index (the training step reads the lists' texts here)."""
    if corpus == "fixture":
        from app.eval.distill_queries import fixture_documents

        return dict(fixture_documents(root))
    from app.eval.beir import load_corpus

    return load_corpus(Path(data_dir) / corpus)


def candidate_list(stage, query: str, source_id: str, top: int = CANDIDATES_TOP) -> tuple[list[str], int | None]:
    """(the fused top `top` ids with the source document appended when absent, the source's rank among the fused top:
    1-based, None when it was appended)."""
    ids = list(stage.top(query, top))
    if source_id in ids:
        return ids, ids.index(source_id) + 1
    if source_id not in stage.texts:
        raise KeyError(f"source document {source_id!r} is not in the corpus")
    return ids + [source_id], None


def shard_of(queries: Sequence[dict], shard: tuple[int, int]) -> list[dict]:
    """Every n-th query by id, starting at i: shard (i, n)."""
    i, n = shard
    if not 0 <= i < n:
        raise ValueError(f"shard {i}/{n}: need 0 <= i < n")
    return sorted(queries, key=lambda q: q["id"])[i::n]


def score_block(model, lists: Sequence[tuple[str, list[str]]], batch_size: int = TEACHER_BATCH) -> list[list[float]]:
    """The model's raw logit for every (query, text) of every list, regrouped per list."""
    from app.retrieval.rerank import cross_encoder_logits

    logits = cross_encoder_logits(model, [(q, t) for q, texts in lists for t in texts], batch_size=batch_size)
    out, at = [], 0
    for _, texts in lists:
        out.append([round(x, 6) for x in logits[at: at + len(texts)]])
        at += len(texts)
    if not all(np.isfinite(row).all() for row in out):
        raise RuntimeError("the teacher returned a score that is not finite")
    return out


def score_shard(queries: Sequence[dict], stages: dict[str, Any], model, out_path: Path, *, block: int = BLOCK) -> dict[str, int]:
    """Append one line per query of `queries` not yet in `out_path`: its candidates and the teacher's logits."""
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [q for q in queries if q["id"] not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), block):
            rows, lists = [], []
            for q in todo[start: start + block]:
                stage = stages[q["corpus"]]
                ids, rank = candidate_list(stage, q["query"], q["doc_id"])
                rows.append({"id": q["id"], "corpus": q["corpus"], "candidates": ids, "source_rank": rank})
                lists.append((q["query"], [stage.texts[i] for i in ids]))
            for row, scores in zip(rows, score_block(model, lists)):
                fh.write(json.dumps({**row, "scores": scores}, ensure_ascii=False) + "\n")
                pairs += len(scores)
            fh.flush()
    return {"already_done": len(done), "scored": len(todo), "pairs": pairs}


def summarise(rows: Sequence[dict]) -> dict[str, Any]:
    """Per corpus: lists, pairs, where the source document ranked in the fused list and with the teacher, and the
    spread of the teacher's logits."""
    out: dict[str, Any] = {}
    for corpus in sorted({r["corpus"] for r in rows}):
        mine = [r for r in rows if r["corpus"] == corpus]
        scores = np.concatenate([np.asarray(r["scores"], dtype=np.float64) for r in mine])
        source_top = 0
        for r in mine:
            at = (r["source_rank"] or len(r["candidates"])) - 1  # an appended source is the last candidate
            source_top += int(np.argmax(r["scores"])) == at
        out[corpus] = {
            "lists": len(mine), "pairs": int(scores.size),
            "source_in_fused_top": sum(1 for r in mine if r["source_rank"] is not None),
            "source_first_in_fused": sum(1 for r in mine if r["source_rank"] == 1),
            "source_first_with_teacher": source_top,
            "logit": {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4), "min": round(float(scores.min()), 4),
                      "max": round(float(scores.max()), 4), "percentiles_5_25_50_75_95": [round(float(x), 4) for x in np.percentile(scores, [5, 25, 50, 75, 95])]},
        }
    return out


def _shard_paths(work: Path, shard: tuple[int, int]) -> tuple[Path, Path]:
    stem = f"teacher.shard-{shard[0]}-of-{shard[1]}"
    return work / f"{stem}.jsonl", work / f"{stem}.meta.json"


def _parse_args(argv: list[str] | None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["score", "merge"])
    p.add_argument("--data-dir", type=Path, default=root / "bench" / "external", help="where the BEIR datasets and the embedding cache are")
    p.add_argument("--work", type=Path, required=True, help="where the shards are written and read")
    p.add_argument("--queries", type=Path, default=root / "bench" / "distill" / QUERIES_FILE)
    p.add_argument("--out-dir", type=Path, default=root / "bench" / "distill", help="merge: where the committed file goes")
    p.add_argument("--shard", default="0/1", help="score: i/n, every n-th query by id starting at i")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = git_state(root)
    queries = read_jsonl(args.queries)
    queries_sha = hashlib.sha256(args.queries.read_bytes()).hexdigest()
    if args.step == "score":
        from app.eval.models_2026 import check_reranker
        from app.retrieval.rerank import cross_encoder_provenance, get_cross_encoder

        i, n = (int(x) for x in args.shard.split("/"))
        mine = shard_of(queries, (i, n))
        out_path, meta_path = _shard_paths(args.work, (i, n))
        t0 = time.perf_counter()
        stages = {corpus: first_stage(corpus, root, args.data_dir, args.device) for corpus in sorted({q["corpus"] for q in mine})}
        model = get_cross_encoder(TEACHER_MODEL, precision=SHAPE["precision"], max_length=SHAPE["max_length"],
                                  device=None if args.device == "auto" else args.device, raise_errors=True)
        teacher = {**model_provenance(TEACHER_MODEL, "teacher"), "precision": SHAPE["precision"], "max_length": SHAPE["max_length"],
                   "device": model_device(model), **cross_encoder_provenance(model), "load_and_score": check_reranker(model, TEACHER_MODEL),
                   "scores": "raw logits (no activation)", "batch_size": TEACHER_BATCH}
        built = time.perf_counter() - t0
        counts = score_shard(mine, stages, model, out_path)
        provenance = run_provenance(root=root, started=started)
        device = device_info(args.device)
        # a shard that ran out of time is submitted again: every run that wrote lines is listed, and merge refuses
        # shards whose runs were not all at one clean commit
        earlier = json.loads(meta_path.read_text())["runs"] if meta_path.exists() else []
        run = {"git_head": provenance["git_head"], "working_tree_dirty": provenance["working_tree_dirty"],
               "timestamp": datetime.now(timezone.utc).isoformat(), "device": device.get("device_name"), **counts,
               "build_seconds": round(built, 1), "score_seconds": round(time.perf_counter() - t0 - built, 1)}
        meta = {"git_sha": git_sha(root), **provenance, "timestamp": run["timestamp"],
                "command": command_line(), "device": device, "shard": [i, n], "queries_sha256": queries_sha,
                "n_queries": len(mine), "runs": [*earlier, run],
                "teacher": teacher,
                "first_stage": {"config": FIRST_STAGE, "candidates_top": CANDIDATES_TOP, "source_document": "appended when it is not among them",
                                "embedder": model_provenance(BGE_SMALL, "embedder"), "corpora": {c: s.provenance for c, s in stages.items()}}}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({"shard": [i, n], "n_queries": len(mine), **run}))
        return 0

    metas = sorted(args.work.glob("teacher.shard-*-of-*.meta.json"))
    if not metas:
        raise SystemExit(f"no shard in {args.work}")
    shards = [json.loads(p.read_text()) for p in metas]
    n = shards[0]["shard"][1]
    if sorted(s["shard"][0] for s in shards) != list(range(n)) or any(s["shard"][1] != n for s in shards):
        raise SystemExit(f"shards {[s['shard'] for s in shards]} are not 0..{n - 1} of {n}")
    runs = [r for s in shards for r in s["runs"]]
    if len({s["queries_sha256"] for s in shards}) != 1 or shards[0]["queries_sha256"] != queries_sha:
        raise SystemExit("the shards were not all scored for this queries file")
    if len({r["git_head"] for r in runs}) != 1:
        raise SystemExit("the shards' runs were not all at one commit")
    if any(r["working_tree_dirty"] for r in runs):
        raise SystemExit("a shard ran with uncommitted changes")
    per_shard = {i: read_jsonl(_shard_paths(args.work, (i, n))[0]) for i in range(n)}
    rows = [r for i in range(n) for r in per_shard[i]]
    by_id = {r["id"]: r for r in rows}
    wanted = [q["id"] for q in sorted(queries, key=lambda q: q["id"])]
    if len(rows) != len(by_id) or set(by_id) != set(wanted):
        raise SystemExit(f"{len(rows)} lines, {len(by_id)} distinct ids, {len(wanted)} queries: the shards do not cover every query exactly once")
    ordered = [by_id[i] for i in wanted]
    sha = write_jsonl_gz(args.out_dir / SCORES_FILE, ordered)
    first = shards[0]
    meta = {"git_sha": first["git_sha"], "git_head": first["git_head"], "working_tree_dirty": False, "source_sha256": first["source_sha256"],
            "timestamp": max(s["timestamp"] for s in shards), "command": first["command"].replace(f"--shard 0/{n}", f"--shard <i>/{n}"),
            "merged_by": command_line(),
            "file": {"name": SCORES_FILE, "sha256": sha, "n": len(ordered),
                     "fields": "id, corpus, candidates (fused order, the source appended last when absent), source_rank (1-based in the fused top; null: appended), scores (the teacher's raw logits, one per candidate)"},
            "queries": {"file": QUERIES_FILE, "sha256": queries_sha},
            "teacher": first["teacher"], "first_stage": {**first["first_stage"], "corpora": {c: p for s in shards for c, p in s["first_stage"]["corpora"].items()}},
            "shards": [{"shard": s["shard"], "device": s["device"].get("device_name"), "cpu_model": s["device"].get("cpu_model"), "torch": s["device"].get("torch"),
                        "sentence_transformers": s["device"].get("sentence_transformers"), "lists": len(per_shard[s["shard"][0]]),
                        "pairs": sum(len(r["scores"]) for r in per_shard[s["shard"][0]]),
                        "runs": [{k: r[k] for k in ("timestamp", "device", "scored", "pairs", "score_seconds")} for r in s["runs"]]} for s in shards],
            "summary": summarise(ordered),
            "labels": "none: a list carries where the source document ranked and what the teacher said, never a relevance judgment"}
    (args.out_dir / SCORES_META).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["summary"], indent=1))
    print(f"wrote {args.out_dir / SCORES_FILE} ({len(ordered)} lists, sha256 {sha[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
