"""BEIR SciFact, NFCorpus and FiQA with reproducible external evaluation.

    python -m app.eval.beir --out bench/results/beir_scifact.json [--device auto|cpu|cuda] [--rerank-top 20]
    python -m app.eval.beir --ladder bm25-ablation --device cpu --out bench/results/beir_scifact_bm25_ablation.json

Dataset: SciFact (Wadden et al., 2020) in the BEIR packaging (Thakur et al.,
2021): 5,183 abstracts, 300 test queries with binary qrels. Downloaded once to
bench/external/ (gitignored); the zip's sha256 is pinned below.

Published reference: Anserini's BEIR regression for SciFact, "flat" BM25 over
title + text concatenated, Lucene English analyzer (Porter stemming +
stopwords), k1=0.9, b=0.4: nDCG@10 = 0.6789
(https://github.com/castorini/anserini/blob/master/docs/regressions/regressions-beir-v1.0.0-scifact.flat.md).
The `bm25` row uses the same document text, analyzer and parameters, so the
two are comparable. A `bm25@default-tokenizer` row is kept next to it because
that tokenizer (no stemming, no stopwords, k1=1.5, b=0.75) is what fixture v1
is scored with. That row changes the analyzer, k1 and b at once; the
bm25-ablation ladder adds one row per change (the analyzer alone; k1 and b
alone), each paired against bm25.

Rows form an explicit ablation chain (each row names its `parent`, and the
paired delta is against that parent, not against list order): bm25 ->
dense@bge-small -> hybrid@bge-small -> hybrid+bge-rerank (top 20, fp32,
library-default token cap) -> fp32 with max_length 512 -> fp16 at the same cap -> rerank depth
50 -> 100. dense@bge-small+instruction is the bge-v1.5 query prefix on the
query side only (document embeddings are cached). The dense leg ranks as the
fixture benchmark does (app/retrieval/hybrid.py): float64 cosine rounded to 10
decimals, exact ties in corpus order. Metrics: nDCG@10,
Recall@10, Recall@100 and MRR@10 with 95% paired bootstrap CIs; per-query
latency p50/p95 end to end and per stage, measured after a warm-up query and
with index / embedding / model build time reported separately (build_ms);
model revisions, precision, max_length and device recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.eval.latency import StageTimer
from app.eval.metrics import mrr_at_k, recall_at_k
from app.eval.provenance import command_line, device_info, git_sha, git_state, model_device, model_provenance, run_provenance, sanitize_path
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, mean_ci, paired_delta_ci, resample_indices, settings as bootstrap_settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedders import HashEmbedder, encode_queries
from app.retrieval.fuse import rrf_score_map
from app.retrieval.hybrid import dense_scores, rank_by_score, unit_rows
from app.retrieval.rerank import OverlapReranker, cross_encoder_provenance, cross_encoder_scores, get_cross_encoder
from app.retrieval.types import Chunk

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
PUBLISHED = {
    "bm25_ndcg@10": 0.6789,
    "source": "Anserini BEIR (v1.0.0) regression, SciFact, 'flat' BM25: title + text concatenated into one field, Lucene English analyzer (Porter stemming + stopwords), k1=0.9, b=0.4",
    "url": "https://github.com/castorini/anserini/blob/master/docs/regressions/regressions-beir-v1.0.0-scifact.flat.md",
    "analyzer": "Lucene English (Porter stemming + stopwords), k1=0.9, b=0.4",
    "document_text": "title + ' ' + text (same as our rows)",
}
BGE_SMALL = "st:BAAI/bge-small-en-v1.5"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
REFERENCE = "bm25"
RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CI_METRICS = ("ndcg@10", "recall@10", "recall@100")
DATASETS = {
    "scifact": {"url": SCIFACT_URL, "sha256": SCIFACT_SHA256, "published": PUBLISHED},
    # Observed official archive identities on 2026-10-07, pinned for replay.
    "nfcorpus": {"url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip", "sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b", "published": None},
    "fiqa": {"url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip", "sha256": "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02", "published": None},
}


@dataclass
class BeirConfig:
    name: str
    mode: str  # bm25 | dense | hybrid
    analyzer: str = "beir"
    k1: float = 0.9
    b: float = 0.4
    embedder: str | None = None  # "st:<model>" or "hash"
    reranker: str | None = None  # "<hf model>" or "overlap"
    rerank_top: int = 20
    k: int = 100
    rrf_k: int = 60
    parent: str | None = None  # the row this one is a single change away from (paired delta target)
    precision: str = "fp32"  # reranker weights: fp32 | fp16 | bf16 (the half precisions on CUDA only)
    max_length: int | None = None  # reranker token cap; None = the model's own
    query_instruction: str | None = None  # prefix added to queries only (bge-v1.5 recommends one)


RERANK_FP32 = "hybrid+bge-rerank@bge-small"
RERANK_FP32_512 = "hybrid+bge-rerank@bge-small/fp32-512/top20"
RERANK_FP16 = "hybrid+bge-rerank@bge-small/fp16-512/top20"
DEFAULT_CONFIGS = [
    BeirConfig(name="bm25", mode="bm25"),
    BeirConfig(name="bm25@default-tokenizer", mode="bm25", analyzer="default", k1=1.5, b=0.75, parent="bm25"),
    BeirConfig(name="dense@bge-small", mode="dense", embedder=BGE_SMALL, parent="bm25"),
    BeirConfig(name="dense@bge-small+instruction", mode="dense", embedder=BGE_SMALL, parent="dense@bge-small", query_instruction=BGE_QUERY_INSTRUCTION),
    BeirConfig(name="hybrid@bge-small", mode="hybrid", embedder=BGE_SMALL, parent="dense@bge-small"),
    BeirConfig(name=RERANK_FP32, mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=20, parent="hybrid@bge-small"),
    BeirConfig(name=RERANK_FP32_512, mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=20, parent=RERANK_FP32, precision="fp32", max_length=512),
    BeirConfig(name=RERANK_FP16, mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=20, parent=RERANK_FP32_512, precision="fp16", max_length=512),
    BeirConfig(name="hybrid+bge-rerank@bge-small/fp16-512/top50", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=50, parent=RERANK_FP16, precision="fp16", max_length=512),
    BeirConfig(name="hybrid+bge-rerank@bge-small/fp16-512/top100", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=100, parent="hybrid+bge-rerank@bge-small/fp16-512/top50", precision="fp16", max_length=512),
]
# bm25@default-tokenizer changes the analyzer, k1 and b at once. These rows change the analyzer
# alone and (k1, b) alone, each paired against bm25, next to the combined row; CPU only, no model.
BM25_ABLATION_CONFIGS = [
    DEFAULT_CONFIGS[0],
    BeirConfig(name="bm25@default-analyzer", mode="bm25", analyzer="default", parent="bm25"),
    BeirConfig(name="bm25@k1-1.5-b-0.75", mode="bm25", k1=1.5, b=0.75, parent="bm25"),
    DEFAULT_CONFIGS[1],
]


def models_2026_configs(which: str = "main") -> list[BeirConfig]:
    """M41: the current open models (app/eval/models_2026.py) beside the committed bge rows, recomputed in the same run.
    Parents: bm25, dense@bge-small, hybrid@bge-small and the bge-reranker-v2-m3 row at the settings both committed files
    have (RERANK_FP32_512: top 20, fp32, max_length 512). An embedder's dense and hybrid rows are paired with
    bge-small's; a reranker over hybrid@bge-small with the bge reranker row, at its settings (Qwen3-Reranker-8B in
    bf16). The 512-token cap counts the causal-LM rerankers' chat template too, so their passages get fewer tokens.
    main: Qwen3-Embedding 0.6B and 8B, four rerankers; harrier: harrier-oss-v1-27b (an 80 GB GPU)."""
    from app.eval.models_2026 import EMBEDDERS, HARRIER, RERANKERS

    if which not in ("main", "harrier"):
        raise ValueError(f"unknown models-2026 set {which!r}; expected main or harrier")
    by = {c.name: c for c in DEFAULT_CONFIGS}
    embedders = [t for t in EMBEDDERS if t != HARRIER] if which == "main" else [HARRIER]
    rerankers = list(RERANKERS) if which == "main" else []
    configs = [replace(by[n]) for n in ("bm25", "dense@bge-small", "hybrid@bge-small")]
    if rerankers:
        configs.append(replace(by[RERANK_FP32_512], parent="hybrid@bge-small"))  # as in beir_fiqa.json (its uncapped parent is not run)
    for tag in embedders:  # a model's rows next to each other: it is released after its last row (Engine.release_unused)
        configs.append(BeirConfig(name=f"dense@{tag}", mode="dense", embedder=EMBEDDERS[tag], parent="dense@bge-small"))
        configs.append(BeirConfig(name=f"hybrid@{tag}", mode="hybrid", embedder=EMBEDDERS[tag], parent="hybrid@bge-small"))
    for tag in rerankers:
        model, precision = RERANKERS[tag]
        configs.append(BeirConfig(name=f"hybrid+{tag}@bge-small/{precision}-512/top20", mode="hybrid", embedder=BGE_SMALL, reranker=model,
                                  rerank_top=20, precision=precision, max_length=512, parent=RERANK_FP32_512))
    return configs


MODELS_2026_LADDERS = {"models-2026": "main", "models-2026-harrier": "harrier"}
# The committed file each dataset's recomputed parents are compared with (app.eval.models_2026.reproduces).
COMMITTED_FILES = {"scifact": "bench/results/beir_scifact.json", "fiqa": "bench/results/beir_fiqa.json"}
MODELS_2026_PARENTS = ("bm25", "dense@bge-small", "hybrid@bge-small", RERANK_FP32_512)


def small_reranker_configs(with_students: bool | None = None) -> list[BeirConfig]:
    """M43: the small rerankers of app/eval/small_rerankers.py over hybrid@bge-small, every one at the served shape (the
    fused top 20, max_length 512, fp32) and every one paired with the unreranked hybrid row, its parent: the app's
    default reranker, the teacher, the candidates as published, then the students (a student's `reranker` is its tag;
    the run resolves it to a verified local checkpoint). with_students None: only once their manifest is committed
    (bench/distill/students.json), so the zero-shot rows can be measured before any student exists."""
    from app.eval import small_rerankers as sr

    if with_students is None:
        with_students = (Path(__file__).resolve().parents[2] / sr.STUDENT_MANIFEST).exists()
    by = {c.name: c for c in DEFAULT_CONFIGS}
    configs = [replace(by[sr.PARENT], parent=None)]
    for tag in [t for t in sr.reranker_tags() if with_students or t not in sr.STUDENTS]:
        configs.append(BeirConfig(name=sr.row_name(tag), mode="hybrid", embedder=BGE_SMALL, reranker=tag if tag in sr.STUDENTS else sr.model_of(tag),
                                  rerank_top=sr.SHAPE["rerank_top"], precision=sr.SHAPE["precision"], max_length=sr.SHAPE["max_length"], parent=sr.PARENT))
    return configs


def small_reranker_comparisons(configs: list[BeirConfig]) -> list[tuple[str, str]]:
    """Beside each row's delta against the unreranked hybrid (its parent, the bar): its delta against the app's default
    reranker and against the teacher, and a student's against the candidate it started from."""
    from app.eval import small_rerankers as sr

    names = [c.name for c in configs]
    teacher = sr.row_name(sr.TEACHER_TAG)
    out = [(n, ref) for n in names for ref in (sr.BGE_ROW, teacher) if n not in (sr.PARENT, ref) and ref in names]
    out += [(sr.row_name(s), sr.row_name(init)) for s, init in sr.STUDENTS.items() if sr.row_name(s) in names and sr.row_name(init) in names]
    return out


def small_reranker_shape_configs(rows: str = "all") -> list[BeirConfig]:
    """M51: the six models of app/eval/small_rerankers_shapes.py at every shape of its grid (depth 5/10/20 x max_length
    256/512, fp32) over hybrid@bge-small, each paired with the unreranked hybrid, its parent. A model's rows are next to
    each other, grouped by cap, so each (model, cap) is loaded once and released after its last row. rows "anchors":
    the parent and the six 20/512 rows only (M43's committed rows, so the smoke computes no new number)."""
    from app.eval import small_rerankers as sr
    from app.eval import small_rerankers_shapes as shapes

    if rows not in ("all", "anchors"):
        raise ValueError(f"unknown row filter {rows!r}; expected all or anchors")
    by = {c.name: c for c in DEFAULT_CONFIGS}
    configs = [replace(by[sr.PARENT], parent=None)]
    for tag, depth, cap in shapes.GRID:
        if rows == "anchors" and (depth, cap) != shapes.ANCHOR:
            continue
        configs.append(BeirConfig(name=shapes.row_name(tag, depth, cap), mode="hybrid", embedder=BGE_SMALL, reranker=tag if tag in sr.STUDENTS else sr.model_of(tag),
                                  rerank_top=depth, precision=shapes.PRECISION, max_length=cap, parent=sr.PARENT))
    return configs


def small_reranker_ml256_configs() -> list[BeirConfig]:
    """M51 Phase 2: the parent, the attempt-1 students and their inits at depth 5/10/20 x max_length 256 (anchors: they
    must equal M51's committed shapes rows) and the two students retrained at 256 (attempt 2) at the same shapes."""
    from app.eval import small_rerankers as sr
    from app.eval import small_rerankers_shapes as shapes

    by = {c.name: c for c in DEFAULT_CONFIGS}
    configs = [replace(by[sr.PARENT], parent=None)]
    for name, tag, depth, cap in shapes.ml256_rows():
        student = tag in sr.STUDENTS or tag in shapes.PHASE2["students"]
        configs.append(BeirConfig(name=name, mode="hybrid", embedder=BGE_SMALL, reranker=tag if student else sr.model_of(tag),
                                  rerank_top=depth, precision=shapes.PRECISION, max_length=cap, parent=sr.PARENT))
    return configs


SMALL_RERANKERS_LADDER = "small-rerankers"
SMALL_RERANKER_SHAPES_LADDER = "small-rerankers-shapes"
SMALL_RERANKER_ML256_LADDER = "small-rerankers-ml256"
# The committed files a small-rerankers run compares its recomputed rows with: the parent and the app's default reranker
# with the dataset's first file, the teacher with M41's row of the same name.
SMALL_RERANKERS_REPRODUCE = {"scifact": (("bench/results/beir_scifact.json", ("hybrid@bge-small", RERANK_FP32_512)),
                                         ("bench/results/beir_scifact_models_2026.json", ("hybrid+mxbai-rerank-large-v2@bge-small/fp32-512/top20",))),
                             "fiqa": (("bench/results/beir_fiqa.json", ("hybrid@bge-small", RERANK_FP32_512)),
                                      ("bench/results/beir_fiqa_models_2026.json", ("hybrid+mxbai-rerank-large-v2@bge-small/fp32-512/top20",)))}
LADDERS = {"default": DEFAULT_CONFIGS, "bm25-ablation": BM25_ABLATION_CONFIGS,
           **{name: models_2026_configs(which) for name, which in MODELS_2026_LADDERS.items()},
           SMALL_RERANKERS_LADDER: small_reranker_configs(), SMALL_RERANKER_SHAPES_LADDER: small_reranker_shape_configs(),
           SMALL_RERANKER_ML256_LADDER: small_reranker_ml256_configs()}


def models_2026_comparisons(configs: list[BeirConfig]) -> list[tuple[str, str]]:
    """Beside the parent deltas: a new embedder's hybrid row against its own dense row, and each new reranker's row
    against the unreranked hybrid@bge-small (what the reranker adds, as the bge row's own parent delta shows)."""
    names = {c.name for c in configs}
    out = []
    for c in configs:
        if c.mode == "hybrid" and not c.reranker and c.name != "hybrid@bge-small" and c.name.replace("hybrid@", "dense@", 1) in names:
            out.append((c.name, c.name.replace("hybrid@", "dense@", 1)))
        if c.reranker and c.parent != "hybrid@bge-small" and "hybrid@bge-small" in names:
            out.append((c.name, "hybrid@bge-small"))
    return out


# ---------------------------------------------------------------- data


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_scifact(data_dir: Path, url: str = SCIFACT_URL, sha256: str | None = SCIFACT_SHA256) -> Path:
    """Backward-compatible SciFact download entry point."""
    return download_dataset(data_dir, "scifact", url=url, sha256=sha256)


def download_dataset(data_dir: Path, dataset: str, *, url: str | None = None, sha256: str | None = None) -> Path:
    """Fetch a supported BEIR archive, record its identity and extract safely.

    All three supported datasets have fixed archive checksum pins. An explicit
    --dataset-sha256 overrides the expected identity for a deliberate replay.
    """
    spec = DATASETS[dataset]
    expected = sha256 or spec["sha256"]
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{dataset}.zip"
    if not zip_path.exists():
        temporary = zip_path.with_suffix(".zip.partial")
        urllib.request.urlretrieve(url or spec["url"], temporary)
        temporary.replace(zip_path)
    actual = sha256_of(zip_path)
    if expected and actual != expected:
        raise RuntimeError(f"{dataset}.zip sha256 {actual} != pinned {expected}; delete it and retry")
    identity = data_dir / f"{dataset}.sha256"
    if identity.exists() and identity.read_text().strip() != actual:
        raise RuntimeError(f"{dataset}.zip changed after first download; sha256 differs from recorded identity")
    root = data_dir / dataset
    if not (root / "corpus.jsonl").exists():
        with zipfile.ZipFile(zip_path) as zf:
            base = data_dir.resolve()
            for member in zf.infolist():
                if not (base / member.filename).resolve().is_relative_to(base):
                    raise ValueError(f"unsafe archive member {member.filename!r}")
            zf.extractall(data_dir)
    identity.write_text(actual + "\n")
    return root


def load_scifact(root: Path, split: str = "test") -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Backward-compatible alias for the generic BEIR format loader."""
    return load_beir(root, split)


def load_corpus(root: Path) -> dict[str, str]:
    """A BEIR corpus alone, {id: title + ' ' + text} in file order. It opens corpus.jsonl and nothing else: the
    synthetic training queries of M43 are written from documents sampled here, by code that must never read
    queries.jsonl (it holds the test queries) or the qrels (app/eval/distill_queries.py)."""
    corpus: dict[str, str] = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                corpus[str(row["_id"])] = (row.get("title", "") + " " + row.get("text", "")).strip()
    return corpus


def load_beir(root: Path, split: str = "test") -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Load BEIR JSONL/TSV, retaining positive graded relevance exactly."""
    corpus = load_corpus(root)
    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        header = fh.readline()
        assert header.lower().startswith("query-id"), header
        for line in fh:
            if not line.strip():
                continue
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    queries: dict[str, str] = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if str(row["_id"]) in qrels:
                    queries[str(row["_id"])] = row["text"]
    return corpus, queries, qrels


def graded_ndcg_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    """trec_eval ndcg_cut: linear relevance grades / log2(rank + 1).

    BEIR uses pytrec_eval, whose default gains are the numeric qrel grades,
    not 2**grade - 1. Duplicate retrieved ids are refused by the run writer.
    """
    gains = [max(0, qrels.get(did, 0)) for did in retrieved[:k]]
    ideal = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)[:k]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def trec_scores(rankings: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    return {qid: {did: float(len(ids) - i) for i, did in enumerate(ids)} for qid, ids in rankings.items()}


def cross_check_metrics(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], per_query: list[dict], *, required: bool = False) -> dict:
    """Cross-check exact per-query scores against independent trec_eval code."""
    try:
        import pytrec_eval
    except ImportError:
        if required:
            raise RuntimeError("--require-trec-eval needs pytrec-eval (or pytrec-eval-terrier)")
        return {"status": "not_run", "reason": "pytrec_eval not installed", "implementation": "pytrec_eval/trec_eval"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10,100"})
    reference = evaluator.evaluate(trec_scores(rankings))
    mapping = {"ndcg@10": "ndcg_cut_10", "recall@10": "recall_10", "recall@100": "recall_100"}
    maximum = 0.0
    for row in per_query:
        if row["id"] not in reference:
            raise RuntimeError(f"trec_eval omitted query {row['id']}")
        for internal, external in mapping.items():
            difference = abs(row[internal] - reference[row["id"]][external])
            maximum = max(maximum, difference)
            if difference > 1e-9:
                raise RuntimeError(f"metric mismatch {row['id']} {internal}: {row[internal]} vs {reference[row['id']][external]}")
    return {"status": "passed", "implementation": "pytrec_eval/trec_eval", "n_queries": len(per_query), "metrics": list(mapping), "max_absolute_difference": maximum}


def trec_run_text(cfg: BeirConfig, rankings: dict[str, list[str]]) -> str:
    """A config's rankings as the TREC run file write_trec_outputs writes; its sha256 is the row's ranking_sha256."""
    return "".join(f"{qid} Q0 {did} {i + 1} {len(ids) - i} {_slug(cfg.name)}\n" for qid, ids in rankings.items() for i, did in enumerate(ids))


def write_trec_outputs(directory: Path, cfg: BeirConfig, rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    run_path = directory / f"{_slug(cfg.name)}.run"
    qrels_path = directory / "qrels.txt"
    query_order_path = directory / "query_ids.json"
    for qid, ids in rankings.items():
        if len(ids) != len(set(ids)) or any(any(c.isspace() for c in value) for value in [qid, *ids]):
            raise ValueError("TREC ids must be unique per ranking and contain no whitespace")
    run_path.write_text(trec_run_text(cfg, rankings))
    qrels_path.write_text("".join(f"{qid} 0 {did} {grade}\n" for qid, docs in qrels.items() for did, grade in docs.items()))
    # TREC run files cannot represent a query with zero retrieved documents.
    # Preserve its identity and the original bootstrap order separately.
    query_order_path.write_text(json.dumps(list(rankings)) + "\n")
    return {"run": sanitize_path(str(run_path)), "run_sha256": sha256_of(run_path), "qrels": sanitize_path(str(qrels_path)), "qrels_sha256": sha256_of(qrels_path),
            "query_order": sanitize_path(str(query_order_path)), "query_order_sha256": sha256_of(query_order_path)}


def chunks_from_corpus(corpus: dict[str, str]) -> list[Chunk]:
    return [Chunk(chunk_id=did, path=did, text=text, start_byte=0, end_byte=len(text.encode("utf-8"))) for did, text in corpus.items()]


# ---------------------------------------------------------------- models


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


class Engine:
    """Shares indexes, embeddings and models across configs; caches doc embeddings on disk."""

    def __init__(self, chunks: list[Chunk], cache_dir: Path | None = None, *, device: str = "auto", check_models: bool = False,
                 resolve_reranker=None, per_query_extras=None):
        """check_models (M41): every model, once loaded, passes its load-and-score check (app/eval/models_2026.py; recorded
        in its provenance) or the run stops. resolve_reranker (M43): name -> (what to load, what the row records about
        it, a check of the loaded model or None); it may raise (a cached revision that is not the pinned one, a
        student's checkpoint whose sha256 is not the manifest's). per_query_extras (M51): (model, query, passages) -> each
        pair's token count; when set, every row also records per query the name-independent hashes of its ranking and,
        for a reranked row, how many reranked pairs are longer than its cap, both computed outside the timed stages."""
        self.check_models = check_models
        self.per_query_extras = per_query_extras
        self.resolve_reranker = resolve_reranker
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.cache_dir = cache_dir
        self.device = device
        # Corpus identity and ordering matter: two datasets of the same size
        # must never silently share a document embedding matrix.
        self.corpus_sha256 = hashlib.sha256(json.dumps([(c.chunk_id, c.text) for c in chunks], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        self._bm25: dict[tuple[str, float, float], BM25Index] = {}
        self._embedders: dict[str, Any] = {}
        self._doc_mats: dict[str, np.ndarray] = {}
        self._unit_mats: dict[str, np.ndarray] = {}
        self._rerankers: dict[str, Any] = {}
        self.provenance: dict[str, dict] = {}

    def bm25(self, analyzer: str, k1: float, b: float) -> BM25Index:
        key = (analyzer, k1, b)
        if key not in self._bm25:
            self._bm25[key] = BM25Index(k1=k1, b=b, analyzer=analyzer).fit(
                [c.chunk_id for c in self.chunks], [c.text for c in self.chunks]
            )
        return self._bm25[key]

    def embedder(self, name: str):
        if name not in self._embedders:
            if name == "hash":
                self._embedders[name] = HashEmbedder()
            elif name.startswith("st:"):
                from app.retrieval.embedders import SentenceTransformerEmbedder

                self._embedders[name] = SentenceTransformerEmbedder(name.split(":", 1)[1])
                if self.device != "auto":
                    self._embedders[name]._model.to(self.device)
            else:
                raise ValueError(f"unknown embedder {name!r}")
            emb = self._embedders[name]
            self.provenance[name] = {
                **model_provenance(name, "embedder"),
                "dim": int(emb.dim),
                "device": model_device(getattr(emb, "_model", None)),
                "query_instruction": None,  # bge-v1.5 recommends a query prefix; not used
                # the query prompt, dtype and batch size of a sentence-transformers embedder (M41)
                **(emb.provenance() if hasattr(emb, "provenance") else {}),
            }
            if self.check_models and name.startswith("st:"):
                from app.eval.models_2026 import check_embedder

                self.provenance[name]["load_and_score"] = check_embedder(emb)
        return self._embedders[name]

    def doc_matrix(self, name: str) -> np.ndarray:
        if name in self._doc_mats:
            return self._doc_mats[name]
        emb = self.embedder(name)
        rev = self.provenance[name].get("revision") or "norev"
        cache = self.cache_dir / f"{self.corpus_sha256}.{_slug(name)}.{rev}.{len(self.chunks)}.npy" if self.cache_dir else None
        if cache and cache.exists():
            mat = np.load(cache)
            if mat.shape != (len(self.chunks), emb.dim) or not np.isfinite(mat).all():
                raise RuntimeError(f"invalid embedding cache {cache}: expected {(len(self.chunks), emb.dim)}, got {mat.shape}")
            source = "read from the on-disk cache (an earlier run computed them)"
        else:
            mat = emb.encode([c.text for c in self.chunks])
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, mat)
            source = "computed in this run"
        self.provenance[name]["doc_embeddings"] = source
        self._doc_mats[name] = mat
        return mat

    def unit_matrix(self, name: str) -> np.ndarray:
        """The document matrix as float64 unit rows, built once per embedder: per query the dense
        leg is then one matrix-vector product (see dense_scores in app/retrieval/hybrid.py)."""
        if name not in self._unit_mats:
            self._unit_mats[name] = unit_rows(self.doc_matrix(name))
        return self._unit_mats[name]

    @staticmethod
    def reranker_key(name: str, precision: str = "fp32", max_length: int | None = None) -> str:
        return f"{name}|{precision}|{max_length or 'uncapped'}"

    def reranker(self, name: str, precision: str = "fp32", max_length: int | None = None):
        key = self.reranker_key(name, precision, max_length)
        if key not in self._rerankers:
            if name == "overlap":
                self._rerankers[key] = OverlapReranker()
                self.provenance[key] = {**model_provenance("overlap", "reranker"), "loaded": True}
            else:
                source, record, after_load = self.resolve_reranker(name) if self.resolve_reranker else (name, {}, None)
                # raise_errors: a model that does not load stops the run with the reason (never a fallback)
                model = get_cross_encoder(source, precision=precision, max_length=max_length, device=None if self.device == "auto" else self.device,
                                          raise_errors=True)
                if model is None:
                    raise RuntimeError(f"reranker {name} not available (set BGE_ALLOW_DOWNLOAD=1 or cache it)")
                inner = getattr(model, "model", None)
                self.provenance[key] = {
                    **model_provenance(name, "reranker"),
                    **record,  # M43: a candidate's pinned revision; a student's weights sha256, never its path
                    **(after_load(model) if after_load else {}),
                    "loaded": True,
                    "device": model_device(model),
                    "precision": precision,
                    "dtype": str(getattr(inner, "dtype", None)) if inner is not None else None,
                    "max_length": max_length or getattr(model, "max_length", None),
                    "score_scale": "sigmoid probability in [0, 1]",
                    # M41: the default prompt a causal-LM reranker puts before every pair (Qwen3-Reranker), its token cap
                    **{k: v for k, v in cross_encoder_provenance(model).items() if k != "dtype"},
                }
                if self.check_models:
                    from app.eval.models_2026 import check_reranker

                    self.provenance[key]["load_and_score"] = check_reranker(model, name)
                self._rerankers[key] = model
        return self._rerankers[key]

    def release_unused(self, later: list) -> None:
        """Drop every model and document matrix that none of the `later` configs uses (their provenance stays), so a
        ladder of large models never holds them all on the GPU at once; a model's rows run next to each other."""
        embedders = {c.embedder for c in later if c.mode in {"dense", "hybrid"} and c.embedder}
        rerankers = {self.reranker_key(c.reranker, c.precision, c.max_length) for c in later if c.reranker}
        dropped = [n for n in self._embedders if n not in embedders] + [k for k in self._rerankers if k not in rerankers]
        for name in dropped:
            self._embedders.pop(name, None)
            self._doc_mats.pop(name, None)
            self._unit_mats.pop(name, None)
            self._rerankers.pop(name, None)
        if dropped:
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def rerank(self, cfg, query: str, pairs: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        model = self.reranker(cfg.reranker, cfg.precision, cfg.max_length)
        if cfg.reranker == "overlap":
            return model(query, pairs)
        scores = cross_encoder_scores(model, [(query, c.text) for c, _ in pairs])
        ranked = [(pairs[i][0], scores[i]) for i in range(len(pairs))]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def build(self, cfg) -> dict[str, float]:
        """Build everything a config needs before any query is timed; return build ms per part."""
        parts: dict[str, float] = {}
        if cfg.mode in {"bm25", "hybrid"}:
            t = time.perf_counter()
            self.bm25(cfg.analyzer, cfg.k1, cfg.b)
            parts["bm25_index_ms"] = round((time.perf_counter() - t) * 1000.0, 2)
        if cfg.mode in {"dense", "hybrid"}:
            t = time.perf_counter()
            self.embedder(cfg.embedder)
            parts["embedder_load_ms"] = round((time.perf_counter() - t) * 1000.0, 2)
            t = time.perf_counter()
            self.unit_matrix(cfg.embedder)  # the cached (or computed) embeddings, then their float64 unit rows
            parts["doc_embeddings_ms"] = round((time.perf_counter() - t) * 1000.0, 2)
        if cfg.reranker:
            t = time.perf_counter()
            self.reranker(cfg.reranker, cfg.precision, cfg.max_length)
            parts["reranker_load_ms"] = round((time.perf_counter() - t) * 1000.0, 2)
        parts["total_ms"] = round(sum(parts.values()), 2)
        return parts


# ---------------------------------------------------------------- run


def retrieve(engine: Engine, cfg: BeirConfig, query: str, timer: StageTimer) -> list[str]:
    ids: list[str]
    t0 = time.perf_counter()
    if cfg.mode in {"bm25", "hybrid"}:
        with timer.stage("bm25"):
            bm25_hits = engine.bm25(cfg.analyzer, cfg.k1, cfg.b).search(query, k=cfg.k)
    if cfg.mode in {"dense", "hybrid"}:
        assert cfg.embedder, f"{cfg.name}: dense/hybrid needs an embedder"
        unit = engine.unit_matrix(cfg.embedder)
        with timer.stage("embed"):
            embedder = engine.embedder(cfg.embedder)
            # a config's typed instruction (the bge rows) is prepended as before; otherwise the model's own query
            # prompt, when it has one (QUERY_PROMPT_NAMES), else the bare query: encode_queries() is encode() then
            qv = (embedder.encode([cfg.query_instruction + query]) if cfg.query_instruction
                  else encode_queries(embedder, [query]))[0]
        with timer.stage("dense"):
            # The fixture benchmark's dense ranking (M10): float64 cosine rounded to DENSE_SCORE_DECIMALS,
            # descending, exact ties in corpus order. np.argsort over float32 cosines left ties to the sort
            # implementation and near-ties to the float32 summation order.
            scores = dense_scores(qv, unit, unit=True)
            dense_hits = [(engine.chunks[i].chunk_id, float(scores[i])) for i in rank_by_score(scores, cfg.k)]
    if cfg.mode == "bm25":
        ids = [d for d, _ in bm25_hits]
    elif cfg.mode == "dense":
        ids = [d for d, _ in dense_hits]
    else:
        with timer.stage("fuse"):
            fused = rrf_score_map([[d for d, _ in bm25_hits], [d for d, _ in dense_hits]], k=cfg.rrf_k)
            ids = [d for d, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: cfg.k]]
    if cfg.reranker and ids:
        with timer.stage("rerank"):
            head = [(engine.by_id[d], 0.0) for d in ids[: cfg.rerank_top]]
            reranked = engine.rerank(cfg, query, head)
            ids = [c.chunk_id for c, _ in reranked] + ids[cfg.rerank_top :]
    timer.record("total", (time.perf_counter() - t0) * 1000.0)
    return ids


def run_config(engine: Engine, cfg: BeirConfig, queries: dict[str, str], qrels: dict[str, dict[str, int]], *, trec_dir: Path | None = None, require_trec_eval: bool = False) -> dict[str, Any]:
    if cfg.precision in {"fp16", "bf16"} and cfg.reranker not in {None, "overlap"}:
        import torch
        if engine.device == "cpu" or not torch.cuda.is_available():
            raise RuntimeError(f"{cfg.precision} ablations require CUDA; CPU fallback would not measure precision")
    build = engine.build(cfg)  # index, embeddings, model loads: reported separately, never inside a query
    first = next(iter(queries.values()))
    retrieve(engine, cfg, first, StageTimer())  # warm-up (CUDA kernels, caches); excluded from the stats
    timer = StageTimer()
    per_query: list[dict[str, float]] = []
    rankings: dict[str, list[str]] = {}
    extra: dict[str, Any] | None = {"truncated_pairs": [], "rank_hash": {"top10": [], "top10_set": [], "ranking": []}} if engine.per_query_extras else None
    t0 = time.perf_counter()
    for qid, text in queries.items():
        ids = retrieve(engine, cfg, text, timer)
        if extra is not None:  # M51, after the timed stages: the reranked pairs the cap cut (the head before the id filter)
            extra["truncated_pairs"].append(_truncated_pairs(engine, cfg, text, ids) if cfg.reranker else None)
        # Match BEIR's default ignore_identical_ids convention.
        ids = [did for did in ids if did != qid]
        rankings[qid] = ids
        if extra is not None:
            from app.eval.small_rerankers_shapes import rank_hashes

            for key, value in rank_hashes(ids).items():  # name-independent: equal for rows that rank this query alike
                extra["rank_hash"][key].append(value)
        relevant = list(qrels[qid])
        per_query.append(
            {
                "id": qid,
                "ndcg@10": graded_ndcg_at_k(ids, qrels[qid], 10),
                "recall@10": recall_at_k(ids, relevant, 10),
                "recall@100": recall_at_k(ids, relevant, 100),
                "mrr@10": mrr_at_k(ids, relevant, 10),
            }
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    n = len(per_query)
    retrieval = {m: round(sum(r[m] for r in per_query) / n, 4) for m in ("ndcg@10", "recall@10", "recall@100", "mrr@10")}
    retrieval["n_queries"] = n
    models = {}
    if cfg.embedder:
        models["embedder"] = {**(engine.provenance.get(cfg.embedder) or {}), "query_instruction": cfg.query_instruction}
        if cfg.query_instruction and models["embedder"].get("query_prompt_name"):  # the typed instruction replaces the model's prompt
            models["embedder"].update(query_prompt_name=None, query_prompt=None)
    if cfg.reranker:
        models["reranker"] = engine.provenance.get(engine.reranker_key(cfg.reranker, cfg.precision, cfg.max_length))
    return {
        "config": cfg.name,
        "parent": cfg.parent,
        "settings": asdict(cfg),
        "retrieval": retrieval,
        "latency": timer.summary(),
        "build_ms": build,
        "wall_ms": round(wall_ms, 2),
        "models": models,
        "metric_cross_check": cross_check_metrics(rankings, qrels, per_query, required=require_trec_eval),
        "trec": write_trec_outputs(trec_dir, cfg, rankings, qrels) if trec_dir is not None else None,
        # the sha256 of the TREC run file this row's rankings make (trec_run_text), written or not: a committed row's
        # trec.run_sha256 says whether a recomputed row ranks every query identically (M41)
        "ranking_sha256": hashlib.sha256(trec_run_text(cfg, rankings).encode("utf-8")).hexdigest(),
        "_per_query": per_query,
        **({"_extra": extra} if extra is not None else {}),
    }


def _truncated_pairs(engine: Engine, cfg: BeirConfig, query: str, ids: list[str]) -> int:
    """How many of a query's reranked (query, passage) pairs are longer than the row's cap, by the loaded model's own
    tokenizer (special tokens on, truncation and padding off): the pairs of the reranked head."""
    model = engine.reranker(cfg.reranker, cfg.precision, cfg.max_length)
    lengths = engine.per_query_extras(model, query, [engine.by_id[d].text for d in ids[: cfg.rerank_top]])
    return sum(1 for n in lengths if n > cfg.max_length)


def attach_confidence(results: list[dict], n_queries: int, seed: int = DEFAULT_SEED, n_boot: int = DEFAULT_N_BOOT, *, published: dict | None = PUBLISHED,
                      comparisons: list[tuple[str, str]] | None = None) -> dict:
    """Per row: 95% CIs, the paired delta against bm25 and against its parent. `comparisons` (row, other) adds the same
    paired delta for other pairs, returned under "comparisons" (M41)."""
    idx = resample_indices(n_queries, n_boot=n_boot, seed=seed)
    by_name = {r["config"]: r for r in results}
    pairs = []
    for a, b in comparisons or []:
        if a in by_name and b in by_name:
            block: dict[str, Any] = {"row": a, "other": b}
            for m in CI_METRICS:
                delta, lo, hi = paired_delta_ci([r[m] for r in by_name[a]["_per_query"]], [r[m] for r in by_name[b]["_per_query"]], idx)
                block[m] = {"delta": round(delta, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
            pairs.append(block)
    ref = by_name.get(REFERENCE)
    for row in results:
        pq = row["_per_query"]
        row["ci"] = {}
        for m in CI_METRICS:
            mean, lo, hi = mean_ci([r[m] for r in pq], idx)
            row["ci"][m] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        if ref is None or row is ref:
            row[f"delta_vs_{REFERENCE}"] = None
        else:
            d: dict[str, Any] = {"reference": REFERENCE}
            for m in CI_METRICS:
                delta, lo, hi = paired_delta_ci([r[m] for r in pq], [r[m] for r in ref["_per_query"]], idx)
                d[m] = {"delta": round(delta, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
            row[f"delta_vs_{REFERENCE}"] = d
        parent = by_name.get(row.get("parent") or "")
        if parent is None or parent is row:
            row["delta_vs_parent"] = None
        else:
            dp: dict[str, Any] = {"reference": parent["config"]}
            for m in CI_METRICS:
                delta, lo, hi = paired_delta_ci([r[m] for r in pq], [r[m] for r in parent["_per_query"]], idx)
                dp[m] = {"delta": round(delta, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
            row["delta_vs_parent"] = dp
        if row["config"] == REFERENCE and published is not None:
            row["vs_published"] = {
                "published_bm25_ndcg@10": published["bm25_ndcg@10"],
                "delta": round(row["retrieval"]["ndcg@10"] - published["bm25_ndcg@10"], 4),
            }
    for row in results:
        row.pop("_per_query", None)
    settings = bootstrap_settings(n_queries, n_boot=n_boot, seed=seed)
    if pairs:
        settings["comparisons"] = pairs
    return settings


def measured_parents(configs: list[BeirConfig]) -> list[BeirConfig]:
    """A bounded sweep may omit the expensive uncapped reranker. In that case,
    evaluate the capped reranker as a component addition against the measured
    hybrid baseline, rather than leaving its paired delta empty."""
    names = {cfg.name for cfg in configs}
    return [replace(cfg, parent="hybrid@bge-small") if cfg.name == RERANK_FP32_512 and cfg.parent not in names and "hybrid@bge-small" in names else cfg for cfg in configs]


def run(
    data_dir: Path,
    configs: list[BeirConfig] | None = None,
    *,
    device: str = "auto",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    root: Path | None = None,
    download: bool = True,
    dataset: str = "scifact",
    split: str = "test",
    dataset_sha256: str | None = None,
    max_queries: int | None = None,
    trec_dir: Path | None = None,
    require_trec_eval: bool = False,
    check_models: bool = False,
    comparisons: list[tuple[str, str]] | None = None,
    reproduce: tuple[str, tuple[str, ...]] | tuple[tuple[str, tuple[str, ...]], ...] | None = None,
    header: dict[str, Any] | None = None,
    resolve_reranker=None,
    keep_per_query: bool = False,
    per_query_extras=None,
    reproduce_extra=None,
) -> dict[str, Any]:
    """M41: `check_models` (Engine), `comparisons` (attach_confidence), `reproduce` = (committed file, row names): whether
    those rows, recomputed here, equal the file's (retrieval block; and the ranking, where the committed row wrote a TREC
    run or recorded its ranking's sha256), recorded whatever the answer; `header` adds fields to the results header.
    M43: `reproduce` may be several such pairs (a list of blocks is recorded); `resolve_reranker` (Engine);
    `keep_per_query` returns every row's per-query scores under "_per_query" for the caller's sidecar (per_query_sidecar).
    M51: `per_query_extras` (Engine) returns every row's per-query extras under "_per_query_extra"; `reproduce_extra`
    replaces _same_ranking as the per-row check of `reproduce`."""
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if configs is not None and not configs:
        raise ValueError("at least one config is required")
    started = git_state(root)  # before the run writes TREC files or its cache (run_provenance)
    configs = measured_parents(configs if configs is not None else DEFAULT_CONFIGS)
    spec = DATASETS[dataset]
    dataset_root = download_dataset(data_dir, dataset, sha256=dataset_sha256) if download else data_dir / dataset
    corpus, queries, qrels = load_beir(dataset_root, split)
    full_query_count = len(queries)
    if max_queries is not None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        queries = dict(list(sorted(queries.items()))[:max_queries])
        qrels = {qid: qrels[qid] for qid in queries}
    if not corpus or not queries:
        raise ValueError("dataset must have documents and positively judged queries")
    missing_docs = {did for docs in qrels.values() for did in docs if did not in corpus}
    if missing_docs:
        raise ValueError(f"qrels reference {len(missing_docs)} missing documents")
    engine = Engine(chunks_from_corpus(corpus), cache_dir=data_dir / "cache", device=device, check_models=check_models,
                    resolve_reranker=resolve_reranker, per_query_extras=per_query_extras)
    results = []
    for i, cfg in enumerate(configs):
        results.append(run_config(engine, cfg, queries, qrels, trec_dir=trec_dir, require_trec_eval=require_trec_eval))
        engine.release_unused(configs[i + 1:])
    per_query = {row["config"]: row["_per_query"] for row in results} if keep_per_query else None  # attach_confidence drops them
    per_query_extra = {row["config"]: row.pop("_extra") for row in results if "_extra" in row} or None
    # A published full-test reference is inapplicable to a query subset.
    published = spec["published"] if split == "test" and len(queries) == full_query_count else None
    boot = attach_confidence(results, len(queries), seed=seed, n_boot=n_boot, published=published, comparisons=comparisons)
    reproduced = None
    if reproduce is not None and len(queries) == full_query_count:
        from app.eval.models_2026 import reproduces

        repo = Path(root or Path(__file__).resolve().parents[2])
        check = reproduce_extra or _same_ranking
        if isinstance(reproduce[0], str):
            reproduced = reproduces(repo, reproduce[0], results, list(reproduce[1]), extra=check)
        else:
            reproduced = [reproduces(repo, rel, results, list(names), extra=check) for rel, names in reproduce]
    zip_path = data_dir / f"{dataset}.zip"
    return {
        "dataset": f"BEIR/{dataset}",
        "split": split,
        "url": spec["url"],
        "zip_sha256": sha256_of(zip_path) if zip_path.exists() else None,
        "expected_zip_sha256": dataset_sha256 or spec["sha256"],
        "files_sha256": {str(p.relative_to(dataset_root)): sha256_of(p) for p in (dataset_root / "corpus.jsonl", dataset_root / "queries.jsonl", dataset_root / "qrels" / f"{split}.tsv")},
        "corpus_sha256": engine.corpus_sha256,
        "runner_sha256": RUNNER_SHA256,
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "full_split_queries": full_query_count,
        "query_selection": "full split" if len(queries) == full_query_count else "bounded subset, query ids sorted ascending",
        "document_text": "title + ' ' + text",
        "published": published,
        "relevance": "graded linear-gain nDCG (trec_eval/BEIR); binary-positive Recall/MRR; ignore identical query/document ids",
        "latency_note": "per-query stats exclude build (index, doc embeddings, model load; see build_ms) and one warm-up query",
        "git_sha": git_sha(root),
        **run_provenance(root=root, started=started),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info(device),
        "bootstrap": boot,
        "results": results,
        **(header or {}),
        **({"reproduces": reproduced} if reproduced is not None else {}),
        **({"_per_query": per_query} if per_query is not None else {}),
        **({"_per_query_extra": per_query_extra} if per_query_extra is not None else {}),
    }


def _same_ranking_and_intervals(row: dict, committed: dict) -> dict:
    """M51's reproduces() extra for M43's anchor rows: _same_ranking, and whether the row's 95% intervals and its delta
    against its parent equal the committed row's (the same queries, seed and resample count draw the same matrix)."""
    return {**_same_ranking(row, committed), "ci_equal": row.get("ci") == committed.get("ci"),
            "delta_vs_parent_equal": row.get("delta_vs_parent") == committed.get("delta_vs_parent")}


def _same_ranking(row: dict, committed: dict) -> dict:
    """reproduces() extra: whether a row ranks every query as the committed row does: its TREC run when it wrote one,
    else the sha256 of its ranking when it recorded one (rows since M41)."""
    sha = (committed.get("trec") or {}).get("run_sha256") or committed.get("ranking_sha256")
    return {"ranking_equal": row.get("ranking_sha256") == sha} if sha else {}


PER_QUERY_SCHEMA = "beir-per-query-scores-v1"
PER_QUERY_METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr@10")


def per_query_sidecar(blob: dict, per_query: dict[str, list[dict]], extra: dict[str, dict] | None = None) -> dict[str, Any]:
    """The per-query scores of a BEIR results file, column by column and unrounded, so that every interval and paired
    delta in it can be recomputed without a model (M43: python -m app.eval.small_rerankers --rescore). Query ids in the
    run's order (the bootstrap's); nothing else: no ranking, no command, no path. M51: `extra` (per row: per-query
    truncation counts and ranking hashes) goes under "per_query_extra", outside "scores", which every rescore reads."""
    first = next(iter(per_query.values()))
    return {"schema": PER_QUERY_SCHEMA,
            "about": "per-query scores of the results file with the same name (.json), in the run's query order and unrounded",
            "git_sha": blob.get("git_sha"), "timestamp": blob.get("timestamp"), "dataset": blob.get("dataset"), "split": blob.get("split"),
            "metrics": list(PER_QUERY_METRICS), "ids": [r["id"] for r in first],
            "scores": {name: {m: [float(r[m]) for r in rows] for m in PER_QUERY_METRICS} for name, rows in per_query.items()},
            **({"per_query_extra": extra} if extra else {})}


def _fmt(block: dict | None, key: str = "mean") -> str:
    if not block:
        return "—"
    if key == "delta":
        return f"{block['delta']:+.3f} [{block['lo']:+.3f}, {block['hi']:+.3f}]"
    return f"{block['mean']:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


def markdown_table(blob: dict) -> str:
    lines = [
        f"BEIR {blob['dataset'].split('/')[-1].replace('scifact', 'SciFact')} {blob['split']}: {blob['n_docs']} docs, {blob['n_queries']} queries." + (f" Published BM25 nDCG@10 = {blob['published']['bm25_ndcg@10']:.3f}." if blob.get("published") else " No matched published reference claimed."),
        "",
        "| config | parent | nDCG@10 [95% CI] | Recall@10 [95% CI] | Recall@100 [95% CI] | Δ nDCG@10 vs parent [95% CI] | Δ vs bm25 | p50 / p95 ms | build s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["results"]:
        ci, lat = row["ci"], row["latency"].get("total", {})
        dp = (row.get("delta_vs_parent") or {}).get("ndcg@10")
        db = (row.get(f"delta_vs_{REFERENCE}") or {}).get("ndcg@10")
        lines.append(
            f"| {row['config']} | {row.get('parent') or '—'} | {_fmt(ci['ndcg@10'])} | {_fmt(ci['recall@10'])} | {_fmt(ci['recall@100'])} | "
            f"{_fmt(dp, 'delta')} | {_fmt(db, 'delta')} | {lat.get('p50_ms', '')} / {lat.get('p95_ms', '')} | {(row.get('build_ms') or {}).get('total_ms', 0) / 1000:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=root / "bench" / "external")
    p.add_argument("--dataset", choices=tuple(DATASETS), default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--dataset-sha256", default=None)
    p.add_argument("--max-queries", type=int, default=None, help="bounded sorted query subset; never reported as the full benchmark")
    p.add_argument("--trec-dir", type=Path, default=None, help="write independently evaluable .run and qrels.txt files")
    p.add_argument("--require-trec-eval", action="store_true", help="fail unless pytrec_eval confirms exact per-query metrics")
    p.add_argument("--out", type=Path, default=root / "bench" / "results" / "beir_scifact.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--rerank-top", type=int, default=None, help="override every reranked row's depth (default: each row's own rerank_top)")
    p.add_argument("--ladder", choices=tuple(LADDERS), default="default", help="the config list: the default ladder, or bm25-ablation (analyzer only, k1/b only, both)")
    p.add_argument("--only", default=None, help="comma-separated config names (from the chosen ladder)")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="small-rerankers: where the students' checkpoints are (default: $SMALL_RERANKER_CHECKPOINTS)")
    p.add_argument("--rows", choices=("all", "anchors"), default="all", help="small-rerankers-shapes: every row, or the parent and M43's six 20/512 rows only")
    p.add_argument("--ml256-checkpoint-dir", type=Path, default=None, help="small-rerankers-ml256: where the attempt-2 checkpoints are (default: $SMALL_RERANKER_ML256_CHECKPOINTS)")
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configs = list(small_reranker_shape_configs(args.rows) if args.ladder == SMALL_RERANKER_SHAPES_LADDER else LADDERS[args.ladder])
    if args.rerank_top is not None:  # the first SciFact sweep silently ran every depth row at 20 because of a default here
        configs = [BeirConfig(**{**asdict(c), "rerank_top": args.rerank_top}) for c in configs]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        configs = [c for c in configs if c.name in wanted]
    if not configs:
        p.error("--only selected no configurations")
    m41: dict[str, Any] = {}
    if args.ladder in MODELS_2026_LADDERS:  # every model must load and rank; the parents' reproduction is recorded
        from app.eval.models_2026 import header

        names = {c.name for c in configs}
        m41 = dict(check_models=True, comparisons=models_2026_comparisons(configs),
                   reproduce=(COMMITTED_FILES[args.dataset], tuple(n for n in MODELS_2026_PARENTS if n in names)),
                   header={"models_2026": header(MODELS_2026_LADDERS[args.ladder])})
    if args.ladder == SMALL_RERANKERS_LADDER:  # M43: M41's conventions, each row against the unreranked hybrid, a per-query sidecar
        from app.eval import small_rerankers as sr

        names = {c.name for c in configs}
        m41 = dict(check_models=True, comparisons=small_reranker_comparisons(configs), keep_per_query=True,
                   reproduce=tuple((rel, tuple(n for n in rows if n in names)) for rel, rows in SMALL_RERANKERS_REPRODUCE[args.dataset]),
                   header={"small_rerankers": sr.header([c.name for c in configs])}, resolve_reranker=sr.resolver(args.checkpoint_dir, root))
    if args.ladder == SMALL_RERANKER_SHAPES_LADDER:  # M51: M43's conventions at every shape, plus per-query extras and the anchors
        from app.eval import small_rerankers as sr
        from app.eval import small_rerankers_shapes as shapes

        names = {c.name for c in configs}
        m41 = dict(check_models=True, comparisons=shapes.comparisons([c.name for c in configs], with_parent=False), keep_per_query=True,
                   reproduce=((sr.RESULTS[args.dataset], tuple(n for n in shapes.anchor_rows() if n in names)),),
                   reproduce_extra=_same_ranking_and_intervals, per_query_extras=shapes.pair_token_lengths,
                   header={"small_rerankers": shapes.header([c.name for c in configs], rows=args.rows)},
                   resolve_reranker=shapes.resolver(args.checkpoint_dir, root))
    if args.ladder == SMALL_RERANKER_ML256_LADDER:  # M51 Phase 2: the same conventions, anchored on M51's committed shapes files
        from app.eval import small_rerankers_shapes as shapes

        names = {c.name for c in configs}
        m41 = dict(check_models=True, comparisons=shapes.comparisons([c.name for c in configs], with_parent=False), keep_per_query=True,
                   reproduce=((shapes.RESULTS[args.dataset], tuple(n for n in shapes.ml256_anchor_rows() if n in names)),),
                   reproduce_extra=_same_ranking_and_intervals, per_query_extras=shapes.pair_token_lengths,
                   header={"small_rerankers": shapes.header([c.name for c in configs], phase2=True)},
                   resolve_reranker=shapes.resolver(args.checkpoint_dir, root, args.ml256_checkpoint_dir))
    blob = run(args.data_dir, configs, device=args.device, n_boot=args.n_boot, root=root, dataset=args.dataset, split=args.split, dataset_sha256=args.dataset_sha256, max_queries=args.max_queries, trec_dir=args.trec_dir, require_trec_eval=args.require_trec_eval, **m41)
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    per_query = blob.pop("_per_query", None)
    per_query_extra = blob.pop("_per_query_extra", None)
    if args.ladder == SMALL_RERANKER_SHAPES_LADDER:
        shapes.annotate_beir(blob, per_query_extra)  # per row: ranking_digest, n_top10_differs_from_parent, truncation totals
    if args.ladder == SMALL_RERANKER_ML256_LADDER:
        shapes.annotate_beir(blob, per_query_extra, shapes.N_ML256_ANCHOR_ROWS)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    if per_query is not None:
        from app.eval import per_question

        per_question.write(per_question.sidecar_path(out), per_query_sidecar(blob, per_query, per_query_extra))
    print(markdown_table(blob))
    if blob.get("reproduces"):  # recorded whatever the answer; a parent that differs is not an error
        print("reproduces", json.dumps(blob["reproduces"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
