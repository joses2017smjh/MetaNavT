"""BEIR SciFact: an external benchmark with a published BM25 baseline.

    python -m app.eval.beir --out bench/results/beir_scifact.json [--device auto|cpu|cuda] [--rerank-top 20]

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
is scored with.

Rows form an explicit ablation chain (each row names its `parent`, and the
paired delta is against that parent, not against list order): bm25 ->
dense@bge-small -> hybrid@bge-small -> hybrid+bge-rerank (top 20, fp32,
uncapped) -> the same reranker in fp16 with max_length 512 -> rerank depth
50 -> 100. dense@bge-small+instruction is the bge-v1.5 query prefix on the
query side only (document embeddings are cached). Metrics: nDCG@10,
Recall@10, Recall@100 and MRR@10 with 95% paired bootstrap CIs; per-query
latency p50/p95 end to end and per stage, measured after a warm-up query and
with index / embedding / model build time reported separately (build_ms);
model revisions, precision, max_length and device recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.eval.latency import StageTimer
from app.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from app.eval.provenance import command_line, device_info, git_sha, model_device, model_provenance
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, mean_ci, paired_delta_ci, resample_indices, settings as bootstrap_settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedders import HashEmbedder, cosine_scores
from app.retrieval.fuse import rrf_score_map
from app.retrieval.rerank import OverlapReranker, cross_encoder_scores, get_cross_encoder
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
CI_METRICS = ("ndcg@10", "recall@10", "recall@100")


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
    precision: str = "fp32"  # reranker weights: fp32 | fp16 (CUDA only)
    max_length: int | None = None  # reranker token cap; None = the model's own
    query_instruction: str | None = None  # prefix added to queries only (bge-v1.5 recommends one)


RERANK_FP32 = "hybrid+bge-rerank@bge-small"
RERANK_FP16 = "hybrid+bge-rerank@bge-small/fp16-512/top20"
DEFAULT_CONFIGS = [
    BeirConfig(name="bm25", mode="bm25"),
    BeirConfig(name="bm25@default-tokenizer", mode="bm25", analyzer="default", k1=1.5, b=0.75, parent="bm25"),
    BeirConfig(name="dense@bge-small", mode="dense", embedder=BGE_SMALL, parent="bm25"),
    BeirConfig(name="dense@bge-small+instruction", mode="dense", embedder=BGE_SMALL, parent="dense@bge-small", query_instruction=BGE_QUERY_INSTRUCTION),
    BeirConfig(name="hybrid@bge-small", mode="hybrid", embedder=BGE_SMALL, parent="dense@bge-small"),
    BeirConfig(name=RERANK_FP32, mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=20, parent="hybrid@bge-small"),
    BeirConfig(name=RERANK_FP16, mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=20, parent=RERANK_FP32, precision="fp16", max_length=512),
    BeirConfig(name="hybrid+bge-rerank@bge-small/fp16-512/top50", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=50, parent=RERANK_FP16, precision="fp16", max_length=512),
    BeirConfig(name="hybrid+bge-rerank@bge-small/fp16-512/top100", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, rerank_top=100, parent="hybrid+bge-rerank@bge-small/fp16-512/top50", precision="fp16", max_length=512),
]


# ---------------------------------------------------------------- data


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_scifact(data_dir: Path, url: str = SCIFACT_URL, sha256: str | None = SCIFACT_SHA256) -> Path:
    """Fetch and unpack the BEIR zip once; verify the checksum every time."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "scifact.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)
    actual = sha256_of(zip_path)
    if sha256 and actual != sha256:
        raise RuntimeError(f"scifact.zip sha256 {actual} != pinned {sha256}; delete it and retry")
    root = data_dir / "scifact"
    if not (root / "corpus.jsonl").exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
    return root


def load_scifact(root: Path, split: str = "test") -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """corpus id -> 'title text', query id -> text (only queries with qrels), qrels id -> {doc: score>0}."""
    corpus: dict[str, str] = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                corpus[str(row["_id"])] = (row.get("title", "") + " " + row.get("text", "")).strip()
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


def chunks_from_corpus(corpus: dict[str, str]) -> list[Chunk]:
    return [Chunk(chunk_id=did, path=did, text=text, start_byte=0, end_byte=len(text.encode("utf-8"))) for did, text in corpus.items()]


# ---------------------------------------------------------------- models


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


class Engine:
    """Shares indexes, embeddings and models across configs; caches doc embeddings on disk."""

    def __init__(self, chunks: list[Chunk], cache_dir: Path | None = None):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.cache_dir = cache_dir
        self._bm25: dict[tuple[str, float, float], BM25Index] = {}
        self._embedders: dict[str, Any] = {}
        self._doc_mats: dict[str, np.ndarray] = {}
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
            else:
                raise ValueError(f"unknown embedder {name!r}")
            self.provenance[name] = {
                **model_provenance(name, "embedder"),
                "dim": int(self._embedders[name].dim),
                "device": model_device(getattr(self._embedders[name], "_model", None)),
                "query_instruction": None,  # bge-v1.5 recommends a query prefix; not used
            }
        return self._embedders[name]

    def doc_matrix(self, name: str) -> np.ndarray:
        if name in self._doc_mats:
            return self._doc_mats[name]
        emb = self.embedder(name)
        rev = self.provenance[name].get("revision") or "norev"
        cache = self.cache_dir / f"scifact.{_slug(name)}.{rev}.{len(self.chunks)}.npy" if self.cache_dir else None
        if cache and cache.exists():
            mat = np.load(cache)
        else:
            mat = emb.encode([c.text for c in self.chunks])
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, mat)
        self._doc_mats[name] = mat
        return mat

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
                model = get_cross_encoder(name, precision=precision, max_length=max_length)
                if model is None:
                    raise RuntimeError(f"reranker {name} not available (set BGE_ALLOW_DOWNLOAD=1 or cache it)")
                inner = getattr(model, "model", None)
                self.provenance[key] = {
                    **model_provenance(name, "reranker"),
                    "loaded": True,
                    "device": model_device(model),
                    "precision": precision,
                    "dtype": str(getattr(inner, "dtype", None)) if inner is not None else None,
                    "max_length": max_length or getattr(model, "max_length", None),
                    "score_scale": "sigmoid probability in [0, 1]",
                }
                self._rerankers[key] = model
        return self._rerankers[key]

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
            self.doc_matrix(cfg.embedder)
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
        mat = engine.doc_matrix(cfg.embedder)
        with timer.stage("embed"):
            qv = engine.embedder(cfg.embedder).encode([(cfg.query_instruction or "") + query])[0]
        with timer.stage("dense"):
            scores = cosine_scores(qv, mat)
            order = np.argsort(-scores)[: cfg.k]
            dense_hits = [(engine.chunks[i].chunk_id, float(scores[i])) for i in order]
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


def run_config(engine: Engine, cfg: BeirConfig, queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    build = engine.build(cfg)  # index, embeddings, model loads: reported separately, never inside a query
    first = next(iter(queries.values()))
    retrieve(engine, cfg, first, StageTimer())  # warm-up (CUDA kernels, caches); excluded from the stats
    timer = StageTimer()
    per_query: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for qid, text in queries.items():
        ids = retrieve(engine, cfg, text, timer)
        relevant = list(qrels[qid])
        per_query.append(
            {
                "id": qid,
                "ndcg@10": ndcg_at_k(ids, relevant, 10),
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
        "_per_query": per_query,
    }


def attach_confidence(results: list[dict], n_queries: int, seed: int = DEFAULT_SEED, n_boot: int = DEFAULT_N_BOOT) -> dict:
    idx = resample_indices(n_queries, n_boot=n_boot, seed=seed)
    by_name = {r["config"]: r for r in results}
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
        if row["config"] == REFERENCE:
            row["vs_published"] = {
                "published_bm25_ndcg@10": PUBLISHED["bm25_ndcg@10"],
                "delta": round(row["retrieval"]["ndcg@10"] - PUBLISHED["bm25_ndcg@10"], 4),
            }
    for row in results:
        row.pop("_per_query", None)
    return bootstrap_settings(n_queries, n_boot=n_boot, seed=seed)


def run(
    data_dir: Path,
    configs: list[BeirConfig] | None = None,
    *,
    device: str = "auto",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    root: Path | None = None,
    download: bool = True,
) -> dict[str, Any]:
    configs = configs or DEFAULT_CONFIGS
    scifact = download_scifact(data_dir) if download else data_dir / "scifact"
    corpus, queries, qrels = load_scifact(scifact)
    engine = Engine(chunks_from_corpus(corpus), cache_dir=data_dir / "cache")
    results = [run_config(engine, cfg, queries, qrels) for cfg in configs]
    boot = attach_confidence(results, len(queries), seed=seed, n_boot=n_boot)
    return {
        "dataset": "BEIR/scifact",
        "split": "test",
        "url": SCIFACT_URL,
        "zip_sha256": SCIFACT_SHA256,
        "n_docs": len(corpus),
        "n_queries": len(queries),
        "document_text": "title + ' ' + text",
        "published": PUBLISHED,
        "latency_note": "per-query stats exclude build (index, doc embeddings, model load; see build_ms) and one warm-up query",
        "git_sha": git_sha(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info(device),
        "bootstrap": boot,
        "results": results,
    }


def _fmt(block: dict | None, key: str = "mean") -> str:
    if not block:
        return "—"
    if key == "delta":
        return f"{block['delta']:+.3f} [{block['lo']:+.3f}, {block['hi']:+.3f}]"
    return f"{block['mean']:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


def markdown_table(blob: dict) -> str:
    lines = [
        f"BEIR SciFact test: {blob['n_docs']} docs, {blob['n_queries']} queries. Published BM25 nDCG@10 = {blob['published']['bm25_ndcg@10']:.3f}.",
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
    p.add_argument("--out", type=Path, default=root / "bench" / "results" / "beir_scifact.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--rerank-top", type=int, default=None, help="override every reranked row's depth (default: each row's own rerank_top)")
    p.add_argument("--only", default=None, help="comma-separated config names")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configs = list(DEFAULT_CONFIGS)
    if args.rerank_top is not None:  # the first SciFact sweep silently ran every depth row at 20 because of a default here
        configs = [BeirConfig(**{**asdict(c), "rerank_top": args.rerank_top}) for c in configs]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        configs = [c for c in configs if c.name in wanted]
    blob = run(args.data_dir, configs, device=args.device, n_boot=args.n_boot, root=root)
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    print(markdown_table(blob))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
