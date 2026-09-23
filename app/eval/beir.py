"""BEIR SciFact: an external benchmark with a published BM25 baseline.

    python -m app.eval.beir --out bench/results/beir_scifact.json [--device auto|cpu|cuda] [--rerank-top 20]

Dataset: SciFact (Wadden et al., 2020) in the BEIR packaging (Thakur et al.,
2021): 5,183 abstracts, 300 test queries with binary qrels. Downloaded once to
bench/external/ (gitignored); the zip's sha256 is pinned below.

Published reference: the BEIR paper reports BM25 (Anserini; Lucene English
analyzer = Porter stemming + stopwords; k1=0.9, b=0.4) nDCG@10 = 0.665 on
SciFact. The `bm25` row uses the same analyzer and parameters so the two are
comparable. A `bm25@default-tokenizer` row is kept next to it because that
tokenizer (no stemming, no stopwords, k1=1.5, b=0.75) is what fixture v1 is
scored with.

Rows: bm25, bm25@default-tokenizer, dense@bge-small, hybrid@bge-small (RRF k=60
over the two top-100 lists), hybrid+bge-rerank@bge-small (bge-reranker-v2-m3
over the fused top-`rerank_top`, then the rest of the RRF order). Metrics:
nDCG@10, Recall@10, Recall@100 and MRR@10 with 95% paired bootstrap CIs; per-query
latency p50/p95 end to end and per stage; model revisions and device recorded.
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
    "bm25_ndcg@10": 0.665,
    "source": "Thakur et al., 'BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models', NeurIPS 2021 Datasets and Benchmarks, Table 2, BM25 (Anserini) on SciFact",
    "analyzer": "Lucene English (Porter stemming + stopwords), k1=0.9, b=0.4",
}
BGE_SMALL = "st:BAAI/bge-small-en-v1.5"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
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


DEFAULT_CONFIGS = [
    BeirConfig(name="bm25", mode="bm25"),
    BeirConfig(name="bm25@default-tokenizer", mode="bm25", analyzer="default", k1=1.5, b=0.75),
    BeirConfig(name="dense@bge-small", mode="dense", embedder=BGE_SMALL),
    BeirConfig(name="hybrid@bge-small", mode="hybrid", embedder=BGE_SMALL),
    BeirConfig(name="hybrid+bge-rerank@bge-small", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER),
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

    def reranker(self, name: str):
        if name not in self._rerankers:
            if name == "overlap":
                self._rerankers[name] = OverlapReranker()
                self.provenance[name] = {**model_provenance("overlap", "reranker"), "loaded": True}
            else:
                model = get_cross_encoder(name)
                self.provenance[name] = {**model_provenance(name, "reranker"), "loaded": model is not None, "device": model_device(model)}
                if model is None:
                    raise RuntimeError(f"reranker {name} not available (set BGE_ALLOW_DOWNLOAD=1 or cache it)")
                self._rerankers[name] = model
        return self._rerankers[name]

    def rerank(self, name: str, query: str, pairs: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        model = self.reranker(name)
        if name == "overlap":
            return model(query, pairs)
        scores = cross_encoder_scores(model, [(query, c.text) for c, _ in pairs])
        ranked = [(pairs[i][0], scores[i]) for i in range(len(pairs))]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


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
            qv = engine.embedder(cfg.embedder).encode([query])[0]
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
            reranked = engine.rerank(cfg.reranker, query, head)
            ids = [c.chunk_id for c, _ in reranked] + ids[cfg.rerank_top :]
    timer.record("total", (time.perf_counter() - t0) * 1000.0)
    return ids


def run_config(engine: Engine, cfg: BeirConfig, queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
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
        models["embedder"] = engine.provenance.get(cfg.embedder)
    if cfg.reranker:
        models["reranker"] = engine.provenance.get(cfg.reranker)
    return {
        "config": cfg.name,
        "settings": asdict(cfg),
        "retrieval": retrieval,
        "latency": timer.summary(),
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
        "published": PUBLISHED,
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
        "| config | nDCG@10 [95% CI] | Recall@10 [95% CI] | Recall@100 [95% CI] | Δ nDCG@10 vs bm25 [95% CI] | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["results"]:
        ci, lat = row["ci"], row["latency"].get("total", {})
        delta = (row.get(f"delta_vs_{REFERENCE}") or {}).get("ndcg@10")
        lines.append(
            f"| {row['config']} | {_fmt(ci['ndcg@10'])} | {_fmt(ci['recall@10'])} | {_fmt(ci['recall@100'])} | "
            f"{_fmt(delta, 'delta')} | {lat.get('p50_ms', '')} | {lat.get('p95_ms', '')} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=root / "bench" / "external")
    p.add_argument("--out", type=Path, default=root / "bench" / "results" / "beir_scifact.json")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--rerank-top", type=int, default=20)
    p.add_argument("--only", default=None, help="comma-separated config names")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = p.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configs = [BeirConfig(**{**asdict(c), "rerank_top": args.rerank_top}) for c in DEFAULT_CONFIGS]
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
