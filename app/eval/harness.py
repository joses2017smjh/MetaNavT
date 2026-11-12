"""One-command retrieval bench over the frozen corpus + gold set."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.corrective import evaluate_retrieval, determine_correction
from app.agent.decompose import heuristic_decompose, should_decompose
from app.agent.deep_research import expand_queries, multi_query_retrieve
from app.agent.retrieval_loop import extractive_answer
from app.eval.corpus import load_manifest, verify_manifest
from app.eval.frontier_llm import GENERATIONS_FILE, context_prefixes, header as generations_header, hyde_completion, load_generations
from app.eval.gold import GoldQuestion, load_gold
from app.eval.hashing import content_hash
from app.eval.index_loader import build_index, mtime_provenance
from app.eval.ingest import CHUNKER_VERSION
from app.eval.jury import KAPPA_GATE, default_jury, kappa_vs_gold
from app.eval.latency import StageTimer
from app.eval.metrics import aggregate_retrieval
from app.eval import per_question
from app.eval.probes import load_probes
# git_sha is also imported from this module by rerank_arms, sweeps, rrf_sweep, hybrid_drops, demo_export and frontier_llm.
from app.eval.provenance import command_line, device_info, git_sha, git_state, model_device, model_provenance, run_provenance, sha256_file
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, mean_ci, paired_delta_ci, resample_indices, settings as bootstrap_settings
from app.eval.ragas_metrics import (
    answer_relevancy,
    aggregate_e2e,
    context_precision,
    context_recall,
    faithfulness,
)
from app.graph.file_graph import build_file_graph, expand_with_graph
from app.graph.graphrag import boost_by_communities, build_graphrag, community_stats, match_communities
from app.graph.hipporag import apply_hipporag, should_apply, triples_from_chunks
from app.graph.staleness import cluster_versions, prefer_current
from app.retrieval.adaptive import RetrievalDecision, should_retrieve
from app.retrieval.cache import SemanticCache
from app.retrieval.distill import bench_log_triples, write_triples
from app.retrieval.hybrid import InMemoryHybridIndex, RetrievalHit
from app.retrieval.hyde import hyde_embed


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class BenchConfig:
    name: str = "hybrid+rerank+router"
    embedder: str = "hash"
    retrieve_k: int = 50
    rerank_n: int = 8
    enable_router: bool = True
    enable_rerank: bool = True
    reranker: str = "overlap"
    chunk_strategy: str = "auto"
    mode: str = "hybrid"  # hybrid | bm25 | dense
    graph_hops: int = 0
    staleness_tier1: bool = True
    e2e: bool = True
    log_triples: bool = True
    jury: bool = False
    hipporag: bool = False
    multi_query: bool = False
    hyde: bool = False
    hyde_stage: str | None = None  # where the HyDE vector enters (HYDE_STAGES); required when hyde is on
    decompose: bool = False
    corrective: bool = False
    adaptive: bool = False  # skip retrieval when app.retrieval.adaptive says SKIP
    graphrag: bool = False  # rank-fuse the files of the community summaries the query matches
    contextual: bool = False  # prepend a situating prefix to every chunk before indexing
    cache: bool = False  # serve near-duplicate questions from app.retrieval.cache.SemanticCache (gold order, empty at the start)
    generations: str | None = None  # stored LLM-written HyDE hypotheses / contextual prefixes (app/eval/frontier_llm.py) instead of the heuristics
    late_chunking: bool = False  # pool document-level token embeddings per chunk (needs a token-level model)
    parent: str | None = None  # the row this one is a single change away from (paired delta target)
    # The served reranker's settings (M17): RERANK_DEPTH, RERANKER_PRECISION, RERANKER_MAX_LENGTH. None: every fused
    # candidate, fp32, the model's own token cap, as every row recorded before M17.
    rerank_depth: int | None = None
    rerank_precision: str | None = None  # fp32 | fp16 (CUDA only)
    rerank_max_length: int | None = None
    # Staleness Tier 1 on the fused list before the reranker, as the served retriever orders them. False: after the
    # reranker, as every row recorded before M17 (the two orders agree when every fused candidate is reranked).
    staleness_before_rerank: bool = False


# first_stage:  the HyDE vector is the dense leg's query, before fusion; the reranker (if any) has the last word.
# after_rerank: the wiring of the rows recorded before M18 as hybrid+rerank+hyde: retrieve and rerank with the raw
#               query, then RRF-fuse the reranked list with the HyDE dense list and re-sort. The re-sort overrides
#               the reranker's order and no HyDE hit outside the reranked list is added.
HYDE_STAGES = ("first_stage", "after_rerank")
# Switches added after bench/results/main.json was recorded. A row's settings carry them only when it sets them,
# so rows that do not use them keep the settings they were recorded with (`make bench-check` compares settings).
LATER_SETTINGS = ("hyde_stage", "cache", "generations", "rerank_depth", "rerank_precision", "rerank_max_length", "staleness_before_rerank")
# The cache arm also looks up these paraphrases of gold questions after the gold replay (app/eval/probes.py).
CACHE_PROBES = "bench/probes/paraphrases.jsonl"


def bench_settings(cfg: BenchConfig) -> dict[str, Any]:
    """The config as a results row records it (`settings`)."""
    defaults = {f.name: f.default for f in fields(BenchConfig)}
    return {k: v for k, v in cfg.__dict__.items() if not (k in LATER_SETTINGS and v == defaults[k])}


def _paths_from_hits(hits: list[RetrievalHit]) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        if hit.chunk.path not in seen:
            seen.append(hit.chunk.path)
    return seen


def _count(fired: dict[str, int] | None, key: str) -> None:
    if fired is not None:
        fired[key] = fired.get(key, 0) + 1


def _retrieve(
    index: InMemoryHybridIndex,
    query: str,
    cfg: BenchConfig,
    timer: StageTimer,
    clusters,
    graph,
    *,
    hippo_triples=None,
    communities=None,
    fired: dict[str, int] | None = None,
    hyde_complete=None,
) -> tuple[list[str], object]:
    """One query through one config. Takes the question text only: no gold field
    (category, answer, relevant paths) may reach retrieval, so every gate below
    uses what a served system would have (the router's own prediction).

    `fired` counts the questions on which each optional mechanism triggered, so
    an arm that ties its parent can say whether it ever ran. `hyde_complete`
    answers the HyDE prompt with a stored LLM-written hypothesis (None: the
    keyword heuristic)."""
    if cfg.hyde and (cfg.hyde_stage not in HYDE_STAGES or (cfg.mode == "dense" and cfg.hyde_stage != "first_stage")):
        raise ValueError(f"{cfg.name}: hyde needs hyde_stage in {HYDE_STAGES} (dense mode: first_stage), got {cfg.hyde_stage!r}")
    stale_first = cfg.staleness_tier1 and cfg.staleness_before_rerank
    if stale_first and (cfg.mode != "hybrid" or cfg.decompose or cfg.multi_query or cfg.hyde or cfg.corrective):
        raise ValueError(f"{cfg.name}: staleness_before_rerank is wired for the plain hybrid retrieval only")
    if cfg.adaptive:
        decision = should_retrieve(query).decision
        _count(fired, f"adaptive_{decision.value}")
        if decision == RetrievalDecision.SKIP:
            return [], []
    if cfg.decompose and should_decompose(query):
        subs = heuristic_decompose(query)
        if len(subs) > 1:
            _count(fired, "decompose")
            all_hits: list[RetrievalHit] = []
            for sub_q in subs:
                sub_result = index.retrieve(sub_q, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
                all_hits.extend(sub_result.hits)
            seen_ids: dict[str, RetrievalHit] = {}
            for h in all_hits:
                if h.chunk.chunk_id not in seen_ids or h.score > seen_ids[h.chunk.chunk_id].score:
                    seen_ids[h.chunk.chunk_id] = h
            hits = sorted(seen_ids.values(), key=lambda h: h.score, reverse=True)
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            result.hits = hits
        else:
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            hits = result.hits
    elif cfg.mode == "bm25":
        with timer.stage("bm25"):
            pairs = index.search_bm25(query, k=cfg.retrieve_k)
        hits = [
            RetrievalHit(chunk=c, score=s, rank=i)
            for i, (c, s) in enumerate(pairs, start=1)
        ]
        result = None
    elif cfg.mode == "dense":
        with timer.stage("embed"):
            if cfg.hyde:
                _count(fired, "hyde")
                hyde_vec = hyde_embed(query, index.embedder, complete=hyde_complete)
                pairs = index.search_dense(query, k=cfg.retrieve_k, query_vec=hyde_vec)
            else:
                pairs = index.search_dense(query, k=cfg.retrieve_k)
        hits = [
            RetrievalHit(chunk=c, score=s, rank=i)
            for i, (c, s) in enumerate(pairs, start=1)
        ]
        result = None
    elif cfg.multi_query:
        if len({q.lower() for q in expand_queries(query, n=3)}) > 1:
            _count(fired, "multi_query")
        with timer.stage("multi_query"):
            result = multi_query_retrieve(index, query, n=3, k=cfg.retrieve_k)
        hits = result.hits
    else:
        if cfg.hyde and cfg.hyde_stage == "first_stage":
            _count(fired, "hyde")
            with timer.stage("hyde"):
                hyde_vec = hyde_embed(query, index.embedder, complete=hyde_complete)
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n, query_vec=hyde_vec)
            hits = result.hits
        elif cfg.hyde:  # after_rerank (HYDE_STAGES)
            _count(fired, "hyde")
            with timer.stage("hyde"):
                hyde_vec = hyde_embed(query, index.embedder, complete=hyde_complete)
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            hyde_dense = index.search_dense(query, k=cfg.retrieve_k, query_vec=hyde_vec)
            hyde_ids = [c.chunk_id for c, _ in hyde_dense]
            from app.retrieval.fuse import rrf_score_map
            orig_ids = [h.chunk.chunk_id for h in result.hits]
            boosted = rrf_score_map([orig_ids, hyde_ids], k=60)
            for h in result.hits:
                if h.chunk.chunk_id in boosted:
                    h.score = boosted[h.chunk.chunk_id]
            result.hits.sort(key=lambda h: h.score, reverse=True)
            hits = result.hits
        else:
            prefilter = (lambda pairs: _prefer_current_pairs(pairs, clusters, query)) if stale_first and clusters else None
            result = index.retrieve(query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n, prefilter=prefilter)
            hits = result.hits

    if cfg.corrective and hits:
        pairs_for_eval = [(h.chunk, h.score) for h in hits[:cfg.rerank_n]]
        evaluated = evaluate_retrieval(query, pairs_for_eval)
        action = determine_correction(query, evaluated)
        if action.strategy == "rewrite" and action.rewritten_query:
            _count(fired, "corrective_rewrite")
            retry = index.retrieve(action.rewritten_query, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
            seen = {h.chunk.chunk_id for h in hits}
            for rh in retry.hits:
                if rh.chunk.chunk_id not in seen:
                    hits.append(rh)
                    seen.add(rh.chunk.chunk_id)
        elif action.strategy == "refine":
            _count(fired, "corrective_refine_no_list_change")  # sentence extraction for the generator; the ranked list is untouched
        elif action.strategy == "decompose" and action.sub_queries:
            _count(fired, "corrective_decompose")
            for sub_q in action.sub_queries:
                sub_r = index.retrieve(sub_q, timer=timer, k=cfg.retrieve_k, n=cfg.rerank_n)
                seen = {h.chunk.chunk_id for h in hits}
                for rh in sub_r.hits:
                    if rh.chunk.chunk_id not in seen:
                        hits.append(rh)
                        seen.add(rh.chunk.chunk_id)

    if cfg.staleness_tier1 and clusters and not stale_first:
        hits = prefer_current(hits, clusters, query)

    if cfg.hipporag and hippo_triples:
        # Gate = the router's prediction. The gold category is never passed (it was, before M7: a label leak).
        if should_apply(query, router=index.router):
            _count(fired, "hipporag")
        with timer.stage("hipporag"):
            hits = apply_hipporag(query, hits, hippo_triples, router=index.router)
        if result is not None:
            result.hits = hits

    if cfg.graphrag and communities:
        matched = match_communities(query, communities)
        if matched:
            _count(fired, "graphrag")
            with timer.stage("graphrag"):
                hits = boost_by_communities(hits, matched)
            if result is not None:
                result.hits = hits

    paths = _paths_from_hits(hits)
    if cfg.graph_hops and graph is not None:
        _count(fired, "graph_hops")
        paths = expand_with_graph(paths, graph, hops=cfg.graph_hops)
    return paths, result if result is not None else hits


def _prefer_current_pairs(pairs: list, clusters, query: str) -> list:
    """Staleness Tier 1 (prefer_current) on (chunk, score) pairs, keeping their order and scores."""
    kept = prefer_current([RetrievalHit(chunk=c, score=s, rank=i) for i, (c, s) in enumerate(pairs, start=1)], clusters, query)
    return [(h.chunk, h.score) for h in kept]


def _serve_from_cache(cache: SemanticCache, query: str, qid: str, fresh: list[str]) -> tuple[list[str], dict[str, Any]]:
    """One lookup of the cache arm. A hit serves the cached file list; a miss serves `fresh` (what retrieval
    returned for this question) and stores it. The event says which entry a hit came from and whether its
    list is the one retrieval returns anyway (`same_list`); a hit with a different list is a wrong hit."""
    cached = cache.get(query)
    sim = None if cache.last_similarity is None else round(cache.last_similarity, 6)
    if cached is None:
        cache.put(query, {"id": qid, "paths": list(fresh)})
        return fresh, {"id": qid, "hit": False, "similarity": sim}
    served = list(cached["paths"])
    return served, {"id": qid, "hit": True, "served_from": cached["id"], "similarity": sim, "same_list": served == list(fresh)}


def _min_margin(events: list[dict[str, Any]], threshold: float) -> float | None:
    sims = [e["similarity"] for e in events if e.get("similarity") is not None]
    return round(min(abs(x - threshold) for x in sims), 6) if sims else None


def _replay_probes(cache: SemanticCache, probes: list, fresh_paths, n_files: int | None) -> dict[str, Any]:
    """Look up each paraphrase probe in the cache the gold replay filled, without adding it. A hit served
    from the rewritten question's own entry is a true hit; from another question's entry, a false one.
    nDCG@10 of what is served against fresh retrieval for the probe, on the relevant paths it inherits."""
    events, served_rows, fresh_rows = [], [], []
    for probe in probes:
        fresh = fresh_paths(probe.question)
        cached = cache.get(probe.question)
        sim = None if cache.last_similarity is None else round(cache.last_similarity, 6)
        served = list(cached["paths"]) if cached is not None else fresh
        event: dict[str, Any] = {"id": probe.id, "source_id": probe.source_id, "rule": probe.rule, "hit": cached is not None,
                                 "similarity": sim}
        if cached is not None:
            event.update(served_from=cached["id"], same_list=served == fresh)
        events.append(event)
        served_rows.append({"id": probe.id, "retrieved": served, "relevant": list(probe.relevant_paths)})
        fresh_rows.append({"id": probe.id, "retrieved": fresh, "relevant": list(probe.relevant_paths)})
    served_scores = [r["ndcg@10"] for r in aggregate_retrieval(served_rows, n_files=n_files).per_query]
    fresh_scores = [r["ndcg@10"] for r in aggregate_retrieval(fresh_rows, n_files=n_files).per_query]
    idx = resample_indices(len(probes))
    d, lo, hi = paired_delta_ci(served_scores, fresh_scores, idx)
    hits = [e for e in events if e["hit"]]
    return {
        "file": CACHE_PROBES,
        "n": len(probes),
        "added_to_cache": False,
        "hits_from_source": sum(1 for e in hits if e["served_from"] == e["source_id"]),
        "hits_from_other": sum(1 for e in hits if e["served_from"] != e["source_id"]),
        "misses": len(events) - len(hits),
        "hits_with_a_list_fresh_retrieval_would_not_return": sum(1 for e in hits if not e["same_list"]),
        "min_margin_to_threshold": _min_margin(events, cache.similarity_threshold),
        "ndcg@10": {"served": round(sum(served_scores) / len(probes), 4), "fresh": round(sum(fresh_scores) / len(probes), 4),
                    "delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                    "note": "served minus fresh retrieval, paired bootstrap over the probes; relevant paths inherited, not judged"},
        "lookups": events,
    }


def _rerank_settings(cfg: BenchConfig, model) -> dict[str, Any]:
    """What a row that sets the served reranker settings records next to the model (nothing for the other rows,
    whose models block is compared field by field by `make bench-check`): the settings, and what loaded."""
    if cfg.rerank_depth is None and cfg.rerank_precision is None and cfg.rerank_max_length is None:
        return {}
    inner = getattr(model, "model", None)
    return {
        "depth": cfg.rerank_depth,
        "precision": cfg.rerank_precision or "fp32",
        "max_length": cfg.rerank_max_length,
        "loaded_dtype": str(getattr(inner, "dtype", None)) if inner is not None else None,
        "loaded_max_length": getattr(model, "max_length", None) if model is not None else None,
    }


def _generations_path(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else project_root() / path


def _sklearn_versions() -> dict[str, str]:
    """The TF-IDF embedder's fit (TfidfVectorizer + randomized TruncatedSVD) depends on these versions."""
    import scipy
    import sklearn

    return {"scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def _jury_complete():
    """Optional Ollama backend. Probe once; stay heuristic if the daemon is down."""
    flag = os.environ.get("JUDGE", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return None
    from app.eval.judge import ollama_complete

    probe = ollama_complete('Reply JSON {"score": 1, "label": "correct"}')
    if not probe:
        return None
    return lambda prompt: ollama_complete(prompt) or ""


BGE_SMALL = "st:BAAI/bge-small-en-v1.5"
BGE_BASE = "st:BAAI/bge-base-en-v1.5"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"

# Real models on fixture v1 (make bench-neural). bm25_only is repeated so the
# paired deltas have the same reference as the hash table; e2e heuristics and
# triple logging are off because they add nothing to the retrieval question.
_N = dict(log_triples=False, e2e=False)
NEURAL_CONFIGS = [
    BenchConfig(name="bm25_only", mode="bm25", enable_rerank=False, enable_router=False, staleness_tier1=False, **_N),
    BenchConfig(name="dense_only@bge-small", mode="dense", embedder=BGE_SMALL, enable_rerank=False, enable_router=False, staleness_tier1=False, parent="bm25_only", **_N),
    BenchConfig(name="hybrid@bge-small", mode="hybrid", embedder=BGE_SMALL, enable_rerank=False, enable_router=False, staleness_tier1=False, parent="dense_only@bge-small", **_N),
    BenchConfig(name="hybrid+bge-rerank@hash", mode="hybrid", embedder="hash", reranker=BGE_RERANKER, enable_rerank=True, enable_router=False, staleness_tier1=False, parent="bm25_only", **_N),
    BenchConfig(name="hybrid+bge-rerank@bge-small", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, enable_rerank=True, enable_router=False, staleness_tier1=False, parent="hybrid@bge-small", **_N),
    BenchConfig(name="hybrid+bge-rerank+router@bge-small", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, enable_rerank=True, enable_router=True, staleness_tier1=False, parent="hybrid+bge-rerank@bge-small", **_N),
    BenchConfig(name="hybrid+bge-rerank+router+staleness@bge-small", mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, enable_rerank=True, enable_router=True, staleness_tier1=True, parent="hybrid+bge-rerank+router@bge-small", **_N),
    BenchConfig(name="dense_only@bge-base", mode="dense", embedder=BGE_BASE, enable_rerank=False, enable_router=False, staleness_tier1=False, parent="bm25_only", **_N),
    BenchConfig(name="hybrid@bge-base", mode="hybrid", embedder=BGE_BASE, enable_rerank=False, enable_router=False, staleness_tier1=False, parent="dense_only@bge-base", **_N),
]

# Extra paired comparisons that are not parent/child rows (row, other): reported in blob["comparisons"].
NEURAL_COMPARISONS = [("hybrid+bge-rerank@bge-small", "hybrid+bge-rerank@hash")]

# M17 (B23): the served reranker settings on fixture v1, one change at a time from neural.json's depth-50 row (every
# fused candidate, fp32, the model's own token cap) to what the API serves on CUDA (RERANK_DEPTH 20,
# RERANKER_MAX_LENGTH 512, RERANKER_PRECISION fp16), then the router and staleness Tier 1, then Tier 1 moved in front of
# the reranker as the served retriever orders them: that last row is the in-memory configuration of
# bench/results/parity_neural.json (app/eval/parity.py). make bench-rerank-depth -> bench/results/rerank_depth.json
# (each row with on how many questions its list differs from its parent's).
_RR = dict(mode="hybrid", embedder=BGE_SMALL, reranker=BGE_RERANKER, enable_rerank=True, **_N)
RERANK_DEPTH_CONFIGS = [
    BenchConfig(name="hybrid@bge-small", mode="hybrid", embedder=BGE_SMALL, enable_rerank=False, enable_router=False, staleness_tier1=False, **_N),
    BenchConfig(name="hybrid+bge-rerank@bge-small", enable_router=False, staleness_tier1=False, parent="hybrid@bge-small", **_RR),
    BenchConfig(name="hybrid+bge-rerank@bge-small/top20", enable_router=False, staleness_tier1=False, rerank_depth=20,
                parent="hybrid+bge-rerank@bge-small", **_RR),
    BenchConfig(name="hybrid+bge-rerank@bge-small/fp32-512/top20", enable_router=False, staleness_tier1=False, rerank_depth=20,
                rerank_max_length=512, parent="hybrid+bge-rerank@bge-small/top20", **_RR),
    BenchConfig(name="hybrid+bge-rerank@bge-small/fp16-512/top20", enable_router=False, staleness_tier1=False, rerank_depth=20,
                rerank_max_length=512, rerank_precision="fp16", parent="hybrid+bge-rerank@bge-small/fp32-512/top20", **_RR),
    BenchConfig(name="hybrid+bge-rerank+router+staleness@bge-small/fp16-512/top20", enable_router=True, staleness_tier1=True,
                rerank_depth=20, rerank_max_length=512, rerank_precision="fp16", parent="hybrid+bge-rerank@bge-small/fp16-512/top20", **_RR),
    BenchConfig(name="hybrid+bge-rerank+router+staleness@bge-small/fp16-512/top20/staleness-first", enable_router=True, staleness_tier1=True,
                rerank_depth=20, rerank_max_length=512, rerank_precision="fp16", staleness_before_rerank=True,
                parent="hybrid+bge-rerank+router+staleness@bge-small/fp16-512/top20", **_RR),
]
# Per-category paired deltas vs parent for the rows whose mechanism targets a category; n is small and said so.
CATEGORY_DELTAS = {"hybrid+rerank+router": "exact_path", "hybrid+rerank+router+staleness": "staleness",
                   "hybrid+bge-rerank+router@bge-small": "exact_path", "hybrid+bge-rerank+router+staleness@bge-small": "staleness"}
SMALL_N = 30


def run_config(
    cfg: BenchConfig,
    gold: list[GoldQuestion],
    files_root: Path,
    n_files: int | None = None,
    *,
    keep_retrieved: bool = False,
    mtimes: str = "auto",
    probes: list | None = None,
) -> dict[str, Any]:
    """Score one config. `mtimes` is where chunk mtimes come from (index_loader.MTIME_MODES).
    `probes` (app.eval.probes.Probe) are looked up after the gold replay by the cache arm only."""
    gen = load_generations(_generations_path(cfg.generations)) if cfg.generations else None
    prefixes = context_prefixes(gen) if gen is not None and cfg.contextual else None
    hyde_complete = hyde_completion(gen) if gen is not None and cfg.hyde else None
    index = build_index(
        files_root,
        embedder_name=cfg.embedder,
        retrieve_k=cfg.retrieve_k,
        rerank_n=cfg.rerank_n,
        enable_router=cfg.enable_router,
        enable_rerank=cfg.enable_rerank,
        reranker=cfg.reranker,
        chunk_strategy=cfg.chunk_strategy,
        contextual=cfg.contextual,
        mtimes=mtimes,
        prefixes=prefixes,
        rerank_depth=cfg.rerank_depth,
        rerank_precision=cfg.rerank_precision,
        rerank_max_length=cfg.rerank_max_length,
    )
    late = None
    if cfg.late_chunking:
        from app.chunking.late import late_chunk_index

        late = late_chunk_index(index, files_root)
    clusters = cluster_versions(index.chunks) if cfg.staleness_tier1 else {}
    graph = None
    if cfg.graph_hops:
        files = []
        seen = set()
        for chunk in index.chunks:
            if chunk.path in seen:
                continue
            seen.add(chunk.path)
            files.append((chunk.path, chunk.text, chunk.mtime))
        graph = build_file_graph(files)
    hippo_triples = triples_from_chunks(index.chunks) if cfg.hipporag else None
    communities = build_graphrag(index.chunks) if cfg.graphrag else None
    fired: dict[str, int] = {}
    jury = default_jury(_jury_complete()) if cfg.jury else None

    timer = StageTimer()
    per_query = []
    e2e_rows = []
    triples = []
    routes: dict[str, int] = {}
    jury_sf_answers: list[str] = []
    jury_sf_golds: list[str] = []
    jury_sf_labels: list[str] = []

    if gold:  # warm-up: caches, CUDA kernels; retrieval is deterministic so results are unaffected
        _retrieve(index, gold[0].question, cfg, StageTimer(), clusters, graph, hippo_triples=hippo_triples, communities=communities,
                  hyde_complete=hyde_complete)
    # Built after the warm-up, so the first gold question meets an empty cache. Fresh retrieval still runs on
    # every question (a hit's list is compared with it), so this row's latency is not the cache's.
    cache = SemanticCache(index.embedder) if cfg.cache else None
    cache_events: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for q in gold:
        paths, payload = _retrieve(
            index,
            q.question,
            cfg,
            timer,
            clusters,
            graph,
            hippo_triples=hippo_triples,
            communities=communities,
            fired=fired,
            hyde_complete=hyde_complete,
        )
        if cache is not None:
            with timer.stage("cache_lookup"):
                paths, event = _serve_from_cache(cache, q.question, q.id, paths)
            cache_events.append(event)
            if event["hit"]:
                _count(fired, "cache_hit")
        per_query.append(
            {
                "id": q.id,
                "retrieved": paths,
                "relevant": q.relevant_ids(),
                "category": q.category,
            }
        )
        result = payload if hasattr(payload, "hits") else None
        if result is not None:
            routes[result.route.route.value] = routes.get(result.route.route.value, 0) + 1
            if cfg.log_triples:  # serving features of the list the reranker received, labelled with its scores
                triples.extend(bench_log_triples(q.question, result))
        if cfg.e2e:
            hits = result.hits[:8] if result is not None else payload[:8]
            contexts = [h.chunk.text for h in hits]
            answer = extractive_answer(q.question, hits)
            row = {
                "id": q.id,
                "faithfulness": faithfulness(answer, contexts),
                "context_precision": context_precision(q.relevant_paths, paths, k=10),
                "context_recall": context_recall(q.relevant_paths, paths, k=50),
                "answer_relevancy": answer_relevancy(q.question, answer),
            }
            if jury is not None:
                verdict = jury.vote(q.question, answer, contexts, q.answer)
                row["jury_label"] = verdict.label
                row["jury_score"] = verdict.score
                if q.category == "simple_factual":
                    jury_sf_answers.append(answer)
                    jury_sf_golds.append(q.answer)
                    jury_sf_labels.append(verdict.label)
            e2e_rows.append(row)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    retrieval = aggregate_retrieval(per_query, n_files=n_files)
    payload: dict[str, Any] = {
        "config": cfg.name,
        "settings": bench_settings(cfg),
        "retrieval": retrieval.as_dict(),
        "latency": timer.summary(),
        "wall_ms": round(wall_ms, 2),
        "routes": routes,
        "n_chunks": len(index.chunks),
    }
    if cfg.e2e:
        payload["e2e"] = aggregate_e2e(e2e_rows).as_dict()
    payload["_per_query_scores"] = retrieval.per_query  # gold order; used for the bootstrap, then dropped
    payload["_ingest"] = index.ingest_report  # what the loader indexed: the header's ingest block (ingest_header), then dropped
    if keep_retrieved:
        payload["_retrieved"] = [row["retrieved"] for row in per_query]  # gold order; compared with the parent, then dropped
    if fired:
        payload["fired"] = dict(sorted(fired.items()))
    if communities is not None:
        payload["graphrag"] = community_stats(communities, len({c.path for c in index.chunks}))
    if late is not None:
        payload["late_chunking"] = late
    if gen is not None:
        llm: dict[str, Any] = {"file": cfg.generations, "model": gen["model"], "options": gen["options"]}
        if cfg.hyde:
            ok = {i["question"]: i["ok"] for i in gen["hyde"]["items"]}
            failed = [q.id for q in gold if not ok[q.question]]
            llm["hyde"] = {"n": len(gold), "n_failed": len(failed), "failed_ids": failed}
        if cfg.contextual:
            llm["context"] = {"n": len(index.chunks), "n_failed": sum(1 for c in index.chunks if not prefixes[c.chunk_id]["ok"])}
        payload["llm"] = llm
    if cache is not None:
        hits = [e for e in cache_events if e["hit"]]
        payload["cache"] = {
            "class": "app.retrieval.cache.SemanticCache",
            "threshold": cache.similarity_threshold,
            "max_entries": cache.max_entries,
            "ttl_seconds": cache.ttl_seconds,
            "similarity": f"cosine of the index embedder's vectors ({cfg.embedder})",
            "cached_value": "the question's final file list",
            "gold": {
                "n": len(cache_events),
                "order": "gold order, cache empty at the start",
                "hits": len(hits),
                "misses": len(cache_events) - len(hits),
                "wrong_hits": sum(1 for e in hits if not e["same_list"]),
                "min_margin_to_threshold": _min_margin(cache_events, cache.similarity_threshold),
                "lookups": cache_events,
            },
        }
        if probes:
            fresh = lambda text: _retrieve(index, text, cfg, StageTimer(), clusters, graph,  # noqa: E731
                                           hippo_triples=hippo_triples, communities=communities, hyde_complete=hyde_complete)[0]
            payload["cache"]["probes"] = _replay_probes(cache, probes, fresh, n_files)
    payload["reranker_name"] = getattr(index, "reranker_name", cfg.reranker)
    payload["reranker_loaded"] = bool(getattr(index, "reranker_loaded", False))
    rerank_model = getattr(getattr(index, "rerank_fn", None), "model", None)
    payload["models"] = {
        "embedder": {
            **model_provenance(cfg.embedder, "embedder"),
            "dim": int(getattr(index.embedder, "dim", 0)),
            "device": model_device(getattr(index.embedder, "_model", None)),
            # bge-v1.5 recommends a query prefix; the benchmark embeds the bare query. The served HuggingFace embedder
            # adds LlamaIndex's prefix unless EMBEDDING_QUERY_INSTRUCTION=none (app/settings.py), as the model-backed
            # parity run sets it.
            "query_instruction": None,
            **({"library": _sklearn_versions()} if cfg.embedder == "tfidf" else {}),
        },
        "reranker": {
            **model_provenance(cfg.reranker, "reranker"),
            "loaded": payload["reranker_loaded"],
            "device": model_device(rerank_model) if payload["reranker_loaded"] else None,
            **_rerank_settings(cfg, rerank_model if payload["reranker_loaded"] else None),
        },
    }
    fallback = getattr(index, "reranker_fallback", None)
    if fallback:
        payload["reranker_fallback"] = fallback
    if cfg.multi_query:
        payload["multi_query"] = True
    if cfg.hipporag:
        payload["hipporag"] = True
    if jury is not None and jury_sf_labels:
        report = kappa_vs_gold(jury_sf_answers, jury_sf_golds, jury_sf_labels)
        payload["jury"] = {
            "mean_score": round(
                sum(r.get("jury_score", 0.0) for r in e2e_rows) / max(len(e2e_rows), 1), 4
            ),
            "kappa_simple_factual": report.as_dict(),
            "readme_ok": report.kappa >= KAPPA_GATE,
            "members": [name for name, _ in jury.members],
        }
        if not report.gated:
            payload["jury"]["note"] = (
                f"kappa {report.kappa:.3f} < {KAPPA_GATE}; do not print jury scores on the README"
            )
    if triples:
        payload["n_triples"] = len(triples)
        payload["_triples"] = triples
    return payload


DEFAULT_CONFIGS = [
    BenchConfig(name="dense_only", mode="dense", enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False),
    BenchConfig(name="bm25_only", mode="bm25", enable_rerank=False, enable_router=False, staleness_tier1=False, log_triples=False, parent="dense_only"),
    BenchConfig(name="hybrid", mode="hybrid", enable_rerank=False, enable_router=False, staleness_tier1=False, parent="bm25_only"),
    BenchConfig(name="hybrid+rerank", mode="hybrid", enable_rerank=True, enable_router=False, staleness_tier1=False, parent="hybrid"),
    BenchConfig(name="hybrid+rerank+router", mode="hybrid", enable_rerank=True, enable_router=True, staleness_tier1=False, parent="hybrid+rerank"),
    BenchConfig(
        name="hybrid+rerank+router+staleness",
        mode="hybrid",
        enable_rerank=True,
        enable_router=True,
        staleness_tier1=True,
        parent="hybrid+rerank+router",
    ),
]


# Prototype arms (make bench-frontier). Each is one switch away from its parent, so the
# paired delta isolates the mechanism. Parents are not listed here: frontier_configs()
# prepends them, and `hybrid+bge-rerank` is itself the reranked parent of the real-model run.
_F = dict(mode="hybrid", enable_router=False, staleness_tier1=False, e2e=False, log_triples=False)
_D = {**_F, "mode": "dense"}
FRONTIER_PARENT, FRONTIER_RERANK_PARENT, FRONTIER_DENSE_PARENT = "hybrid", "hybrid+rerank", "dense_only"
FRONTIER_CONFIGS = [
    BenchConfig(name="hybrid+bge-rerank", enable_rerank=True, reranker=BGE_RERANKER, parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+rankgpt", enable_rerank=True, reranker="rankgpt", parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+multiquery", enable_rerank=False, multi_query=True, parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+hipporag", enable_rerank=False, hipporag=True, parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+hops=1", enable_rerank=False, graph_hops=1, parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+hops=2", enable_rerank=False, graph_hops=2, parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+fixed-chunks", enable_rerank=False, chunk_strategy="fixed", parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+contextual", enable_rerank=False, contextual=True, parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+late-chunking", enable_rerank=False, late_chunking=True, parent=FRONTIER_PARENT, **_F),
    # HyDE in the first stage: the HyDE vector is the dense leg's query (HYDE_STAGES).
    BenchConfig(name="hybrid+hyde", enable_rerank=False, hyde=True, hyde_stage="first_stage", parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="dense_only+hyde", enable_rerank=False, hyde=True, hyde_stage="first_stage", parent=FRONTIER_DENSE_PARENT, **_D),
    BenchConfig(name="hybrid+hyde+rerank", enable_rerank=True, hyde=True, hyde_stage="first_stage", parent=FRONTIER_RERANK_PARENT, **_F),
    # HyDE fused after the reranker: the rows recorded under these names before M18, kept as that wiring's result.
    BenchConfig(name="hybrid+rerank+hyde", enable_rerank=True, hyde=True, hyde_stage="after_rerank", parent=FRONTIER_RERANK_PARENT, **_F),
    BenchConfig(name="hybrid+rerank+decompose", enable_rerank=True, decompose=True, parent=FRONTIER_RERANK_PARENT, **_F),
    BenchConfig(name="hybrid+rerank+corrective", enable_rerank=True, corrective=True, parent=FRONTIER_RERANK_PARENT, **_F),
    BenchConfig(name="hybrid+rerank+hyde+corrective", enable_rerank=True, hyde=True, hyde_stage="after_rerank", corrective=True,
                parent=FRONTIER_RERANK_PARENT, **_F),
    BenchConfig(name="hybrid+rerank+adaptive", enable_rerank=True, adaptive=True, parent=FRONTIER_RERANK_PARENT, **_F),
    BenchConfig(name="hybrid+rerank+graphrag", enable_rerank=True, graphrag=True, parent=FRONTIER_RERANK_PARENT, **_F),
    BenchConfig(name="hybrid+rerank+cache", enable_rerank=True, cache=True, parent=FRONTIER_RERANK_PARENT, **_F),
    # A second model-free embedder (TF-IDF + truncated SVD, fitted on the indexed chunks) against the hash embedder.
    BenchConfig(name="dense_only@tfidf", embedder="tfidf", enable_rerank=False, parent=FRONTIER_DENSE_PARENT, **_D),
    BenchConfig(name="hybrid@tfidf", embedder="tfidf", enable_rerank=False, parent=FRONTIER_PARENT, **_F),
    # LLM-written generations (make frontier-generations), stored and read back: no model call while scoring.
    BenchConfig(name="hybrid+hyde-llm", enable_rerank=False, hyde=True, hyde_stage="first_stage", generations=GENERATIONS_FILE,
                parent=FRONTIER_PARENT, **_F),
    BenchConfig(name="hybrid+hyde-llm+rerank", enable_rerank=True, hyde=True, hyde_stage="first_stage", generations=GENERATIONS_FILE,
                parent=FRONTIER_RERANK_PARENT, **_F),
    BenchConfig(name="hybrid+contextual-llm", enable_rerank=False, contextual=True, generations=GENERATIONS_FILE, parent=FRONTIER_PARENT, **_F),
]
# Pairs of arms compared directly (row minus other), with the same paired bootstrap: the two HyDE wirings, and
# each LLM-written arm against the heuristic arm it replaces. Real-model names are derived with _neural_name.
FRONTIER_COMPARISONS = [
    ("hybrid+hyde+rerank", "hybrid+rerank+hyde"),
    ("hybrid+hyde-llm", "hybrid+hyde"),
    ("hybrid+hyde-llm+rerank", "hybrid+hyde+rerank"),
    ("hybrid+contextual-llm", "hybrid+contextual"),
]
_LLM_REASON = ("LLM-written generations (make frontier-generations) are scored with the real models only "
               "(make bench-frontier-neural); the hash embedder rows keep the keyword and path-words heuristics")
# Arms a run cannot score, with the reason; written into the results file instead of a row.
FRONTIER_NOT_EVALUATED = {
    "hybrid+late-chunking": "late chunking pools a model's token embeddings; the hash embedder (character n-grams) has none, "
                            "so this arm exists only in the real-model run (make bench-frontier-neural)",
    "hybrid+hyde-llm": _LLM_REASON,
    "hybrid+hyde-llm+rerank": _LLM_REASON,
    "hybrid+contextual-llm": _LLM_REASON,
}
# Arms of the CPU run only: the real-model run swaps every embedder for bge-small, which would turn them into their parents.
FRONTIER_CPU_ONLY = {
    "dense_only@tfidf": "TF-IDF is a model-free embedder compared with the hash embedder; its rows are in the CPU run (make bench-frontier)",
    "hybrid@tfidf": "TF-IDF is a model-free embedder compared with the hash embedder; its rows are in the CPU run (make bench-frontier)",
}


def _neural_name(name: str) -> str:
    return name.replace("+rerank", "+bge-rerank") + "@bge-small"


def _frontier_parents() -> list[BenchConfig]:
    """The rows the arms are compared with. dense_only's own delta is against hybrid (the BM25 leg removed)."""
    return [
        BenchConfig(name=FRONTIER_PARENT, enable_rerank=False, **_F),
        BenchConfig(name=FRONTIER_RERANK_PARENT, enable_rerank=True, parent=FRONTIER_PARENT, **_F),
        BenchConfig(name=FRONTIER_DENSE_PARENT, enable_rerank=False, parent=FRONTIER_PARENT, **_D),
    ]


FRONTIER_PARENTS = tuple(c.name for c in _frontier_parents())


def frontier_configs(neural: bool = False) -> list[BenchConfig]:
    """Parent rows first, then every prototype arm with its parent named.

    CPU fallbacks: hash embedder and overlap reranker (the parents are the
    `hybrid`, `hybrid+rerank` and `dense_only` rows of the default bench). neural=True: the
    same arms over bge-small and the real bge-reranker-v2-m3, with their own
    parent rows, so each results file is self-contained.
    """
    parents = _frontier_parents()
    arms = [c for c in FRONTIER_CONFIGS if c.name != "hybrid+bge-rerank"]
    if not neural:
        return parents + [c for c in arms if c.name not in FRONTIER_NOT_EVALUATED]
    out = []
    for c in parents + [c for c in arms if c.name not in FRONTIER_CPU_ONLY]:
        out.append(replace(
            c,
            name=_neural_name(c.name),
            parent=_neural_name(c.parent) if c.parent else None,
            embedder=BGE_SMALL,
            reranker=BGE_RERANKER if c.reranker == "overlap" else c.reranker,
        ))
    return out


def fallback_labels(row: dict[str, Any]) -> list[str]:
    """Every stand-in a row's number depends on, so tables can print them next to it."""
    st = row.get("settings") or {}
    labels = []
    if st.get("embedder") in {"hash", "tfidf"}:
        labels.append(f"{st['embedder']} embedder (no model)")
    if st.get("enable_rerank"):
        if st.get("reranker") == "overlap":
            labels.append("overlap reranker (no model)")
        elif st.get("reranker") in {"rankgpt", "listwise"} and not row.get("reranker_loaded"):
            labels.append("listwise rerank by token overlap (no LLM)")
        elif not row.get("reranker_loaded"):
            labels.append("overlap reranker (cross-encoder not loaded)")
    llm_written = {"hyde", "contextual"} if st.get("generations") else set()  # stored LLM output, not the heuristic
    for flag, label in (
        ("multi_query", "query expansion by keyword heuristic (no LLM)"),
        ("hipporag", "PPR over regex triples with a hand-written path prior (no LLM extraction)"),
        ("hyde", "HyDE hypothesis by keyword heuristic (no LLM)"),
        ("decompose", "decomposition by heuristic (no LLM)"),
        ("corrective", "corrective grader by heuristic (no LLM)"),
        ("adaptive", "retrieve-or-skip by regex (no model)"),
        ("graphrag", "regex triples, connected components, extractive summaries (no LLM, no Leiden)"),
        ("contextual", "chunk prefix from path words (no LLM)"),
    ):
        if st.get(flag) and flag not in llm_written:
            labels.append(label)
    llm = row.get("llm") or {}
    hyde, context = llm.get("hyde") or {}, llm.get("context") or {}
    if hyde.get("n_failed"):
        labels.append(f"HyDE hypothesis by keyword heuristic on {hyde['n_failed']} of {hyde['n']} questions (failed LLM calls)")
    if context.get("n_failed"):
        labels.append(f"no prefix on {context['n_failed']} of {context['n']} chunks (failed LLM calls)")
    return labels


def attach_list_changes(results: list[dict[str, Any]], retrieved: dict[str, list[list[str]]]) -> None:
    """Per row with a parent in the run: on how many questions its ranked file list differs from the parent's."""
    for row in results:
        parent = (row.get("settings") or {}).get("parent")
        mine, theirs = retrieved.get(row["config"]), retrieved.get(parent or "")
        if not mine or not theirs or len(mine) != len(theirs):
            continue
        row["vs_parent"] = {
            "parent": parent,
            "n_questions": len(mine),
            "n_top10_differs": sum(1 for a, b in zip(mine, theirs) if a[:10] != b[:10]),
            "n_list_differs": sum(1 for a, b in zip(mine, theirs) if a != b),
        }


def attach_frontier(results: list[dict[str, Any]], retrieved: dict[str, list[list[str]]]) -> dict[str, Any]:
    """Per arm: its fallback labels and on how many questions its list differs from the parent's."""
    names = {row["config"] for row in results}
    for row in results:
        row["fallbacks"] = fallback_labels(row)
    attach_list_changes(results, retrieved)
    parents = [n for n in (*FRONTIER_PARENTS, *(_neural_name(p) for p in FRONTIER_PARENTS)) if n in names]
    neural = any(n.endswith("@bge-small") for n in names)
    return {
        "parents": parents,
        "arms": [row["config"] for row in results if row["config"] not in parents],
        "not_evaluated": [{"arm": k, "reason": v} for k, v in (FRONTIER_CPU_ONLY if neural else FRONTIER_NOT_EVALUATED).items()],
        "verdict_rule": "paired bootstrap interval of nDCG@10 (arm minus parent): above zero, covers zero, or below zero",
    }


# The loader's ingest report fields a results header keeps (app/eval/ingest.py ingest_report); an in-memory build
# has no previous run, so its unchanged / removed / changed counts are always zero and are left out.
INGEST_FIELDS = ("schema", "chunker", "files_seen", "files_indexed", "files_rejected", "rejected_by_reason", "rejected_sample")


def ingest_header(reports: list[tuple[str, dict]]) -> dict:
    """What a run indexed, from the ingest report of each row's index ((chunk strategy, report) pairs): the scan's
    counts, which every row shares (one fixture, one scan rule), and the chunks embedded per chunk strategy."""
    if not reports:
        return {}
    first = reports[0][1]
    block: dict[str, Any] = {k: first[k] for k in INGEST_FIELDS}
    block["chunker_version"] = CHUNKER_VERSION  # the structure-aware ("auto") chunker's label
    chunks: dict[str, int] = {}
    for strategy, report in reports:
        if any(report[k] != first[k] for k in INGEST_FIELDS):
            raise RuntimeError(f"rows indexed different files: {[(k, report[k], first[k]) for k in INGEST_FIELDS if report[k] != first[k]]}")
        if chunks.setdefault(strategy, report["chunks_embedded"]) != report["chunks_embedded"]:
            raise RuntimeError(f"two {strategy} rows embedded {chunks[strategy]} and {report['chunks_embedded']} chunks")
    block["chunks_embedded"] = chunks
    return block


def run_bench(
    *,
    root: Path | None = None,
    configs: list[BenchConfig] | None = None,
    e2e: bool | None = None,
    out: Path | None = None,
    device: str = "auto",
    frontier: bool = False,
    compare_lists: bool = False,
) -> dict[str, Any]:
    """Score every config on fixture v1.

    Without `out`, results go to bench/results/<sha>.json and latest.json (the
    default bench). With `out`, only that file is written, so neural runs never
    touch the CI baseline. Each results file gets its per-question sidecar
    (app/eval/per_question.py: <name>.per_query.json), from which every interval
    in it is recomputed exactly. `frontier` adds what a prototype arm needs next to
    its delta: fallback labels, per-category deltas vs parent for every
    category, and the number of questions whose list differs from the parent's.
    `compare_lists` adds only that last count (vs_parent) to every row with a parent.
    """
    root = root or project_root()
    started = git_state(root)  # before the run writes anything (run_provenance)
    files_root = root / "bench" / "corpus" / "files"
    gold_path = root / "bench" / "gold" / "questions.jsonl"
    manifest_path = root / "bench" / "corpus" / "MANIFEST.json"

    gold = load_gold(gold_path)
    manifest = load_manifest(manifest_path)
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))

    configs = configs or DEFAULT_CONFIGS
    if e2e is False:
        for c in configs:
            c.e2e = False
    # The staleness filter and the file graph compare mtimes. A checkout sets them to the moment
    # each file was written, so every config indexes the mtimes the MANIFEST records instead
    # (an error if it has none), and the header says so.
    mtimes = mtime_provenance(files_root, "manifest", base=root)

    probe_path = root / CACHE_PROBES
    probes = load_probes(probe_path) if any(c.cache for c in configs) and probe_path.exists() else None

    results = []
    all_triples = []
    ingest_reports: list[tuple[str, dict]] = []
    scores_by_config: dict[str, list[dict[str, float]]] = {}
    retrieved_by_config: dict[str, list[list[str]]] = {}
    for cfg in configs:
        row = run_config(cfg, gold, files_root, n_files=manifest["n_files"], keep_retrieved=frontier or compare_lists, mtimes="manifest", probes=probes)
        ingest_reports.append((cfg.chunk_strategy, row.pop("_ingest")))
        triples = row.pop("_triples", [])
        all_triples.extend(triples)
        scores_by_config[cfg.name] = row.pop("_per_query_scores", [])
        retrieved_by_config[cfg.name] = row.pop("_retrieved", [])
        results.append(row)
    names = {c.name for c in configs}
    boot = attach_confidence(
        results,
        scores_by_config,
        n_queries=len(gold),
        categories=[q.category for q in gold],
        comparisons=[pair for pair in NEURAL_COMPARISONS + FRONTIER_COMPARISONS + [(_neural_name(a), _neural_name(b)) for a, b in FRONTIER_COMPARISONS]
                     if set(pair) <= names],
        all_categories=frontier,
    )

    sha = git_sha(root)
    blob = {
        "git_sha": sha,
        **run_provenance(root=root, started=started),  # full commit, dirty tree, sha256 of the modules that ran
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info(device),
        "corpus_sha256": manifest["aggregate_sha256"],
        "gold_sha256": sha256_file(gold_path),  # the MANIFEST lists corpus files only
        "n_gold": len(gold),
        "n_files": manifest["n_files"],
        "mtimes": mtimes,  # what the staleness filter and the file graph compared
        "ingest": ingest_header(ingest_reports),  # files seen, indexed, rejected by reason; chunks embedded per strategy
        "bootstrap": boot,
        "results": results,
    }
    gens = sorted({c.generations for c in configs if c.generations})
    if len(gens) > 1:
        raise ValueError(f"one generations file per run, got {gens}")
    if gens:
        text = _generations_path(gens[0]).read_text()
        blob["generations"] = generations_header(json.loads(text), gens[0], text)
    if probes:
        blob["probes"] = {"file": CACHE_PROBES, "sha256": content_hash(probe_path.read_text()), "n": len(probes),
                          "made_by": "python -m app.eval.probes", "labels": "inherited from the rewritten gold question, not judged"}
    if frontier:
        blob["frontier"] = attach_frontier(results, retrieved_by_config)
    elif compare_lists:
        attach_list_changes(results, retrieved_by_config)
    sidecar = per_question.build(blob, scores_by_config, [q.id for q in gold], [q.category for q in gold], CI_METRICS)
    out_dir = root / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out is not None:
        out = out if out.is_absolute() else root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        per_question.write(per_question.sidecar_path(out), sidecar)
        blob["output"] = str(out)
        return blob
    out_path = out_dir / f"{sha}.json"
    out_path.write_text(json.dumps(blob, indent=2) + "\n")
    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(blob, indent=2) + "\n")
    for path in (out_path, latest):
        per_question.write(per_question.sidecar_path(path), sidecar)
    if all_triples:
        write_triples(out_dir / f"{sha}.triples.jsonl", all_triples)
    blob["output"] = str(out_path)
    return blob


CI_METRICS = ("ndcg@10", "recall@10", "recall@50")
REFERENCE_CONFIG = "bm25_only"


def attach_confidence(
    results: list[dict[str, Any]],
    scores_by_config: dict[str, list[dict[str, float]]],
    *,
    n_queries: int,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    categories: list[str] | None = None,
    comparisons: list[tuple[str, str]] | None = None,
    all_categories: bool = False,
    metrics: tuple[str, ...] = CI_METRICS,
) -> dict:
    """Add per-config 95% CIs and paired deltas in place, for each of `metrics`.

    delta_vs_bm25_only: every row against the bm25_only row.
    delta_vs_previous:  against the row's explicit `parent` (settings.parent); a row
                        without a parent falls back to the previous list entry, so
                        older result files and ad-hoc runs still get a delta.
    delta_vs_parent_by_category: for rows in CATEGORY_DELTAS, the paired delta vs
                        parent restricted to that category, with n and small_n.
    delta_vs_parent_by_category_all: with all_categories (the frontier run), the
                        same paired delta for every category, so an arm aimed at one
                        kind of question shows where it moved.
    One index matrix is shared by every config, so all deltas are paired. Missing
    references yield None rather than an error. Returns the bootstrap settings;
    `comparisons` (extra row pairs) are returned under "comparisons".
    """
    if n_queries <= 0 or not results:
        return bootstrap_settings(n_queries, n_boot=n_boot, seed=seed)
    idx = resample_indices(n_queries, n_boot=n_boot, seed=seed)

    def column(name: str, metric: str) -> list[float] | None:
        rows = scores_by_config.get(name)
        if not rows or len(rows) != n_queries:
            return None
        return [r[metric] for r in rows]

    def deltas(name: str, other: str | None) -> dict | None:
        if other is None or other == name:
            return None
        out: dict[str, Any] = {"reference": other}
        for metric in metrics:
            a, b = column(name, metric), column(other, metric)
            if a is None or b is None:
                return None
            d, lo, hi = paired_delta_ci(a, b, idx)
            out[metric] = {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        return out

    def category_delta(name: str, other: str, category: str) -> dict | None:
        if not categories or len(categories) != n_queries:
            return None
        rows_a, rows_b = scores_by_config.get(name), scores_by_config.get(other)
        if not rows_a or not rows_b:
            return None
        sel = [i for i, c in enumerate(categories) if c == category]
        if len(sel) < 2:
            return None
        sub_idx = resample_indices(len(sel), n_boot=n_boot, seed=seed)
        out: dict[str, Any] = {"reference": other, "category": category, "n": len(sel), "small_n": len(sel) < SMALL_N}
        for metric in metrics:
            a = [rows_a[i][metric] for i in sel]
            b = [rows_b[i][metric] for i in sel]
            d, lo, hi = paired_delta_ci(a, b, sub_idx)
            out[metric] = {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        return out

    names = [row["config"] for row in results]
    reference = REFERENCE_CONFIG if REFERENCE_CONFIG in scores_by_config else None
    for i, row in enumerate(results):
        name = row["config"]
        ci: dict[str, Any] = {}
        for metric in metrics:
            values = column(name, metric)
            if values is None:
                continue
            mean, lo, hi = mean_ci(values, idx)
            ci[metric] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        row["ci"] = ci
        row["delta_vs_bm25_only"] = deltas(name, reference)
        parent = (row.get("settings") or {}).get("parent")
        previous = parent if parent in scores_by_config else (names[i - 1] if i > 0 and parent is None else None)
        row["delta_vs_previous"] = deltas(name, previous)
        category = CATEGORY_DELTAS.get(name)
        row["delta_vs_parent_by_category"] = category_delta(name, previous, category) if category and previous else None
        if all_categories and previous:
            row["delta_vs_parent_by_category_all"] = [
                block for block in (category_delta(name, previous, c) for c in sorted(set(categories or []))) if block
            ]
    settings = bootstrap_settings(n_queries, n_boot=n_boot, seed=seed)
    if comparisons:
        settings["comparisons"] = [
            {"row": a, "other": b, **(deltas(a, b) or {"reference": b, "note": "one side missing"})}
            for a, b in comparisons
        ]
    return settings


def _fmt_ci(block: dict | None, key: str = "mean") -> str:
    if not block:
        return "—"
    sign = "+" if key == "delta" and block[key] >= 0 else ""
    return f"{sign}{block[key]:.3f} [{block['lo']:+.3f}, {block['hi']:+.3f}]" if key == "delta" else f"{block[key]:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


def markdown_table(blob: dict) -> str:
    lines = [
        "| config | nDCG@10 [95% CI] | Recall@10 [95% CI] | Recall@50 (random list, same length) | unique files in top 50 | MRR@10 | Δ nDCG@10 vs bm25_only [95% CI] | p95 search ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["results"]:
        lat = row.get("latency") or {}
        search = lat.get("vector_search") or lat.get("bm25") or {}
        p95 = search.get("p95_ms", "")
        r = row["retrieval"]
        ci = row.get("ci") or {}
        extra = ""
        if row.get("reranker_fallback"):
            extra = f" (fallback={row['reranker_fallback']})"
        rand = r.get("random_recall@50")
        rand_txt = f"{r['recall@50']:.3f} ({rand:.3f})" if rand is not None else f"{r['recall@50']:.3f}"
        delta = (row.get("delta_vs_bm25_only") or {}).get("ndcg@10")
        lines.append(
            f"| {row['config']}{extra} | {_fmt_ci(ci.get('ndcg@10'))} | {_fmt_ci(ci.get('recall@10'))} | {rand_txt} | "
            f"{r.get('mean_unique_paths@50', 0):.1f} | {r['mrr@10']:.3f} | {_fmt_ci(delta, 'delta')} | {p95} |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(description="Frozen-corpus retrieval bench")
    p.add_argument(
        "--jury",
        action="store_true",
        help="extra Phase 6 jury columns (heuristic; LLM if JUDGE=1 and Ollama is up)",
    )
    p.add_argument(
        "--frontier",
        action="store_true",
        help="prototype arms with their parent rows -> bench/results/frontier.json (with --neural: real models -> frontier_neural.json)",
    )
    p.add_argument(
        "--neural",
        action="store_true",
        help="real embedders (bge-small, bge-base) and the real bge-reranker-v2-m3 on fixture v1 (needs .[ml])",
    )
    p.add_argument(
        "--rerank-depth",
        action="store_true",
        help="the served reranker settings (depth 20, max_length 512, fp16) one at a time on fixture v1 (needs .[ml] and CUDA) -> --out",
    )
    p.add_argument("--out", type=Path, default=None, help="write only this file (latest.json and the sha file are left alone)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="recorded in the results header; cpu forces CPU for torch models")
    p.add_argument("--only", default=None, help="comma-separated config names to run (subset of the selected list)")
    p.add_argument("--backend", default="memory", choices=["memory", "api"], help="api: replay the gold set through POST /api/retrieve/ (see app/eval/parity.py)")
    p.add_argument("--url", default="http://localhost:8000", help="API base URL for --backend api")
    p.add_argument("--parity-config", default=None,
                   help="--backend api: the named in-memory configuration to compare with (app/eval/parity.py IN_MEMORY_CONFIGS; default the hash one CI runs)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.backend == "api":
        from app.eval.parity import DEFAULT_OUT, IN_MEMORY, markdown_table as parity_table, run as parity_run

        blob = parity_run(args.url, in_memory=args.parity_config or IN_MEMORY.name)
        sidecar = blob.pop("_sidecar")
        out = args.out or (project_root() / DEFAULT_OUT)  # untracked; a committed parity file is named with --out
        out = out if out.is_absolute() else project_root() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        per_question.write(per_question.sidecar_path(out), sidecar)
        print(parity_table(blob))
        print(f"wrote {out}")
        if not blob["api_ok"]:  # a broken API, not the size of the gap (the table above lists the problems)
            raise SystemExit(1)
        return
    jury = args.jury or os.environ.get("JUDGE", "").strip().lower() in {"1", "true", "yes"}
    if args.frontier:
        configs = frontier_configs(neural=args.neural)
        if args.out is None:  # never the default bench outputs: latest.json belongs to `make bench`
            args.out = Path("bench/results/frontier_neural.json" if args.neural else "bench/results/frontier.json")
    elif args.rerank_depth:
        configs = list(RERANK_DEPTH_CONFIGS)
        if args.out is None:  # never the default bench outputs: latest.json belongs to `make bench`
            args.out = Path("bench/results/rerank_depth.json")
    elif args.neural:
        configs = list(NEURAL_CONFIGS)
    else:
        configs = list(DEFAULT_CONFIGS)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        if args.frontier:  # an arm's delta needs its parent row (and the reranked parent needs plain hybrid)
            for _ in range(2):
                wanted |= {c.parent for c in configs if c.name in wanted and c.parent}
        configs = [c for c in configs if c.name in wanted]
    if jury:
        configs = [replace(c, jury=True) for c in configs]
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    blob = run_bench(configs=configs, out=args.out, device=args.device, frontier=args.frontier, compare_lists=args.rerank_depth)
    print(markdown_table(blob))
    print(f"wrote {blob['output']}")


if __name__ == "__main__":
    main()
