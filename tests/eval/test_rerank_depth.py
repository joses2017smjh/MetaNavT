"""M17 (B23 and the served reranker settings): the in-memory index can rerank the fused top N only, as the served
retriever does with RERANK_DEPTH, and a bench row can carry the served precision, token cap and staleness order.

With none of them set every row runs as before: the reranker gets every fused candidate (the order these tests
reproduce from the fused list), and `make bench-check` still finds main.json in every field.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from app.eval.gold import load_gold
from app.eval.harness import DEFAULT_CONFIGS, LATER_SETTINGS, RERANK_DEPTH_CONFIGS, BenchConfig, bench_settings, run_config
from app.eval.index_loader import build_index
from app.retrieval.rerank import OverlapReranker

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "bench" / "corpus" / "files"


@pytest.fixture(scope="module")
def gold():
    return load_gold(ROOT / "bench" / "gold" / "questions.jsonl")


def _pairs(hits):
    return [(h.chunk.chunk_id, h.score) for h in hits]


def test_the_default_depth_reranks_every_fused_candidate_as_before(gold):
    """No depth: the reranker gets the whole fused list, so the hits are rerank_fn(query, fused), the order every
    recorded row has (and the same as a depth at least the list's length)."""
    index = build_index(FILES, embedder_name="hash", reranker="overlap", enable_router=True, mtimes="manifest")
    deep = build_index(FILES, embedder_name="hash", reranker="overlap", enable_router=True, mtimes="manifest", rerank_depth=50)
    assert index.rerank_depth is None
    reranked = 0
    for q in gold:
        result = index.retrieve(q.question)
        expected = result.fused if result.skipped_rerank else OverlapReranker()(q.question, list(result.fused))
        assert _pairs(result.hits) == [(c.chunk_id, s) for c, s in expected], q.id
        assert _pairs(deep.retrieve(q.question).hits) == _pairs(result.hits), q.id
        reranked += not result.skipped_rerank
    assert reranked > 90  # the router skips the reranker on lexical_path and aggregation questions only


def test_a_depth_reranks_the_head_and_keeps_the_tail_in_fused_order(gold):
    index = build_index(FILES, embedder_name="hash", reranker="overlap", enable_router=False, mtimes="manifest", rerank_depth=5)
    result = index.retrieve(gold[0].question)
    head = OverlapReranker()(gold[0].question, list(result.fused[:5]))
    assert _pairs(result.hits[:5]) == [(c.chunk_id, s) for c, s in head]
    assert _pairs(result.hits[5:]) == [(c.chunk_id, s) for c, s in result.fused[5:]]  # fused order, RRF scores
    assert all(h.rerank is None for h in result.hits[5:]) and all(h.rerank is not None for h in result.hits[:5])
    assert len(result.hits) == len(result.fused) > 5
    with pytest.raises(ValueError, match="rerank_depth"):
        build_index(FILES, embedder_name="hash", reranker="overlap", rerank_depth=0)


def test_the_new_settings_are_recorded_only_on_rows_that_set_them():
    """`make bench-check` compares every field of main.json's rows, settings included."""
    assert {"rerank_depth", "rerank_precision", "rerank_max_length", "staleness_before_rerank"} <= set(LATER_SETTINGS)
    for cfg in DEFAULT_CONFIGS:
        assert not set(bench_settings(cfg)) & {"rerank_depth", "rerank_precision", "rerank_max_length", "staleness_before_rerank"}
    served = bench_settings(RERANK_DEPTH_CONFIGS[-1])
    assert (served["rerank_depth"], served["rerank_precision"], served["rerank_max_length"], served["staleness_before_rerank"]) == (20, "fp16", 512, True)


def test_fp16_without_cuda_stops_instead_of_measuring_fp32(monkeypatch):
    monkeypatch.setattr("app.eval.index_loader._cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs CUDA"):
        build_index(FILES, embedder_name="hash", reranker="BAAI/bge-reranker-v2-m3", rerank_precision="fp16")
    with pytest.raises(ValueError, match="fp32 or fp16"):
        build_index(FILES, embedder_name="hash", reranker="overlap", rerank_precision="bf16")


def test_the_rerank_depth_rows_change_one_thing_at_a_time():
    by_name = {c.name: c for c in RERANK_DEPTH_CONFIGS}
    for cfg in RERANK_DEPTH_CONFIGS[1:]:
        parent = by_name[cfg.parent]
        assert RERANK_DEPTH_CONFIGS.index(parent) < RERANK_DEPTH_CONFIGS.index(cfg)
        changed = {k for k, v in vars(cfg).items() if k not in {"name", "parent"} and vars(parent)[k] != v}
        assert changed in ({"enable_rerank", "reranker"}, {"rerank_depth"}, {"rerank_max_length"}, {"rerank_precision"},
                           {"enable_router", "staleness_tier1"}, {"staleness_before_rerank"}), (cfg.name, changed)


def test_staleness_before_or_after_a_full_rerank_gives_the_same_lists(gold):
    """Tier 1 drops a superseded chunk when its current version is anywhere in the list, and keeps the order; with
    every fused candidate reranked the two orders give the same lists (they can differ with a depth cut)."""
    cfg = next(c for c in DEFAULT_CONFIGS if c.name == "hybrid+rerank+router+staleness")
    cfg = replace(cfg, e2e=False, log_triples=False)
    after = run_config(cfg, gold, FILES, keep_retrieved=True, mtimes="manifest")
    before = run_config(replace(cfg, staleness_before_rerank=True), gold, FILES, keep_retrieved=True, mtimes="manifest")
    assert before["_retrieved"] == after["_retrieved"]
    assert before["retrieval"] == after["retrieval"]


class _Recording(OverlapReranker):
    """The overlap reranker, remembering the files of every list it was given."""

    def __init__(self):
        self.seen = []

    def __call__(self, query, pairs):
        self.seen.append({c.path for c, _ in pairs})
        return super().__call__(query, pairs)


def test_staleness_before_the_rerank_keeps_superseded_chunks_away_from_the_reranker(gold):
    """With Tier 1 first (the served order), the reranker never scores a chunk Tier 1 drops."""
    from app.eval.harness import StageTimer, _retrieve, cluster_versions

    index = build_index(FILES, embedder_name="hash", reranker="overlap", enable_router=False, mtimes="manifest")
    clusters = cluster_versions(index.chunks)
    superseded = {m.path for cluster in clusters.values() for m in cluster.superseded}
    q = next(q for q in gold if q.category == "staleness")
    cfg = BenchConfig(name="t", mode="hybrid", enable_rerank=True, enable_router=False, staleness_tier1=True, e2e=False, log_triples=False)
    lists = {}
    for first in (True, False):
        index.rerank_fn = _Recording()
        lists[first] = _retrieve(index, q.question, replace(cfg, staleness_before_rerank=first), StageTimer(), clusters, None)[0]
        lists[first, "reranked"] = index.rerank_fn.seen[0]
    kept_away = lists[False, "reranked"] - lists[True, "reranked"]
    assert kept_away and kept_away <= superseded  # only superseded files were kept away from the reranker
    assert lists[True] == lists[False]  # every candidate reranked: the same list either way
