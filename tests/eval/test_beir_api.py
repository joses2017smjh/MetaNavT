"""BEIR qrels through the served API at the document level (app/eval/beir_api.py), with a stub API."""

import json
import math

import pytest

from app.eval import beir_api
from app.eval.stats import resample_indices


def test_document_files_are_plain_names_and_map_back():
    assert beir_api.doc_file_name("31") == "31.txt"
    for bad in ("", "..", "../31", "a/b", "a b", "-x"):
        with pytest.raises(ValueError):
            beir_api.doc_file_name(bad)
    assert beir_api.doc_id_from_path("31.txt") == "31"
    assert beir_api.doc_id_from_path("shard/31.txt") == "31"
    assert beir_api.doc_id_from_path("31.md") is None
    assert beir_api.doc_id_from_path(".txt") is None
    assert beir_api.doc_id_from_path(None) is None


def test_write_corpus_writes_every_document_byte_for_byte(tmp_path):
    corpus = {"2": "café crème", "10": "", "1": "same", "3": "same"}
    info = beir_api.write_corpus(corpus, tmp_path / "files")
    assert sorted(p.name for p in (tmp_path / "files").iterdir()) == ["1.txt", "10.txt", "2.txt", "3.txt"]
    assert (tmp_path / "files" / "2.txt").read_bytes() == "café crème".encode("utf-8")
    assert (tmp_path / "files" / "10.txt").read_bytes() == b""  # written: the served scanner rejects it and counts it
    assert info["n_files"] == 4 and info["bytes"] == len("café crème".encode()) + 8
    assert beir_api.write_corpus(corpus, tmp_path / "again")["listing_sha256"] == info["listing_sha256"]


def test_a_document_ranks_at_its_best_chunk_and_unmapped_hits_are_counted():
    hits = [{"path": "a.txt"}, {"path": "b.txt"}, {"path": "a.txt"}, {"path": "notes.md"}, {"path": "b.txt"}, {"path": "c.txt"}]
    assert beir_api.doc_ranking(hits) == (["a", "b", "c"], 1)


def test_score_ranking_drops_the_query_id_and_uses_graded_gains():
    scores = beir_api.score_ranking(["q1", "d2", "d1"], {"d1": 2, "d2": 1}, "q1")
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert scores["ndcg@10"] == pytest.approx(dcg / ideal)
    assert scores["recall@10"] == 1.0 and scores["recall@50"] == 1.0
    assert scores["mrr@10"] == 1.0  # d2 is first once the query's own id is dropped
    far = beir_api.score_ranking([f"x{i}" for i in range(20)] + ["d1"], {"d1": 1}, "q")
    assert (far["recall@10"], far["recall@50"]) == (0.0, 1.0)


def stub_post(responses):
    def post(query, k):
        assert k == 50
        return responses[query]
    return post


def body(paths, route="semantic", degraded=()):
    return {"route": route, "counts": {"fused": len(paths)}, "staleness": {"enabled": False}, "degraded": list(degraded),
            "latency_ms": {"embed": 1.0, "hybrid_sql": 2.0, "total": 3.5}, "hits": [{"path": p} for p in paths]}


QUERIES = {"q1": "first", "q2": "slow", "q3": "broken", "q4": "offline", "q5": "lexical"}
QRELS = {"q1": {"d1": 1}, "q2": {"d1": 1}, "q3": {"d2": 1}, "q4": {"d3": 1}, "q5": {"d4": 1}}
RESPONSES = {
    "first": (200, body(["d9.txt", "d1.txt", "d9.txt"]), None),
    "slow": (504, None, "http_504: retrieval timed out: stage hybrid_sql"),
    "broken": (200, {"hits": None}, None),
    "offline": (None, None, "URLError: refused"),
    "lexical": (200, body(["d4.txt"], route="lexical_path"), None),
}


def test_replay_keeps_failed_requests_in_place_with_zero_scores():
    rows = beir_api.replay(stub_post(RESPONSES), QUERIES, QRELS, k=50)
    assert [r["id"] for r in rows] == list(QUERIES)
    first, slow, broken, offline, lexical = rows
    assert first["ok"] and first["ranking"] == ["d9", "d1"] and first["n_hits"] == 3 and first["n_docs"] == 2
    assert first["ndcg@10"] == pytest.approx(1 / math.log2(3))
    for failed in (slow, broken, offline):
        assert not failed["ok"] and failed["ranking"] == []
        assert all(failed[m] == 0.0 for m in beir_api.METRICS)
    assert slow["status"] == 504 and slow["error"].startswith("http_504")
    assert broken["error"] == "invalid_response: no hits list"
    assert lexical["route"] == "lexical_path" and lexical["recall@10"] == 1.0


def test_summary_counts_failures_documents_and_routes():
    rows = beir_api.replay(stub_post(RESPONSES), QUERIES, QRELS, k=50)
    summary = beir_api.summarize(rows, resample_indices(len(rows), n_boot=200, seed=0))
    assert summary["n_queries"] == 5
    assert summary["requests"]["failed"] == 3 and summary["requests"]["failed_ids"] == ["q2", "q3", "q4"]
    assert summary["requests"]["status_counts"] == {"200": 3, "504": 1, "None": 1}
    assert summary["routes"] == {"semantic": 1, "lexical_path": 1}
    assert summary["documents_per_response"] == {"mean": 1.5, "min": 1, "queries_under_10": 2, "queries_under_50": 2}
    assert summary["latency"]["e2e"]["n"] == 2 and summary["latency"]["server_stages"]["hybrid_sql"]["n"] == 2
    assert summary["retrieval"]["recall@10"] == pytest.approx(0.4)  # two of five queries find their document
    assert summary["ci"]["recall@10"]["lo"] <= 0.4 <= summary["ci"]["recall@10"]["hi"]


def test_paired_comparison_requires_the_same_queries_in_the_same_order():
    a = [{"id": "q1", "ndcg@10": 1.0, "recall@10": 1.0, "recall@50": 1.0}, {"id": "q2", "ndcg@10": 0.0, "recall@10": 0.0, "recall@50": 1.0}]
    b = [{"id": "q1", "ndcg@10": 0.5, "recall@10": 1.0, "recall@50": 1.0}, {"id": "q2", "ndcg@10": 0.0, "recall@10": 0.0, "recall@50": 1.0}]
    idx = resample_indices(2, n_boot=100, seed=0)
    d = beir_api.paired(a, b, idx)
    assert d["ndcg@10"]["delta"] == 0.25
    assert d["ndcg@10"]["covers_zero"] == (d["ndcg@10"]["lo"] <= 0.0 <= d["ndcg@10"]["hi"])
    assert d["recall@50"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0, "covers_zero": True}
    with pytest.raises(ValueError):
        beir_api.paired(a, list(reversed(b)), idx)


def test_route_breakdown_puts_failed_requests_in_their_own_group():
    rows = beir_api.replay(stub_post(RESPONSES), QUERIES, QRELS, k=50)
    reference = [{"id": r["id"], **{m: 0.5 for m in beir_api.CI_METRICS}} for r in rows]
    split = beir_api.by_route(rows, reference, n_boot=50)
    assert {k: v["n"] for k, v in split.items()} == {"failed": 3, "lexical_path": 1, "semantic": 1}
    assert split["failed"]["served"]["ndcg@10"] == 0.0 and split["failed"]["delta"]["ndcg@10"]["delta"] == -0.5


def test_public_rows_carry_no_ranking_and_no_absolute_path():
    rows = [{"id": "q1", "status": None, "error": "URLError: [Errno 2] /home/someone/sock", "ranking": ["d1"], "ok": False, "n_hits": 0, "n_docs": 0, "e2e_ms": 1.0,
             **{m: 0.0 for m in beir_api.METRICS}}]
    public = beir_api.public_rows(rows)
    assert "ranking" not in public[0] and "/home/someone" not in json.dumps(public)
