"""SciFact runner on a synthetic six-document dataset: no network, no models."""

import json
from pathlib import Path

import pytest

pytest.importorskip("snowballstemmer")

from app.eval import beir  # noqa: E402


def _write_dataset(root: Path) -> Path:
    d = root / "scifact"
    (d / "qrels").mkdir(parents=True)
    docs = {
        "d1": ("Cell division", "Cells divide rapidly during embryonic growth."),
        "d2": ("Star formation", "Stars form in collapsing molecular clouds."),
        "d3": ("Vaccines", "Vaccination reduces measles incidence in children."),
        "d4": ("Diet", "Dietary fibre intake lowers colon cancer risk."),
        "d5": ("Sleep", "Sleep deprivation impairs memory consolidation."),
        "d6": ("Exercise", "Aerobic exercise improves cardiac output."),
    }
    with (d / "corpus.jsonl").open("w") as fh:
        for did, (title, text) in docs.items():
            fh.write(json.dumps({"_id": did, "title": title, "text": text}) + "\n")
    queries = {"q1": "do cells divide during growth", "q2": "vaccination and measles in children", "q3": "unused query"}
    with (d / "queries.jsonl").open("w") as fh:
        for qid, text in queries.items():
            fh.write(json.dumps({"_id": qid, "text": text}) + "\n")
    (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td3\t1\nq2\td5\t0\n")
    return d


def test_load_scifact_keeps_only_positive_qrels_and_their_queries(tmp_path):
    corpus, queries, qrels = beir.load_scifact(_write_dataset(tmp_path))
    assert len(corpus) == 6 and corpus["d1"].startswith("Cell division Cells divide")
    assert set(queries) == {"q1", "q2"}
    assert qrels == {"q1": {"d1": 1}, "q2": {"d3": 1}}


def test_checksum_mismatch_is_refused(tmp_path):
    (tmp_path / "scifact.zip").write_bytes(b"not a zip")
    with pytest.raises(RuntimeError, match="sha256"):
        beir.download_scifact(tmp_path)


def test_run_bm25_dense_hybrid_and_rerank_offline(tmp_path):
    _write_dataset(tmp_path)
    configs = [
        beir.BeirConfig(name="bm25", mode="bm25"),
        beir.BeirConfig(name="bm25@default-tokenizer", mode="bm25", analyzer="default", k1=1.5, b=0.75),
        beir.BeirConfig(name="dense@hash", mode="dense", embedder="hash"),
        beir.BeirConfig(name="hybrid@hash", mode="hybrid", embedder="hash"),
        beir.BeirConfig(name="hybrid+overlap@hash", mode="hybrid", embedder="hash", reranker="overlap", rerank_top=3),
    ]
    blob = beir.run(tmp_path, configs, device="cpu", n_boot=50, download=False)

    assert blob["n_docs"] == 6 and blob["n_queries"] == 2 and blob["published"]["bm25_ndcg@10"] == 0.665
    names = [r["config"] for r in blob["results"]]
    assert names == [c.name for c in configs]
    bm25 = blob["results"][0]
    assert bm25["retrieval"]["ndcg@10"] == 1.0 and bm25["delta_vs_bm25"] is None
    assert bm25["vs_published"]["delta"] == round(1.0 - 0.665, 4)
    for row in blob["results"]:
        assert set(row["ci"]) == {"ndcg@10", "recall@10", "recall@100"}
        assert 0.0 <= row["retrieval"]["recall@100"] <= 1.0
        assert "total" in row["latency"] and row["latency"]["total"]["n"] == 2
        assert "_per_query" not in row
    hybrid = blob["results"][3]
    assert hybrid["delta_vs_bm25"]["reference"] == "bm25" and "delta" in hybrid["delta_vs_bm25"]["ndcg@10"]
    assert hybrid["models"]["embedder"]["model"] == "hash" and hybrid["models"]["embedder"]["neural"] is False
    reranked = blob["results"][4]
    assert reranked["models"]["reranker"]["loaded"] is True and "rerank" in reranked["latency"]
    assert (tmp_path / "cache").exists()  # hash doc matrix cached on disk
    assert "seed" in blob["bootstrap"] and "device" in blob and "command" in blob
    assert "BEIR SciFact" in beir.markdown_table(blob)
