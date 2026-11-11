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
        beir.BeirConfig(name="bm25@default-tokenizer", mode="bm25", analyzer="default", k1=1.5, b=0.75, parent="bm25"),
        beir.BeirConfig(name="dense@hash", mode="dense", embedder="hash", parent="bm25"),
        beir.BeirConfig(name="hybrid@hash", mode="hybrid", embedder="hash", parent="dense@hash", query_instruction="query: "),
        beir.BeirConfig(name="hybrid+overlap@hash", mode="hybrid", embedder="hash", reranker="overlap", rerank_top=3, parent="hybrid@hash"),
    ]
    blob = beir.run(tmp_path, configs, device="cpu", n_boot=50, download=False)

    assert blob["n_docs"] == 6 and blob["n_queries"] == 2 and blob["published"]["bm25_ndcg@10"] == 0.6789
    assert blob["published"]["url"].startswith("https://github.com/castorini/anserini") and blob["document_text"] == "title + ' ' + text"
    names = [r["config"] for r in blob["results"]]
    assert names == [c.name for c in configs]
    bm25 = blob["results"][0]
    assert bm25["retrieval"]["ndcg@10"] == 1.0 and bm25["delta_vs_bm25"] is None
    assert bm25["vs_published"]["delta"] == round(1.0 - 0.6789, 4)
    assert bm25["delta_vs_parent"] is None and "build_ms" in bm25 and bm25["build_ms"]["bm25_index_ms"] >= 0
    for row in blob["results"]:
        assert set(row["ci"]) == {"ndcg@10", "recall@10", "recall@100"}
        assert 0.0 <= row["retrieval"]["recall@100"] <= 1.0
        assert "total" in row["latency"] and row["latency"]["total"]["n"] == 2
        assert "_per_query" not in row
    hybrid = blob["results"][3]
    assert hybrid["delta_vs_bm25"]["reference"] == "bm25" and "delta" in hybrid["delta_vs_bm25"]["ndcg@10"]
    assert hybrid["delta_vs_parent"]["reference"] == "dense@hash" and hybrid["parent"] == "dense@hash"
    assert hybrid["models"]["embedder"]["query_instruction"] == "query: "
    assert hybrid["models"]["embedder"]["model"] == "hash" and hybrid["models"]["embedder"]["neural"] is False
    reranked = blob["results"][4]
    assert reranked["models"]["reranker"]["loaded"] is True and "rerank" in reranked["latency"]
    assert reranked["delta_vs_parent"]["reference"] == "hybrid@hash"
    assert reranked["latency"]["total"]["n"] == 2  # warm-up query excluded
    assert (tmp_path / "cache").exists()  # hash doc matrix cached on disk
    assert "seed" in blob["bootstrap"] and "device" in blob and "command" in blob
    assert "BEIR SciFact" in beir.markdown_table(blob)


def test_default_rows_keep_their_own_rerank_depth():
    depths = {c.name: c.rerank_top for c in beir.DEFAULT_CONFIGS if c.reranker}
    assert depths["hybrid+bge-rerank@bge-small"] == 20
    assert depths["hybrid+bge-rerank@bge-small/fp16-512/top50"] == 50
    assert depths["hybrid+bge-rerank@bge-small/fp16-512/top100"] == 100
    assert {c.name: c.parent for c in beir.DEFAULT_CONFIGS}["hybrid+bge-rerank@bge-small/fp16-512/top100"] == "hybrid+bge-rerank@bge-small/fp16-512/top50"


def test_graded_relevance_and_trec_output_are_preserved(tmp_path):
    import math
    root = _write_dataset(tmp_path)
    root.rename(tmp_path / "nfcorpus")
    root = tmp_path / "nfcorpus"
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\nq2\td3\t1\n")
    _, _, qrels = beir.load_beir(root)
    assert qrels["q1"] == {"d1": 2, "d2": 1}
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert beir.graded_ndcg_at_k(["d2", "d1"], qrels["q1"], 10) == pytest.approx(expected)
    blob = beir.run(tmp_path, [beir.BeirConfig(name="bm25", mode="bm25")], dataset="nfcorpus", download=False, device="cpu", n_boot=20, trec_dir=tmp_path / "trec")
    assert blob["dataset"] == "BEIR/nfcorpus" and blob["published"] is None
    assert "vs_published" not in blob["results"][0]
    assert blob["results"][0]["trec"]["run_sha256"]
    assert (tmp_path / "trec" / "qrels.txt").read_text().startswith("q1 0 d1 2")


def test_cache_identity_changes_with_corpus_content_and_order(tmp_path):
    corpus_a = beir.chunks_from_corpus({"d1": "cats", "d2": "dogs"})
    corpus_b = beir.chunks_from_corpus({"d1": "boats", "d2": "trucks"})
    a = beir.Engine(corpus_a, tmp_path)
    b = beir.Engine(corpus_b, tmp_path)
    assert a.corpus_sha256 != b.corpus_sha256
    assert a.corpus_sha256 != beir.Engine(list(reversed(corpus_a))).corpus_sha256
    a.doc_matrix("hash")
    b.doc_matrix("hash")
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_cap_and_precision_are_isolated_ablations():
    from dataclasses import asdict
    configs = {cfg.name: cfg for cfg in beir.DEFAULT_CONFIGS}
    for name, factor in [(beir.RERANK_FP32_512, "max_length"), (beir.RERANK_FP16, "precision")]:
        cfg = configs[name]
        parent = configs[cfg.parent]
        differences = {key for key, value in asdict(cfg).items() if key not in {"name", "parent"} and value != asdict(parent)[key]}
        assert differences == {factor}


def test_independent_trec_eval_metrics_if_available():
    pytest.importorskip("pytrec_eval")
    qrels = {"q": {"high": 2, "low": 1}}
    rankings = {"q": ["low", "noise", "high"]}
    pq = [{"id": "q", "ndcg@10": beir.graded_ndcg_at_k(rankings["q"], qrels["q"], 10), "recall@10": 1.0, "recall@100": 1.0}]
    checked = beir.cross_check_metrics(rankings, qrels, pq, required=True)
    assert checked["status"] == "passed" and checked["max_absolute_difference"] < 1e-9


def test_empty_config_selection_and_subset_published_claim_refused(tmp_path):
    _write_dataset(tmp_path)
    with pytest.raises(ValueError, match="config"):
        beir.run(tmp_path, [], download=False)
    blob = beir.run(tmp_path, [beir.BeirConfig(name="bm25", mode="bm25")], download=False, max_queries=1, n_boot=10)
    assert blob["n_queries"] == 1 and blob["full_split_queries"] == 2
    assert blob["published"] is None and "subset" in blob["query_selection"]


def test_dense_ties_keep_corpus_order():
    """Forty documents in two groups of identical text tie exactly within a group; corpus order decides, as in
    app/retrieval/hybrid.py (M10). numpy's default argsort over float32 scores need not keep it."""
    from app.eval.latency import StageTimer

    texts = ("learning_rate: 3e-4 encoder dinov2", "bark_type: birch fusion true")
    corpus = {f"d{i:02d}": texts[i % 3 == 0] for i in range(40)}
    first, second = [d for d, t in corpus.items() if t == texts[0]], [d for d, t in corpus.items() if t == texts[1]]
    cfg = beir.BeirConfig(name="dense@hash", mode="dense", embedder="hash", k=40)
    query = "learning_rate for the dinov2 encoder"

    forward = beir.Engine(beir.chunks_from_corpus(corpus))
    backward = beir.Engine(beir.chunks_from_corpus(dict(reversed(list(corpus.items())))))

    assert beir.retrieve(forward, cfg, query, StageTimer()) == first + second
    assert beir.retrieve(backward, cfg, query, StageTimer()) == first[::-1] + second[::-1]
    assert beir.retrieve(forward, beir.BeirConfig(name="dense@hash", mode="dense", embedder="hash", k=5), query, StageTimer()) == first[:5]


def test_dense_ranking_is_the_fixture_benchmarks_ranking():
    """The BEIR dense leg ranks with app/retrieval/hybrid.py's dense_scores and rank_by_score: same scores, same order."""
    import numpy as np

    from app.eval.latency import StageTimer
    from app.retrieval.hybrid import dense_scores, rank_by_score

    rng = np.random.RandomState(7)
    corpus = {f"d{i:03d}": " ".join(rng.choice(["cell", "star", "fibre", "sleep", "vaccine", "heart", "diet"], size=6)) for i in range(120)}
    engine = beir.Engine(beir.chunks_from_corpus(corpus))
    cfg = beir.BeirConfig(name="dense@hash", mode="dense", embedder="hash", k=50)
    mat = engine.doc_matrix("hash")
    for query in ("cell star", "sleep and diet", "vaccine heart fibre"):
        qv = engine.embedder("hash").encode([query])[0]
        assert np.array_equal(dense_scores(qv, engine.unit_matrix("hash"), unit=True), dense_scores(qv, mat))
        expected = [engine.chunks[i].chunk_id for i in rank_by_score(dense_scores(qv, mat), 50)]
        assert beir.retrieve(engine, cfg, query, StageTimer()) == expected


def test_bm25_ablation_changes_one_setting_group_per_row():
    """B03: the SciFact BM25 rows split bm25@default-tokenizer's three changes into the analyzer and (k1, b)."""
    from dataclasses import asdict

    rows = {c.name: c for c in beir.BM25_ABLATION_CONFIGS}
    assert list(rows) == ["bm25", "bm25@default-analyzer", "bm25@k1-1.5-b-0.75", "bm25@default-tokenizer"]
    base = asdict(rows["bm25"])
    changed = {name: {k for k, v in asdict(c).items() if k not in {"name", "parent"} and v != base[k]} for name, c in rows.items()}
    assert changed == {"bm25": set(), "bm25@default-analyzer": {"analyzer"}, "bm25@k1-1.5-b-0.75": {"k1", "b"},
                       "bm25@default-tokenizer": {"analyzer", "k1", "b"}}
    assert (base["analyzer"], base["k1"], base["b"]) == ("beir", 0.9, 0.4)
    assert (rows["bm25@default-analyzer"].analyzer, rows["bm25@k1-1.5-b-0.75"].k1, rows["bm25@k1-1.5-b-0.75"].b) == ("default", 1.5, 0.75)
    assert {c.parent for name, c in rows.items() if name != "bm25"} == {"bm25"} and {c.mode for c in rows.values()} == {"bm25"}
    assert rows["bm25@default-tokenizer"] == {c.name: c for c in beir.DEFAULT_CONFIGS}["bm25@default-tokenizer"]
    assert len(beir.DEFAULT_CONFIGS) == 10 and not {"bm25@default-analyzer", "bm25@k1-1.5-b-0.75"} & {c.name for c in beir.DEFAULT_CONFIGS}


def test_cli_ladder_runs_the_bm25_ablation(tmp_path):
    """`--ladder bm25-ablation` (make bench-beir-bm25-ablation) writes the four rows with paired deltas against bm25."""
    import zipfile

    _write_dataset(tmp_path)
    with zipfile.ZipFile(tmp_path / "scifact.zip", "w") as zf:  # a local archive: no download, its own sha256 pinned on the command line
        for p in sorted((tmp_path / "scifact").rglob("*")):
            zf.write(p, p.relative_to(tmp_path).as_posix())
    out = tmp_path / "ablation.json"
    args = ["--data-dir", str(tmp_path), "--dataset-sha256", beir.sha256_of(tmp_path / "scifact.zip"), "--ladder", "bm25-ablation", "--n-boot", "20", "--out", str(out)]
    assert beir.main(args) == 0
    blob = json.loads(out.read_text())
    assert [r["config"] for r in blob["results"]] == [c.name for c in beir.BM25_ABLATION_CONFIGS]
    for row in blob["results"][1:]:
        for key in ("delta_vs_parent", "delta_vs_bm25"):
            assert row[key]["reference"] == "bm25" and set(row[key]) == {"reference", "ndcg@10", "recall@10", "recall@100"}
    assert blob["results"][0]["vs_published"]["published_bm25_ndcg@10"] == 0.6789
    from app.eval.report import beir_bm25_table

    table = beir_bm25_table(blob)
    assert all(f"| {c.name} |" in table for c in beir.BM25_ABLATION_CONFIGS) and "make bench-beir-bm25-ablation" in table
    assert "The analyzer alone: nDCG@10" in table and "k1 and b alone: nDCG@10" in table and "All three together: nDCG@10" in table
    assert beir.main([*args[:-2], "--only", "bm25@default-analyzer", "--out", str(out)]) == 0
    assert [r["config"] for r in json.loads(out.read_text())["results"]] == ["bm25@default-analyzer"]
