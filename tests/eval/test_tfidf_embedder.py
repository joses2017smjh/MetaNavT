"""B11: the TF-IDF embedder (app/retrieval/embedders.py), a model-free dense proxy whose rows are in make bench-frontier.

scikit-learn is in the eval extra; the tests that fit the vectorizer skip cleanly where it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.embedders import HashEmbedder, TfidfEmbedder, cosine_scores

DOCS = [
    "run_id: 40 encoder: resnet50 learning_rate: 1e-4 bark: oak",
    "run_id: 47 encoder: dinov2 learning_rate: 3e-4 bark: birch fusion: on",
    "run_id: 48 encoder: dinov2 learning_rate: 3e-4 bark: pine fusion: on",
    "#!/bin/bash sbatch job for run 47 with 3 stereo pairs",
    "the fusion module combines left and right stereo features",
    "draft v2 reports val RMSE 0.0412 for run 47",
]


def test_encode_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        TfidfEmbedder().encode(["learning rate"])


def test_fit_is_deterministic_and_gives_unit_rows_capped_by_the_corpus():
    pytest.importorskip("sklearn")
    first = TfidfEmbedder(dim=128).fit(DOCS)
    a, b = first.encode(DOCS), TfidfEmbedder(dim=128).fit(DOCS).encode(DOCS)
    assert np.array_equal(a, b)  # TruncatedSVD(random_state=0): the same fit twice
    assert first.dim == len(DOCS) - 1 and a.shape == (len(DOCS), first.dim)  # fewer documents than the requested 128
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_a_query_is_closest_to_the_document_that_shares_its_terms():
    pytest.importorskip("sklearn")
    emb = TfidfEmbedder().fit(DOCS)
    scores = cosine_scores(emb.encode(["sbatch job stereo pairs"])[0], emb.encode(DOCS))
    assert int(np.argmax(scores)) == 3


def test_build_index_defaults_to_the_hash_embedder():
    # The default used to be "tfidf", so a bare call needed scikit-learn; every caller passes an embedder.
    from app.eval.harness import project_root
    from app.eval.index_loader import build_index

    index = build_index(project_root() / "bench" / "corpus" / "files", enable_rerank=False)
    assert isinstance(index.embedder, HashEmbedder)


def test_tfidf_arms_are_cpu_only_parented_on_the_hash_rows_and_labelled_non_neural():
    from app.eval.harness import fallback_labels, frontier_configs
    from app.eval.provenance import model_provenance

    cpu = {c.name: c for c in frontier_configs()}
    assert (cpu["dense_only@tfidf"].parent, cpu["dense_only@tfidf"].mode, cpu["dense_only@tfidf"].embedder) == ("dense_only", "dense", "tfidf")
    assert (cpu["hybrid@tfidf"].parent, cpu["hybrid@tfidf"].mode, cpu["hybrid@tfidf"].embedder) == ("hybrid", "hybrid", "tfidf")
    assert cpu["dense_only"].embedder == cpu["hybrid"].embedder == "hash"  # one switch: the embedder
    assert not [c.name for c in frontier_configs(neural=True) if "tfidf" in c.name]
    assert "tfidf embedder (no model)" in fallback_labels({"settings": {"embedder": "tfidf"}})
    assert model_provenance("tfidf", "embedder")["neural"] is False


def test_a_tfidf_row_records_the_library_versions_it_was_fitted_with():
    pytest.importorskip("sklearn")
    import scipy
    import sklearn

    from app.eval.gold import load_gold
    from app.eval.harness import BenchConfig, project_root, run_config

    root = project_root()
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")[:3]
    cfg = BenchConfig(name="dense_only@tfidf", mode="dense", embedder="tfidf", enable_rerank=False, enable_router=False,
                      staleness_tier1=False, e2e=False, log_triples=False)
    emb = run_config(cfg, gold, root / "bench" / "corpus" / "files")["models"]["embedder"]
    assert emb["neural"] is False and emb["model"] == "tfidf" and emb["dim"] > 0
    assert emb["library"] == {"scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def test_the_frontier_block_names_the_tfidf_fit(tmp_path):
    pytest.importorskip("sklearn")
    import json

    from app.eval.harness import frontier_configs, run_bench
    from app.eval.report import frontier_table

    out = tmp_path / "frontier.json"
    run_bench(configs=[c for c in frontier_configs() if c.name in {"hybrid", "dense_only", "dense_only@tfidf"}], out=out, frontier=True)
    blob = json.loads(out.read_text())
    assert blob["frontier"]["parents"] == ["hybrid", "dense_only"] and blob["frontier"]["arms"] == ["dense_only@tfidf"]
    row = next(r for r in blob["results"] if r["config"] == "dense_only@tfidf")
    assert row["delta_vs_previous"]["reference"] == "dense_only" and "tfidf embedder (no model)" in row["fallbacks"]
    note = next(line for line in frontier_table(blob).splitlines() if line.startswith("- `dense_only@tfidf`"))
    assert "model-free embedder `tfidf`" in note and "scikit-learn" in note
