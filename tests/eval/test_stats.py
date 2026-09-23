"""Seeded paired bootstrap: deterministic, paired, and honest at the edges."""

import numpy as np

from app.eval.stats import DEFAULT_SEED, mean_ci, paired_delta_ci, resample_indices, settings


def test_resample_indices_are_deterministic_and_in_range():
    a = resample_indices(136, n_boot=50, seed=DEFAULT_SEED)
    b = resample_indices(136, n_boot=50, seed=DEFAULT_SEED)
    assert a.shape == (50, 136)
    assert (a == b).all()
    assert a.min() >= 0 and a.max() < 136
    assert not (a == resample_indices(136, n_boot=50, seed=1)).all()


def test_legacy_stream_is_the_documented_one():
    # RandomState streams are stable across numpy versions (NEP 19); this pins seed 0.
    assert resample_indices(5, n_boot=2, seed=0).tolist() == np.random.RandomState(0).randint(0, 5, size=(2, 5)).tolist()


def test_mean_ci_contains_the_mean_and_collapses_for_constants():
    idx = resample_indices(100, n_boot=500)
    rng = np.random.RandomState(3)
    values = rng.uniform(0, 1, size=100)
    mean, lo, hi = mean_ci(values, idx)
    assert abs(mean - values.mean()) < 1e-12
    assert lo <= mean <= hi
    assert hi - lo < 0.3
    assert mean_ci([0.5] * 100, idx) == (0.5, 0.5, 0.5)


def test_paired_delta_is_zero_for_identical_and_exact_for_constant_shift():
    idx = resample_indices(20, n_boot=200)
    a = list(np.linspace(0, 1, 20))
    assert paired_delta_ci(a, a, idx) == (0.0, 0.0, 0.0)
    b = [x + 0.1 for x in a]
    delta, lo, hi = paired_delta_ci(b, a, idx)
    assert abs(delta - 0.1) < 1e-12 and abs(lo - 0.1) < 1e-9 and abs(hi - 0.1) < 1e-9


def test_length_mismatch_is_an_error():
    idx = resample_indices(10, n_boot=5)
    try:
        mean_ci([1.0] * 9, idx)
    except ValueError as exc:
        assert "index matrix" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_settings_record_what_reproduces_the_intervals():
    s = settings(136)
    assert s["seed"] == DEFAULT_SEED and s["n_queries"] == 136 and s["rng"].endswith("randint") and s["level"] == 0.95


def test_attach_confidence_uses_parent_and_reports_small_n_categories():
    from app.eval.harness import attach_confidence

    n = 12
    cats = ["exact_path"] * 4 + ["staleness"] * 4 + ["semantic"] * 4
    base = [{"ndcg@10": 0.5, "recall@10": 0.5, "recall@50": 0.9} for _ in range(n)]
    better = [{"ndcg@10": 0.5 + (0.2 if c == "exact_path" else 0.0), "recall@10": 0.5, "recall@50": 0.9} for c in cats]
    results = [
        {"config": "bm25_only", "settings": {"parent": None}},
        {"config": "unrelated", "settings": {"parent": "bm25_only"}},
        {"config": "hybrid+rerank+router", "settings": {"parent": "bm25_only"}},
    ]
    scores = {"bm25_only": base, "unrelated": base, "hybrid+rerank+router": better}
    boot = attach_confidence(results, scores, n_queries=n, n_boot=50, categories=cats, comparisons=[("hybrid+rerank+router", "unrelated")])

    router = results[2]
    assert router["delta_vs_previous"]["reference"] == "bm25_only"  # explicit parent, not the list neighbour
    cat = router["delta_vs_parent_by_category"]
    assert cat["category"] == "exact_path" and cat["n"] == 4 and cat["small_n"] is True
    assert abs(cat["ndcg@10"]["delta"] - 0.2) < 1e-9
    assert results[1]["delta_vs_parent_by_category"] is None
    assert boot["comparisons"][0]["row"] == "hybrid+rerank+router" and abs(boot["comparisons"][0]["ndcg@10"]["delta"] - 0.2 / 3) < 1e-4  # stored to 4 dp
