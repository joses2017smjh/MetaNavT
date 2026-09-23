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
