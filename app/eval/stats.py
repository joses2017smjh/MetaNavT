"""Seeded, paired bootstrap confidence intervals over the gold queries.

One (n_boot x n_queries) index matrix is drawn once per bench run and reused
for every config and every metric, so a delta between two configs is computed
on identical resamples (paired) and every interval in a results file comes
from the same draw. Intervals are percentile intervals (2.5th / 97.5th).

The index matrix comes from numpy's legacy RandomState, whose stream is
guaranteed stable across numpy versions (NEP 19); Generator.integers is not.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_SEED = 0
DEFAULT_N_BOOT = 2000
RNG_NAME = "numpy.random.RandomState.randint"


def resample_indices(n: int, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> np.ndarray:
    """(n_boot, n) matrix of query indices drawn with replacement."""
    if n <= 0:
        raise ValueError("need at least one query to resample")
    return np.random.RandomState(seed).randint(0, n, size=(n_boot, n))


def mean_ci(values: Sequence[float], idx: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi): the sample mean and the percentile bootstrap interval of the mean."""
    v = np.asarray(values, dtype=float)
    if v.shape[0] != idx.shape[1]:
        raise ValueError(f"{v.shape[0]} values but the index matrix has {idx.shape[1]} columns")
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_delta_ci(
    a: Sequence[float], b: Sequence[float], idx: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(delta, lo, hi) for mean(a) - mean(b), resampling the per-query differences."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return mean_ci(diff, idx, alpha=alpha)


def settings(n_queries: int, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """What to record next to the intervals so they can be reproduced."""
    return {
        "method": "percentile bootstrap over queries, one index matrix shared by all configs (paired deltas)",
        "rng": RNG_NAME,
        "seed": seed,
        "n_boot": n_boot,
        "n_queries": n_queries,
        "level": 0.95,
    }
