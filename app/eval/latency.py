"""Per-stage latency logging. Always record p50/p95 for route, embed, search, rerank, generate."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass
class StageStats:
    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples_ms.append(ms)

    def summary(self) -> dict:
        return {
            "n": len(self.samples_ms),
            "p50_ms": round(percentile(self.samples_ms, 50), 2),
            "p95_ms": round(percentile(self.samples_ms, 95), 2),
            "mean_ms": round(
                sum(self.samples_ms) / len(self.samples_ms), 2
            ) if self.samples_ms else 0.0,
        }


class StageTimer:
    """Thread-unsafe timer for a single bench run."""

    def __init__(self) -> None:
        self.stages: dict[str, StageStats] = defaultdict(StageStats)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stages[name].add(elapsed_ms)

    def record(self, name: str, ms: float) -> None:
        self.stages[name].add(ms)

    def summary(self) -> dict[str, dict]:
        return {name: stats.summary() for name, stats in self.stages.items()}
