"""One retrieval request's time budget: the deadline the route and the retrieving thread share.

Standard library only, so the routes can import it without the LlamaIndex stack
(app/mobile/demo.py serves the same route over an in-memory index).
"""

from __future__ import annotations

import math
import time
from typing import Optional


class StageTimeout(TimeoutError):
    """A stage of one retrieval ran past the request's time budget (POST /api/retrieve/ answers 504)."""

    def __init__(self, stage: str, budget_s: float):
        super().__init__(f"stage {stage} ran past the {budget_s:g} s retrieval budget")
        self.stage = stage
        self.budget_s = budget_s


class Deadline:
    """One request's time budget, shared by the route and the thread that retrieves.

    The retrieval records the stage it is in and checks the budget when a stage
    ends, so a stage that ran over stops the request there and the stages after it
    do not run. The hybrid statement gets the rest of the budget as its
    statement_timeout, so Postgres cancels it. A model call cannot be interrupted:
    the route stops waiting shortly after the budget and names `stage`, and the
    call keeps its worker thread until it returns.
    """

    def __init__(self, budget_s: float = math.inf):
        self.budget_s = float(budget_s)
        self.started = time.monotonic()
        self.stage: Optional[str] = None

    @property
    def bounded(self) -> bool:
        return math.isfinite(self.budget_s)

    def remaining_s(self) -> float:
        return self.budget_s - (time.monotonic() - self.started)

    def check(self, stage: str) -> None:
        """Raise StageTimeout(stage) if the budget is spent (call when `stage` ends)."""
        if self.remaining_s() <= 0:
            raise StageTimeout(stage, self.budget_s)


__all__ = ["Deadline", "StageTimeout"]
