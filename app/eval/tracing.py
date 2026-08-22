"""No-op OpenTelemetry spans. If opentelemetry is installed, use a real tracer."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


try:
    from opentelemetry import trace  # type: ignore

    _tracer = trace.get_tracer("metanavit.retrieval")
except Exception:  # pragma: no cover
    _tracer = None


@contextmanager
def span(name: str, **attrs) -> Iterator[None]:
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as current:
        for k, v in attrs.items():
            current.set_attribute(k, v)
        yield
