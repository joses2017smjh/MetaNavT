"""Spans and structured logs for the API, both optional at runtime.

- span(name, **attrs): an OpenTelemetry span when the SDK is installed and
  OTEL_EXPORTER_OTLP_ENDPOINT is set (docker compose --profile otel starts
  Jaeger, which accepts OTLP on 4318); a no-op context manager otherwise.
  HybridRetriever wraps route / embed / hybrid_sql / bm25 / vector_search /
  staleness / rerank with it, so a trace shows the same stages StageTimer
  measures.
- configure_logging(): LOG_FORMAT=json turns every record from the app and
  uvicorn loggers into one JSON object per line (ts, level, logger, msg, plus
  any `extra` fields); the default keeps uvicorn's text format.

pip install -e ".[otel]" for the SDK; without it the API behaves the same.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

_tracer = None
_tracing_state = {"configured": False, "enabled": False, "reason": "not configured"}


def _configure_tracer():
    """Build a tracer once. Enabled only with the SDK installed and an OTLP endpoint set."""
    global _tracer
    if _tracing_state["configured"]:
        return _tracer
    _tracing_state["configured"] = True
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        _tracing_state["reason"] = "OTEL_EXPORTER_OTLP_ENDPOINT not set"
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        _tracing_state["reason"] = f"opentelemetry SDK not installed ({exc.name}); pip install -e '.[otel]'"
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "metanavit-api")}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # endpoint from the env var
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("metanavit")
    _tracing_state.update(enabled=True, reason=f"exporting to {endpoint}")
    return _tracer


def tracing_status() -> dict[str, Any]:
    _configure_tracer()
    return dict(_tracing_state)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    tracer = _configure_tracer()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            if value is not None:
                s.set_attribute(key, value)
        yield


class JsonFormatter(logging.Formatter):
    """One JSON object per record; `extra` fields are included as top-level keys."""

    _STANDARD = set(vars(logging.LogRecord("x", 0, "x", 0, "", (), None)).keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(fmt: str | None = None, level: str | None = None) -> str:
    """Apply LOG_FORMAT (json | text) and LOG_LEVEL to the app and uvicorn loggers. Returns the format used."""
    fmt = (fmt or os.getenv("LOG_FORMAT", "text")).strip().lower()
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).strip().upper()
    if fmt != "json":
        return "text"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "app"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.propagate = False
        logger.setLevel(level_name)
    return "json"
