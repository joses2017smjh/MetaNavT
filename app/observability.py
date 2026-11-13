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
- report_failure(log, what, exc): the one place an exception becomes what a
  client may be told. The exception's text (a driver error names the database
  host, port and user; an OS error names paths) goes to the server log under an
  id: the request's (app/api/errors.py gives every request one and sends it as
  X-Request-ID), else a new one (start-up, model loading, offline runs). The
  client gets the class and the id, never the text. Routes turn it into a fixed
  HTTP answer with app.api.errors.failure(). The log line stays one line
  (one_line): a client's text in it cannot start a line of its own.

pip install -e ".[otel]" for the SDK; without it the API behaves the same.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
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


# ---------------------------------------------------------------- request ids and what a client is told of a failure

_request_id: ContextVar[str | None] = ContextVar("metanavit_request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def current_request_id() -> str | None:
    """The id of the request being served (set by app.api.errors.RequestIdMiddleware), None outside a request.
    Worker threads started for the request (run_in_threadpool, anyio.to_thread) see it too."""
    return _request_id.get()


def bind_request_id(value: str) -> Token:
    return _request_id.set(value)


def unbind_request_id(token: Token) -> None:
    _request_id.reset(token)


def public_failure(kind: str, error_id: str | None = None) -> str:
    """What a client is told of a failure: its kind (an exception's class name), never the exception's text."""
    return f"{kind} (the server log has the reason{f', id {error_id}' if error_id else ''})"


def one_line(text: str) -> str:
    """`text` with every character that is not printable (a newline, a carriage return, any other line break or
    control character, a bidirectional control) written as its escape, as repr() writes it; printable text, other
    scripts included, is left as it is."""
    if text.isprintable():
        return text
    return "".join(ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in text)


def report_failure(log: logging.Logger, what: str, exc: BaseException, *, error_id: str | None = None,
                   level: int = logging.ERROR, traceback: bool = False) -> tuple[str, str]:
    """Log that `what` failed, with the exception's full text, under an id: `error_id`, else the current request's,
    else a new one. Returns (the id, what a client may be told: the class and the id).

    The line is written through one_line: `what` and the exception's text can hold a client's own text (a path, a
    plan id from the URL), and a newline or a carriage return in it would otherwise start a log line of the client's
    choosing, with any id. With `traceback`, the traceback that follows the line is Python's, as before."""
    error_id = error_id or current_request_id() or new_request_id()
    log.log(level, one_line(f"{what} failed [id {error_id}]: {type(exc).__name__}: {exc}"),
            exc_info=exc if traceback else None)
    return error_id, public_failure(type(exc).__name__, error_id)
