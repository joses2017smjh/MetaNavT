"""Spans are no-ops without the SDK / endpoint; JSON logs are one parseable object per line."""

import json
import logging

from app import observability
from app.observability import JsonFormatter, configure_logging, span, tracing_status


def test_span_is_a_no_op_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(observability, "_tracing_state", {"configured": False, "enabled": False, "reason": "not configured"})
    monkeypatch.setattr(observability, "_tracer", None)
    with span("retrieve", mode="sql"):
        pass
    status = tracing_status()
    assert status["enabled"] is False and "OTEL_EXPORTER_OTLP_ENDPOINT" in status["reason"]


def test_json_formatter_emits_one_object_per_record():
    record = logging.LogRecord("app.test", logging.INFO, "x.py", 1, "hello %s", ("world",), None)
    record.stage = "rerank"
    line = JsonFormatter().format(record)
    payload = json.loads(line)
    assert payload["msg"] == "hello world" and payload["level"] == "INFO" and payload["logger"] == "app.test"
    assert payload["stage"] == "rerank" and payload["ts"].endswith("+00:00")


def test_configure_logging_respects_the_env(monkeypatch):
    assert configure_logging(fmt="text") == "text"
    assert configure_logging(fmt="json", level="WARNING") == "json"
    assert isinstance(logging.getLogger("uvicorn").handlers[0].formatter, JsonFormatter)
    assert logging.getLogger("uvicorn").level == logging.WARNING
    # restore uvicorn's default handlers for the rest of the suite
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "app"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
