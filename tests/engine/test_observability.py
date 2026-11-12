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


def test_report_failure_writes_one_line_whatever_the_text_holds(caplog):
    """`what` and the exception's text can hold a client's own text (a path, a plan id from the URL). A character
    that is not printable, from a newline or a carriage return to an escape sequence's ESC or a bidirectional control,
    is written as its escape (as repr() writes it), so one failure is one log line and a client cannot start a line of
    its own; printable text, other scripts included, is written as it is. The JSON format stays one valid object."""
    breaks = "\n\r\v\f\x1c\x1d\x1e\x85  "  # every character str.splitlines() breaks on
    others = "\x00\x1b[31m\t‮\udcff"  # NUL, a terminal colour, a tab, a right-to-left override, a lone surrogate
    text = f"calibration α μ{breaks}{others}end"
    escaped = ("calibration α μ\\n\\r\\x0b\\x0c\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"
               "\\x00\\x1b[31m\\t\\u202e\\udcffend")
    logger = logging.getLogger("app.test.report_failure")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        error_id, told = observability.report_failure(logger, f"route {text}", ValueError(text), level=logging.WARNING)
    [record] = [record for record in caplog.records if record.name == logger.name]
    message = record.getMessage()
    assert message == f"route {escaped} failed [id {error_id}]: ValueError: {escaped}"
    assert message.isprintable() and message.splitlines() == [message]
    assert told == f"ValueError (the server log has the reason, id {error_id})"  # the client's answer is unchanged
    line = JsonFormatter().format(record)
    assert line.splitlines() == [line] and json.loads(line)["msg"] == message
