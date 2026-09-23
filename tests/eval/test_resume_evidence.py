"""docs/RESUME_EVIDENCE.md must equal `make evidence` output; every claim row carries its provenance."""

from pathlib import Path

from app.eval.evidence import claims, render

ROOT = Path(__file__).resolve().parents[2]


def test_document_is_in_sync():
    assert (ROOT / "docs" / "RESUME_EVIDENCE.md").read_text() == render(), "run `make evidence`"


def test_every_claim_has_command_source_and_limits():
    rows = claims()
    assert len(rows) >= 10
    for r in rows:
        assert r["command"].startswith(("make ", "docker ")) and "bench/results/" in r["source"] and r["not_shown"]


def test_prose_sections_type_no_metric_digits():
    from app.eval.evidence import BULLETS, QA
    import re

    # a metric typed by hand looks like a decimal with two or more places, a signed delta, or scientific notation
    metric = re.compile(r"(?<![\w.])[-+]?\d*\.\d{2,}(?![\w.])|(?<![\w.])\d+(?:\.\d+)?[eE][-+]?\d+(?![\w.])")
    for text in (BULLETS, QA):
        found = metric.findall(text)
        assert not found, f"metric-like numbers typed by hand in prose: {found}"
