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

    for text in (BULLETS, QA):
        assert not re.search(r"\b0\.\d{2,}\b", text), "metric-like number typed by hand in prose"
