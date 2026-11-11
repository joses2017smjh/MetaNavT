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


def test_rrf_sweep_row_reads_the_held_out_delta_and_says_what_happens_to_the_default():
    """B12: one row for the RRF sweep, read from rrf_sweep.json: the tune-half winner minus k = 60 on the held-out half,
    the verdict word from its interval, and that the default stays unless that interval excludes zero in its favour."""
    import json

    from app.eval.evidence import _ci
    from app.eval.report import verdict

    blob = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())
    rows = [c for c in claims() if c["command"] == "make bench-rrf-sweep"]
    assert len(rows) == 1
    row, d = rows[0], blob["decision"]
    assert row["source"] == f"bench/results/rrf_sweep.json @ {blob['git_sha']}"
    assert not d["chosen_is_default"]  # the recorded run; the other branch is tested on a copy below
    block = d["held_out_ndcg@10"]
    assert (row["value"], row["ci"]) == _ci(block, "delta", digits=4) and row["n"] == len(blob["split"]["held_out_ids"])
    assert f"k = {d['chosen']['k']}, dense weight {d['chosen']['weights']['dense']:g}" in row["claim"]
    assert f"verdict from the interval: {verdict(block)}" in row["not_shown"]
    assert ("the default stays k = 60 with equal weights" in row["not_shown"]) == (not d["owner_decision_needed"])


def test_rrf_sweep_row_when_the_tune_half_chose_the_default(tmp_path):
    import json

    from app.eval.evidence import _rrf_sweep_claims

    blob = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())
    blob["decision"]["chosen"] = {"k": 60, "weights": {"bm25": 1.0, "dense": 1.0}}
    blob["decision"]["chosen_is_default"] = True
    blob["decision"]["held_out_ndcg@10"] = None
    blob["decision"]["owner_decision_needed"] = False
    (tmp_path / "bench" / "results").mkdir(parents=True)
    (tmp_path / "bench" / "results" / "rrf_sweep.json").write_text(json.dumps(blob))
    (row,) = _rrf_sweep_claims(tmp_path)
    assert row["ci"] == "—" and row["n"] == len(blob["split"]["tune_ids"]) and "chose the default" in row["claim"]
    assert "nothing to compare on the held-out half" in row["not_shown"]
