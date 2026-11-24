"""M43's generated outputs: the README block, the chart and the evidence rows are what the committed files give, the
bar's verdicts are read off the intervals, and a row whose dataset is missing is open, never passed."""

import json
import shutil
from pathlib import Path

from app.eval import figures
from app.eval import small_rerankers as sr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = [*sr.RESULTS.values(), "bench/results/small_rerankers_cpu.json"]


def _copy_results(dest: Path) -> None:
    for rel in RESULTS:
        if (ROOT / rel).exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, dest / rel)


def test_the_chart_is_what_make_figures_draws_from_the_committed_results(tmp_path):
    _copy_results(tmp_path)
    written = figures.write_small_rerankers_figure(tmp_path, tmp_path / "out")
    assert [p.name for p in written] == [figures.SMALL_RERANKERS_FIGURE]
    svg = written[0].read_text()
    assert svg == (ROOT / "doc" / "figures" / figures.SMALL_RERANKERS_FIGURE).read_text(), "run `make figures`"
    assert f"](doc/figures/{figures.SMALL_RERANKERS_FIGURE})" in (ROOT / "README.md").read_text()
    cpu = json.loads((ROOT / "bench" / "results" / "small_rerankers_cpu.json").read_text())
    for tag in {r["tag"] for r in cpu["results"]}:  # every timed model is drawn, losers included
        assert f">{tag}<" in svg, tag
    assert "not the demo's host" in svg  # the time axis is labelled indicative


def test_the_chart_draws_nothing_without_its_files(tmp_path):
    assert figures.write_small_rerankers_figure(tmp_path, tmp_path / "out") == []


def test_the_readme_block_keeps_every_row_and_reads_the_bar_off_the_intervals():
    from app.eval.report import small_rerankers_table

    text = small_rerankers_table(ROOT)
    table = sr.bar_table(ROOT)
    assert len(table["datasets"]) == 3
    for row in table["rows"]:
        line = next(line for line in text.splitlines() if line.startswith(f"| {row['tag']} ("))  # ties and losses stay
        above = [c["vs_parent"]["lo"] > 0 for c in row["datasets"].values()]
        assert row["passes"] is all(above)
        assert line.endswith("| **passes** |" if row["passes"] else "| no |")
    assert "**What this does not show.**" in text and "indicative only" in text


def test_a_missing_dataset_leaves_the_verdict_open(tmp_path):
    _copy_results(tmp_path)
    (tmp_path / sr.RESULTS["fiqa"]).unlink()
    table = sr.bar_table(tmp_path)
    by = {r["tag"]: r for r in table["rows"]}
    assert set(table["datasets"]) == {"fixture", "scifact"}
    for row in by.values():
        lost_or_tied = any(not c["above_zero"] for c in row["datasets"].values())
        assert row["passes"] is (False if lost_or_tied else None)  # never True with a dataset unmeasured


def test_evidence_rows_only_where_an_interval_excludes_zero():
    from app.eval.evidence import _small_reranker_claims

    claims = _small_reranker_claims(ROOT)
    assert claims
    for c in claims:
        lo, hi = (float(x) for x in c["ci"].strip("[]").split(","))
        assert lo > 0 or hi < 0, c["claim"]
        assert "measured on the host" in c["not_shown"]
    table = sr.bar_table(ROOT)
    for row in table["rows"]:
        if row["passes"]:  # a passing row has its three rows
            assert sum(f"`{row['tag']}`" in c["claim"] for c in claims) >= 3, row["tag"]


def test_the_generic_sidecar_check_rescores_these_files_and_catches_a_changed_interval():
    """tests/eval/test_per_question.py checks every committed sidecar with per_question.differences: the small-rerankers
    sidecars carry their own schemas, so that check hands them to the same recomputation as make small-rerankers-rescore."""
    from app.eval.per_question import differences, sidecar_path

    for rel in sr.RESULTS.values():
        path = ROOT / rel
        blob, side = json.loads(path.read_text()), json.loads(sidecar_path(path).read_text())
        assert differences(blob, side) == [], rel
        scope = blob["held_out"] if "held_out" in blob else blob
        row = next(r for r in scope["results"] if r.get("deltas") or r.get("delta_vs_parent"))
        block = row["deltas"][0] if row.get("deltas") else row["delta_vs_parent"]
        block["ndcg@10"]["lo"] = round(block["ndcg@10"]["lo"] - 0.01, 4)
        assert differences(blob, side), rel  # a hand-edited interval is caught
    assert differences({"results": []}, {"schema": "something-else"}) == ["sidecar schema 'something-else', expected 'per-question-scores-v1'"]
