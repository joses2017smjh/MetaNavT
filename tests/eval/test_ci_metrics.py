"""B33: MRR@10 and Recall@5 get bootstrap intervals and paired deltas like nDCG@10, Recall@10 and Recall@50; the
regression gate still gates the three it gated; the README's main and real-model tables print the new intervals
for a file that has them, and a file recorded before keeps its columns."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval import report
from app.eval.compare import GATED_METRICS
from app.eval.harness import CI_METRICS, DEFAULT_CONFIGS, run_bench
from app.eval.per_question import recompute, sidecar_path

ROOT = Path(__file__).resolve().parents[2]
NEW = ("mrr@10", "recall@5")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    out = tmp_path_factory.mktemp("b33") / "bench.json"
    blob = run_bench(configs=[replace(c) for c in DEFAULT_CONFIGS], out=out)
    return blob, json.loads(sidecar_path(out).read_text())


def test_mrr_and_recall_at_5_have_intervals_and_the_gate_is_unchanged(fresh):
    blob, sidecar = fresh
    assert CI_METRICS == ("ndcg@10", "recall@10", "recall@50", "mrr@10", "recall@5")
    assert GATED_METRICS == ("ndcg@10", "recall@10", "recall@50")
    for row in blob["results"]:
        assert set(row["ci"]) == set(CI_METRICS)
        for metric in NEW:
            block = row["ci"][metric]
            assert block["mean"] == row["retrieval"][metric] and block["lo"] <= block["mean"] <= block["hi"], (row["config"], metric)
        if row["delta_vs_previous"]:
            assert set(NEW) <= set(row["delta_vs_previous"]) and set(NEW) <= set(row["delta_vs_bm25_only"] or NEW)


def test_adding_metrics_moved_no_interval_of_the_others(fresh):
    """One resample matrix serves every metric, so the intervals of the three older metrics are those a run with the
    old list gives: recomputed from the same per-question scores with the old metric list, they are identical."""
    blob, sidecar = fresh
    old = ("ndcg@10", "recall@10", "recall@50")
    rows, _ = recompute(blob, {**sidecar, "metrics": list(old)})
    for stored, again in zip(blob["results"], rows):
        assert {m: stored["ci"][m] for m in old} == again["ci"], stored["config"]
        for key in ("delta_vs_bm25_only", "delta_vs_previous"):
            if stored.get(key):
                assert {k: v for k, v in stored[key].items() if k not in NEW} == again[key], (stored["config"], key)


def _without_new_intervals(blob: dict) -> dict:
    old = copy.deepcopy(blob)
    for row in old["results"]:
        for metric in NEW:
            row["ci"].pop(metric, None)
    return old


def test_the_main_table_prints_the_new_intervals_and_an_older_file_keeps_its_columns(fresh):
    blob, _ = fresh
    header = report.table(blob).splitlines()[0]
    assert "Recall@5 [95% CI]" in header and "MRR@10 [95% CI]" in header
    hybrid = next(r for r in blob["results"] if r["config"] == "hybrid")
    line = next(t for t in report.table(blob).splitlines() if t.startswith("| hybrid |"))
    assert report._ci(hybrid, "mrr@10") in line and report._ci(hybrid, "recall@5") in line
    old_header = report.table(_without_new_intervals(blob)).splitlines()[0]
    assert old_header == ("| config | nDCG@10 [95% CI] | Recall@10 [95% CI] | random list @10 | Recall@50 | random list @50 | "
                          "unique files in top 50 | MRR@10 |")


def test_the_real_model_table_prints_them_when_its_file_has_them():
    neural = json.loads((ROOT / report.NEURAL_FILE).read_text())
    has_new = all(set(NEW) <= set(r.get("ci") or {}) for r in neural["results"])
    old = _without_new_intervals(neural)
    assert "MRR@10 [95% CI]" not in report.neural_table(old) and "Recall@5 [95% CI]" not in report.neural_table(old)
    widened = copy.deepcopy(old)
    for row in widened["results"]:
        row["ci"].update({m: {"mean": row["retrieval"][m], "lo": 0.0, "hi": 1.0} for m in NEW})
    text = report.neural_table(widened)
    assert "MRR@10 [95% CI]" in text and "Recall@5 [95% CI]" in text
    assert ("MRR@10 [95% CI]" in report.neural_table(neural)) == has_new
