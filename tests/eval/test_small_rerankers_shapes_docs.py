"""M51's committed files and what the docs say about them: the invariants re-read from the committed sidecars, the
frontier recomputed, the README block, the evidence rows and the chart."""

import json
from pathlib import Path

import pytest

from app.eval import small_rerankers_shapes as shapes

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = all((ROOT / rel).exists() for rel in (*shapes.RESULTS.values(), shapes.FRONTIER_FILE))
needs_files = pytest.mark.skipif(not COMMITTED, reason="the shapes results and the frontier are committed after the code")


@needs_files
@pytest.mark.parametrize("dataset", shapes.DATASETS)
def test_the_committed_sidecars_hold_every_invariant_and_per_row_total(dataset):
    from app.eval import per_question

    path = ROOT / shapes.RESULTS[dataset]
    blob, side = json.loads(path.read_text()), json.loads(per_question.sidecar_path(path).read_text())
    assert shapes.extra_differences(dataset, blob, side) == []
    assert shapes.invariant_problems(dataset, side["per_query_extra"], side["scores"]) == []
    assert blob["working_tree_dirty"] is False and (blob.get("device") or {}).get("device_name") == "NVIDIA GeForce RTX 2080 Ti"
    anchors = blob["anchors"]
    assert anchors["all"] if dataset == "fixture" else anchors["metrics_equal"]
    if dataset != "fixture":  # the cap bites: where a 256 row cut a pair, some query ranks unlike its 512 twin
        for tag in shapes.MODELS:
            mine, twin = side["per_query_extra"][shapes.row_name(tag, 5, 256)], side["per_query_extra"][shapes.row_name(tag, 5, 512)]
            assert sum(mine["truncated_pairs"]) > 0 and mine["rank_hash"]["ranking"] != twin["rank_hash"]["ranking"]


@needs_files
def test_the_committed_frontier_is_recomputed_from_the_committed_files():
    checked, problems = shapes.rescore(ROOT)
    phase2 = [rel for rel in (*shapes.ML256_RESULTS.values(), shapes.ML256_FRONTIER_FILE) if (ROOT / rel).exists()]
    assert checked == 4 + len(phase2) and problems == [], problems[:10]  # the three shapes files, the frontier, and Phase 2's once committed
    front = json.loads((ROOT / shapes.FRONTIER_FILE).read_text())
    assert front["working_tree_dirty"] is False and front["headline"]["family"] == "F1"
    assert [a for a, _ in shapes.ANSWER_LEVELS].count(front["answers"]["F1"]["answer"]) == 1


@needs_files
def test_the_readme_block_states_both_answers_and_what_it_does_not_show():
    from app.eval.report import small_rerankers_shapes_table

    front = json.loads((ROOT / shapes.FRONTIER_FILE).read_text())
    text = small_rerankers_shapes_table(ROOT)
    assert f"**{front['answers']['F1']['answer']}**" in text and f"**{front['answers']['F2']['answer']}**" in text
    assert "not jointly controlled" in text and "not host budgets" in text and "**What this does not show.**" in text
    assert text.count("\n| ") >= 36 + 30 and "Phase 2" in text and "time_reranker_on_host.py" in text
    assert "is missing" in small_rerankers_shapes_table(ROOT / "nowhere")


@needs_files
def test_an_evidence_row_never_renders_without_its_family_answer():
    from app.eval.evidence import _shape_row, _small_reranker_shape_claims

    front = json.loads((ROOT / shapes.FRONTIER_FILE).read_text())
    cell = next(c for c in front["cells"] if "F1" in c["families"])
    meta = {"name": "x", "n": 1, "command": "make x", "source": "y"}
    with pytest.raises(ValueError, match="never stands alone"):
        _shape_row(cell, "scifact", "F1", {}, meta, "")
    row = _shape_row(cell, "scifact", "F1", front["answers"]["F1"], meta, "")
    assert front["answers"]["F1"]["answer"] in row["not_shown"] and "one of 18 cell x dataset tests in F1" in row["claim"]
    rows = _small_reranker_shape_claims(ROOT)
    assert [r["value"] for r in rows[:2]] == [front["answers"]["F1"]["answer"], front["answers"]["F2"]["answer"]]
    for r in rows[2:]:  # nothing for an over-budget cell: every cell row is one of its family's tests
        assert "tests in F1" in r["claim"] or "tests in F2" in r["claim"] or "cost of cheapening" in r["claim"]


@needs_files
def test_the_chart_is_drawn_from_the_frontier(tmp_path):
    from app.eval.figures import write_small_rerankers_shapes_figure

    out = write_small_rerankers_shapes_figure(ROOT, tmp_path)
    svg = out[0].read_text()
    assert svg.startswith("<svg") and "1,000 ms" in svg and "2,500 ms" in svg and svg.count("<polyline") == 3 * len(shapes.MODELS)
    assert (ROOT / "doc" / "figures" / "small-rerankers-shapes.svg").read_text() == svg
