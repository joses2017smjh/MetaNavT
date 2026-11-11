"""Generated hard-case demo stays tied to gold, code, and committed sweeps."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval.demo_export import _checks, _rank
from app.eval.gold import load_gold

ROOT = Path(__file__).resolve().parents[2]


def test_demo_manifest_ids_exist_in_frozen_gold():
    manifest = json.loads((ROOT / "bench" / "demo" / "manifest.json").read_text())
    gold = {question.id for question in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")}
    selected = {
        row["id"]
        for group in ("gold_cases", "controls")
        for row in manifest[group]
    }
    assert selected <= gold
    assert len(manifest["gold_cases"]) == 8
    assert len(manifest["controls"]) == 1
    assert len(manifest["synthetic_cases"]) == 3


def test_rank_and_coverage_checks_are_deterministic():
    paths = ["noise", "gold-a", "gold-b"]
    assert _rank(paths, ["gold-b"]) == 3
    assert _rank(paths, ["missing"]) is None
    checks = _checks(paths, ["gold-a", "gold-b"])
    assert checks[0]["pass"] is True
    assert checks[0]["value"] == "2/2"


def test_generated_demo_has_hard_cases_controls_and_current_artifact_evidence():
    payload = json.loads((ROOT / "doc" / "demo" / "traces.json").read_text())
    cases = {case["id"]: case for case in payload["cases"]}
    assert payload["summary"] == {
        "n_cases": 12,
        "n_gold_cases": 9,
        "n_synthetic_cases": 3,
        "passed": 12,
        "mixed": 0,
    }
    assert {"q115", "q117", "q107", "q108", "q114", "q100", "q120", "q130", "q036"} <= cases.keys()
    assert cases["q107"]["control"]["gold_rank"] > cases["q107"]["method_result"]["gold_rank"]
    assert cases["q108"]["control"]["gold_rank"] > cases["q108"]["method_result"]["gold_rank"]
    hops = {row["config"]: row for row in cases["q036"]["rows"]}
    assert hops["hybrid+hops=0"]["ndcg@10"] > hops["hybrid+hops=1"]["ndcg@10"]

    artifact_paths = {
        citation["path"] for citation in cases["artifact-run47"]["spec"]["citations"]
    }
    assert {
        "configs/run_047.yaml",
        "paper/draft_v2.md",
        "src/fusion.py",
    } <= artifact_paths
    assert not any("archive/" in path or "draft_v1" in path for path in artifact_paths)
    assert cases["artifact-run47"]["execution"]["ok"] is True
    assert cases["matlab-visualization"]["execution"]["ok"] is True
    assert cases["matlab-visualization"]["recommended_chart"] == "dot"
    assert all(check["pass"] for case in cases.values() for check in case["checks"])


def test_html_consumes_generated_trace_script():
    html = (ROOT / "doc" / "demo.html").read_text()
    assert '<script src="demo/traces.js"></script>' in html
    assert 'id="challenge-chips"' in html
    script = (ROOT / "doc" / "demo" / "traces.js").read_text()
    assert script.startswith("window.METANAVIT_DEMO = {")
    for name in (
        "hard-graph.gif",
        "hard-staleness.gif",
        "artifacts.gif",
        "visualization.gif",
    ):
        path = ROOT / "doc" / "gifs" / name
        assert path.is_file()
        assert path.stat().st_size > 20_000


def _demo_inputs():
    from app.eval.demo_export import demo_index

    index, clusters, triples = demo_index(ROOT / "bench" / "corpus" / "files")
    gold = {question.id: question for question in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")}
    specs = {row["id"]: row for row in json.loads((ROOT / "bench" / "demo" / "manifest.json").read_text())["gold_cases"]}
    return index, clusters, triples, gold, specs


def test_demo_export_never_gates_ppr_on_the_gold_category(monkeypatch):
    """The PPR gate is the router's own prediction, as in the benchmark arm (app/eval/harness.py).

    q114 is multi_hop in the gold set, a PPR category, while the router sends it down the semantic route: it is
    the demo case where the two gates disagree. Its case must be the same whatever category the gold set gives it,
    and the gold category must never reach apply_hipporag."""
    import app.eval.demo_export as demo

    index, clusters, triples, gold, specs = _demo_inputs()
    seen = []
    real = demo.apply_hipporag

    def spy(*args, **kwargs):
        seen.append(kwargs.get("category", ""))
        return real(*args, **kwargs)

    monkeypatch.setattr(demo, "apply_hipporag", spy)

    def case(qid, category):
        out = demo._gold_case(index, replace(gold[qid], category=category), specs[qid], clusters, triples)
        assert out.pop("category") == category  # echoed for the reader, used for nothing else
        return out

    assert gold["q114"].category == "multi_hop"
    q114 = [case("q114", category) for category in ("multi_hop", "aggregation", "simple_factual")]
    assert q114[0] == q114[1] == q114[2]
    assert seen and not any(seen)  # no call passed a category
    assert q114[0]["route"] == "semantic" and q114[0]["method_result"]["ppr_applied"] is False
    assert q114[0]["method_result"]["label"] == "current-version filter; PPR gate off"

    q107 = [case("q107", category) for category in ("multi_hop", "simple_factual")]  # the router picks PPR here
    assert q107[0] == q107[1]
    assert q107[0]["route"] == "aggregation_query" and q107[0]["method_result"]["ppr_applied"] is True
    assert q107[0]["method_result"]["label"] == "typed PPR rank fusion"


def test_hard_graph_gif_panels_are_titled_from_the_trace(tmp_path, monkeypatch):
    """The METHOD panel says what ran (the trace's label), so q114's frame does not claim PPR."""
    pytest.importorskip("PIL")
    from PIL import ImageFont

    from app.eval import gifs

    titles = []
    monkeypatch.setattr(gifs, "_font", lambda path, size: ImageFont.load_default())
    monkeypatch.setattr(gifs, "_rank_panel", lambda draw, xy, title, result, color: titles.append(title))
    monkeypatch.setattr(gifs, "_save", lambda path, frames, durations: None)
    payload = json.loads((ROOT / "doc" / "demo" / "traces.json").read_text())
    labels = {"q107": "typed PPR rank fusion", "q108": "typed PPR rank fusion", "q114": "current-version filter; PPR gate off"}
    for case in payload["cases"]:
        if case["id"] in labels:
            case["method_result"]["label"] = labels[case["id"]]
    gifs.gif_hard_graph(tmp_path / "hard-graph.gif", payload)
    assert titles == [title for qid in ("q107", "q108", "q114")
                      for title in ("CONTROL  single-query hybrid", f"METHOD  {labels[qid]}")]
