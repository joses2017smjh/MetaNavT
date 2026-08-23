"""Generated hard-case demo stays tied to gold, code, and committed sweeps."""

import json
from pathlib import Path

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
