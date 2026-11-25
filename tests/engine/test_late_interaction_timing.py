"""M49: what a ColBERT model would cost the demo's API on a CPU. scripts/late_interaction_timing_payload.py is the
measuring program, scripts/measure_late_interaction_cpu.py runs it in a fresh process per model (indicative only).
Checked without torch: the child process is replaced, and the payload's timing helpers are exercised directly.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


payload = _load("late_interaction_timing_payload")
cpu = _load("measure_late_interaction_cpu")
rerank_payload = _load("rerank_timing_payload")

SHAPES = [{"shape": "query_encode", "p50_ms": 20.0, "p95_ms": 25.0, "n": 30, "memory_mib": {"VmRSS": 900, "VmHWM": 950}},
          {"shape": "rerank_on_the_fly", "depth": 20, "p50_ms": 900.0, "p95_ms": 950.0, "n": 30, "memory_mib": {"VmRSS": 990, "VmHWM": 1010}}]


def test_the_payload_sets_the_single_hosts_settings_as_the_reranker_payload_does():
    theirs = {k: v for k, v in rerank_payload.SETTINGS.items() if not k.startswith("RERANK")}
    assert payload.SETTINGS == theirs  # the reranker's own settings do not apply: no CrossEncoder is loaded
    assert payload.WORDS == rerank_payload.WORDS and payload.PASSAGE_WORDS == rerank_payload.PASSAGE_WORDS
    assert payload.DEPTHS == (5, 10, 20)
    assert payload.query(3) == payload.query(3) and payload.query(3) != payload.query(4)  # seeded, never a benchmark question
    assert all(len(p.split()) == payload.PASSAGE_WORDS for p in payload.passages(1, 5))


def test_the_timing_helper_warms_up_then_times_within_its_budget():
    calls = []
    out = payload.timed(lambda k: calls.append(k), repeats=7, min_repeats=3, budget=1000.0)
    assert calls == [0, 1, 2, 3, 4, 5, 6, 7] and out["n"] == 7  # seed 0 is the warm-up
    assert set(out) >= {"p50_ms", "p95_ms", "min_ms", "max_ms", "warm_up_ms", "memory_mib", "cgroup_peak_mib"}
    assert payload.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_the_cpu_record_is_labelled_indicative_and_hides_cuda_from_the_child_only(tmp_path, monkeypatch):
    from app.eval import late_interaction as li

    runs = []

    def run(env):
        runs.append((env["TIMING_TAG"], env["TIMING_THREADS"], env["CUDA_VISIBLE_DEVICES"]))
        return subprocess.CompletedProcess([], 0, stdout="noise\nRESULT " + json.dumps({"tag": env["TIMING_TAG"], "shapes": SHAPES}) + "\n", stderr="")

    monkeypatch.setattr(cpu, "run_payload", run)
    monkeypatch.setattr(li, "check_revision", lambda tag, revision: None)
    out = tmp_path / "cpu.json"
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert cpu.main(["--out", str(out)]) == 0
    assert runs == [("answerai-colbert-small", "2", ""), ("colbertv2", "2", "")]  # a fresh process per model
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before
    record = json.loads(out.read_text())
    assert "indicative only" in record["label"] and record["device"]["threads"] == 2
    assert [r["tag"] for r in record["results"]] == ["answerai-colbert-small", "colbertv2"]
    assert set(record["source_sha256"]) >= {"scripts/measure_late_interaction_cpu.py", "scripts/late_interaction_timing_payload.py"}


def test_a_failed_child_stops_the_cpu_measurement(tmp_path, monkeypatch):
    from app.eval import late_interaction as li

    monkeypatch.setattr(cpu, "run_payload", lambda env: subprocess.CompletedProcess([], 1, stdout="", stderr="Killed"))
    monkeypatch.setattr(li, "check_revision", lambda tag, revision: None)
    with pytest.raises(SystemExit, match="failed"):
        cpu.main(["--models", "colbertv2", "--out", str(tmp_path / "cpu.json")])
    assert not (tmp_path / "cpu.json").exists()
