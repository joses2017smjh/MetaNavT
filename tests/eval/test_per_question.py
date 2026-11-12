"""B32: a per-question sidecar next to each harness results file, and every interval recomputed from it exactly.

The results files store means and intervals; the scores behind them were dropped after the bootstrap, so checking an
interval meant re-running the benchmark. app/eval/per_question.py writes them beside the file and recomputes.
"""

import copy
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval import per_question
from app.eval.gold import load_gold
from app.eval.harness import CI_METRICS, DEFAULT_CONFIGS, run_bench
from app.eval.per_question import SUFFIX, differences, sidecar_path

ROOT = Path(__file__).resolve().parents[2]
GOLD = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def test_the_bench_writes_a_sidecar_from_which_every_interval_is_recomputed_exactly(tmp_path, capsys):
    out = tmp_path / "bench.json"
    run_bench(configs=[replace(c) for c in DEFAULT_CONFIGS], out=out)
    stored, sidecar = _load(out), _load(tmp_path / "bench.per_query.json")
    assert sidecar_path(out) == tmp_path / "bench.per_query.json"
    assert sidecar["ids"] == [q.id for q in GOLD] and sidecar["categories"] == [q.category for q in GOLD]
    assert sidecar["metrics"] == list(CI_METRICS) and list(sidecar["scores"]) == [r["config"] for r in stored["results"]]
    assert all(len(col) == len(GOLD) for cols in sidecar["scores"].values() for col in cols.values())
    assert (sidecar["git_sha"], sidecar["timestamp"]) == (stored["git_sha"], stored["timestamp"])
    assert set(sidecar) == {"schema", "about", "git_sha", "timestamp", "metrics", "ids", "categories", "scores"}  # nothing else
    assert differences(stored, sidecar) == []
    assert per_question.main([str(out)]) == 0 and "intervals: OK" in capsys.readouterr().out

    moved = copy.deepcopy(stored)  # one unit in the fourth decimal of one stored bound
    moved["results"][3]["delta_vs_previous"]["ndcg@10"]["hi"] = round(moved["results"][3]["delta_vs_previous"]["ndcg@10"]["hi"] + 0.0001, 4)
    assert [d for d in differences(moved, sidecar) if d.startswith("hybrid+rerank.delta_vs_previous.ndcg@10.hi")]
    changed = copy.deepcopy(sidecar)  # one question's score
    column = changed["scores"]["hybrid"]["ndcg@10"]
    column[0] = 1.0 - column[0]
    found = differences(stored, changed)
    assert any(d.startswith("hybrid.ci.ndcg@10") for d in found) and any(d.startswith("hybrid.retrieval.ndcg@10") for d in found)
    assert "not the same run" in differences(stored, {**sidecar, "timestamp": "another run"})[0]
    out.with_name("bench.per_query.json").unlink()
    assert per_question.main([str(out)]) == 1 and "no sidecar" in capsys.readouterr().out


def test_make_bench_writes_a_sidecar_beside_latest_and_the_sha_file(tmp_path, monkeypatch):
    """Without --out the bench writes bench/results/<sha>.json and latest.json; each gets its sidecar."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    root = tmp_path / "checkout"
    shutil.copytree(ROOT / "bench" / "corpus", root / "bench" / "corpus")
    shutil.copytree(ROOT / "bench" / "gold", root / "bench" / "gold")
    blob = run_bench(root=root, configs=[replace(DEFAULT_CONFIGS[1])])
    results = root / "bench" / "results"
    for name in (f"{blob['git_sha']}.json", "latest.json"):
        assert differences(_load(results / name), _load(sidecar_path(results / name))) == [], name


def test_the_baseline_takes_its_sidecar_and_the_committed_sidecars_are_tracked():
    makefile = (ROOT / "Makefile").read_text().split("bench-baseline:", 1)[1].split("\n\n", 1)[0]
    assert "cp bench/results/latest.json bench/results/main.json" in makefile
    assert "cp bench/results/latest.per_query.json bench/results/main.per_query.json" in makefile
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    for name in ("main", "latest", "neural", "frontier", "frontier_neural", "rerank_depth", "parity_shared_chunker",
                 "parity_shared_chunker_restart", "parity_neural"):  # every tracked harness file's: trackable once negated
        assert subprocess.run(["git", "check-ignore", "-q", f"bench/results/{name}{SUFFIX}"], cwd=ROOT).returncode == 1, name
    assert subprocess.run(["git", "check-ignore", "-q", f"bench/results/check{SUFFIX}"], cwd=ROOT).returncode == 0


def test_the_parity_file_gets_a_sidecar_too(tmp_path, monkeypatch):
    from app.eval import harness, parity
    from app.eval.metrics import aggregate_retrieval

    def fake_run_config(cfg, gold, files_root, n_files=None, **kwargs):
        per_query = [{"id": q.id, "retrieved": q.relevant_ids()[:1], "relevant": q.relevant_ids(), "category": q.category} for q in gold]
        retrieval = aggregate_retrieval(per_query, n_files=n_files)
        return {"config": cfg.name, "settings": {"name": cfg.name}, "retrieval": retrieval.as_dict(), "latency": {},
                "_per_query_scores": retrieval.per_query}

    by_question = {q.question: q for q in GOLD}
    monkeypatch.setattr(parity, "wait_for_health", lambda url: {"status": "ok"})
    monkeypatch.setattr(parity, "run_config", fake_run_config)
    monkeypatch.setattr(parity, "post_retrieve", lambda url, query, k=50, timeout=60.0: {
        "route": "semantic", "hits": [{"path": p} for p in by_question[query].relevant_ids()[::-1]]})
    out = tmp_path / "parity.json"
    harness.main(["--backend", "api", "--url", "http://127.0.0.1:9", "--out", str(out)])
    stored, sidecar = _load(out), _load(sidecar_path(out))
    assert "_per_query_scores" not in json.dumps(stored) and list(sidecar["scores"]) == [parity.IN_MEMORY.name, parity.API_ROW]
    assert differences(stored, sidecar) == []


def _committed_results_with_a_sidecar() -> list[Path]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        return []
    listed = subprocess.run(["git", "ls-files", f"bench/results/*{SUFFIX}"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    return [ROOT / name.replace(SUFFIX, ".json") for name in listed]


@pytest.mark.parametrize("path", _committed_results_with_a_sidecar() or [None], ids=lambda p: p.name if p else "none")
def test_every_committed_sidecar_reproduces_its_results_file(path):
    if path is None:
        pytest.skip("no committed sidecar yet")
    assert differences(_load(path), _load(sidecar_path(path))) == []
