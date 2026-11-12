"""The bench regression gate: metric-only comparison of latest.json vs main.json."""

import json
from pathlib import Path

from app.eval.compare import DEFAULT_THRESHOLD, compare, exact_differences, main, regressions


def _blob(sha, **per_config):
    return {
        "git_sha": sha,
        "timestamp": "ignored",
        "results": [
            {
                "config": name,
                "retrieval": {"recall@50": r50, "recall@10": r50 - 0.2, "ndcg@10": n10, "mrr@10": 0.4},
                "latency": {"bm25": {"p95_ms": 1.0}},
            }
            for name, (r50, n10) in per_config.items()
        ],
    }


BASE = _blob("aaaaaaa", bm25_only=(0.843, 0.505), hybrid=(0.938, 0.454))


def test_identical_results_pass():
    assert regressions(BASE, BASE) == []


def test_drop_within_threshold_passes():
    cur = _blob("bbbbbbb", bm25_only=(0.843, 0.505 - DEFAULT_THRESHOLD), hybrid=(0.938, 0.454))
    assert regressions(cur, BASE) == []


def test_ndcg_drop_beyond_threshold_fails():
    cur = _blob("bbbbbbb", bm25_only=(0.843, 0.495), hybrid=(0.938, 0.454))
    problems = regressions(cur, BASE)
    assert len(problems) == 1
    assert problems[0].startswith("bm25_only: ndcg@10 dropped")


def test_recall_drop_beyond_threshold_fails():
    cur = _blob("bbbbbbb", bm25_only=(0.843, 0.505), hybrid=(0.900, 0.454))
    assert any("hybrid: recall@50 dropped" in p for p in regressions(cur, BASE))


def test_improvement_never_fails():
    cur = _blob("bbbbbbb", bm25_only=(0.900, 0.600), hybrid=(0.999, 0.600))
    assert regressions(cur, BASE) == []


def test_missing_config_fails_and_new_config_is_ignored():
    cur = _blob("bbbbbbb", bm25_only=(0.843, 0.505), brand_new=(0.1, 0.1))
    problems = regressions(cur, BASE)
    assert problems == ["hybrid: present in baseline but missing from current result"]


def test_recall_at_10_is_gated():
    cur = json.loads(json.dumps(BASE))
    cur["results"][1]["retrieval"]["recall@10"] -= 0.05
    assert any("hybrid: recall@10 dropped" in p for p in regressions(cur, BASE))


def test_metric_missing_from_baseline_is_skipped_with_a_note():
    base = json.loads(json.dumps(BASE))
    for row in base["results"]:
        del row["retrieval"]["recall@10"]
    notes = []
    assert regressions(BASE, base, notes=notes) == []
    assert notes and "recall@10 not in baseline" in notes[0]
    assert "recall@10" in compare(BASE, base)  # reported as — rather than raising


def test_mrr_is_reported_but_not_gated():
    cur = json.loads(json.dumps(BASE))
    cur["results"][0]["retrieval"]["mrr@10"] = 0.0
    assert regressions(cur, BASE) == []
    assert "mrr@10" in compare(cur, BASE)


def test_cli_exit_codes(tmp_path, capsys):
    base_path = tmp_path / "main.json"
    cur_path = tmp_path / "latest.json"
    base_path.write_text(json.dumps(BASE))

    cur_path.write_text(json.dumps(BASE))
    assert main(["--current", str(cur_path), "--baseline", str(base_path), "--fail-on-regression"]) == 0
    assert "gate: OK" in capsys.readouterr().out

    cur_path.write_text(json.dumps(_blob("ccccccc", bm25_only=(0.843, 0.400), hybrid=(0.938, 0.454))))
    assert main(["--current", str(cur_path), "--baseline", str(base_path)]) == 0  # report only
    assert main(["--current", str(cur_path), "--baseline", str(base_path), "--fail-on-regression"]) == 1
    out = capsys.readouterr().out
    assert "gate: REGRESSION" in out and "bm25_only: ndcg@10 dropped" in out

    missing = tmp_path / "nope.json"
    assert main(["--current", str(cur_path), "--baseline", str(missing), "--fail-on-regression"]) == 2
    assert main(["--current", str(cur_path), "--baseline", str(missing)]) == 0
    assert main(["--current", str(missing), "--baseline", str(base_path)]) == 2


# ---- make bench-check: exact reproduction of main.json

MAIN = Path(__file__).resolve().parents[2] / "bench" / "results" / "main.json"


def _main_blob() -> dict:
    return json.loads(MAIN.read_text())


def test_exact_passes_on_identical_results_whatever_the_timings_and_provenance():
    base, cur = _main_blob(), _main_blob()
    for row in cur["results"]:
        row["latency"], row["wall_ms"] = {}, row["wall_ms"] + 1.0
    cur.update(git_sha="0000000", timestamp="later", command="python -m app.eval.harness --out x.json", device={"device": "cuda"})
    assert exact_differences(cur, base) == []


def test_exact_fails_on_one_unit_in_the_last_decimal_anywhere():
    base = _main_blob()
    for mutate, expected in (
        (lambda b: b["results"][2]["retrieval"].__setitem__("ndcg@10", round(b["results"][2]["retrieval"]["ndcg@10"] + 0.0001, 4)), "hybrid.retrieval.ndcg@10"),
        (lambda b: b["results"][0]["retrieval"].__setitem__("random_recall@50", round(b["results"][0]["retrieval"]["random_recall@50"] - 0.0001, 4)), "dense_only.retrieval.random_recall@50"),
        (lambda b: b["results"][5]["ci"]["recall@10"].__setitem__("hi", round(b["results"][5]["ci"]["recall@10"]["hi"] + 0.0001, 4)), "ci.recall@10.hi"),
        (lambda b: b["results"][5]["retrieval"]["by_category"]["staleness"].__setitem__("mrr@10", 0.0), "by_category.staleness.mrr@10"),
        (lambda b: b["bootstrap"].__setitem__("seed", 1), "bootstrap.seed"),
        (lambda b: b["mtimes"].__setitem__("source", "disk"), "mtimes.source"),
        (lambda b: b["results"].pop(), "rows:"),
    ):
        cur = _main_blob()
        mutate(cur)
        problems = exact_differences(cur, base)
        assert problems and any(expected in p for p in problems), (expected, problems)


def test_exact_cli_exit_codes(tmp_path):
    base = _main_blob()
    (tmp_path / "main.json").write_text(json.dumps(base))
    (tmp_path / "same.json").write_text(json.dumps(base))
    base["results"][1]["retrieval"]["mrr@10"] = round(base["results"][1]["retrieval"]["mrr@10"] + 0.0001, 4)
    (tmp_path / "moved.json").write_text(json.dumps(base))
    assert main(["--exact", "--current", str(tmp_path / "same.json"), "--baseline", str(tmp_path / "main.json")]) == 0
    assert main(["--exact", "--current", str(tmp_path / "moved.json"), "--baseline", str(tmp_path / "main.json")]) == 1


# ---- A26: the gate's labels name every gated metric

ROOT = Path(__file__).resolve().parents[2]
METRIC_NAMES = {"ndcg@10": "nDCG@10", "recall@10": "Recall@10", "recall@50": "Recall@50"}  # as the docs write them


def test_the_gate_step_and_the_make_help_name_every_gated_metric():
    """The CI step that runs `make bench-gate` and that target's help line used to name nDCG@10 and Recall@50 only."""
    from app.eval.compare import GATED_METRICS

    names = [METRIC_NAMES[m] for m in GATED_METRICS]  # a newly gated metric needs its name here and in both labels
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text().splitlines()
    (at,) = [i for i, line in enumerate(ci) if line.strip() == "run: make bench-gate"]
    step = ci[at - 1].strip()
    assert step.startswith("- name: ")
    (help_line,) = [line for line in (ROOT / "Makefile").read_text().splitlines() if line.startswith("bench-gate:")]
    for name in names:
        assert name in step, (name, step)
        assert name in help_line.split("##", 1)[1], (name, help_line)
