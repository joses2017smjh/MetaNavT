"""The bench regression gate: metric-only comparison of latest.json vs main.json."""

import json

from app.eval.compare import DEFAULT_THRESHOLD, compare, main, regressions


def _blob(sha, **per_config):
    return {
        "git_sha": sha,
        "timestamp": "ignored",
        "results": [
            {
                "config": name,
                "retrieval": {"recall@50": r50, "ndcg@10": n10, "mrr@10": 0.4},
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
