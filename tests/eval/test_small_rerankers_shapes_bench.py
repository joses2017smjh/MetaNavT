"""M51: the rows at every shape (make bench-small-rerankers-shapes and the BEIR `small-rerankers-shapes` ladder), their
per-query extras, the anchors against M43's committed rows, the invariants, the re-score and the frontier.

No torch: the cross-encoder is M43's token-overlap stand-in loaded at the row's cap, with a stand-in tokenizer; the
embedder is the hash embedder. What is tested is the wiring: one load per (model, cap), which rows exist and what each
is paired with, that a depth or truncation bug is caught per query, and that every number is re-derived from the
sidecars alone.
"""

import copy
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval import beir, per_question
from app.eval import small_rerankers as sr
from app.eval import small_rerankers_shapes as shapes
from app.retrieval.bm25 import tokenize

sys.path.insert(0, str(Path(__file__).parent))
from test_small_rerankers_bench import _Overlap, _write_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ALL = [shapes.row_name(*c) for c in shapes.GRID]


class _Tokenizer:
    """Counts a pair as its words plus three special tokens; records how it was called."""

    calls: list = []

    def __call__(self, queries, passages, **kw):
        _Tokenizer.calls.append(kw)
        return {"input_ids": [[0] * (len(tokenize(q)) + len(tokenize(p)) + 3) for q, p in zip(queries, passages)]}


class _Capped(_Overlap):
    tokenizer = _Tokenizer()

    def __init__(self, cap, bias=0.5):
        super().__init__(bias)
        self.max_seq_length = cap


LOADS: list = []


def _load(tag, cap):
    LOADS.append((tag, cap))
    return _Capped(cap, bias=0.5 if "student" not in tag else 0.4), {"model": tag, "stand_in": True}


@pytest.fixture(scope="module")
def fixture_run():
    LOADS.clear()
    blob, sidecar = shapes.run_fixture(ROOT, embedder="hash", load=_load)
    return blob, sidecar, list(LOADS)


def test_every_shape_is_a_row_and_each_model_cap_is_loaded_once(fixture_run):
    blob, sidecar, loads = fixture_run
    assert loads == [(t, m) for t in shapes.MODELS for m in shapes.MAX_LENGTHS]
    assert list(blob["loads"]) == [f"{t}|{m}" for t, m in loads]
    names = [shapes.PARENT, *ALL]
    assert blob["small_rerankers"]["rows"] == names and blob["small_rerankers"]["served_defaults_changed"] is False
    assert list(blob["rerankers"]) == ALL  # keyed by row name: six shapes of one model never overwrite each other
    for scope, n in (("held_out", 68), ("tune", 68), ("all_questions", 136)):
        rows = {r["config"]: r for r in blob[scope]["results"]}
        assert list(rows) == names and all(r["n"] == n for r in rows.values())
        refs = {name: [d["reference"] for d in row.get("deltas", [])] for name, row in rows.items()}
        assert refs[shapes.PARENT] == []
        assert refs[shapes.row_name("minilm-l6", 20, 512)] == [shapes.PARENT]
        assert refs[shapes.row_name("minilm-l6", 5, 256)] == [shapes.PARENT, shapes.row_name("minilm-l6", 20, 512)]
        assert refs[shapes.row_name("student-minilm-l6", 5, 256)] == [shapes.PARENT, shapes.row_name("student-minilm-l6", 20, 512), shapes.row_name("minilm-l6", 5, 256)]
        assert refs[shapes.row_name("student-minilm-l12", 20, 512)] == [shapes.PARENT, shapes.row_name("minilm-l12", 20, 512)]
    info = blob["rerankers"][shapes.row_name("minilm-l12", 10, 256)]
    assert (info["depth"], info["max_length"], info["load"]) == (10, 256, "minilm-l12|256")
    assert blob["loads"]["minilm-l12|256"]["loaded_max_length"] == 256 and blob["loads"]["minilm-l12|256"]["load_and_score"]["passed"]
    assert "not the server" in info["rerank_latency"]["note"] and info["rerank_latency"]["n"] == 136
    assert sidecar["schema"] == sr.SIDECAR_SCHEMA and set(sidecar["per_query_extra"]) == set(names)
    assert str(ROOT) not in json.dumps(blob) and blob["does_not_show"]


def test_the_token_count_has_padding_and_truncation_off(fixture_run):
    assert _Tokenizer.calls and all(kw == {"add_special_tokens": True, "truncation": False, "padding": False} for kw in _Tokenizer.calls)
    model = _Capped(256)
    assert shapes.pair_token_lengths(model, "a b", ["c d e", "f"]) == [8, 6] and shapes.pair_token_lengths(model, "a", []) == []


def test_the_fixture_file_is_re_derived_from_its_sidecar_and_its_invariants_hold(fixture_run):
    blob, sidecar, _ = fixture_run
    assert sr.rescore_fixture(blob, sidecar) == []
    assert per_question.differences(blob, sidecar) == []  # the generic sidecar check routes it to M43's rescore
    assert shapes.extra_differences("fixture", blob, sidecar) == []
    assert shapes.invariant_problems("fixture", sidecar["per_query_extra"], sidecar["scores"]) == []
    extra = sidecar["per_query_extra"]
    for name in ALL:  # no fixture pair is cut, and at a cap of 256 every question ranks as at 512
        assert sum(extra[name]["truncated_pairs"]) == 0
    assert extra[shapes.row_name("minilm-l6", 5, 256)]["rank_hash"] == extra[shapes.row_name("minilm-l6", 5, 512)]["rank_hash"]
    assert blob["rerankers"][shapes.row_name("minilm-l6", 5, 256)]["ranking_digest"] == blob["rerankers"][shapes.row_name("minilm-l6", 5, 512)]["ranking_digest"]
    moved = copy.deepcopy(blob)
    moved["rerankers"][ALL[0]]["n_top10_differs_from_parent"] += 1
    assert any("n_top10_differs_from_parent" in p for p in shapes.extra_differences("fixture", moved, sidecar))


def test_a_depth_bug_and_a_cap_bug_are_caught_per_query(fixture_run):
    _, sidecar, _ = fixture_run
    extra = copy.deepcopy(sidecar["per_query_extra"])
    name = shapes.row_name("minilm-l6", 5, 256)
    extra[name]["rank_hash"]["top10_set"][3] = "0" * 12  # the tail reordered at depth 5
    extra[shapes.row_name("minilm-l6", 10, 256)]["rank_hash"]["ranking"][0] = "f" * 12  # 256 ranks unlike 512
    extra[shapes.row_name("minilm-l12", 20, 256)]["truncated_pairs"][0] = 2  # a fixture pair cut
    problems = shapes.invariant_problems("fixture", extra, sidecar["scores"])
    assert any(name in p and "top-10 set" in p for p in problems)
    assert any("ranked differently" in p for p in problems) and any("longer than 256 tokens" in p for p in problems)
    scores = copy.deepcopy(sidecar["scores"])
    scores[name]["recall@50"][0] += 0.5
    assert any("recall@50" in p for p in shapes.invariant_problems("fixture", sidecar["per_query_extra"], scores))


def test_on_beir_a_query_with_a_cut_pair_may_rank_differently():
    a, b = shapes.row_name("minilm-l6", 5, 256), shapes.row_name("minilm-l6", 5, 512)
    h = {"top10": ["x", "y"], "top10_set": ["s", "t"], "ranking": ["r1", "r2"]}
    extra = {shapes.PARENT: {"truncated_pairs": None, "rank_hash": h}, b: {"truncated_pairs": [0, 3], "rank_hash": h},
             a: {"truncated_pairs": [0, 1], "rank_hash": {**h, "ranking": ["r1", "other"]}}}
    assert shapes.invariant_problems("scifact", extra) == []
    extra[a]["truncated_pairs"] = [0, 0]  # nothing was cut, so the inputs were identical: the rankings must be too
    assert any("with no pair cut" in p for p in shapes.invariant_problems("scifact", extra))


def test_a_model_whose_cap_does_not_read_back_stops_the_run():
    with pytest.raises(RuntimeError, match="loaded_max_length"):
        shapes.run_fixture(ROOT, tags=["minilm-l6"], embedder="hash", load=lambda tag, cap: (_Capped(512), {}))


def test_rows_of_one_model_and_cap_must_be_next_to_each_other():
    rows = [("a", "minilm-l6", 5, 256), ("b", "minilm-l6", 5, 512), ("c", "minilm-l6", 10, 256)]
    with pytest.raises(ValueError, match="not next to each other"):
        sr.rerank_rows([], [], rows, load=_load)


# ---------------------------------------------------------------- anchors


def _m43_scopes():
    m43 = json.loads((ROOT / sr.RESULTS["fixture"]).read_text())
    return {scope: m43[scope] for scope in sr.SCOPES}


def test_m43s_own_fixture_rows_fed_back_are_equal_in_every_scope():
    got = shapes.fixture_anchors(ROOT, _m43_scopes())
    assert got["all"] is True
    for scope in sr.SCOPES:
        assert got["scopes"][scope]["n_compared"] == shapes.N_ANCHOR_ROWS == 7 and got["scopes"][scope]["all"]
    moved = _m43_scopes()
    moved["tune"]["results"][1]["ci"]["ndcg@10"]["lo"] += 0.0001
    got = shapes.fixture_anchors(ROOT, moved)
    assert got["all"] is False and got["scopes"]["held_out"]["all"] and not got["scopes"]["tune"]["all"]


def test_an_anchor_that_compares_no_row_fails():
    empty = {scope: {"results": []} for scope in sr.SCOPES}
    got = shapes.fixture_anchors(ROOT, empty)
    assert got["all"] is False and all(s["n_compared"] == 0 for s in got["scopes"].values())
    assert shapes.beir_anchors({"reproduces": []})["metrics_equal"] is False
    assert shapes.beir_anchors({"reproduces": [{"rows": {}}]})["metrics_equal"] is False


def test_the_fixture_run_compares_seven_anchor_rows(fixture_run):
    blob, _, _ = fixture_run
    for scope in sr.SCOPES:
        assert blob["anchors"]["scopes"][scope]["n_compared"] == 7
    assert blob["anchors"]["all"] is False  # recorded whatever the answer: the hash embedder is not bge-small


@pytest.mark.parametrize("dataset", ["scifact", "fiqa"])
def test_m43s_own_beir_rows_fed_back_are_equal(dataset):
    committed = json.loads((ROOT / sr.RESULTS[dataset]).read_text())
    from app.eval.models_2026 import reproduces

    blob = {"reproduces": [reproduces(ROOT, sr.RESULTS[dataset], committed["results"], shapes.anchor_rows(), extra=beir._same_ranking_and_intervals)]}
    got = shapes.beir_anchors(blob)
    assert got == {"file": sr.RESULTS[dataset], "n_compared": 7, "metrics_equal": True, "metric_differences": [], "ranking_differences": []}
    rows = copy.deepcopy(committed["results"])
    by = {r["config"]: r for r in rows}
    moved, reordered = shapes.row_name("minilm-l12", 20, 512), shapes.row_name("gte-modernbert-base", 20, 512)
    by[moved]["delta_vs_parent"]["ndcg@10"]["hi"] += 0.0001
    by[reordered]["ranking_sha256"] = "0" * 64  # a ranking-only difference is reported, never a metric difference
    got = shapes.beir_anchors({"reproduces": [reproduces(ROOT, sr.RESULTS[dataset], rows, shapes.anchor_rows(), extra=beir._same_ranking_and_intervals)]})
    assert got["metrics_equal"] is False and got["metric_differences"] == [moved] and got["ranking_differences"] == [reordered]


# ---------------------------------------------------------------- the BEIR ladder


def test_the_beir_ladder_has_every_shape_and_the_anchor_filter_has_seven_rows():
    configs = beir.small_reranker_shape_configs()
    assert [c.name for c in configs] == [shapes.PARENT, *ALL] and configs[0].parent is None
    for c in configs[1:]:
        tag, d, m = shapes.parse_row(c.name)
        assert (c.rerank_top, c.max_length, c.precision, c.parent, c.embedder) == (d, m, "fp32", shapes.PARENT, beir.BGE_SMALL)
        assert c.reranker == (tag if tag in sr.STUDENTS else sr.model_of(tag))
    anchors = beir.small_reranker_shape_configs("anchors")
    assert [c.name for c in anchors] == shapes.anchor_rows() and len(anchors) == 7
    assert beir.small_reranker_shape_configs()[1:][5].name == shapes.row_name("minilm-l6", 20, 512)
    with pytest.raises(ValueError):
        beir.small_reranker_shape_configs("some")
    pairs = shapes.comparisons([c.name for c in configs], with_parent=False)
    assert (shapes.row_name("minilm-l6", 5, 256), shapes.row_name("minilm-l6", 20, 512)) in pairs
    assert (shapes.row_name("student-minilm-l6", 5, 256), shapes.row_name("minilm-l6", 5, 256)) in pairs
    assert not any(shapes.PARENT in p for p in pairs) and len(pairs) == 30 + 12
    assert beir.SMALL_RERANKER_SHAPES_LADDER in beir.LADDERS


def _beir_run(tmp_path, monkeypatch, configs=None):
    _write_dataset(tmp_path)
    monkeypatch.setattr(beir, "get_cross_encoder", lambda source, max_length=None, **kw: _Capped(max_length))
    configs = configs or [replace(c, embedder="hash") for c in beir.small_reranker_shape_configs()]
    blob = beir.run(tmp_path, configs, device="cpu", n_boot=50, download=False, check_models=True, keep_per_query=True,
                    comparisons=shapes.comparisons([c.name for c in configs], with_parent=False), resolve_reranker=lambda n: (n, {}, None),
                    per_query_extras=shapes.pair_token_lengths, reproduce=((sr.RESULTS["scifact"], tuple(shapes.anchor_rows())),),
                    reproduce_extra=beir._same_ranking_and_intervals, header={"small_rerankers": shapes.header([c.name for c in configs])})
    per_query, extra = blob.pop("_per_query"), blob.pop("_per_query_extra")
    shapes.annotate_beir(blob, extra)
    return blob, beir.per_query_sidecar(blob, per_query, extra)


def test_a_beir_shapes_run_keeps_its_extras_outside_the_scores_and_rescores(tmp_path, monkeypatch):
    blob, sidecar = _beir_run(tmp_path, monkeypatch)
    assert set(sidecar["per_query_extra"]) == {shapes.PARENT, *ALL} and set(sidecar["scores"]) == {shapes.PARENT, *ALL}
    assert sidecar["per_query_extra"][shapes.PARENT]["truncated_pairs"] == [None] * 12
    assert sr.rescore_beir(blob, sidecar) == [] and per_question.differences(blob, sidecar) == []
    assert shapes.extra_differences("scifact", blob, sidecar) == []
    assert shapes.invariant_problems("scifact", sidecar["per_query_extra"], sidecar["scores"]) == []
    rows = {r["config"]: r for r in blob["results"]}
    assert rows[shapes.row_name("minilm-l6", 5, 256)]["truncated_pairs"] == {"total": 0, "queries_with_any": 0}
    assert "n_top10_differs_from_parent" in rows[ALL[0]] and rows[ALL[0]]["ranking_digest"]
    assert blob["anchors"]["n_compared"] == 7 and blob["anchors"]["metrics_equal"] is False  # a 12-query toy is not SciFact
    moved = copy.deepcopy(blob)
    moved["results"][3]["ranking_digest"] = "x"
    assert any("ranking_digest" in p for p in shapes.extra_differences("scifact", moved, sidecar))


def test_the_truncation_count_runs_outside_the_timed_rerank(tmp_path, monkeypatch):
    counted = []

    def count(model, query, passages):
        counted.append(len(passages))
        return [300] * len(passages)  # every pair longer than 256 tokens and none longer than 512

    _write_dataset(tmp_path)
    monkeypatch.setattr(beir, "get_cross_encoder", lambda source, max_length=None, **kw: _Capped(max_length))
    configs = [replace(c, embedder="hash") for c in beir.small_reranker_shape_configs() if c.name in (shapes.PARENT, shapes.row_name("minilm-l6", 5, 256), shapes.row_name("minilm-l6", 5, 512))]
    blob = beir.run(tmp_path, configs, device="cpu", n_boot=20, download=False, keep_per_query=True, per_query_extras=count)
    extra = blob["_per_query_extra"]
    assert extra[shapes.row_name("minilm-l6", 5, 256)]["truncated_pairs"] == [5] * 12
    assert extra[shapes.row_name("minilm-l6", 5, 512)]["truncated_pairs"] == [0] * 12
    assert counted == [5] * 24  # the reranked head only, once per query and reranked row (not for the warm-up)
    plain = beir.run(tmp_path, configs, device="cpu", n_boot=20, download=False, keep_per_query=True)
    assert "_per_query_extra" not in plain  # off for every other ladder


# ---------------------------------------------------------------- the frontier and the re-score


@pytest.fixture(scope="module")
def frontier_root(tmp_path_factory, fixture_run):
    """A copy of the repository's inputs with the stand-in fixture file and two stand-in BEIR files in place of the
    shapes files, and an exposure file over the stand-in queries."""
    root = tmp_path_factory.mktemp("root")
    for rel in (shapes.CPU_FILE, shapes.LIMIT_FILE, *sr.RESULTS.values(), *(shapes._sidecar_rel(r) for r in sr.RESULTS.values())):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    blob, sidecar, _ = fixture_run
    (root / shapes.RESULTS["fixture"]).write_text(json.dumps(blob))
    per_question.write(root / shapes._sidecar_rel(shapes.RESULTS["fixture"]), sidecar)
    mp = pytest.MonkeyPatch()
    data = tmp_path_factory.mktemp("data")
    b, s = _beir_run(data, mp)
    mp.undo()
    for d in ("scifact", "fiqa"):
        (root / shapes.RESULTS[d]).write_text(json.dumps({**b, "dataset": f"BEIR/{d}"}))
        per_question.write(root / shapes._sidecar_rel(shapes.RESULTS[d]), s)
    exposure = {"datasets": {d: {"exposed_query_ids": ["q0", "q1", "q2"]} for d in ("scifact", "fiqa")}}
    (root / shapes.EXPOSURE_FILE).write_text(json.dumps(exposure))
    return root


def test_the_frontier_names_every_cell_its_band_and_the_answers(frontier_root):
    blob = shapes.frontier(frontier_root)
    assert [c["config"] for c in blob["cells"]] == ALL
    by = {c["config"]: c for c in blob["cells"]}
    assert {b: len(v) for b, v in blob["bands"].items()} == {"under-indicative-budget": 6, "borderline": 10, "over-budget": 20}
    f1 = by[shapes.row_name("student-minilm-l6", 5, 256)]
    assert f1["band"] == "under-indicative-budget" and f1["families"] == ["F1", "F2"]
    for d, e in f1["datasets"].items():
        assert set(e["adjusted"]) == {"F1", "F2"} and 0 < e["sign_flip_p"] <= 1
        assert e["vs_20_512"] and e["vs_init"] and e["n_queries_ranked_unlike_512"] == 0 or d != "fixture"
        if d != "fixture":
            assert e["exposure_excluded"]["n"] == 9  # 12 stand-in queries, 3 exposed
    border = by[shapes.row_name("minilm-l6", 20, 256)]
    assert border["families"] == ["F2"] and all(set(e["adjusted"]) == {"F2"} and "sign_flip_p" not in e for e in border["datasets"].values())
    over = by[shapes.row_name("bge-rerank", 5, 256)]
    assert over["families"] == [] and all("adjusted" not in e for e in over["datasets"].values())
    assert "vs_20_512" not in by[shapes.row_name("minilm-l6", 20, 512)]["datasets"]["fixture"]
    for f in ("F1", "F2"):
        cells = {c["config"]: {d: {"ci95": e["vs_parent"], "adjusted": e["adjusted"][f]} for d, e in c["datasets"].items()} for c in blob["cells"] if f in c["families"]}
        assert {k: v for k, v in blob["answers"][f].items() if k in ("answer", "cells")} == {k: v for k, v in shapes.family_answer(cells).items() if k in ("answer", "cells")}
    assert blob["headline"]["family"] == "F1" and blob["answers"]["F1"]["k"] == 6 and blob["answers"]["F2"]["k"] == 16
    assert len(blob["host_commands"]) >= 1 and all("--stop-api" in c and "time_reranker_on_host.py" in c for c in blob["host_commands"])
    assert any("--model cross-encoder/ms-marco-MiniLM-L6-v2 --depths 5 --max-length 256" in c for c in blob["host_commands"])  # student-L6 by its init
    assert set(blob["inputs_sha256"]) >= {shapes.CPU_FILE, shapes.LIMIT_FILE, shapes.EXPOSURE_FILE, *shapes.RESULTS.values()}
    assert blob["does_not_show"] and "not jointly controlled" in blob["family_control"]


def test_the_rescore_recomputes_the_frontier_and_catches_a_drift(frontier_root):
    (frontier_root / shapes.FRONTIER_FILE).write_text(json.dumps(shapes.frontier(frontier_root)))
    checked, problems = shapes.rescore(frontier_root)
    assert checked == 4 and problems == [], problems[:5]
    stored = json.loads((frontier_root / shapes.FRONTIER_FILE).read_text())
    stored["timestamp"], stored["git_sha"] = "another time", "other"  # the run-provenance header is not compared
    stored["answers"]["F1"]["answer"] = "YES"
    (frontier_root / shapes.FRONTIER_FILE).write_text(json.dumps(stored))
    _, problems = shapes.rescore(frontier_root)
    assert problems and all("answers" in p for p in problems)


def test_the_sign_flip_p_value():
    assert shapes.sign_flip_p([1.0] * 30, [0.0] * 30) == round(1 / 20001, 5)
    assert shapes.sign_flip_p([0.5, 0.0], [0.5, 0.0]) == 1.0


def test_the_cli_needs_a_step_and_the_files_are_whitelisted():
    import subprocess

    assert shapes.main([]) == 2
    files = [*shapes.RESULTS.values(), shapes.FRONTIER_FILE, shapes.REPLICATE_FILE, shapes.EXPOSURE_FILE]
    files += [shapes._sidecar_rel(r) for r in shapes.RESULTS.values()]
    files += [shapes.replicate_path(r) for r in shapes.RESULTS.values()] + [shapes._sidecar_rel(shapes.replicate_path(r)) for r in shapes.RESULTS.values()]
    for rel in files:
        assert subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=ROOT).returncode == 1, f"{rel} is ignored by .gitignore"
