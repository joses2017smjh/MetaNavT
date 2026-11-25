"""M51 (app/eval/small_rerankers_shapes.py): what was declared before any small reranker was measured at a cheaper shape.

No torch here. The grid, the budget bands, the families, the answer levels and their order, the replicate and Phase 2
are pinned value by value: they were frozen at the declarations commit, before any job scored a test query at a new
shape, so a change after that has to be made on purpose (and is then a second reading beside the first).
"""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.eval import small_rerankers as sr
from app.eval import small_rerankers_shapes as shapes

ROOT = Path(__file__).resolve().parents[2]
UNDER, BORDER, OVER = shapes.BAND_NAMES


def test_the_grid_is_six_models_by_three_depths_by_two_caps():
    assert shapes.MODELS == ("minilm-l6", "minilm-l12", "student-minilm-l6", "student-minilm-l12", "gte-modernbert-base", "bge-rerank")
    assert shapes.DEPTHS == (5, 10, 20) and shapes.MAX_LENGTHS == (256, 512) and shapes.PRECISION == "fp32"
    assert len(shapes.GRID) == 36 and len(set(shapes.GRID)) == 36
    # a model's rows run next to each other, grouped by cap (one load per (model, cap))
    assert shapes.GRID[:6] == [("minilm-l6", d, m) for m in (256, 512) for d in (5, 10, 20)]
    assert shapes.PARENT == "hybrid@bge-small" and shapes.ANCHOR == (20, 512) and shapes.METRIC == "ndcg@10"


def test_row_names_follow_m43_and_the_served_shape_keeps_m43s_names():
    assert shapes.row_name("minilm-l6", 5, 256) == "hybrid+minilm-l6@bge-small/fp32-256/top5"
    assert shapes.row_name("bge-rerank", 10, 512) == "hybrid+bge-rerank@bge-small/fp32-512/top10"
    for tag in shapes.MODELS:
        assert shapes.row_name(tag, 20, 512) == sr.row_name(tag)
    assert len({shapes.row_name(*c) for c in shapes.GRID}) == 36


def test_every_model_is_pinned():
    assert shapes.PINNED == {"minilm-l6": "233902d25c440f23af6f7d6e94d2946bac0bee0a", "minilm-l12": "7b0235231ca2674cb8ca8f022859a6eba2b1c968",
                             "gte-modernbert-base": "f7481e6055501a30fb19d090657df9ec1f79ab2c", "bge-rerank": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"}
    assert "bge-rerank" not in sr.CANDIDATES  # M43's list stays as it was; M51 pins bge in its own module
    students = json.loads((ROOT / sr.STUDENT_MANIFEST).read_text())["students"]
    for tag, sha in shapes.STUDENT_WEIGHTS_SHA256.items():
        assert students[tag]["weights_sha256"] == sha and students[tag]["attempt"] == 1


# ---------------------------------------------------------------- the budget bands


def test_the_memory_limit_is_read_from_terraform():
    assert shapes.memory_limit_mib(ROOT) == 1024


def test_the_band_membership_is_the_declared_one():
    found = shapes.bands(ROOT)
    assert found[UNDER] == shapes.BAND_MEMBERSHIP[UNDER] and len(found[UNDER]) == 6
    assert sorted(found[BORDER]) == sorted(shapes.BAND_MEMBERSHIP[BORDER]) and len(found[BORDER]) == 10
    assert len(found[OVER]) == 20
    for tag in ("gte-modernbert-base", "bge-rerank"):  # the two references are over budget at every shape
        assert all(c in found[OVER] for c in shapes.GRID if c[0] == tag)
    assert [len(shapes.family_cells(f, ROOT)) for f in ("F1", "F2", "F3")] == [6, 16, 3]
    assert [shapes.FAMILIES[f]["k"] for f in ("F1", "F2", "F3")] == [6, 16, 3]


def test_the_named_edge_cells_are_where_the_brief_says():
    cpu, limit = shapes.load_cpu(ROOT), shapes.memory_limit_mib(ROOT)
    cost = lambda *c: shapes.cpu_cost(*c, cpu)  # noqa: E731
    assert cost("minilm-l6", 10, 256)["p95_ms"] == 899.2 and cost("minilm-l12", 5, 256)["p95_ms"] == 901.0
    assert cost("minilm-l6", 5, 512)["p95_ms"] == 1035.4 and cost("student-minilm-l6", 5, 512)["p95_ms"] == 1027.8
    assert cost("student-minilm-l6", 20, 256)["peak_mib"] == 1019 and shapes.band_of("student-minilm-l6", 20, 256, cpu, limit) == BORDER
    assert cost("minilm-l12", 5, 512)["peak_mib"] == 922 and shapes.band_of("minilm-l12", 5, 512, cpu, limit) == BORDER
    assert cost("student-minilm-l12", 5, 256)["peak_mib"] == 905 and shapes.band_of("student-minilm-l12", 5, 256, cpu, limit) == UNDER
    # the references' cheapest shapes (5/256) are over budget on both axes
    assert (cost("gte-modernbert-base", 5, 256)["p95_ms"], cost("gte-modernbert-base", 5, 256)["peak_mib"]) == (4125.3, 1324)
    assert (cost("bge-rerank", 5, 256)["p95_ms"], cost("bge-rerank", 5, 256)["peak_mib"]) == (10229.8, 2372)
    # no M51 cell lies between 2,039 and 3,485 ms, so none is near the 2,500 ms edge
    p95 = sorted(cost(*c)["p95_ms"] for c in shapes.GRID)
    assert not [x for x in p95 if 2039 < x < 3485]


def _cpu(entries):
    """A CPU file with one entry per (tag, max_length), every entry's config string saying 512 as the real file's do."""
    out: dict = {}
    for t, m, d, p, mem in entries:
        out.setdefault((t, m), {"tag": t, "max_length": m, "config": "hybrid+x@bge-small/fp32-512/top20", "shapes": []})["shapes"].append(
            {"depth": d, "p50_ms": p, "p95_ms": p, "n": 30, "memory_mib": {"VmHWM": mem}})
    return {"results": list(out.values())}


def test_the_band_rule_at_its_edges_and_the_join_ignores_the_config_string():
    blob = _cpu([("a", 256, 5, 1000.0, 921.6), ("a", 256, 10, 1000.1, 900), ("a", 512, 5, 900, 921.7),
                 ("a", 512, 10, 2500.0, 1024), ("a", 512, 20, 2500.1, 900), ("a", 256, 20, 900, 1025)])
    band = lambda d, m: shapes.band_of("a", d, m, blob, 1024)  # noqa: E731
    assert band(5, 256) == UNDER  # both edges inclusive
    assert band(10, 256) == BORDER and band(5, 512) == BORDER and band(10, 512) == BORDER
    assert band(20, 512) == OVER and band(20, 256) == OVER
    with pytest.raises(KeyError):
        shapes.band_of("a", 50, 256, blob, 1024)  # no such depth: never a guess
    with pytest.raises(KeyError):
        shapes.band_of("b", 5, 256, blob, 1024)


# ---------------------------------------------------------------- verdicts, families, answers


def test_the_verdict_reads_the_stored_bounds():
    assert shapes.verdict({"delta": 0.01, "lo": 0.0001, "hi": 0.02}) == "win"
    assert shapes.verdict({"delta": 0.01, "lo": 0.0, "hi": 0.02}) == "tie"  # lo must be above zero
    assert shapes.verdict({"delta": -0.01, "lo": -0.02, "hi": -0.0001}) == "loss"
    assert shapes.verdict({"delta": -0.01, "lo": -0.02, "hi": 0.0}) == "tie"
    assert shapes.verdict(None) is None


def test_the_adjusted_levels():
    assert shapes.ADJUSTED["n_boot"] == 20_000 and shapes.ADJUSTED["seed"] == 0
    for family, (lo, hi) in {"F1": (0.417, 99.583), "F2": (0.156, 99.844), "F3": (0.833, 99.167)}.items():
        got = shapes.adjusted_percentiles(shapes.FAMILIES[family]["k"])
        assert (round(got[0], 3), round(got[1], 3)) == (lo, hi)
    assert shapes.FAMILIES["F1"]["headline"] and not shapes.FAMILIES["F2"]["headline"] and not shapes.FAMILIES["F3"]["headline"]
    assert "not jointly controlled" in shapes.FAMILY_CONTROL


def _cell(ci, adj):
    return {d: {"ci95": {"delta": 0, "lo": a, "hi": b}, "adjusted": {"delta": 0, "lo": c, "hi": e}} for d, (a, b), (c, e) in zip(shapes.DATASETS, ci, adj)}


def test_the_answer_levels_and_their_order():
    assert [a for a, _ in shapes.ANSWER_LEVELS] == ["YES", "PASSES PER CELL, NOT AFTER ADJUSTMENT", "NO", "NOT SHOWN at this n"]
    win, tie, loss = (0.01, 0.05), (-0.01, 0.05), (-0.05, -0.01)
    yes = _cell([win] * 3, [win] * 3)
    passes = _cell([win] * 3, [tie, win, win])
    losing = _cell([win, loss, win], [tie, loss, tie])
    open_ = _cell([tie, win, win], [tie, tie, tie])
    assert shapes.family_answer({"a": yes, "b": losing}) == {"answer": "YES", "cells": ["a"]}
    assert shapes.family_answer({"a": passes, "b": losing}) == {"answer": "PASSES PER CELL, NOT AFTER ADJUSTMENT", "cells": ["a"]}
    got = shapes.family_answer({"a": losing, "b": _cell([loss, tie, tie], [loss, tie, tie])})
    assert got["answer"] == "NO" and got["losses"] == {"a": ["scifact"], "b": ["fixture"]}
    assert shapes.family_answer({"a": losing, "b": open_})["answer"] == "NOT SHOWN at this n"  # one cell shows no loss
    # a cell missing a dataset is neither YES nor a pass
    partial = {d: v for d, v in yes.items() if d != "fiqa"}
    assert shapes.family_answer({"a": partial})["answer"] == "NOT SHOWN at this n"
    assert shapes.family_answer({})["answer"] == "NOT SHOWN at this n"


def test_the_bar_replicate_and_phase2_are_declared():
    assert "ANY shape" in shapes.BAR["question"] and "under the indicative HPC-CPU budget" in shapes.BAR["question"]
    assert "never modifies it" in shapes.BAR["headline"] and "over-budget cell never answers" in shapes.BAR["headline"]
    assert "NOT equivalence" in shapes.BAR["against_the_served_shape"] and "descriptive only" in shapes.BAR["against_the_served_shape"]
    assert shapes.SENSITIVITY["permutation"]["draws"] == 20_000 and shapes.SENSITIVITY["permutation"]["seed"] == 0
    assert "never replaces a result row" in shapes.REPLICATE["role"] and "36 rows" in shapes.REPLICATE["configs"]
    assert "eecs" in shapes.DEVICE["result"] and "rtx2080" in shapes.DEVICE["result"] and "24 h" in shapes.DEVICE["fallback"]
    p2 = shapes.PHASE2
    assert p2["attempt"] == 2 and p2["recipe_delta"] == {"max_length": 256} and p2["family"] == "F3"
    assert p2["students"] == {"student-minilm-l6-ml256": "student-minilm-l6", "student-minilm-l12-ml256": "student-minilm-l12"}
    assert p2["shapes"] == ((5, 256), (10, 256), (20, 256)) and p2["manifest"] == "bench/distill/students_ml256.json"
    assert shapes.FAMILIES["F3"]["cells"] == (("student-minilm-l6-ml256", 5, 256), ("student-minilm-l6-ml256", 10, 256), ("student-minilm-l12-ml256", 5, 256))
    # F3's cells are exactly the Phase-2 twins of the attempt-1 students' under-indicative-budget cells
    twins = [(p2["students"][t], d, m) for t, d, m in shapes.FAMILIES["F3"]["cells"]]
    assert twins == [c for c in shapes.BAND_MEMBERSHIP[UNDER] if c[0] in sr.STUDENTS]
    assert sr.RECIPE["max_length"] == 512 and sr.ATTEMPT == 1  # M43's recipe and attempt stay as they were
    assert shapes.GPU_HOURS_CAP == {"core": 8, "phase2": 7}


def test_the_design_half_widths_are_recomputed_from_m43s_sidecars():
    widths = shapes.design_half_widths(ROOT)
    for dataset in shapes.DATASETS:
        assert widths[dataset]["n"] == shapes.DESIGN_HALF_WIDTHS[dataset]["n"] == shapes.N_QUERIES[dataset]
        for family in ("F1", "F2"):
            assert widths[dataset][family] == shapes.DESIGN_HALF_WIDTHS[dataset][family], (dataset, family)
    assert widths["scifact"]["F1_interval"]["student-minilm-l6"] == {"delta": 0.0118, "lo": -0.0225, "hi": 0.0476}
    assert widths["fixture"]["delta_20_512"]["student-minilm-l6"] == 0.1145
    table = shapes.mde_table(widths)
    assert "| scifact (300) | 0.0350 to 0.0358 | 0.0392 to 0.0405 | +0.0118, +0.0185 |" in table
    assert table.endswith("matches the pinned DESIGN_HALF_WIDTHS: True")


# ---------------------------------------------------------------- the exposure file


def _beir(tmp_path, name, docs, qrels):
    root = tmp_path / "data" / name
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": d, "title": "", "text": "t " + d}) + "\n" for d in docs))
    (root / "queries.jsonl").write_text("".join(json.dumps({"_id": q, "text": "q " + q}) + "\n" for q in qrels))
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t1\n" for q, ds in qrels.items() for d in ds))


def test_the_exposure_counts_on_a_small_tree(tmp_path, monkeypatch):
    from app.eval import beir

    (tmp_path / "bench" / "distill").mkdir(parents=True)
    rows = [{"corpus": "scifact", "doc_id": "d1"}, {"corpus": "scifact", "doc_id": "d2"}, {"corpus": "fiqa", "doc_id": "f9"},
            {"corpus": "fixture", "doc_id": "configs/run_040.yaml::abc"}]
    with gzip.open(tmp_path / shapes.DISTILL_QUERIES, "wt") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "bench" / "gold").mkdir()
    (tmp_path / "bench" / "gold" / "questions.jsonl").write_text((ROOT / "bench" / "gold" / "questions.jsonl").read_text())
    _beir(tmp_path, "scifact", ["d1", "d2", "d3"], {"q1": ["d1", "d3"], "q2": ["d3"], "q3": ["d2"]})
    _beir(tmp_path, "fiqa", ["f1", "f9"], {"a": ["f1"]})
    for name in ("scifact", "fiqa"):
        (tmp_path / "data" / f"{name}.zip").write_bytes(name.encode())
        monkeypatch.setitem(beir.DATASETS, name, {**beir.DATASETS[name], "sha256": hashlib.sha256(name.encode()).hexdigest()})
    got = shapes.exposure(tmp_path, tmp_path / "data")
    assert got["scifact"]["n_exposed_queries"] == 2 and got["scifact"]["exposed_query_ids"] == ["q1", "q3"]
    assert got["scifact"]["n_unexposed_queries"] == 1 and got["scifact"]["relevant_documents"] == {"unique": 3, "unique_in_source": 2, "pairs": 4, "pairs_in_source": 2}
    assert got["fiqa"]["n_exposed_queries"] == 0 and got["fiqa"]["n_source_documents"] == 1
    assert got["fixture"]["n_source_documents"] == 1 and 0 < got["fixture"]["n_exposed_questions"] < 136
    (tmp_path / "data" / "fiqa.zip").write_bytes(b"another archive")
    with pytest.raises(RuntimeError, match="not the pinned"):
        shapes.exposure(tmp_path, tmp_path / "data")


def test_the_committed_exposure_file_says_what_the_brief_computed():
    path = ROOT / shapes.EXPOSURE_FILE
    if not path.exists():
        pytest.skip("the exposure file is generated after the declarations commit")
    d = json.loads(path.read_text())["datasets"]
    assert (d["scifact"]["n_exposed_queries"], d["scifact"]["n_test_queries"], d["scifact"]["n_source_documents"]) == (78, 300, 1500)
    assert (d["scifact"]["relevant_documents"]["unique_in_source"], d["scifact"]["relevant_documents"]["unique"]) == (76, 283)
    assert (d["fiqa"]["n_exposed_queries"], d["fiqa"]["n_test_queries"], d["fiqa"]["n_source_documents"]) == (52, 648, 1800)
    assert (d["fiqa"]["relevant_documents"]["unique_in_source"], d["fiqa"]["relevant_documents"]["unique"]) == (56, 1706)
    assert d["fixture"]["n_exposed_questions"] == 136 and d["fixture"]["n_source_documents"] == 71
    assert len(d["scifact"]["exposed_query_ids"]) == 78 and len(d["fiqa"]["exposed_query_ids"]) == 52


def test_the_m51_results_files_are_not_ignored_by_git():
    # --no-index: check-ignore never reports a tracked file, so without it this could not fail once they are committed
    for rel in (shapes.EXPOSURE_FILE,):
        assert subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=ROOT).returncode == 1, f"{rel} is ignored by .gitignore"


def test_the_cli_needs_a_step():
    assert shapes.main([]) == 2


# ---------------------------------------------------------------- M43's fixture run, unchanged by M51's refactor

VOLATILE = ("git_sha", "git_head", "working_tree_dirty", "source_sha256", "timestamp", "command", "device")


def test_m43s_fixture_run_is_byte_for_byte_what_it_was_before_m51():
    """Digests of M43's run_fixture output (hash embedder, the overlap stand-in, run-time fields removed), captured at
    7a6aa83 before M51 touched app/eval/small_rerankers.py: the refactor must not change a field."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_small_rerankers_bench import TAGS, _Overlap

    blob, side = sr.run_fixture(ROOT, tags=TAGS, embedder="hash", load=lambda tag: (_Overlap(), {"model": tag, "stand_in": True}))
    blob = {k: v for k, v in blob.items() if k not in VOLATILE}
    for info in blob["rerankers"].values():
        info.pop("rerank_latency")
        info.pop("device", None)
    side = {k: v for k, v in side.items() if k not in ("git_sha", "timestamp")}
    digest = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()  # noqa: E731
    golden = json.loads((Path(__file__).parent / "fixtures" / "m43_run_fixture_standin.sha256.json").read_text())
    assert {k: digest(v) for k, v in blob.items()} == golden["blob"]
    assert {k: digest(v) for k, v in side.items()} == golden["sidecar"]
