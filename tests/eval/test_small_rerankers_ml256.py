"""M51 Phase 2 (attempt 2): the students retrained at max_length 256, run beside the attempt-1 students and their inits
at depth 5/10/20 x 256 (the anchors: they must equal M51's committed shapes rows), their checkpoint and manifest
checks, the BEIR ladder, and the F3 frontier with its re-score.

No torch: the cross-encoder is M43's token-overlap stand-in at the row's cap and the embedder the hash embedder, as in
test_small_rerankers_shapes_bench.py. What is tested is the wiring, never a number of a real run.
"""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval import beir, per_question
from app.eval import small_rerankers as sr
from app.eval import small_rerankers_shapes as shapes

sys.path.insert(0, str(Path(__file__).parent))
from test_small_rerankers_bench import _write_dataset  # noqa: E402
from test_small_rerankers_shapes_bench import _Capped  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
P2 = list(shapes.PHASE2["students"])
ROWS = [name for name, _, _, _ in shapes.ml256_rows()]
LOADS: list = []


def _load(tag, cap):
    LOADS.append((tag, cap))
    bias = 0.3 if tag in P2 else (0.4 if "student" in tag else 0.5)
    return _Capped(cap, bias=bias), {"model": tag, "stand_in": True}


@pytest.fixture(scope="module")
def p2_run():
    LOADS.clear()
    blob, sidecar = shapes.run_fixture(ROOT, embedder="hash", load=_load, phase2=True)
    return blob, sidecar, list(LOADS)


def test_the_phase2_rows_are_the_declared_ones_and_each_model_is_loaded_once(p2_run):
    blob, sidecar, loads = p2_run
    assert loads == [(t, 256) for t in (*shapes.ML256_ANCHOR_TAGS, *P2)]
    assert len(ROWS) == 18 and shapes.N_ML256_ANCHOR_ROWS == 13 and len(shapes.ml256_anchor_rows()) == 13
    assert shapes.PHASE2["shapes"] == ((5, 256), (10, 256), (20, 256)) and set(P2) == {"student-minilm-l6-ml256", "student-minilm-l12-ml256"}
    names = [shapes.PARENT, *ROWS]
    assert blob["small_rerankers"]["rows"] == names and blob["small_rerankers"]["phase2"]["attempt"] == 2
    held = {r["config"]: r for r in blob["held_out"]["results"]}
    refs = {name: [d["reference"] for d in row.get("deltas", [])] for name, row in held.items()}
    p2 = shapes.row_name("student-minilm-l6-ml256", 10, 256)
    assert refs[p2] == [shapes.PARENT, shapes.row_name("student-minilm-l6", 10, 256), shapes.row_name("minilm-l6", 10, 256)]
    assert refs[shapes.row_name("student-minilm-l12", 5, 256)] == [shapes.PARENT, shapes.row_name("minilm-l12", 5, 256)]
    assert sr.rescore_fixture(blob, sidecar) == [] and per_question.differences(blob, sidecar) == []
    assert shapes.extra_differences("fixture", blob, sidecar) == []
    assert shapes.invariant_problems("fixture", sidecar["per_query_extra"], sidecar["scores"]) == []
    assert any("Phase 2" in x or "attempt-2" in x for x in blob["does_not_show"])


def test_the_phase2_anchors_compare_thirteen_rows_against_m51s_file(p2_run):
    blob, _, _ = p2_run
    assert blob["anchors"]["file"] == shapes.RESULTS["fixture"]
    for scope in sr.SCOPES:
        assert blob["anchors"]["scopes"][scope]["n_compared"] == 13
    assert blob["anchors"]["all"] is False  # recorded whatever the answer: the hash embedder is not bge-small


def test_m51s_own_rows_fed_back_are_equal_and_a_drift_is_caught():
    committed = json.loads((ROOT / shapes.RESULTS["fixture"]).read_text())
    scopes = {scope: committed[scope] for scope in sr.SCOPES}
    got = shapes.fixture_anchors(ROOT, scopes, shapes.RESULTS["fixture"], shapes.ml256_anchor_rows(), shapes.N_ML256_ANCHOR_ROWS)
    assert got["all"] is True and all(s["n_compared"] == 13 for s in got["scopes"].values())
    row = got["scopes"]["held_out"]["rows"][shapes.row_name("student-minilm-l6", 5, 256)]
    assert row[f"delta_vs {shapes.row_name('minilm-l6', 5, 256)} equal"] is True  # the init delta at the row's own shape
    moved = copy.deepcopy(scopes)
    by = {r["config"]: r for r in moved["held_out"]["results"]}
    by[shapes.row_name("minilm-l12", 10, 256)]["ci"]["ndcg@10"]["hi"] += 0.0001
    assert shapes.fixture_anchors(ROOT, moved, shapes.RESULTS["fixture"], shapes.ml256_anchor_rows(), 13)["all"] is False
    core = shapes.fixture_anchors(ROOT, scopes)  # the default stays the core's: the seven 20/512 rows against M43's file
    assert core["file"] == sr.RESULTS["fixture"] and all(s["n_compared"] == 7 for s in core["scopes"].values()) and core["all"] is True


@pytest.mark.parametrize("dataset", ["scifact", "fiqa"])
def test_m51s_own_beir_rows_fed_back_are_equal(dataset):
    from app.eval.models_2026 import reproduces

    committed = json.loads((ROOT / shapes.RESULTS[dataset]).read_text())
    blob = {"reproduces": [reproduces(ROOT, shapes.RESULTS[dataset], committed["results"], shapes.ml256_anchor_rows(), extra=beir._same_ranking_and_intervals)]}
    got = shapes.beir_anchors(blob, shapes.N_ML256_ANCHOR_ROWS)
    assert got == {"file": shapes.RESULTS[dataset], "n_compared": 13, "metrics_equal": True, "metric_differences": [], "ranking_differences": []}
    assert shapes.beir_anchors(blob)["metrics_equal"] is False  # 13 rows are not the core's 7


def test_the_ml256_ladder_and_its_comparisons():
    configs = beir.small_reranker_ml256_configs()
    assert [c.name for c in configs] == [shapes.PARENT, *ROWS] and configs[0].parent is None
    for c in configs[1:]:
        tag, d, m = shapes.parse_row(c.name)
        assert (c.rerank_top, c.max_length, c.precision, c.parent) == (d, m, "fp32", shapes.PARENT) and m == 256
        assert c.reranker == (tag if tag in sr.STUDENTS or tag in P2 else sr.model_of(tag))
    assert beir.SMALL_RERANKER_ML256_LADDER in beir.LADDERS
    pairs = shapes.comparisons([c.name for c in configs], with_parent=False)
    assert len(pairs) == 6 + 6 * 2  # attempt-1 students vs init; attempt-2 vs attempt 1 and vs init
    assert shapes.comparisons(shapes.anchor_rows() + [shapes.row_name("student-minilm-l6", 5, 256), shapes.row_name("minilm-l6", 5, 256)], with_parent=False) == [
        (shapes.row_name("student-minilm-l6", 20, 512), shapes.row_name("minilm-l6", 20, 512)),
        (shapes.row_name("student-minilm-l12", 20, 512), shapes.row_name("minilm-l12", 20, 512)),
        (shapes.row_name("student-minilm-l6", 5, 256), shapes.row_name("student-minilm-l6", 20, 512)),
        (shapes.row_name("student-minilm-l6", 5, 256), shapes.row_name("minilm-l6", 5, 256)),
        (shapes.row_name("minilm-l6", 5, 256), shapes.row_name("minilm-l6", 20, 512))]  # the core's pairs are unchanged


def _manifest(tmp_path, attempt=2, max_length=256, weights=b"w"):
    entry = {"directory": "student-minilm-l6-ml256", "weights_sha256": hashlib.sha256(b"w").hexdigest(), "attempt": attempt,
             "recipe": {**sr.RECIPE, "max_length": max_length}, "reference_logits": []}
    (tmp_path / "student-minilm-l6-ml256").mkdir(parents=True, exist_ok=True)
    (tmp_path / "student-minilm-l6-ml256" / sr.WEIGHTS_FILE).write_bytes(weights)
    return {"students": {"student-minilm-l6-ml256": entry}}


def test_a_phase2_checkpoint_must_be_attempt_2_at_256_with_its_sha256(tmp_path, monkeypatch):
    monkeypatch.delenv("SMALL_RERANKER_ML256_CHECKPOINTS", raising=False)
    good = _manifest(tmp_path)
    directory, entry = shapes.ml256_checkpoint("student-minilm-l6-ml256", tmp_path, manifest=good)
    assert directory == tmp_path / "student-minilm-l6-ml256" and entry["attempt"] == 2
    with pytest.raises(sr.CheckpointError, match="no checkpoint directory"):
        shapes.ml256_checkpoint("student-minilm-l6-ml256", None, manifest=good)
    with pytest.raises(sr.CheckpointError, match="not Phase 2's"):
        shapes.ml256_checkpoint("student-minilm-l6-ml256", tmp_path, manifest=_manifest(tmp_path, attempt=1))
    with pytest.raises(sr.CheckpointError, match="not Phase 2's"):
        shapes.ml256_checkpoint("student-minilm-l6-ml256", tmp_path, manifest=_manifest(tmp_path, max_length=512))
    with pytest.raises(sr.CheckpointError, match="has sha256"):
        shapes.ml256_checkpoint("student-minilm-l6-ml256", tmp_path, manifest=_manifest(tmp_path, weights=b"other"))
    with pytest.raises(KeyError):
        shapes.ml256_checkpoint("student-minilm-l6", tmp_path, manifest=good)
    if not (ROOT / shapes.PHASE2["manifest"]).exists():
        with pytest.raises(sr.CheckpointError, match="is missing"):
            shapes.ml256_checkpoint("student-minilm-l12-ml256", tmp_path)
    resolve = shapes.resolver(None, ROOT, tmp_path)
    monkeypatch.setattr(shapes, "ml256_checkpoint", lambda tag, d, root=None, manifest=None: (Path(d) / tag, good["students"]["student-minilm-l6-ml256"]))
    source, record, after = resolve("student-minilm-l6-ml256")
    assert record["attempt"] == 2 and record["attempt_1_twin"] == "student-minilm-l6" and record["student_of"] == sr.model_of("minilm-l6")
    assert str(tmp_path) not in json.dumps(record) and callable(after)


# ---------------------------------------------------------------- the F3 frontier


def _beir_p2(tmp_path):
    _write_dataset(tmp_path)
    mp = pytest.MonkeyPatch()
    mp.setattr(beir, "get_cross_encoder", lambda source, max_length=None, **kw: _Capped(max_length, bias=0.3 if "ml256" in str(source) else 0.5))
    configs = [replace(c, embedder="hash") for c in beir.small_reranker_ml256_configs()]
    blob = beir.run(tmp_path, configs, device="cpu", n_boot=50, download=False, check_models=True, keep_per_query=True,
                    comparisons=shapes.comparisons([c.name for c in configs], with_parent=False), resolve_reranker=lambda n: (n, {}, None),
                    per_query_extras=shapes.pair_token_lengths, reproduce=((shapes.RESULTS["scifact"], tuple(shapes.ml256_anchor_rows())),),
                    reproduce_extra=beir._same_ranking_and_intervals, header={"small_rerankers": shapes.header([c.name for c in configs], phase2=True)})
    mp.undo()
    per_query, extra = blob.pop("_per_query"), blob.pop("_per_query_extra")
    shapes.annotate_beir(blob, extra, shapes.N_ML256_ANCHOR_ROWS)
    return blob, beir.per_query_sidecar(blob, per_query, extra)


@pytest.fixture(scope="module")
def p2_root(tmp_path_factory, p2_run):
    """The repository's inputs, M51's committed shapes files (the anchors) and stand-in Phase-2 files."""
    root = tmp_path_factory.mktemp("p2root")
    for rel in (shapes.CPU_FILE, shapes.LIMIT_FILE, shapes.EXPOSURE_FILE, shapes.PHASE2["manifest"] if (ROOT / shapes.PHASE2["manifest"]).exists() else shapes.CPU_FILE,
                *shapes.RESULTS.values(), *(shapes._sidecar_rel(r) for r in shapes.RESULTS.values())):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    if not (root / shapes.PHASE2["manifest"]).exists():
        (root / shapes.PHASE2["manifest"]).parent.mkdir(parents=True, exist_ok=True)
        (root / shapes.PHASE2["manifest"]).write_text(json.dumps({"students": {}}))
    blob, sidecar, _ = p2_run
    (root / shapes.ML256_RESULTS["fixture"]).write_text(json.dumps(blob))
    per_question.write(root / shapes._sidecar_rel(shapes.ML256_RESULTS["fixture"]), sidecar)
    b, s = _beir_p2(tmp_path_factory.mktemp("data"))
    for d in ("scifact", "fiqa"):
        (root / shapes.ML256_RESULTS[d]).write_text(json.dumps({**b, "dataset": f"BEIR/{d}"}))
        per_question.write(root / shapes._sidecar_rel(shapes.ML256_RESULTS[d]), s)
    exposure = json.loads((root / shapes.EXPOSURE_FILE).read_text())
    for d in ("scifact", "fiqa"):
        exposure["datasets"][d]["exposed_query_ids"] = ["q0", "q1", "q2"]  # the stand-in queries
    (root / shapes.EXPOSURE_FILE).write_text(json.dumps(exposure))
    return root


def test_the_f3_frontier_names_its_cells_their_attempt_1_cost_and_the_answer(p2_root):
    blob = shapes.ml256_frontier(p2_root)
    by = {c["config"]: c for c in blob["cells"]}
    assert list(by) == [shapes.row_name(t, d, m) for t in P2 for d, m in shapes.PHASE2["shapes"]]
    f3 = {shapes.row_name(*c) for c in shapes.FAMILIES["F3"]["cells"]}
    assert {n for n, c in by.items() if c["families"] == ["F3"]} == f3 and len(f3) == 3
    cpu = shapes.load_cpu(ROOT)
    for name, c in by.items():
        twin = shapes.PHASE2["students"][c["tag"]]
        assert c["attempt_1_twin"] == shapes.row_name(twin, c["depth"], c["max_length"]) and c["cost_source"] == shapes.PHASE2["cost_source"]
        assert c["indicative_cpu"] == shapes.cpu_cost(twin, c["depth"], c["max_length"], cpu)
        assert (c["band"] == "under-indicative-budget") == (name in f3)
        for d, e in c["datasets"].items():
            assert e["vs_attempt_1"] and e["vs_init"] and e["verdict"] in ("win", "tie", "loss")
            assert ("adjusted" in e) == (name in f3) and (("exposure_excluded" in e) == (name in f3 and d != "fixture"))
    cells = {n: {d: {"ci95": e["vs_parent"], "adjusted": e["adjusted"]["F3"]} for d, e in c["datasets"].items()} for n, c in by.items() if c["families"]}
    assert {k: v for k, v in blob["answers"]["F3"].items() if k in ("answer", "cells")} == {k: v for k, v in shapes.family_answer(cells).items() if k in ("answer", "cells")}
    assert blob["answers"]["F3"]["k"] == 3 and blob["answers"]["F3"]["percentiles"] == [0.8333, 99.1667]
    assert "never merged into F1 or F2" in blob["family_control"] and "headline" not in blob
    assert set(blob["inputs_sha256"]) >= {*shapes.ML256_RESULTS.values(), *shapes.RESULTS.values(), shapes.PHASE2["manifest"]}


def test_the_rescore_covers_phase2_and_catches_a_drift(p2_root):
    (p2_root / shapes.ML256_FRONTIER_FILE).write_text(json.dumps(shapes.ml256_frontier(p2_root)))
    checked, problems = shapes.rescore(p2_root)
    assert checked == 3 + 3 + 1 and problems == [], problems[:5]  # M51's three files (no core frontier here), Phase 2's three and its frontier
    stored = json.loads((p2_root / shapes.ML256_FRONTIER_FILE).read_text())
    stored["timestamp"] = "another time"
    stored["answers"]["F3"]["answer"] = "YES"
    (p2_root / shapes.ML256_FRONTIER_FILE).write_text(json.dumps(stored))
    _, problems = shapes.rescore(p2_root)
    assert problems and all("answers" in p for p in problems)


def test_the_phase2_files_are_whitelisted_and_the_cli_reports_a_missing_frontier(tmp_path, monkeypatch):
    files = [*shapes.ML256_RESULTS.values(), *(shapes._sidecar_rel(r) for r in shapes.ML256_RESULTS.values()), shapes.ML256_FRONTIER_FILE, shapes.PHASE2["manifest"]]
    for rel in files:
        assert subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=ROOT).returncode == 1, f"{rel} is ignored by .gitignore"
    monkeypatch.setattr(shapes, "ROOT", tmp_path)
    assert shapes.main(["--bar", "--phase2"]) == 1
