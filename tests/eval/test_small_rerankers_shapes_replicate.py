"""M51's replicate comparison (make small-rerankers-shapes-replicate): the committed files fed in as both sides compare
equal, every kind of difference is counted (a score, a ranking hash, a results-file field, a query or row one side
lacks), a missing replicate file is reported as not compared, and the committed comparison file is re-derived from the
committed results and replicate files."""

import json
import shutil
from pathlib import Path

import pytest

from app.eval import small_rerankers_shapes as shapes

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = all((ROOT / p).exists() for rel in shapes.RESULTS.values() for p in (rel, shapes.replicate_path(rel)))
needs_files = pytest.mark.skipif(not COMMITTED, reason="the shapes results and the replicate files are committed after the code")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


@pytest.fixture
def same_root(tmp_path):
    """Each committed results file and its sidecar as the result AND as the replicate: nothing may differ."""
    for rel in shapes.RESULTS.values():
        for result, replica in ((rel, shapes.replicate_path(rel)), (shapes._sidecar_rel(rel), shapes._sidecar_rel(shapes.replicate_path(rel)))):
            _copy(ROOT / result, tmp_path / result)
            _copy(ROOT / result, tmp_path / replica)
    return tmp_path


def _edit(path: Path, change) -> None:
    blob = json.loads(path.read_text())
    change(blob)
    path.write_text(json.dumps(blob))


def _totals(entry: dict) -> tuple[int, int, int]:
    rows = entry["rows"].values()
    return (sum(1 for r in rows if not r["fields_equal"]), sum(r["per_query_metric_values_differ"] for r in rows),
            sum(r["ranking_hashes_differ"] for r in rows))


@needs_files
def test_the_committed_files_as_both_sides_compare_equal(same_root):
    out = shapes.replicate(same_root)
    for dataset in shapes.DATASETS:
        entry = out["datasets"][dataset]
        assert entry["compared"] and entry["rows_fields_equal"] == len(entry["rows"]) == 37
        assert _totals(entry) == (0, 0, 0)


@needs_files
def test_a_changed_score_hash_or_field_is_counted(same_root):
    row = shapes.row_name("minilm-l6", 5, 256)
    rep = same_root / shapes.replicate_path(shapes.RESULTS["scifact"])
    side = same_root / shapes._sidecar_rel(shapes.replicate_path(shapes.RESULTS["scifact"]))

    def score_and_hash(s):
        s["scores"][row]["ndcg@10"][3] += 0.125
        s["per_query_extra"][row]["rank_hash"]["ranking"][7] = "another ranking"

    _edit(side, score_and_hash)
    _edit(rep, lambda b: next(r for r in b["results"] if r["config"] == row)["retrieval"].update({"ndcg@10": -1.0}))
    entry = shapes.replicate(same_root)["datasets"]["scifact"]
    assert entry["rows"][row] == {"fields_equal": False, "per_query_metric_values_differ": 1, "ranking_hashes_differ": 1}
    assert entry["rows_fields_equal"] == 36 and _totals(entry) == (1, 1, 1)
    assert _totals(shapes.replicate(same_root)["datasets"]["fiqa"]) == (0, 0, 0)

    fixture_rep = same_root / shapes.replicate_path(shapes.RESULTS["fixture"])  # a fixture field differs in one scope only
    _edit(fixture_rep, lambda b: next(r for r in b["tune"]["results"] if r["config"] == row)["ci"].update({"x": 1}))
    assert shapes.replicate(same_root)["datasets"]["fixture"]["rows"][row]["fields_equal"] is False


@needs_files
def test_a_query_or_row_one_side_lacks_counts_as_a_difference(same_root):
    """A zip over the two columns would call a shorter column, a renamed query or a missing row equal."""
    row = shapes.row_name("student-minilm-l12", 10, 512)
    side = same_root / shapes._sidecar_rel(shapes.replicate_path(shapes.RESULTS["fiqa"]))
    n_metrics = len(json.loads(side.read_text())["metrics"])

    _edit(side, lambda s: s["scores"][row]["ndcg@10"].pop())  # one value short
    assert shapes.replicate(same_root)["datasets"]["fiqa"]["rows"][row]["per_query_metric_values_differ"] == 1

    _copy(ROOT / shapes._sidecar_rel(shapes.RESULTS["fiqa"]), side)
    _edit(side, lambda s: s["ids"].__setitem__(0, "a query the result never ran"))
    entry = shapes.replicate(same_root)["datasets"]["fiqa"]
    assert all(r["per_query_metric_values_differ"] == 2 * n_metrics and r["ranking_hashes_differ"] == 2 for r in entry["rows"].values())

    _copy(ROOT / shapes._sidecar_rel(shapes.RESULTS["fiqa"]), side)

    def reorder(s):  # the same queries in another order: matched by id, nothing differs
        order = list(reversed(range(len(s["ids"]))))
        s["ids"] = [s["ids"][i] for i in order]
        for name in s["scores"]:
            s["scores"][name] = {m: [v[i] for i in order] for m, v in s["scores"][name].items()}
            ranking = s["per_query_extra"][name]["rank_hash"]["ranking"]
            s["per_query_extra"][name]["rank_hash"]["ranking"] = [ranking[i] for i in order]

    _edit(side, reorder)
    assert _totals(shapes.replicate(same_root)["datasets"]["fiqa"]) == (0, 0, 0)

    _copy(ROOT / shapes._sidecar_rel(shapes.RESULTS["fiqa"]), side)
    _edit(side, lambda s: (s["scores"].pop(row), s["per_query_extra"].pop(row)))
    gone = shapes.replicate(same_root)["datasets"]["fiqa"]["rows"][row]
    n = len(json.loads(side.read_text())["ids"])
    assert gone == {"fields_equal": False, "per_query_metric_values_differ": n_metrics * n, "ranking_hashes_differ": n}


@needs_files
def test_a_missing_replicate_file_is_not_compared(same_root):
    (same_root / shapes.replicate_path(shapes.RESULTS["scifact"])).unlink()
    out = shapes.replicate(same_root)["datasets"]
    assert out["scifact"] == {"compared": False, "reason": f"{shapes.replicate_path(shapes.RESULTS['scifact'])} is missing"}
    assert out["fixture"]["compared"] and out["fiqa"]["compared"]


@needs_files
def test_the_committed_replicate_file_is_re_derived_from_the_committed_files():
    from app.eval.compare import _leaf_differences

    stored = json.loads((ROOT / shapes.REPLICATE_FILE).read_text())
    again = shapes.replicate(ROOT)
    differences: list[str] = []
    _leaf_differences("replicate", {k: v for k, v in stored.items() if k not in shapes.PROVENANCE_KEYS},
                      {k: v for k, v in again.items() if k not in shapes.PROVENANCE_KEYS}, differences)
    assert differences == [], differences[:10]
    assert stored["working_tree_dirty"] is False and set(stored["inputs_sha256"]) == {
        p for rel in shapes.RESULTS.values() for p in (rel, shapes._sidecar_rel(rel), shapes.replicate_path(rel), shapes._sidecar_rel(shapes.replicate_path(rel)))}
