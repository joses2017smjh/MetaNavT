"""D04: every ladder row carries its paired delta against its parent for every gold category, and each slice says on
how many of its questions nDCG@10 rose and fell. Before, only the frontier run stored all categories, and the
default and neural ladders stored one targeted slice per row (exact_path for the router, staleness for staleness),
with no way to tell how many questions a slice's gain rests on.
"""

import json
from dataclasses import replace
from pathlib import Path

from app.eval.harness import DEFAULT_CONFIGS, attach_confidence, run_bench
from app.eval.per_question import sidecar_path

ROOT = Path(__file__).resolve().parents[2]


def test_the_default_ladder_stores_every_category_and_the_questions_each_slice_rests_on(tmp_path):
    out = tmp_path / "bench.json"
    blob = run_bench(configs=[replace(c) for c in DEFAULT_CONFIGS], out=out)
    rows = {r["config"]: r for r in blob["results"]}
    sidecar = json.loads(sidecar_path(out).read_text())
    categories = sorted(set(sidecar["categories"]))
    router = rows["hybrid+rerank+router"]
    by_category = {b["category"]: b for b in router["delta_vs_parent_by_category_all"]}
    assert sorted(by_category) == categories and by_category["aggregation"]["reference"] == "hybrid+rerank"
    assert {"delta", "lo", "hi"} <= set(by_category["aggregation"]["ndcg@10"])  # the aggregation slice's interval
    for name, row in rows.items():
        parent = row["settings"].get("parent")
        if parent is None:
            assert "delta_vs_parent_by_category_all" not in row  # the first row has nothing to be compared with
            continue
        for block in [*row["delta_vs_parent_by_category_all"], *([row["delta_vs_parent_by_category"]] if row["delta_vs_parent_by_category"] else [])]:
            sel = [i for i, c in enumerate(sidecar["categories"]) if c == block["category"]]
            mine = [sidecar["scores"][name]["ndcg@10"][i] for i in sel]
            theirs = [sidecar["scores"][parent]["ndcg@10"][i] for i in sel]
            assert block["questions_changed"] == {"metric": "ndcg@10", "up": sum(a > b for a, b in zip(mine, theirs)),
                                                  "down": sum(a < b for a, b in zip(mine, theirs))}, (name, block["category"])
    exact = router["delta_vs_parent_by_category"]  # the router's own slice: its gain, and on how many questions
    assert exact["category"] == "exact_path" and exact["questions_changed"] == by_category["exact_path"]["questions_changed"]
    assert exact["questions_changed"]["up"] + exact["questions_changed"]["down"] <= exact["n"]


def test_a_slice_counts_only_questions_whose_score_moved():
    scores = {"a": [{"ndcg@10": v, "recall@10": 0.0, "recall@50": 0.0} for v in (0.5, 0.5, 1.0, 0.0)],
              "b": [{"ndcg@10": v, "recall@10": 0.0, "recall@50": 0.0} for v in (0.5, 0.25, 0.0, 0.0)]}
    rows = [{"config": "b", "settings": {}}, {"config": "a", "settings": {"parent": "b"}}]
    attach_confidence(rows, scores, n_queries=4, categories=["x", "x", "x", "y"], all_categories=True, n_boot=50,
                      metrics=("ndcg@10", "recall@10", "recall@50"))
    blocks = {b["category"]: b for b in rows[1]["delta_vs_parent_by_category_all"]}
    assert blocks["x"]["questions_changed"] == {"metric": "ndcg@10", "up": 2, "down": 0} and "y" not in blocks  # n = 1: no slice
