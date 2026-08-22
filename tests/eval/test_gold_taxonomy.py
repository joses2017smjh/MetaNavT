from bench.corpus.build import build_gold
from bench.corpus.registry import BASELINE_RMSE, RUNS, lowest_rmse_run


def test_gold_covers_taxonomy_and_size():
    gold = build_gold()
    cats = {q.category for q in gold}
    assert len(gold) >= 100
    for required in (
        "simple_factual",
        "conditional",
        "comparative",
        "aggregation",
        "multi_hop",
        "staleness",
    ):
        assert required in cats
    staleness = [q for q in gold if q.category == "staleness"]
    assert any("configs/run_047.yaml" in q.relevant_paths for q in staleness)
    assert not any("archive" in p for q in staleness for p in q.relevant_paths)


def test_best_run_beats_baseline():
    best = lowest_rmse_run()
    assert best.val_rmse < BASELINE_RMSE
    assert best.run_id == 47
    assert len(RUNS) == 16
