from app.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k, aggregate_retrieval


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["a", "x"], k=1) == 0.5
    assert recall_at_k(retrieved, ["a", "x"], k=50) == 0.5
    assert recall_at_k(retrieved, ["a", "b"], k=2) == 1.0
    assert recall_at_k([], ["a"], k=10) == 0.0
    assert recall_at_k(["a"], [], k=10) == 0.0


def test_mrr_first_relevant_at_rank_2():
    assert mrr_at_k(["x", "a", "y"], ["a"], k=10) == 0.5
    assert mrr_at_k(["a"], ["a"], k=10) == 1.0
    assert mrr_at_k(["x", "y"], ["a"], k=10) == 0.0


def test_ndcg_perfect_and_swapped():
    perfect = ndcg_at_k(["a", "b", "c"], ["a", "b"], k=10)
    swapped = ndcg_at_k(["c", "a", "b"], ["a", "b"], k=10)
    assert perfect == 1.0
    assert 0 < swapped < perfect


def test_aggregate_by_category():
    rows = [
        {"retrieved": ["a", "b"], "relevant": ["a"], "category": "simple_factual"},
        {"retrieved": ["x"], "relevant": ["a"], "category": "simple_factual"},
        {"retrieved": ["a"], "relevant": ["a"], "category": "staleness"},
    ]
    scores = aggregate_retrieval(rows)
    assert scores.n_queries == 3
    assert scores.by_category["staleness"]["recall@50"] == 1.0
    assert 0 < scores.recall_50 < 1


def test_recall_at_5_and_10_and_unique_paths():
    from app.eval.metrics import score_query, unique_count_at_k

    retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    row = score_query(retrieved, ["a", "k"])
    assert row["recall@5"] == 0.5 and row["recall@10"] == 0.5 and row["recall@50"] == 1.0
    assert row["unique_paths@50"] == 12.0
    assert unique_count_at_k(["a", "a", "b"], 50) == 2


def test_random_baseline_is_list_length_over_corpus_size():
    from app.eval.metrics import random_recall_at_k, score_query

    assert random_recall_at_k(46, 61, 50) == 46 / 61
    assert random_recall_at_k(46, 61, 10) == 10 / 61
    assert random_recall_at_k(3, 61, 10) == 3 / 61
    assert random_recall_at_k(10, 0, 10) == 0.0
    row = score_query(["x"] * 1 + [f"p{i}" for i in range(45)], ["p1"], n_files=61)
    assert abs(row["random_recall@50"] - 46 / 61) < 1e-12
    assert abs(row["random_recall@5"] - 5 / 61) < 1e-12


def test_aggregate_reports_random_baseline_and_recall_at_10():
    rows = [
        {"id": "q1", "retrieved": [f"p{i}" for i in range(46)], "relevant": ["p0"], "category": "simple_factual"},
        {"id": "q2", "retrieved": [f"p{i}" for i in range(36)], "relevant": ["p40"], "category": "staleness"},
    ]
    scores = aggregate_retrieval(rows, n_files=61)
    d = scores.as_dict()
    assert d["n_files"] == 61
    assert abs(d["random_recall@50"] - ((46 + 36) / 2) / 61) < 1e-3
    assert d["mean_unique_paths@50"] == 41.0
    assert d["recall@10"] == 0.5 and d["recall@5"] == 0.5
    assert d["by_category"]["staleness"]["recall@10"] == 0.0 and d["by_category"]["staleness"]["n"] == 1.0
    assert [r["id"] for r in scores.per_query] == ["q1", "q2"]
    assert "random_recall@50" not in aggregate_retrieval(rows).as_dict()
