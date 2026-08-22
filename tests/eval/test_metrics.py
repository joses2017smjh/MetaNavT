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
