"""HNSW recall against the exact scan (app/eval/pgvector_hnsw.py): the parts that need no database."""

import pytest

from app.eval import pgvector_hnsw as hnsw


def test_recall_counts_rows_tied_at_the_exact_cut():
    exact = [("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.3), ("e", 0.4)]
    assert hnsw.recall_at([("a", 0.1), ("b", 0.2), ("d", 0.3)], exact, 3) == 1.0  # d ties c at the cut
    assert hnsw.recall_at([("a", 0.1), ("e", 0.4), ("x", 0.5)], exact, 3) == pytest.approx(1 / 3)
    assert hnsw.recall_at([("a", 0.1)], exact, 3) == pytest.approx(1 / 3)  # a short list cannot hold more
    assert hnsw.recall_at([("b", 0.2), ("a", 0.1), ("c", 0.3), ("e", 0.4)], exact, 3) == 1.0  # only the first k rows count
    with pytest.raises(ValueError):
        hnsw.recall_at([], exact[:2], 3)


PLAN_INDEX = """Limit
  ->  Index Scan using t_embedding_hnsw on data_t
        Order By: (embedding <=> '[1,2]'::vector)"""
PLAN_SEQ = """Limit
  ->  Gather Merge
        ->  Sort
              ->  Parallel Seq Scan on data_t"""


def test_check_plan_names_the_scan_and_refuses_the_wrong_one():
    assert hnsw.check_plan(PLAN_INDEX, expect_index="t_embedding_hnsw", table="data_t") == "Index Scan using t_embedding_hnsw"
    assert hnsw.check_plan(PLAN_SEQ, expect_index=None, table="data_t") == "Parallel Seq Scan"
    with pytest.raises(RuntimeError, match="expected an index scan"):
        hnsw.check_plan(PLAN_SEQ, expect_index="t_embedding_hnsw", table="data_t")
    with pytest.raises(RuntimeError, match="expected a sequential scan"):
        hnsw.check_plan(PLAN_INDEX, expect_index=None, table="data_t")
    with pytest.raises(RuntimeError, match="no scan of"):
        hnsw.check_plan(PLAN_INDEX, expect_index="t_embedding_hnsw", table="other")


def test_summary_reads_recall_at_10_from_the_same_list():
    from app.eval.stats import resample_indices

    exact = [[(f"d{i}", i / 100) for i in range(hnsw.K)] for _ in range(2)]
    full = [list(e) for e in exact]
    cut = [e[:40] for e in exact]  # an HNSW scan with ef_search 40 returns 40 rows
    out = hnsw.summarize("x", [full[0], cut[1]], exact, [1.0, 3.0], resample_indices(2, n_boot=50, seed=0), ef_search=40)
    assert out["recall@10"]["mean"] == 1.0 and out["recall@50"]["mean"] == pytest.approx(0.9)
    assert out["rows_returned"] == {"mean": 45.0, "min": 40, "queries_under_k": 1}
    assert out["queries_with_full_recall@50"] == 1 and out["ef_search"] == 40
    assert out["latency"]["p50_ms"] == 2.0


def test_statements_order_by_distance_alone_for_the_index():
    assert "ORDER BY embedding <=> %(qvec)s::vector LIMIT" in hnsw.float32_sql('"public"."t"')
    assert "ORDER BY embedding <=> %(qvec)s::vector, node_id LIMIT" in hnsw.exact_sql('"public"."t"')
    assert hnsw.literal([0.5, -1e-9, 2]) == "[0.5,-1e-09,2]"
