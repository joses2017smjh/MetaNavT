"""The Python side of the served hybrid statement, without a database.

What VectorStoreManager sends along with app/database/sql/hybrid_*.sql: the
per-statement settings. The live behaviour (a forced HNSW scan returning the full
depth) is checked against Postgres in tests/database/test_dense_parity.py. The
committed plans in doc/sql/ must have been made with the current settings.
"""

import re
from pathlib import Path

import pytest

from app.database.vector_store_manager import HNSW_EF_SEARCH_DEFAULT, HNSW_EF_SEARCH_MAX, VectorStoreManager


def manager(pgvector=(0, 8, 6)):
    """A manager that never connects: only the attributes these methods read."""
    vsm = VectorStoreManager.__new__(VectorStoreManager)
    vsm._pgvector_version = pgvector
    return vsm


class _Cursor:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))


@pytest.mark.parametrize(
    "k, ef_search",
    [(50, 100), (10, HNSW_EF_SEARCH_DEFAULT), (20, HNSW_EF_SEARCH_DEFAULT), (21, 42), (500, HNSW_EF_SEARCH_MAX), (5000, HNSW_EF_SEARCH_MAX)],
)
def test_ef_search_is_twice_the_depth_within_pgvectors_bounds(k, ef_search):
    assert manager().search_settings(k) == {"hnsw.ef_search": ef_search, "hnsw.iterative_scan": "strict_order"}


def test_settings_follow_the_installed_pgvector():
    assert manager((0, 7, 4)).search_settings(50) == {"hnsw.ef_search": 100}  # iterative scans arrived in 0.8.0
    assert manager((0, 4, 4)).search_settings(50) == {}  # no HNSW before 0.5.0
    assert manager(()).search_settings(50) == {}  # extension not installed


def test_settings_are_set_for_the_transaction_only():
    cur = _Cursor()
    applied = manager().apply_search_settings(cur, 50)
    assert applied == {"hnsw.ef_search": 100, "hnsw.iterative_scan": "strict_order"}
    assert cur.executed == [
        ("SELECT set_config(%s, %s, true);", ("hnsw.ef_search", "100")),
        ("SELECT set_config(%s, %s, true);", ("hnsw.iterative_scan", "strict_order")),
    ]


# ---------------------------------------------------------------- committed plans (doc/sql, `make explain-sql*`)

PLANS = Path(__file__).resolve().parents[2] / "doc" / "sql"


def plan_settings(text: str) -> tuple[tuple[int, ...], int, str]:
    """(pgvector version, k, the settings line) from a committed plan's header."""
    version = tuple(int(p) for p in re.search(r"^-- pgvector (\d+)\.(\d+)\.(\d+);", text, re.M).groups())
    k = int(re.search(r"^-- query: .*, k=(\d+),", text, re.M).group(1))
    line = re.search(r"^-- per-statement settings \(VectorStoreManager.search_settings\): (.*)$", text, re.M).group(1)
    return version, k, line


@pytest.mark.parametrize("name", ["hybrid_explain.txt", "hybrid_explain_synthetic_10k.txt"])
def test_committed_plans_ran_with_the_current_statement_and_settings(name):
    text = (PLANS / name).read_text()
    version, k, line = plan_settings(text)
    assert line == str(manager(version).search_settings(k)), f"regenerate doc/sql/{name} (make explain-sql / explain-sql-synthetic)"
    assert "CTE params" not in text  # params is inlined (NOT MATERIALIZED), not a materialized CTE scan


def test_synthetic_plan_reads_the_dense_list_from_the_hnsw_index():
    text = (PLANS / "hybrid_explain_synthetic_10k.txt").read_text()
    rows = int(re.search(r"\((\d+) SYNTHETIC rows", text).group(1))
    assert rows >= 10_000
    assert re.search(r"Index Scan using \S+_embedding_hnsw on ", text)
