"""The Python side of the served hybrid statement, without a database.

What VectorStoreManager sends along with app/database/sql/hybrid_*.sql: the
per-statement settings. The live behaviour (a forced HNSW scan returning the full
depth) is checked against Postgres in tests/database/test_dense_parity.py.
"""

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
