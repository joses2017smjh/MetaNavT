"""The connection pool's size and wait come from the environment (C52); the live behaviour is tests/database/test_pool.py."""

import pytest

from app.database.db_base_manager import pool_settings


@pytest.mark.parametrize("env, expected", [
    ({}, (1, 10, 30.0)),
    ({"PG_POOL_MIN": "2", "PG_POOL_MAX": "20", "PG_POOL_WAIT_SECONDS": "5"}, (2, 20, 5.0)),
    ({"PG_POOL_MIN": "0", "PG_POOL_MAX": "1", "PG_POOL_WAIT_SECONDS": "0.25"}, (0, 1, 0.25)),
    ({"PG_POOL_MIN": "8", "PG_POOL_MAX": "4"}, (4, 4, 30.0)),  # the minimum is never above the maximum
    ({"PG_POOL_MIN": "-1", "PG_POOL_MAX": "0", "PG_POOL_WAIT_SECONDS": "0"}, (1, 10, 30.0)),
    ({"PG_POOL_MIN": "two", "PG_POOL_MAX": "3.5", "PG_POOL_WAIT_SECONDS": "inf"}, (1, 10, 30.0)),
])
def test_pool_settings_take_valid_values_and_keep_the_defaults_otherwise(monkeypatch, env, expected):
    for name in ("PG_POOL_MIN", "PG_POOL_MAX", "PG_POOL_WAIT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert pool_settings() == expected
