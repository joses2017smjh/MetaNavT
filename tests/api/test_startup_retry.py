"""A start-up that fails is retried in the background (C38): /health is 503 with the error and the
attempt count, then 200, in the same process. No database: build_retrieval_state and the plan store
are replaced by stubs; tests/database/test_startup.py runs the same path against Postgres (a pending
migration, then the deploy step).
"""

import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

BAD_DB = "postgresql://postgres:password@127.0.0.1:1/metanavit"


class _ReadyVSM:
    hybrid_backend = "ts_rank_cd"
    last_bm25_backend = None
    pgvector_version = (0, 8, 6)

    def count_nodes(self):
        return 61

    def search_settings(self, k):
        return {"hnsw.ef_search": 2 * k}


class _ReadyRetriever:
    _reranker = None
    mode = "sql"
    depth = 50
    reranker_info = {"configured": False, "loaded": False}


class _ReadyState:
    def __init__(self):
        self.vsm, self.retriever, self.index, self.indexing = _ReadyVSM(), _ReadyRetriever(), None, {"indexed": False}


@pytest.fixture
def app_module(monkeypatch):
    """main, with fast retries; the stubs go in before the lifespan runs."""
    for name, value in {"MODEL_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "hash", "PG_CONNECTION_STRING": BAD_DB,
                        "PSYCOPG2_CONNECTION_STRING": "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1",
                        "INDEX_ON_START": "false", "STARTUP_RETRY_SECONDS": "0.05", "STARTUP_RETRY_MAX_SECONDS": "0.1"}.items():
        monkeypatch.setenv(name, value)
    for name in ("FRONTEND_ENDPOINT", "RETRIEVAL_REQUIRE_DEVICE", "STARTUP_RETRY_ATTEMPTS"):
        monkeypatch.delenv(name, raising=False)
    import main  # noqa: WPS433 - the FastAPI app module

    return main


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_a_failed_start_is_retried_and_health_turns_200_without_a_restart(app_module, monkeypatch):
    """The first two attempts fail as a database that is still starting would; the third succeeds once the
    test lets it. Before, the first error was stored for the life of the process and /health stayed 503."""
    import psycopg2

    from app.api.routers import plans
    from app.engine import bootstrap

    builds, settings_built, release = [], [], threading.Event()

    def build_retrieval_state():
        builds.append(1)
        if len(builds) <= 2:
            raise psycopg2.OperationalError("the database system is starting up")
        release.wait(10)
        return _ReadyState()

    real_init_settings = app_module.init_settings
    monkeypatch.setattr(app_module, "init_settings", lambda: settings_built.append(1) or real_init_settings())
    monkeypatch.setattr(bootstrap, "build_retrieval_state", build_retrieval_state)
    monkeypatch.setattr(plans, "build_plan_store", lambda vsm, root=None: ("plan store", vsm))
    with TestClient(app_module.app) as client:
        first = client.get("/health")
        assert first.status_code == 503, first.text
        assert first.json()["status"] == "unavailable"
        # C47: the class and the id the attempt was logged under; the text stays in the server log and in app.state
        assert re.fullmatch(r"OperationalError \(the server log has the reason, id [0-9a-f]{16}\)", first.json()["error"])
        assert client.app.state.startup_error == "OperationalError: the database system is starting up"
        assert first.json()["startup"]["attempts"] >= 1 and first.json()["startup"]["retrying"] is True
        # The third attempt is under way (it waits for the release): /health still says why the last one failed.
        assert _wait_for(lambda: client.app.state.startup["attempts"] == 3)
        waiting = client.get("/health")
        assert waiting.status_code == 503 and waiting.json()["startup"]["attempts"] == 3
        assert waiting.json()["error"] == f"OperationalError (the server log has the reason, id {client.app.state.startup_error_id})"
        assert client.post("/api/retrieve/", json={"query": "q"}).status_code == 503
        release.set()
        assert _wait_for(lambda: client.get("/health").status_code == 200), client.get("/health").text
        # 200 as soon as retrieval exists; the same attempt then builds the plan store
        assert _wait_for(lambda: not client.app.state.startup["retrying"])
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["n_nodes"] == 61
        assert body["startup"] == {"attempts": 3, "retrying": False, "retry_delay_s": None}
        assert client.app.state.startup_error is None and client.app.state.plans[0] == "plan store"
    assert len(builds) == 3 and len(settings_built) == 1  # one process, one lifespan; the settings were built once


def test_a_plan_store_failure_is_retried_and_retrieval_is_built_once(app_module, monkeypatch):
    """Retrieval is served as soon as it is built (as before); the plan store is retried until it is built too."""
    import psycopg2

    from app.api.routers import plans
    from app.engine import bootstrap

    builds, stores = [], []

    def build_plan_store(vsm, root=None):
        stores.append(1)
        if len(stores) == 1:
            raise psycopg2.OperationalError("decision log unreachable")
        return "plan store"

    monkeypatch.setattr(bootstrap, "build_retrieval_state", lambda: builds.append(1) or _ReadyState())
    monkeypatch.setattr(plans, "build_plan_store", build_plan_store)
    with TestClient(app_module.app) as client:
        assert client.get("/health").status_code == 200
        # retrying turns False last, after the plan store is assigned and the error cleared (event-loop thread)
        assert _wait_for(lambda: not client.app.state.startup["retrying"])
        assert client.app.state.plans == "plan store" and client.app.state.startup_error is None
    assert builds == [1] and stores == [1, 1]


def test_one_attempt_means_no_retry_and_the_retry_ends_with_the_process(app_module, monkeypatch):
    """STARTUP_RETRY_ATTEMPTS=1 keeps the old behaviour: one attempt, then 503 until a restart. With retries
    on and a long delay, leaving the lifespan cancels the waiting retry instead of waiting for it."""
    import psycopg2

    from app.engine import bootstrap

    builds = []

    def build_retrieval_state():
        builds.append(1)
        raise psycopg2.OperationalError("the database system is starting up")

    monkeypatch.setattr(bootstrap, "build_retrieval_state", build_retrieval_state)
    monkeypatch.setenv("STARTUP_RETRY_ATTEMPTS", "1")
    with TestClient(app_module.app) as client:
        time.sleep(0.3)  # several retry delays (0.05 s, then 0.1 s)
        body = client.get("/health").json()
        assert body["startup"] == {"attempts": 1, "retrying": False, "retry_delay_s": None}
    assert builds == [1]

    builds.clear()
    monkeypatch.setenv("STARTUP_RETRY_ATTEMPTS", "0")
    monkeypatch.setenv("STARTUP_RETRY_SECONDS", "60")
    monkeypatch.setenv("STARTUP_RETRY_MAX_SECONDS", "120")
    started = time.monotonic()
    with TestClient(app_module.app) as client:
        assert client.get("/health").json()["startup"] == {"attempts": 1, "retrying": True, "retry_delay_s": 60.0}
    assert time.monotonic() - started < 10 and builds == [1]


def test_the_retry_delays_double_up_to_the_cap():
    from itertools import islice

    from app.engine.bootstrap import retry_delays

    assert list(islice(retry_delays(0.5, 4.0), 6)) == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0]
    assert list(islice(retry_delays(10.0, 4.0), 2)) == [4.0, 4.0]  # a cap below the first delay wins


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, (1.0, 30.0, 0)),
        ({"STARTUP_RETRY_SECONDS": "0.25", "STARTUP_RETRY_MAX_SECONDS": "5", "STARTUP_RETRY_ATTEMPTS": "4"}, (0.25, 5.0, 4)),
        ({"STARTUP_RETRY_SECONDS": "abc", "STARTUP_RETRY_MAX_SECONDS": "-1", "STARTUP_RETRY_ATTEMPTS": "x"}, (1.0, 30.0, 0)),
        ({"STARTUP_RETRY_SECONDS": "inf", "STARTUP_RETRY_MAX_SECONDS": "nan", "STARTUP_RETRY_ATTEMPTS": "-3"}, (1.0, 30.0, 0)),
    ],
)
def test_the_retry_settings(monkeypatch, env, expected):
    from app.engine.bootstrap import startup_retry_policy

    for name in ("STARTUP_RETRY_SECONDS", "STARTUP_RETRY_MAX_SECONDS", "STARTUP_RETRY_ATTEMPTS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    policy = startup_retry_policy()
    assert (policy.initial_s, policy.max_s, policy.attempts) == expected
