"""API key authentication and the per-replica request budget (app/api/security.py).

Open by default; 401 without the key once one is configured; 200 with it; 429 past
the budget with Retry-After; /health always reachable and honest about both.
No database: same closed-port setup as tests/api/test_plans.py, so /health is 503
here for the backend's sake, which is still "reachable" (not 401, not 429).
"""

import os
import re

import pytest
from fastapi.testclient import TestClient

from app.api import security
from app.api.routers.plans import PlanStore
from app.mcp.filesystem import FilesystemTools

KEY = "test-key-7f3a9c-do-not-log"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://postgres:password@127.0.0.1:1/metanavit")
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1")
    monkeypatch.setenv("INDEX_ON_START", "false")
    monkeypatch.delenv("FRONTEND_ENDPOINT", raising=False)
    for name in (security.API_KEY_ENV, security.RATE_ENV, security.BURST_ENV):
        monkeypatch.delenv(name, raising=False)
    security.reset_limiter()
    import main

    from app.database import vector_store

    vector_store.get_vector_store_manager.cache_clear()
    with TestClient(main.app) as c:
        c.app.state.plans = PlanStore(FilesystemTools(root=tmp_path), None)
        yield c
        c.app.state.plans = None
    security.reset_limiter()


def _api_calls(client):
    """(method, concrete path) for every route under /api: the operations in the OpenAPI
    schema (FastAPI's own list of what it serves) plus the static mounts."""
    calls = []
    for path, operations in client.get("/openapi.json").json()["paths"].items():
        if path.startswith("/api"):
            concrete = re.sub(r"\{[^}]+\}", "x", path)
            calls.extend((method.upper(), concrete) for method in sorted(operations))
    for route in client.app.routes:
        path = getattr(route, "path", None) or ""
        if path.startswith("/api") and getattr(route, "methods", None) is None:  # a mount
            calls.append(("GET", path.rstrip("/") + "/anything.txt"))
    return calls


# ---------------------------------------------------------------- open by default


def test_open_by_default_and_health_says_so(client):
    assert client.get("/api/plans/").status_code == 200
    assert client.post("/api/retrieve/", json={"query": "learning rate"}).status_code == 503  # backend down, not 401
    body = client.get("/health").json()
    assert body["auth"]["enabled"] is False
    assert body["rate_limit"]["enabled"] is False and body["rate_limit"]["scope"].startswith("per replica")


# ---------------------------------------------------------------- 401 without the key, 200 with it


def test_401_without_the_key_and_200_with_it(client, monkeypatch):
    monkeypatch.setenv(security.API_KEY_ENV, KEY)

    missing = client.get("/api/plans/")
    assert missing.status_code == 401 and missing.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/plans/", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/plans/", headers={"X-API-Key": KEY[:-1]}).status_code == 401
    assert client.get("/api/plans/", headers={"Authorization": f"Basic {KEY}"}).status_code == 401

    assert client.get("/api/plans/", headers={"X-API-Key": KEY}).status_code == 200
    assert client.get("/api/plans/", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_every_api_route_and_mount_requires_the_key(client, monkeypatch):
    calls = _api_calls(client)
    paths = {path for _, path in calls}
    assert {"/api/chat", "/api/chat/upload", "/api/chat/config", "/api/query/", "/api/retrieve/", "/api/plans/",
            "/api/plans/decisions/recent", "/api/plans/x/approve", "/api/plans/x/reject"} <= paths, sorted(paths)
    assert any(path.startswith("/api/files/") for path in paths), "no static mount under /api in this checkout"

    monkeypatch.setenv(security.API_KEY_ENV, KEY)
    for method, path in calls:
        resp = client.request(method, path)
        assert resp.status_code == 401, (method, path, resp.status_code)
    for method, path in calls:  # with the key, whatever each route answers, it is not 401
        resp = client.request(method, path, headers={"X-API-Key": KEY})
        assert resp.status_code != 401, (method, path)


def test_static_files_under_api_are_served_only_with_the_key(client, monkeypatch):
    from app.config import DATA_DIR

    files = [os.path.join(root, name) for root, _, names in os.walk(DATA_DIR) for name in names if "#" not in name and "?" not in name]
    if not files:
        pytest.skip(f"no file under {DATA_DIR} to request")
    url = "/api/files/data/" + os.path.relpath(sorted(files)[0], DATA_DIR).replace(os.sep, "/")
    assert client.get(url).status_code == 200  # open by default
    monkeypatch.setenv(security.API_KEY_ENV, KEY)
    assert client.get(url).status_code == 401
    assert client.get(url, headers={"X-API-Key": KEY}).status_code == 200


def test_the_key_is_compared_in_constant_time_and_never_echoed(client, monkeypatch):
    monkeypatch.setenv(security.API_KEY_ENV, KEY)
    seen = []
    real = security.hmac.compare_digest

    def spy(a, b):
        seen.append((len(a), len(b)))
        return real(a, b)

    monkeypatch.setattr(security.hmac, "compare_digest", spy)
    rejected = client.get("/api/plans/", headers={"X-API-Key": "short"})
    assert rejected.status_code == 401 and KEY not in rejected.text
    assert seen == [(32, 32)]  # both sides hashed to the same length before comparing
    assert KEY not in client.get("/health").text


# ---------------------------------------------------------------- 429 past the budget


def test_429_past_the_budget_with_retry_after(client, monkeypatch):
    monkeypatch.setenv(security.RATE_ENV, "6")   # one token every 10 s
    monkeypatch.setenv(security.BURST_ENV, "3")
    security.reset_limiter()

    assert [client.get("/api/plans/").status_code for _ in range(3)] == [200, 200, 200]
    limited = client.get("/api/plans/")
    assert limited.status_code == 429
    assert 1 <= int(limited.headers["retry-after"]) <= 10
    assert "retry" in limited.json()["detail"]

    health = client.get("/health")
    assert health.status_code == 503  # backend down in this suite; the point is: neither 401 nor 429
    assert health.json()["rate_limit"] == {"enabled": True, "per_minute": 6.0, "burst": 3, "scope": "per replica (in-process buckets)"}


def test_budget_is_per_identity_and_a_bad_key_does_not_spend_it(client, monkeypatch):
    monkeypatch.setenv(security.API_KEY_ENV, KEY)
    monkeypatch.setenv(security.RATE_ENV, "6")
    monkeypatch.setenv(security.BURST_ENV, "2")
    security.reset_limiter()

    for _ in range(5):  # rejected before the budget is consulted
        assert client.get("/api/plans/", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = {"X-API-Key": KEY}
    assert [client.get("/api/plans/", headers=ok).status_code for _ in range(3)] == [200, 200, 429]
    assert client.get("/health").json()["auth"]["enabled"] is True  # still reachable, without a key


# ---------------------------------------------------------------- the bucket itself (fake clock)


def test_token_bucket_refills_and_separates_identities():
    now = [1000.0]
    limiter = security.TokenBucketLimiter(rate_per_minute=60, burst=2, clock=lambda: now[0])  # 1 token / s

    assert limiter.take("a") == 0.0 and limiter.take("a") == 0.0
    assert limiter.take("a") == pytest.approx(1.0)  # empty: one second until the next token
    assert limiter.take("b") == 0.0  # another identity has its own bucket

    now[0] += 0.5
    assert limiter.take("a") == pytest.approx(0.5)
    now[0] += 0.5
    assert limiter.take("a") == 0.0  # refilled one token
    now[0] += 3600
    assert limiter.take("a") == 0.0 and limiter.take("a") == 0.0 and limiter.take("a") > 0  # refill is capped at burst


def test_token_bucket_memory_is_bounded():
    now = [0.0]
    limiter = security.TokenBucketLimiter(rate_per_minute=60, burst=1, clock=lambda: now[0], max_identities=50)
    for i in range(500):
        limiter.take(f"addr:{i}")
        now[0] += 0.01
    assert len(limiter._buckets) <= 51


def test_budget_config_is_off_unless_positive(monkeypatch):
    for value in ("", "0", "-5", "abc"):
        monkeypatch.setenv(security.RATE_ENV, value)
        assert security.budget_config() is None
    monkeypatch.setenv(security.RATE_ENV, "90")
    monkeypatch.delenv(security.BURST_ENV, raising=False)
    assert security.budget_config() == (90.0, 90)
    assert "PER REPLICA" in security.TokenBucketLimiter.__doc__ and "PER REPLICA" in security.__doc__
