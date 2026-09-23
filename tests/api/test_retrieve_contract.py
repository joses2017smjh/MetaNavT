"""API contract with the backend down: the app starts, /health says why, retrieve is 503.

No Postgres in this suite: PG_CONNECTION_STRING points at a closed port so the
lifespan records a startup error instead of a retriever. The happy path and
the parity run need a database and live in CI's docker-smoke job (M0) and the
parity job (M3).
"""

import os

import pytest
from fastapi.testclient import TestClient

BAD_DB = "postgresql://postgres:password@127.0.0.1:1/metanavit"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("PG_CONNECTION_STRING", BAD_DB)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1")
    monkeypatch.setenv("INDEX_ON_START", "false")
    monkeypatch.delenv("FRONTEND_ENDPOINT", raising=False)
    import main  # noqa: WPS433 - the FastAPI app module

    from app.database import vector_store

    vector_store.get_vector_store_manager.cache_clear()
    with TestClient(main.app) as c:  # runs the lifespan
        yield c


def test_routes_are_mounted(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {"/health", "/api/retrieve/", "/api/chat", "/api/query/"} <= paths, sorted(paths)


def test_health_reports_the_startup_error(client):
    resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unavailable"
    assert body["error"], body


def test_retrieve_is_503_when_the_backend_is_down(client):
    resp = client.post("/api/retrieve/", json={"query": "current learning rate for run 47"})
    assert resp.status_code == 503
    assert "retrieval backend unavailable" in resp.json()["detail"]


def test_retrieve_validates_the_body(client):
    assert client.post("/api/retrieve/", json={}).status_code == 422
    assert client.post("/api/retrieve/", json={"query": 42}).status_code == 422
