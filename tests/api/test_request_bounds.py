"""Request bounds on two routes that used to accept anything.

GET  /api/plans/decisions/recent  `limit` is bounded (it was an unbounded int passed to LIMIT).
POST /api/chat/upload             the body is read with a byte cap and answers 413 past it
                                  (it was an unbounded base64 string held in memory).
No database: same closed-port setup as tests/api/test_plans.py.
"""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from app.api.routers.plans import MAX_RECENT_DECISIONS, PlanStore
from app.mcp.filesystem import FilesystemTools


class _CountingLog:
    def __init__(self):
        self.limits = []

    def recent(self, limit=50):
        self.limits.append(limit)
        return []


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://postgres:password@127.0.0.1:1/metanavit")
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1")
    monkeypatch.setenv("INDEX_ON_START", "false")
    monkeypatch.delenv("FRONTEND_ENDPOINT", raising=False)
    import main

    from app.database import vector_store

    vector_store.get_vector_store_manager.cache_clear()
    with TestClient(main.app) as c:
        c.app.state.plans = PlanStore(FilesystemTools(root=tmp_path), _CountingLog())
        yield c
        c.app.state.plans = None


# ---------------------------------------------------------------- decisions limit


def test_recent_decisions_limit_is_bounded(client):
    log = client.app.state.plans.log
    assert client.get("/api/plans/decisions/recent").status_code == 200
    assert client.get(f"/api/plans/decisions/recent?limit={MAX_RECENT_DECISIONS}").status_code == 200
    assert log.limits == [50, MAX_RECENT_DECISIONS]

    for bad in (MAX_RECENT_DECISIONS + 1, 10**9, 0, -1):
        resp = client.get(f"/api/plans/decisions/recent?limit={bad}")
        assert resp.status_code == 422, (bad, resp.text)
    assert log.limits == [50, MAX_RECENT_DECISIONS]  # a rejected limit never reaches the database


# ---------------------------------------------------------------- upload size cap


@pytest.fixture
def uploads(client, monkeypatch):
    from app.api.routers import upload
    from app.engine.loaders.file import DocumentFile

    calls = []

    def fake_process(name, content, params=None):
        calls.append((name, len(content)))
        return DocumentFile(id="f1", name=name, size=len(content))

    monkeypatch.setattr(upload.FileService, "process_private_file", staticmethod(fake_process))
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "3000")
    return client, calls, upload


def _body(n_bytes: int) -> dict:
    return {"name": "notes.txt", "base64": "data:text/plain;base64," + base64.b64encode(b"x" * n_bytes).decode()}


def test_upload_under_the_cap_is_processed(uploads):
    client, calls, _ = uploads
    resp = client.post("/api/chat/upload", json=_body(3000))
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "notes.txt" and len(calls) == 1


def test_upload_over_the_cap_is_413_with_a_clear_error(uploads):
    client, calls, upload = uploads
    resp = client.post("/api/chat/upload", json=_body(200_000))
    assert resp.status_code == 413, resp.text
    detail = resp.json()["detail"]
    assert "UPLOAD_MAX_BYTES" in detail and str(upload.max_body_bytes()) in detail
    assert calls == []  # refused before the file service ran


def test_upload_cap_also_holds_without_a_content_length(uploads):
    client, calls, _ = uploads
    payload = json.dumps(_body(200_000)).encode()

    def chunks():  # an iterator body is sent chunked: no Content-Length to trust
        for start in range(0, len(payload), 4096):
            yield payload[start:start + 4096]

    resp = client.post("/api/chat/upload", content=chunks(), headers={"content-type": "application/json"})
    assert resp.status_code == 413, resp.text
    assert calls == []


def test_upload_still_validates_the_body(uploads):
    client, calls, _ = uploads
    assert client.post("/api/chat/upload", json={"name": "x"}).status_code == 422
    assert client.post("/api/chat/upload", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
    assert calls == []


def test_upload_body_is_still_documented(client):
    schema = client.get("/openapi.json").json()["paths"]["/api/chat/upload"]["post"]
    assert "requestBody" in schema
