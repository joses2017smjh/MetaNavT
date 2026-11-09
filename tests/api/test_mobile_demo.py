"""The companion demo uses actual retrieval with the production HTTP contract."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from app.api.routers.retrieve import RetrieveResponse
from app.mobile.demo import CORPUS, STARTER_QUERIES, create_app


@pytest.fixture
def client():
    with TestClient(create_app(allowed_origins=("http://localhost:8081",))) as client:
        yield client


def test_demo_health_is_explicit_about_synthetic_fixture(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "demo" and body["synthetic"] is True
    assert body["n_files"] == 5 and body["n_nodes"] >= 5
    assert body["embedding_provider"] == "hash-demo-fixture"
    assert body["starter_queries"] == STARTER_QUERIES


def test_fixture_provenance_matches_copied_benchmark_bytes():
    provenance = json.loads(CORPUS.with_name("fixture-provenance.json").read_text())
    assert provenance["synthetic"] is True
    assert len(provenance["files"]) == 5
    for entry in provenance["files"]:
        payload = (CORPUS / entry["path"]).read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert payload == (Path("bench/corpus/files") / entry["path"]).read_bytes()


@pytest.mark.parametrize("query", STARTER_QUERIES)
def test_queries_reuse_live_contract_with_real_source_spans(client, query):
    response = client.post("/api/retrieve/", json={"query": query, "k": 3})
    assert response.status_code == 200, response.text
    body = RetrieveResponse.model_validate(response.json())
    assert body.query == query and body.k == 3
    assert body.retrieval_mode == "mobile_demo_hash"
    assert body.embedding_provider == "hash-demo-fixture"
    assert body.hits and len(body.hits) <= 3
    assert body.latency_ms["total"] >= 0
    for hit in body.hits:
        raw = (CORPUS / hit.path).read_bytes()
        assert raw[hit.start_byte:hit.end_byte].decode("utf-8")[:800] == hit.text
        # Lexical path hits may bypass both BM25 and dense scores; fusion is retained.
        assert hit.scores.rrf is not None


def test_learning_rate_query_opens_the_actual_config(client):
    body = client.post("/api/retrieve/", json={"query": STARTER_QUERIES[0], "k": 2}).json()
    assert body["hits"][0]["path"] == "configs/run_047.yaml"
    assert "learning_rate: 0.0003" in body["hits"][0]["text"]


@pytest.mark.parametrize("payload", [{}, {"query": " "}, {"query": "x", "k": 0},
                                      {"query": "x", "k": 51}, {"query": "x" * 2001}])
def test_mobile_demo_keeps_live_validation(client, payload):
    assert client.post("/api/retrieve/", json=payload).status_code == 422


def test_demo_routes_do_not_expose_agent_or_mutation_capabilities(client):
    assert set(client.get("/openapi.json").json()["paths"]) == {"/health", "/api/retrieve/"}
    for path in ["/api/chat", "/api/plans", "/api/chat/upload", "/api/files/data/configs/run_047.yaml"]:
        assert client.post(path, json={}).status_code == 404


def test_demo_cors_allows_configured_origin_only(client):
    headers = {"Origin": "http://localhost:8081", "Access-Control-Request-Method": "POST",
               "Access-Control-Request-Headers": "content-type"}
    response = client.options("/api/retrieve/", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8081"
    headers["Origin"] = "https://unconfigured.example"
    assert client.options("/api/retrieve/", headers=headers).status_code == 400


@pytest.mark.parametrize("origin", ["*", "null", "https://*.example.com", "https://example.com/path",
                                   "https://user:secret@example.com", "https://example.com?key=value"])
def test_demo_cors_rejects_wildcards_and_non_origin_values(origin):
    with pytest.raises(ValueError):
        create_app(allowed_origins=(origin,))


def test_demo_import_and_retrieval_need_no_database_or_model_stack():
    # A fresh process proves previously imported app packages cannot mask dependencies.
    script = r'''
import importlib.abc
import sys
class RefuseHeavyDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'llama_index', 'psycopg2', 'torch', 'sklearn', 'sentence_transformers'}:
            raise AssertionError('demo imported heavy dependency: ' + fullname)
sys.meta_path.insert(0, RefuseHeavyDependencies())
from fastapi.testclient import TestClient
from app.mobile.demo import create_app
with TestClient(create_app()) as client:
    assert client.get('/health').json()['mode'] == 'demo'
    result = client.post('/api/retrieve/', json={'query': 'learning_rate run_047.yaml', 'k': 2})
    assert result.status_code == 200 and result.json()['hits']
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=30, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
