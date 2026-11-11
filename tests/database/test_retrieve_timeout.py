"""POST /api/retrieve/ against a live Postgres: a statement past its budget is cancelled and answered 504.

Needs PG_TEST_DSN; skipped otherwise. The served retriever (SQL mode, hash
embedder, no reranker) runs over a small table of its own; the app's start-up
points at a closed port and the retrieval state is put in place by the test, so
nothing here depends on the environment other modules leave behind. A second
connection holds an ACCESS EXCLUSIVE lock on the table, so the hybrid statement
waits until its statement_timeout (the rest of the request's budget) cancels it,
however fast the machine is.
"""

import os
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

DOCS = {
    "cfg47": "learning_rate: 3e-4\nencoder: dinov2\nrun_id: 47",
    "log47": "run 47 epoch 12 val_rmse 0.055 learning rate schedule cosine",
    "readme": "Repository layout: configs, logs, checkpoints, paper.",
}
CLOSED_PORT = "postgresql://postgres:password@127.0.0.1:1/metanavit"


@pytest.fixture
def served_api(monkeypatch):
    from llama_index.core.schema import TextNode

    from app.database import vector_store
    from app.database.vector_store_manager import VectorStoreManager
    from app.engine.bootstrap import RetrievalState
    from app.engine.embeddings import HashEmbedding
    from app.engine.retriever import HybridRetriever

    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PGVECTOR_TABLE", "timeouttest_" + uuid.uuid4().hex[:8])
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    vsm = VectorStoreManager(conn_string=dsn)
    embed = HashEmbedding(256)
    vsm.get_vector_store().add(
        [TextNode(id_=nid, text=text, embedding=embed.get_text_embedding(text), metadata={"path": f"{nid}.txt"}) for nid, text in DOCS.items()]
    )
    vsm.ensure_text_index()
    retriever = HybridRetriever(None, vsm, reranker=None, enable_router=False, mode="sql", embed_fn=embed.get_query_embedding)

    # Start-up against a closed port records an error and builds nothing; the state is ours.
    monkeypatch.setenv("PG_CONNECTION_STRING", CLOSED_PORT)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1")
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("INDEX_ON_START", "false")
    monkeypatch.delenv("FRONTEND_ENDPOINT", raising=False)
    import main

    vector_store.get_vector_store_manager.cache_clear()
    with TestClient(main.app) as client:
        client.app.state.retrieval = RetrievalState(vsm=vsm, index=None, retriever=retriever)
        yield client, vsm
        client.app.state.retrieval = None
    vector_store.get_vector_store_manager.cache_clear()
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{vsm.schema_name}"."{vsm.data_table}";')
    vsm.close()


def test_a_statement_past_its_budget_is_504_and_the_next_request_is_served(served_api, monkeypatch):
    import psycopg2

    from app.api.routers.retrieve import TIMEOUT_GRACE_SECONDS

    client, vsm = served_api
    body = {"query": "current learning rate for run 47"}
    assert client.post("/api/retrieve/", json=body).status_code == 200  # warm: connections open, library loaded

    monkeypatch.setenv("RETRIEVE_TIMEOUT_SECONDS", "0.5")
    locker = psycopg2.connect(os.environ["PG_TEST_DSN"])
    release = threading.Timer(15, locker.rollback)  # if nothing else ends the wait, fail instead of hanging
    try:
        with locker.cursor() as cur:
            cur.execute(f'LOCK TABLE "{vsm.schema_name}"."{vsm.data_table}" IN ACCESS EXCLUSIVE MODE;')
        release.start()
        started = time.monotonic()
        resp = client.post("/api/retrieve/", json=body)
        elapsed = time.monotonic() - started
    finally:
        release.cancel()
        locker.rollback()  # releases the lock
        locker.close()
    assert resp.status_code == 504, resp.text
    # Postgres cancelled it: the stage's own StageTimeout ("ran past"), not the route giving up on
    # a thread that is still in the stage ("still running past"), which would also name hybrid_sql.
    assert "stage hybrid_sql ran past" in resp.json()["detail"], resp.text
    assert elapsed < 0.5 + TIMEOUT_GRACE_SECONDS, elapsed

    resp = client.post("/api/retrieve/", json=body)  # the cancelled statement's connection went back to the pool clean
    assert resp.status_code == 200, resp.text
    assert resp.json()["hits"] and resp.json()["hits"][0]["path"]
