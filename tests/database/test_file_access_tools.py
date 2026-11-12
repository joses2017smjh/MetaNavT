"""The chat agent's file-access tools over fixture v1 indexed through the served path.

list_available_files and get_file_content (app/engine/tools/index_file_access.py) are called the way the agent
workflow calls them: get_tools(), then FunctionTool.acall with only the model's arguments. The table they read is
filled by app/engine/ingest.index_directory (what the index step runs) from a temporary copy of fixture v1, with the
hash embedder, in a table of its own built by the schema migrations. The tools reach it through the API's cached
vector store manager (app/database/vector_store.get_vector_store_manager), pointed at that table.

Needs a live Postgres with pgvector: set PG_TEST_DSN. Skipped otherwise.
"""

import asyncio
import os
import shutil
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

FIXTURE = Path(__file__).resolve().parents[2] / "bench" / "corpus" / "files"


def _call(name, **kwargs):
    from app.engine.tools import index_file_access as files

    tool = next(tool for tool in files.get_tools() if tool.metadata.name == name)
    return asyncio.run(tool.acall(**kwargs))


def _close_cached_manager():
    from app.database import vector_store

    if vector_store.get_vector_store_manager.cache_info().currsize:
        vector_store.get_vector_store_manager().close()
    vector_store.get_vector_store_manager.cache_clear()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """(the corpus copy, its relative paths, the indexing report): a copy of fixture v1 indexed by the served indexer."""
    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PGVECTOR_TABLE", "filetools_" + uuid.uuid4().hex[:8])
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    for name in ("PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES", "INGEST_MAX_REJECTED"):
        monkeypatch.delenv(name, raising=False)
    corpus = tmp_path / "files"
    shutil.copytree(FIXTURE, corpus)
    listing = sorted(p.relative_to(corpus).as_posix() for p in corpus.rglob("*") if p.is_file())

    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager
    from app.engine.ingest import index_directory
    from app.retrieval.embedders import HashEmbedder

    _close_cached_manager()
    vsm = VectorStoreManager(conn_string=dsn)
    try:
        with vsm.get_connection() as conn:
            migrate_all(conn, vsm.vector_table)  # the deploy step: the data table and the manifest
        embedder = HashEmbedder(256)
        report = index_directory(vsm, str(corpus), embed_fn=lambda texts: embedder.encode(texts).tolist())
        yield corpus, listing, report
    finally:
        _close_cached_manager()
        with vsm.get_connection() as conn:
            drop_vector_table(conn, vsm.vector_table)
        vsm.close()


def test_list_without_a_pattern_lists_every_file_the_served_index_holds(served):
    _, listing, report = served
    assert report["files_rejected"] == 0 and report["files_indexed"] == len(listing)  # all of fixture v1
    out = _call("list_available_files")  # no arguments: the model gave no pattern
    assert not out.is_error
    assert out.raw_output == listing


def test_a_pattern_filters_the_listed_files(served):
    _, listing, _ = served
    assert _call("list_available_files", pattern="configs/*.yaml").raw_output == [
        p for p in listing if p.startswith("configs/") and p.endswith(".yaml")]
    assert _call("list_available_files", pattern="*.py").raw_output == [p for p in listing if p.endswith(".py")]
    assert _call("list_available_files", pattern="configs/run_047.yaml").raw_output == ["configs/run_047.yaml"]
    assert _call("list_available_files", pattern="nothing/*").raw_output == []
