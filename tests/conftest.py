import sys
import os
from unittest.mock import MagicMock

import pytest

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


LEGACY_SUITES = ("/tests/unit_tests/", "/tests/database_tests/")


def _is_legacy_test(request) -> bool:
    """Only the legacy suites get the llama_index / psycopg2 MagicMock modules.

    tests/eval, tests/engine and tests/api import the real packages.
    """
    path = str(getattr(request, "fspath", "")).replace("\\", "/")
    return any(marker in path for marker in LEGACY_SUITES)


def pytest_configure(config):
    """Mock llama_index/psycopg2 for the legacy unit tests only (see _is_legacy_test)."""
    # Defer sys.modules mutation until collection knows the path is awkward,
    # so we mock lazily in a fixture instead of at import time.
    return


@pytest.fixture
def mock_vector_store(monkeypatch):
    mock_store = MagicMock()
    monkeypatch.setattr("app.database.vector_store.get_vector_store", lambda: mock_store)
    return mock_store


@pytest.fixture
def mock_database_reader(monkeypatch):
    mock_reader = MagicMock()
    monkeypatch.setattr(
        "llama_index.readers.database.DatabaseReader",
        lambda *a, **k: mock_reader,
        raising=False,
    )
    return mock_reader


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch, request):
    if not _is_legacy_test(request):
        return
    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://user:pass@localhost:5432/test")
    monkeypatch.setenv("STORAGE_DIR", "test_storage")
    monkeypatch.setenv("DATA_DIR", "/test_data")


@pytest.fixture(autouse=True)
def mock_llama_index(monkeypatch, request):
    if not _is_legacy_test(request):
        return
    mock_modules = {
        "psycopg2": MagicMock(),
        "psycopg2.sql": MagicMock(),
        "psycopg2.pool": MagicMock(),
        "llama_index": MagicMock(),
        "llama_index.core": MagicMock(),
        "llama_index.core.readers": MagicMock(),
        "llama_index.core.readers.base": MagicMock(),
        "llama_index.core.readers.database": MagicMock(),
        "llama_index.core.indices": MagicMock(),
        "llama_index.core.indices.base": MagicMock(),
        "llama_index.core.indices.composability": MagicMock(),
        "llama_index.core.indices.composability.graph": MagicMock(),
        "llama_index.core.ingestion": MagicMock(),
        "llama_index.core.ingestion.pipeline": MagicMock(),
        "llama_index.core.multi_modal_llms": MagicMock(),
        "llama_index.core.settings": MagicMock(),
        "llama_index.vector_stores": MagicMock(),
        "llama_index.core.storage": MagicMock(),
        "llama_index.core.storage.docstore": MagicMock(),
        "llama_index.core.node_parser": MagicMock(),
        "llama_index.vector_stores.postgres": MagicMock(),
    }
    mock_modules["llama_index.core.multi_modal_llms"].MultiModalLLM = MagicMock()
    for mod_name, mock in mock_modules.items():
        # via monkeypatch so the real modules come back after the test (tests/api imports them)
        monkeypatch.setitem(sys.modules, mod_name, mock)
    monkeypatch.setattr("llama_index.core.indices.VectorStoreIndex", MagicMock(), raising=False)
    monkeypatch.setattr("llama_index.core.storage.StorageContext", MagicMock(), raising=False)
    monkeypatch.setattr("llama_index.core.Document", MagicMock(), raising=False)
    monkeypatch.setattr("llama_index.core.settings.Settings", MagicMock(), raising=False)
