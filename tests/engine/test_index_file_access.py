"""The chat agent's file-access tools (app/engine/tools/index_file_access.py), called the way the agent workflow
calls them: through get_tools() and the FunctionTool, with only the arguments the model gave.

No database: the vector store manager the tools read is a stand-in (or points at a closed port).
tests/database/test_file_access_tools.py runs the same tools over the fixture indexed through the served path.
"""

import asyncio
import inspect

import pytest

from app.engine.tools import index_file_access as files

MANIFEST = ["src/fusion.py", "configs/run_047.yaml", "configs/archive/run_047_v1.yaml", "README.md", "Zeta.md"]


def _tool(name):
    return next(tool for tool in files.get_tools() if tool.metadata.name == name)


def _call(name, **kwargs):
    """What the workflow's _call_tool does with a tool call: await tool.acall(**the model's arguments)."""
    return asyncio.run(_tool(name).acall(**kwargs))


class FakeManager:
    """The ingest manifest the list tool reads ({path: row}), without a database."""

    def __init__(self, paths=MANIFEST, error=None):
        self.paths, self.error = list(paths), error

    def ingest_manifest(self):
        if self.error:
            raise self.error
        return {path: {"content_hash": "0" * 64, "n_chunks": 1, "pipeline": "test"} for path in self.paths}


@pytest.fixture
def manifest(monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: manager)
    return manager


def test_the_pattern_default_is_a_string_not_a_pydantic_field():
    # The default used to be pydantic's Field(default="*"): the FieldInfo object itself, not "*".
    assert inspect.signature(files.list_available_files).parameters["pattern"].default == "*"


def test_list_available_files_called_without_a_pattern_returns_the_paths(manifest):
    out = _call("list_available_files")  # the model gave no pattern: the agent passes no arguments
    assert not out.is_error
    assert out.raw_output == sorted(MANIFEST)  # code point order: upper case before lower case


def test_a_pattern_is_a_glob_over_the_whole_relative_path(manifest):
    assert _call("list_available_files", pattern="configs/*.yaml").raw_output == [
        "configs/archive/run_047_v1.yaml", "configs/run_047.yaml"]  # `*` also crosses "/", as LIKE '%' did
    assert _call("list_available_files", pattern="*.py").raw_output == ["src/fusion.py"]
    assert _call("list_available_files", pattern="configs/run_04?.yaml").raw_output == ["configs/run_047.yaml"]
    assert _call("list_available_files", pattern="run_047*").raw_output == []  # matched against the whole path
    assert _call("list_available_files", pattern="*.MD").raw_output == []  # case-sensitive
    assert _call("list_available_files", pattern="").raw_output == sorted(MANIFEST)
    assert _call("list_available_files", pattern=" * ").raw_output == sorted(MANIFEST)


def test_one_call_lists_at_most_the_first_hundred_paths(monkeypatch):
    many = [f"logs/run_{i:03d}.out" for i in range(150)]
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: FakeManager(many))
    assert _call("list_available_files").raw_output == sorted(many)[: files.MAX_LISTED_FILES]
    assert files.MAX_LISTED_FILES == 100  # the team's LIMIT 100


def test_a_failed_manifest_read_returns_a_fixed_text_without_the_reason(monkeypatch, caplog):
    internals = RuntimeError('relation "public.ingest_manifest_llamaindex_embedding" does not exist')
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: FakeManager(error=internals))
    out = _call("list_available_files")
    assert not out.is_error and out.raw_output == files.LIST_FAILED and out.raw_output.startswith("Error")
    assert "ingest_manifest" not in str(out.content)
    assert "ingest_manifest_llamaindex_embedding" in caplog.text  # the reason is in the API's log


def test_an_unreachable_database_is_not_described_to_the_model(monkeypatch):
    # The real accessor and manager, pointed at a closed port: psycopg2's message names the host and port.
    from app.database import vector_store

    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://someone@127.0.0.1:1/metanavit")
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=someone host=127.0.0.1 port=1")
    vector_store.get_vector_store_manager.cache_clear()
    try:
        out = _call("list_available_files")
    finally:
        vector_store.get_vector_store_manager.cache_clear()
    assert not out.is_error and out.raw_output == files.LIST_FAILED
    assert not any(word in str(out.content) for word in ("127.0.0.1", "port", "someone", "refused"))
