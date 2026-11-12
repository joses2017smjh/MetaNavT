"""The chat agent's file-access tools (app/engine/tools/index_file_access.py), called the way the agent workflow
calls them: through get_tools() and the FunctionTool, with only the arguments the model gave.

No database: the source of the indexed paths is a stand-in. tests/database/test_file_access_tools.py runs the same
tools over the fixture indexed through the served path.
"""

import asyncio
import inspect
from contextlib import contextmanager

from app.engine.tools import index_file_access as files


def _tool(name):
    return next(tool for tool in files.get_tools() if tool.metadata.name == name)


def _call(name, **kwargs):
    """What the workflow's _call_tool does with a tool call: await tool.acall(**the model's arguments)."""
    return asyncio.run(_tool(name).acall(**kwargs))


class _Cursor:
    def __init__(self, rows, queries):
        self.rows, self.queries = rows, queries

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, statement, params=None):
        self.queries.append(params)

    def fetchall(self):
        return list(self.rows)


class FakeIndexManager:
    """The indexed_files rows the tool reads, without a database."""

    rows = [("configs/run_047.yaml",), ("src/fusion.py",)]
    queries: list = []

    def __init__(self, *args, **kwargs):
        pass

    @contextmanager
    def get_connection(self):
        class _Conn:
            def cursor(conn_self):
                return _Cursor(self.rows, self.queries)

        yield _Conn()


def test_the_pattern_default_is_a_string_not_a_pydantic_field():
    # The default used to be pydantic's Field(default="*"): the FieldInfo object itself, not "*".
    assert inspect.signature(files.list_available_files).parameters["pattern"].default == "*"


def test_list_available_files_called_without_a_pattern_returns_the_paths(monkeypatch):
    monkeypatch.setattr(files, "IndexManager", FakeIndexManager)
    out = _call("list_available_files")  # the model gave no pattern: the agent passes no arguments
    assert not out.is_error
    assert out.raw_output == ["configs/run_047.yaml", "src/fusion.py"]
