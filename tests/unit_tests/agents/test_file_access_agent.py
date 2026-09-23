"""Unit tests for create_file_access_agent.

The agent gathers tools from ToolFactory.from_env(), appends a query-engine
tool when an index exists, drops python_exec and duplicates, and builds a
FunctionAgent with Settings.llm. Everything external is patched.
"""

from unittest.mock import MagicMock

import pytest

from app.engine.agents.file_access_agent import FILE_ACCESS_PROMPT, create_file_access_agent


def _tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.metadata.name = name
    return tool


@pytest.fixture
def patched(mocker):
    """Patch the collaborators; return the mocks so tests can shape them."""
    factory = mocker.patch("app.engine.agents.file_access_agent.ToolFactory")
    get_index = mocker.patch("app.engine.agents.file_access_agent.get_index", return_value=None)
    query_tool = mocker.patch("app.engine.agents.file_access_agent.get_query_engine_tool")
    agent_cls = mocker.patch("app.engine.agents.file_access_agent.FunctionAgent")
    settings = mocker.patch("app.engine.agents.file_access_agent.Settings")
    return {
        "factory": factory,
        "get_index": get_index,
        "query_tool": query_tool,
        "agent_cls": agent_cls,
        "settings": settings,
    }


def _tool_names(agent_cls: MagicMock) -> list[str]:
    return [t.metadata.name for t in agent_cls.call_args.kwargs["tools"]]


class TestFileAccessAgent:
    def test_filters_python_exec(self, patched):
        patched["factory"].from_env.return_value = [_tool("python_exec"), _tool("other_tool")]

        create_file_access_agent()

        assert _tool_names(patched["agent_cls"]) == ["other_tool"]

    def test_appends_query_engine_tool_when_index_exists(self, patched):
        patched["factory"].from_env.return_value = [_tool("other_tool")]
        patched["get_index"].return_value = MagicMock(name="index")
        patched["query_tool"].return_value = _tool("query_engine")

        create_file_access_agent()

        patched["query_tool"].assert_called_once()
        assert sorted(_tool_names(patched["agent_cls"])) == ["other_tool", "query_engine"]

    def test_skips_query_engine_tool_without_index(self, patched):
        patched["factory"].from_env.return_value = [_tool("other_tool")]
        patched["get_index"].return_value = None

        create_file_access_agent()

        patched["query_tool"].assert_not_called()
        assert _tool_names(patched["agent_cls"]) == ["other_tool"]

    def test_prevents_duplicate_tools(self, patched):
        patched["factory"].from_env.return_value = [_tool("duplicate_tool"), _tool("duplicate_tool")]

        create_file_access_agent()

        assert _tool_names(patched["agent_cls"]) == ["duplicate_tool"]

    def test_uses_settings_llm(self, patched):
        patched["factory"].from_env.return_value = []

        create_file_access_agent()

        assert patched["agent_cls"].call_args.kwargs["llm"] is patched["settings"].llm

    def test_agent_parameters(self, patched):
        patched["factory"].from_env.return_value = []

        agent = create_file_access_agent()

        kwargs = patched["agent_cls"].call_args.kwargs
        assert kwargs["name"] == "FileAccessAgent"
        assert kwargs["description"] == "Retrieves file contents and uses all tools except code execution"
        assert kwargs["system_prompt"] == FILE_ACCESS_PROMPT
        assert kwargs["can_handoff_to"] == ["PythonCodeAgent"]
        assert agent is patched["agent_cls"].return_value

    def test_logs_tool_count(self, patched, mocker):
        patched["factory"].from_env.return_value = [_tool("a"), _tool("b"), _tool("python_exec")]
        logger = mocker.patch("app.engine.agents.file_access_agent.logger")

        create_file_access_agent()

        logger.info.assert_called_once()
        assert "2 tools" in logger.info.call_args.args[0]
