"""Every tool module named in config/tools.yaml must load through the real ToolFactory.

The agent tests patch ToolFactory, so before this test nothing imported the
configured modules: a module that did not parse sat in the config with the
suite green, and POST /api/chat could not build its agent.
"""

from pathlib import Path

import pytest
import yaml
from llama_index.core.tools.function_tool import FunctionTool

from app.engine.tools import ToolFactory

CONFIG = Path("config/tools.yaml")
ENTRIES = [
    (tool_type, tool_name, config or {})
    for tool_type, entries in yaml.safe_load(CONFIG.read_text()).items()
    for tool_name, config in entries.items()
]


def test_the_config_names_tools():
    assert ENTRIES, f"{CONFIG} lists no tools"


@pytest.mark.parametrize("tool_type,tool_name,config", ENTRIES, ids=[name for _, name, _ in ENTRIES])
def test_configured_tool_module_loads(tool_type, tool_name, config):
    tools = ToolFactory.load_tools(tool_type, tool_name, config)
    assert tools and all(isinstance(tool, FunctionTool) for tool in tools)


def test_from_env_loads_the_whole_config():
    tools = ToolFactory.from_env(map_result=True)
    assert "artifact" in tools, sorted(tools)
    assert len(tools) >= len(ENTRIES)


def test_a_module_that_does_not_parse_is_reported_as_a_config_error(tmp_path, monkeypatch):
    # The loader used to translate only ImportError/AttributeError, so a SyntaxError
    # escaped as-is; it now names the tool that failed.
    package = tmp_path / "brokentools"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "broken.py").write_text('"""doc"""\nstray text that is not python\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(ToolFactory.TOOL_SOURCE_PACKAGE_MAP, "local", "brokentools")
    with pytest.raises(ValueError, match="broken"):
        ToolFactory.load_tools("local", "broken", {})
