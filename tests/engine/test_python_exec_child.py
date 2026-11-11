"""The code-execution tool is off by default and, when on, runs code in a child process (C42 / D35).

Before this change config/tools.yaml listed python_exec and get_chat_engine() always
built the code agent, so model-written code reached exec() inside the API process. Both
agents held the tool: the file-access agent's filter drops a tool named "python_exec",
but the tool is named "execute_code". The tests identify the tool by its name and by
the function behind it, never by the module name.
"""

import os
import time
from pathlib import Path

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.settings import Settings

import app.engine.agents.file_access_agent as file_access_agent
from app.api import security
from app.engine.engine import get_chat_engine
from app.engine.tools import ToolFactory, python_exec
from app.engine.tools.python_exec import PythonExecTool

KEY = "test-key-exec-4c1d-do-not-log"
# The settings by name (they are the documented interface, and main has no constants for them).
FLAG, TIMEOUT, MEMORY = "ENABLE_PYTHON_EXEC", "PYTHON_EXEC_TIMEOUT_SECONDS", "PYTHON_EXEC_MAX_MEMORY_MB"


@pytest.fixture
def engine_env(monkeypatch):
    """get_chat_engine() as the API builds it, minus the database: no index, so no query tool."""
    for name in (FLAG, security.API_KEY_ENV, TIMEOUT, MEMORY):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Settings, "_llm", MockLLM())  # agents only store it; nothing calls it here
    monkeypatch.setattr(file_access_agent, "get_index", lambda config=None: None)
    return monkeypatch


def _enable(monkeypatch, key=KEY):
    monkeypatch.setenv(FLAG, "true")
    if key:
        monkeypatch.setenv(security.API_KEY_ENV, key)


def _is_exec_tool(tool) -> bool:
    return tool.metadata.name == "execute_code" or getattr(tool, "real_fn", None) == PythonExecTool.execute_code


def _exec_tools(workflow) -> dict:
    return {name: [t.metadata.name for t in agent.tools or [] if _is_exec_tool(t)] for name, agent in workflow.agents.items()}


# ---------------------------------------------------------------- off by default


def test_default_chat_engine_exposes_no_exec_tool(engine_env):
    workflow = get_chat_engine()
    assert list(workflow.agents) == ["FileAccessAgent"]
    assert _exec_tools(workflow) == {"FileAccessAgent": []}


def test_default_tool_config_loads_no_exec_tool(engine_env):
    assert python_exec.get_tools() == []
    loaded = ToolFactory.from_env()
    assert loaded and not [t.metadata.name for t in loaded if _is_exec_tool(t)]


def test_flag_without_an_api_key_keeps_the_tool_off(engine_env, caplog):
    _enable(engine_env, key=None)
    caplog.set_level("WARNING", logger=python_exec.__name__)
    assert python_exec.get_tools() == []
    assert _exec_tools(get_chat_engine()) == {"FileAccessAgent": []}
    assert any(security.API_KEY_ENV in r.getMessage() for r in caplog.records), caplog.text


def test_flag_and_key_add_the_code_agent_and_only_it_holds_the_tool(engine_env):
    _enable(engine_env)
    workflow = get_chat_engine()
    assert _exec_tools(workflow) == {"FileAccessAgent": [], "PythonCodeAgent": ["execute_code"]}


# ---------------------------------------------------------------- on: a child process with limits


@pytest.fixture
def exec_tool(engine_env):
    _enable(engine_env)
    (tool,) = python_exec.get_tools()
    return tool


def test_enabled_code_runs_in_another_process(exec_tool, tmp_path):
    out = exec_tool.call(code="import os\nprint(os.getpid())\nresult = 6 * 7", workspace_dir=str(tmp_path)).raw_output
    assert out["error"] == "" and out["result"] == "42", out
    child_pid = int(out["output"])
    assert child_pid != os.getpid()
    assert list(tmp_path.glob("exec_*.py")), "the code is still saved in the workspace"


def _gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state in {"Z", "X"}  # exited; only waiting to be reaped by its new parent


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="reads process state from /proc")
def test_code_past_its_time_limit_is_killed_with_its_process_group(exec_tool, engine_env, tmp_path):
    engine_env.setenv(TIMEOUT, "1")
    pids = tmp_path / "pids.txt"
    code = (
        "import os, subprocess, time\n"
        "grandchild = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{grandchild.pid}}')\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()
    out = exec_tool.call(code=code, workspace_dir=str(tmp_path)).raw_output
    elapsed = time.monotonic() - started
    assert "time limit" in out["error"] and out["output"] == "", out
    assert elapsed < 15, elapsed
    child, grandchild = (int(p) for p in pids.read_text().split())
    assert os.getpid() not in (child, grandchild)
    deadline = time.monotonic() + 5
    while not (_gone(child) and _gone(grandchild)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(child) and _gone(grandchild)


def test_the_time_limit_holds_where_os_has_no_waitid(exec_tool, engine_env, tmp_path):
    # CPython has no os.waitid on macOS; the tool then waits with Popen.wait.
    engine_env.delattr(os, "waitid")
    engine_env.setenv(TIMEOUT, "1")
    started = time.monotonic()
    out = exec_tool.call(code="import time\ntime.sleep(60)", workspace_dir=str(tmp_path)).raw_output
    assert "time limit" in out["error"] and time.monotonic() - started < 15, out
    out = exec_tool.call(code="import os\nprint(os.getpid())", workspace_dir=str(tmp_path)).raw_output
    assert out["error"] == "" and int(out["output"]) != os.getpid(), out


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="reads process state from /proc")
def test_a_background_process_in_its_group_does_not_outlive_the_call(exec_tool, tmp_path):
    pid_file = tmp_path / "background.pid"
    code = (
        "import subprocess\n"
        "background = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(background.pid))\n"
        "result = 'returned'\n"
    )
    out = exec_tool.call(code=code, workspace_dir=str(tmp_path)).raw_output
    assert out["error"] == "" and out["result"] == "returned", out
    background = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while not _gone(background) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(background)


def test_the_child_has_the_limits_and_a_reduced_environment(exec_tool, engine_env, tmp_path):
    engine_env.setenv(MEMORY, "256")
    engine_env.setenv(TIMEOUT, "5")
    engine_env.setenv("PG_CONNECTION_STRING", "postgresql://postgres:hunter2@127.0.0.1:1/metanavit")
    code = (
        "import os, resource\n"
        "limits = [resource.getrlimit(getattr(resource, n)) for n in ('RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_FSIZE', 'RLIMIT_CORE')]\n"
        "print(limits)\n"
        "print(sorted(k for k in os.environ if 'KEY' in k or 'CONNECTION' in k))\n"
    )
    out = exec_tool.call(code=code, workspace_dir=str(tmp_path)).raw_output
    assert out["error"] == "", out
    limits, secrets = out["output"].splitlines()
    assert secrets == "[]"
    mib = 256 * 1024 * 1024
    assert limits == str([(mib, mib), (5, 5), (python_exec.MAX_FILE_BYTES, python_exec.MAX_FILE_BYTES), (0, 0)])


def test_numpy_and_pandas_run_under_the_default_limits(exec_tool, tmp_path):
    # The code agent's prompt offers pandas and numpy; the default limits must leave room for them.
    pytest.importorskip("pandas")
    code = "import numpy as np, pandas as pd\nresult = float(np.mean(pd.Series([1e-3, 3e-4])))"
    out = exec_tool.call(code=code, workspace_dir=str(tmp_path)).raw_output
    assert out["error"] == "" and out["result"] == "0.00065", out


def test_memory_past_the_limit_is_reported_not_fatal(exec_tool, engine_env, tmp_path):
    engine_env.setenv(MEMORY, "256")
    out = exec_tool.call(code="block = bytearray(1024 * 1024 * 1024)\nprint('allocated')", workspace_dir=str(tmp_path)).raw_output
    assert "MemoryError" in out["error"] and out["output"] == "", out
