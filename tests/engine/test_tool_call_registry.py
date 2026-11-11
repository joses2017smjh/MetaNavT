"""The tool-call registry is per request, and tool outputs are not logged at INFO (C27).

Every tool callback used to append to one module-level list that nothing cleared, and
to log the whole call, output included, at INFO. A sync tool and its callback run on
an executor thread (FunctionTool.acall -> run_in_executor), which does not see the
request's context variables, so these tests call tools the way the agents do.
"""

import asyncio
import logging

from llama_index.core.tools import FunctionTool

import app.engine.tools as tools_module
from app.engine.tools import clear_tool_calls, create_tool_callback, get_tool_call_registry


def secret_echo(x: int) -> str:
    """Echo a value with a marker that must not reach an INFO log line."""
    return f"secret output {x}"


async def _request(n: int) -> list:
    """What one chat request does: start a registry, build its tools, call one."""
    clear_tool_calls()
    tool = FunctionTool.from_defaults(fn=secret_echo, name="echo", callback=create_tool_callback("echo", "Echo its input"))
    await tool.acall(x=n)  # the function and the callback run on an executor thread
    return get_tool_call_registry()


def _module_lists() -> dict:
    return {name: len(value) for name, value in vars(tools_module).items() if isinstance(value, list)}


def test_two_request_contexts_get_separate_registries():
    before = _module_lists()

    async def two_requests():
        return await asyncio.gather(_request(1), _request(2))  # one task, so one context, each

    first, second = asyncio.run(two_requests())
    assert [call["output"] for call in first] == ["secret output 1"]
    assert [call["output"] for call in second] == ["secret output 2"]
    assert first is not second
    assert _module_lists() == before  # no module-level list grew


def test_a_tool_call_is_logged_at_info_without_its_output(caplog):
    caplog.set_level(logging.INFO, logger=tools_module.__name__)
    calls = asyncio.run(_request(7))
    assert calls[0]["output"] == "secret output 7"  # still recorded in the request's registry
    info = [r.getMessage() for r in caplog.records if r.name == tools_module.__name__ and r.levelno == logging.INFO]
    assert any("echo" in message for message in info), info
    assert not any("secret output" in message for message in info), info
