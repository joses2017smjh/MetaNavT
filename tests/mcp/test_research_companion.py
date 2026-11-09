"""Official SDK stdio round trip for the read-only companion tool."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("fastapi")
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def drive():
    root = Path(__file__).resolve().parents[2]
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp.research_companion"],
                                  cwd=str(root), env={"PYTHONPATH": str(root)})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert [tool.name for tool in tools.tools] == ["ask_research"]
            result = await session.call_tool("ask_research", {
                "question": "What is the learning rate in run_047.yaml?", "mode": "demo"})
            structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
            if structured:
                return structured.get("result", structured)
            return json.loads("".join(getattr(item, "text", "") for item in result.content))


def test_read_only_sdk_tool_executes_real_demo_retrieval():
    result = asyncio.run(drive())
    assert result["status"] == "complete"
    assert "0.0003" in result["answer"]
    assert result["citation_check"]["passed"]
    assert [event["tool"] for event in result["trace"]] == ["search_research", "inspect_source"]
