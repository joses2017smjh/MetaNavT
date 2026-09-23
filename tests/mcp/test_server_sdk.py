"""Spawn app/mcp/server_sdk.py over stdio with the official SDK client; list and call tools."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


def _payload(result):
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if structured:
        return structured.get("result", structured) if isinstance(structured, dict) and set(structured) == {"result"} else structured
    return json.loads(_text(result))


async def _drive(corpus: Path) -> dict:
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp.server_sdk", "--root", str(corpus)], cwd=str(ROOT), env={"PYTHONPATH": str(ROOT)})
    out: dict = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "serverInfo", None) or getattr(init, "server_info", None)
            out["server_name"] = info.name
            tools = await session.list_tools()
            out["tools"] = {t.name: (getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)) for t in tools.tools}
            listed = await session.call_tool("list_dir", {"path": "."})
            out["list_dir"] = _payload(listed)
            plan = await session.call_tool("propose_move", {"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"})
            out["plan"] = _payload(plan)
            refused = await session.call_tool("apply_plan", {"plan_id": out["plan"]["plan_id"]})
            out["refused"] = _payload(refused)
            applied = await session.call_tool("apply_plan", {"plan_id": out["plan"]["plan_id"], "approved": True})
            out["applied"] = _payload(applied)
    return out


def test_sdk_server_lists_and_calls_tools(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "run_047.yaml").write_text("learning_rate: 3e-4\n")
    out = asyncio.run(_drive(tmp_path))

    assert out["server_name"] == "metanavit-filesystem"
    assert {"search_lexical", "read_file", "list_dir", "propose_move", "apply_plan"} <= set(out["tools"])
    assert "src" in out["tools"]["propose_move"]["properties"] and "src" in out["tools"]["propose_move"].get("required", [])
    listed = out["list_dir"]
    entries = listed if isinstance(listed, list) else listed.get("result", listed)
    assert any("configs" in json.dumps(e) for e in (entries if isinstance(entries, list) else [entries]))
    assert out["plan"]["status"] == "pending_approval"
    assert out["refused"].get("error") == "approval_required"  # the gate survives the SDK port
    assert out["applied"]["status"] == "applied" and (tmp_path / "configs" / "archive" / "run_047.yaml").exists()
