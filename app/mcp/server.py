"""Minimal MCP stdio server for FilesystemTools (JSON-RPC 2.0).

Any MCP client can drive the corpus: search, read, list, propose_move.
apply_plan requires approved=true — the destructive action never auto-fires.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from app.mcp.filesystem import FilesystemTools


class MCPServer:
    def __init__(self, tools: FilesystemTools):
        self.tools = tools

    def handle(self, message: dict) -> dict | None:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        if method == "initialize":
            return _ok(
                msg_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "metanavit-filesystem", "version": "2.0.0"},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return _ok(msg_id, {"tools": self.tools.tool_specs()})
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                result = self.tools.call(name, args)
                text = json.dumps(result, default=str)
                return _ok(
                    msg_id,
                    {"content": [{"type": "text", "text": text}], "isError": False},
                )
            except Exception as exc:
                return _ok(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": str(exc)}],
                        "isError": True,
                    },
                )
        if msg_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def _ok(msg_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(tools: FilesystemTools, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = MCPServer(tools)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bench/corpus/files")
    tools = FilesystemTools(root=root)
    serve(tools)


if __name__ == "__main__":
    main()
