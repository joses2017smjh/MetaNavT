"""Minimal MCP stdio server for FilesystemTools (JSON-RPC 2.0).

    python -m app.mcp.server [ROOT] [--embedder NAME | --no-index]

Any MCP client can drive the corpus: search, read, list, propose_move,
collect_run_artifact, propose_artifact / patch. apply_* requires a separately issued operator capability.
Search and propose_artifact use an index built over ROOT at start
(filesystem.build_root_index: hash embedder and overlap reranker unless
--embedder names another); --no-index, or a root with no indexable file,
starts without one and those tools refuse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from app.mcp.filesystem import DEFAULT_EMBEDDER, FilesystemTools, build_root_index, index_summary


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
                    "serverInfo": {"name": "metanavit-filesystem", "version": "2.2.0"},
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", type=Path, default=Path("bench/corpus/files"))
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER, help="hash (default), tfidf, st or st:<model>")
    parser.add_argument("--no-index", action="store_true", help="start without a search index (large roots)")
    args = parser.parse_args(argv)
    index = None if args.no_index else build_root_index(args.root, args.embedder)
    print(index_summary(index), file=sys.stderr, flush=True)
    serve(FilesystemTools(root=args.root, index=index))


if __name__ == "__main__":
    main()
