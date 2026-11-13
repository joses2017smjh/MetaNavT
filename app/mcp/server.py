"""Minimal MCP stdio server for FilesystemTools (JSON-RPC 2.0).

    python -m app.mcp.server [ROOT] [--embedder NAME | --no-index]

Any MCP client can drive the corpus: search, read, list, propose_move,
collect_run_artifact, propose_artifact / patch. apply_* requires a separately issued operator capability.
Search and propose_artifact use an index built over ROOT at start
(filesystem.build_root_index: hash embedder and overlap reranker unless
--embedder names another); --no-index, or a root with no indexable file,
starts without one and those tools refuse, saying so.

A call that raises is answered with isError and a text written in this
repository, never the text of any other exception (an operating system error
names the root's absolute path, and an error from below the tools can name a
host or a user): a refusal names the check that refused it and the check's fixed
description (REFUSAL_CHECKS, the text app/mcp/server_sdk.py sends for it); an
exception of a class in SHOWN_IN_FULL, whose text is written here for the client
(IndexRequired: "semantic search requires an index"), is answered with that
text; any other exception with its class and an id, its text going to stderr
under that id, on one line (app.observability.report_failure). stdout carries
only the protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

from app.mcp.approvals import REFUSAL_CHECKS
from app.mcp.filesystem import DEFAULT_EMBEDDER, FilesystemTools, IndexRequired, build_root_index, index_summary, refusal_check
from app.observability import report_failure

# Nothing in this process configures logging, so a WARNING goes to Python's last-resort handler: stderr.
logger = logging.getLogger("app.mcp.server")
# The exception classes whose text this repository writes for the client and which name nothing of the server (no
# path, host, address or user): a search or proposal tool on a server started without an index. Answered as
# written, the rule app/api/errors.py applies to its SHOWN_IN_FULL; any other exception by its class and an id.
SHOWN_IN_FULL = (IndexRequired,)


def tool_error(name: Any, exc: Exception) -> str:
    """What the client is told of a tool call that raised: a refusal's check and its fixed description, the text of
    an exception whose class is in SHOWN_IN_FULL, or else the exception's class and the id its text is logged under."""
    check = refusal_check(exc)
    if check is not None:  # anticipated: answered, not logged (apply_* refusals are in the audit trail)
        reason = f"refused by the {check} check: {REFUSAL_CHECKS[check]}"
    elif type(exc) in SHOWN_IN_FULL:  # the client gets the whole reason, so there is nothing to log
        reason = str(exc)
    else:
        _, reason = report_failure(logger, f"MCP tool {name!r}", exc, level=logging.WARNING)
    return f"Error executing tool {name}: {reason}"


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
                        "content": [{"type": "text", "text": tool_error(name, exc)}],
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
