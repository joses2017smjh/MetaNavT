"""MetaNaviT filesystem tools as an MCP server on the official Python SDK (mcp >= 2).

    python -m app.mcp.server_sdk [--root DIR]      # stdio transport, for any MCP client

Every FilesystemTools tool spec (search, read, list, propose_move, apply_plan,
artifacts, patches, visualizations) is registered from tool_specs(), so the
tool list is the same as the hand-rolled server in app/mcp/server.py, which is
kept for clients pinned to its 2024-11-05 framing. The approval gate is
unchanged: apply_* refuses without approved=true.

tests/mcp/test_server_sdk.py spawns this module over stdio with the SDK's
client, lists the tools and calls them.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, Optional

from mcp.server.mcpserver import MCPServer
from app.mcp.filesystem import ApprovalRequired, FilesystemTools


def build_server(root: str | Path, name: str = "metanavit-filesystem") -> MCPServer:
    tools = FilesystemTools(root=Path(root))
    server = MCPServer(name=name, instructions="Search, read and propose changes to a research file tree. apply_* needs approved=true.")

    for spec in tools.tool_specs():
        tool_name = spec["name"]

        def handler(arguments: dict[str, Any] | None = None, _name: str = tool_name) -> Any:
            try:
                return tools.call(_name, arguments or {})
            except ApprovalRequired as exc:
                return {"error": "approval_required", "plan_id": exc.plan_id, "detail": str(exc)}

        _register(server, tool_name, spec, handler)
    server._metanavit_tools = tools  # type: ignore[attr-defined] - handy for tests
    return server


_JSON_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


def _function_for(name: str, schema: dict[str, Any], handler) -> Any:
    """A Python function whose signature mirrors the tool's JSON schema.

    The SDK derives a tool's input schema from the Python signature and type
    hints, so a keyword-only parameter per schema property (required ones
    without defaults) gives clients the same schema the hand-rolled server
    advertised.
    """
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = []
    annotations: dict[str, Any] = {}
    for pname, pspec in props.items():
        typ = _JSON_TYPES.get((pspec or {}).get("type"), Any)
        if pname in required:
            params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY, annotation=typ))
            annotations[pname] = typ
        else:
            params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY, default=(pspec or {}).get("default"), annotation=Optional[typ]))
            annotations[pname] = Optional[typ]

    def fn(**kwargs: Any) -> Any:
        return handler({k: v for k, v in kwargs.items() if v is not None})

    fn.__name__ = name
    fn.__qualname__ = name
    fn.__signature__ = inspect.Signature(params, return_annotation=Any)  # type: ignore[attr-defined]
    fn.__annotations__ = {**annotations, "return": Any}
    return fn


def _register(server: MCPServer, name: str, spec: dict[str, Any], handler) -> None:
    schema = spec.get("inputSchema") or spec.get("input_schema") or {"type": "object", "properties": {}}
    server.add_tool(_function_for(name, schema, handler), name=name, description=spec.get("description", ""))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.getenv("DATA_DIR", "bench/corpus/files"))
    p.add_argument("--transport", default="stdio", choices=["stdio"])
    args = p.parse_args(argv)
    server = build_server(args.root)
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
