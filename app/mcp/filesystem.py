"""Filesystem MCP tools.

search_semantic, search_lexical, read_file, list_dir, stat,
propose_move (plan only), apply_plan (human approval required).
"""

from __future__ import annotations

import json
import os
import stat as statmod
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.retrieval.hybrid import InMemoryHybridIndex


class ApprovalRequired(Exception):
    def __init__(self, plan_id: str):
        super().__init__(f"Plan {plan_id} requires human approval before apply_plan")
        self.plan_id = plan_id


@dataclass
class MovePlan:
    plan_id: str
    src: str
    dst: str
    created_at: float
    approved: bool = False
    applied: bool = False


@dataclass
class FilesystemTools:
    root: Path
    index: InMemoryHybridIndex | None = None
    plans: dict[str, MovePlan] = field(default_factory=dict)
    allow_apply: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def _safe(self, path: str) -> Path:
        candidate = (self.root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"path escapes corpus root: {path}") from exc
        return candidate

    def search_semantic(self, query: str, k: int = 8, filters: dict | None = None) -> list[dict]:
        if self.index is None:
            raise RuntimeError("semantic search requires an index")
        result = self.index.retrieve(query, n=k)
        hits = []
        for hit in result.hits[:k]:
            if filters:
                ft = filters.get("filetype")
                if ft and not hit.chunk.path.endswith(ft):
                    continue
            hits.append(
                {
                    "path": hit.chunk.path,
                    "chunk_id": hit.chunk.chunk_id,
                    "score": hit.score,
                    "text": hit.chunk.text[:500],
                }
            )
        return hits

    def search_lexical(self, query: str, k: int = 8) -> list[dict]:
        if self.index is None:
            raise RuntimeError("lexical search requires an index")
        hits = self.index.search_bm25(query, k=k)
        return [
            {"path": c.path, "chunk_id": c.chunk_id, "score": s, "text": c.text[:500]}
            for c, s in hits[:k]
        ]

    def read_file(self, path: str, byte_range: Sequence[int] | None = None) -> dict:
        target = self._safe(path)
        data = target.read_bytes()
        start, end = 0, len(data)
        if byte_range:
            start = max(0, int(byte_range[0]))
            end = min(len(data), int(byte_range[1]))
        return {
            "path": str(target.relative_to(self.root)),
            "start_byte": start,
            "end_byte": end,
            "text": data[start:end].decode("utf-8", errors="replace"),
        }

    def list_dir(self, path: str = ".") -> list[dict]:
        target = self._safe(path)
        if not target.is_dir():
            raise NotADirectoryError(path)
        entries = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            rel = str(child.relative_to(self.root))
            entries.append(
                {
                    "path": rel,
                    "is_dir": child.is_dir(),
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return entries

    def stat(self, paths: Sequence[str]) -> list[dict]:
        out = []
        for p in paths:
            target = self._safe(p)
            st = target.stat()
            out.append(
                {
                    "path": str(target.relative_to(self.root)),
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "is_dir": statmod.S_ISDIR(st.st_mode),
                    "mode": oct(st.st_mode),
                }
            )
        return out

    def propose_move(self, src: str, dst: str) -> dict:
        src_p = self._safe(src)
        dst_p = self._safe(dst)
        if not src_p.exists():
            raise FileNotFoundError(src)
        plan = MovePlan(
            plan_id=str(uuid.uuid4()),
            src=str(src_p.relative_to(self.root)),
            dst=str(dst_p.relative_to(self.root)),
            created_at=time.time(),
        )
        self.plans[plan.plan_id] = plan
        return {
            "plan_id": plan.plan_id,
            "src": plan.src,
            "dst": plan.dst,
            "status": "pending_approval",
            "note": "Destructive action. Call apply_plan only after human approval.",
        }

    def apply_plan(self, plan_id: str, approved: bool = False) -> dict:
        plan = self.plans.get(plan_id)
        if plan is None:
            raise KeyError(f"unknown plan {plan_id}")
        if plan.applied:
            return {"plan_id": plan_id, "status": "already_applied"}
        if not (approved or self.allow_apply):
            raise ApprovalRequired(plan_id)
        plan.approved = True
        src = self._safe(plan.src)
        dst = self._safe(plan.dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
        plan.applied = True
        return {"plan_id": plan_id, "status": "applied", "src": plan.src, "dst": plan.dst}

    def tool_specs(self) -> list[dict]:
        return [
            {
                "name": "search_semantic",
                "description": "Semantic / hybrid search over the corpus",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 8},
                        "filters": {"type": "object"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_lexical",
                "description": "BM25 / exact-token search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 8},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "read_file",
                "description": "Read a file, optionally a byte range",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "byte_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "list_dir",
                "description": "List a directory under the corpus root",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": "."}},
                },
            },
            {
                "name": "stat",
                "description": "Stat one or more paths",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "paths": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["paths"],
                },
            },
            {
                "name": "propose_move",
                "description": "Propose a file move. Returns a plan; never executes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                    },
                    "required": ["src", "dst"],
                },
            },
            {
                "name": "apply_plan",
                "description": "Apply a move plan. Requires approved=true from a human.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "approved": {"type": "boolean", "default": False},
                    },
                    "required": ["plan_id"],
                },
            },
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "search_semantic":
            return self.search_semantic(
                arguments["query"],
                k=int(arguments.get("k", 8)),
                filters=arguments.get("filters"),
            )
        if name == "search_lexical":
            return self.search_lexical(arguments["query"], k=int(arguments.get("k", 8)))
        if name == "read_file":
            return self.read_file(arguments["path"], arguments.get("byte_range"))
        if name == "list_dir":
            return self.list_dir(arguments.get("path", "."))
        if name == "stat":
            return self.stat(arguments["paths"])
        if name == "propose_move":
            return self.propose_move(arguments["src"], arguments["dst"])
        if name == "apply_plan":
            return self.apply_plan(
                arguments["plan_id"], approved=bool(arguments.get("approved", False))
            )
        raise KeyError(f"unknown tool {name}")
