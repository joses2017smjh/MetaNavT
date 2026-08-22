"""Filesystem MCP tools.

search_semantic, search_lexical, read_file, list_dir, stat,
propose_move / apply_plan,
collect_run_artifact, propose_artifact / apply_artifact,
propose_patch / apply_patch, exec_sandboxed.

Destructive writes never auto-fire: apply_* requires approved=true.
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
    artifacts: dict = field(default_factory=dict)
    patches: dict = field(default_factory=dict)
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

    def collect_run_artifact(self, run_id: str) -> dict:
        from app.artifacts.pipeline import ArtifactAgent
        from app.artifacts.manifest import collect_run_artifact as collect

        if self.index is not None:
            return ArtifactAgent(self.index).collect_run(self.root, run_id).as_dict()
        return collect(self.root, run_id).as_dict()

    def propose_artifact(self, query: str) -> dict:
        from app.artifacts.pipeline import ArtifactAgent

        if self.index is None:
            raise RuntimeError("propose_artifact requires an index")
        prop = ArtifactAgent(self.index).produce(query)
        self.artifacts[prop.plan_id] = prop
        return prop.as_dict()

    def apply_artifact(self, plan_id: str, approved: bool = False) -> dict:
        prop = self.artifacts.get(plan_id)
        if prop is None:
            raise KeyError(f"unknown artifact {plan_id}")
        if prop.applied:
            return {"plan_id": plan_id, "status": "already_applied"}
        if not (approved or self.allow_apply):
            raise ApprovalRequired(plan_id)
        if not prop.exec_result.ok:
            raise RuntimeError(
                f"refusing to write an artifact that failed the sandbox: {prop.exec_result.error}"
            )
        dest = self._safe(prop.spec.file_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(prop.code)
        prop.approved = True
        prop.applied = True
        return {
            "plan_id": plan_id,
            "status": "applied",
            "path": str(dest.relative_to(self.root)),
            "kind": prop.kind,
        }

    def propose_patch(self, path: str, old: str, new: str) -> dict:
        from app.artifacts.patch import FilePatch, unified_hunk

        target = self._safe(path)
        if not target.exists():
            raise FileNotFoundError(path)
        plan_id = str(uuid.uuid4())
        patch = FilePatch(path=str(target.relative_to(self.root)), old=old, new=new)
        self.patches[plan_id] = patch
        return {
            "plan_id": plan_id,
            "path": patch.path,
            "status": "pending_approval",
            "diff": unified_hunk(patch.path, old, new),
            "note": "Call apply_patch only after human approval. Never auto-applies.",
        }

    def apply_patch(self, plan_id: str, approved: bool = False) -> dict:
        from app.artifacts.patch import apply_search_replace

        patch = self.patches.get(plan_id)
        if patch is None:
            raise KeyError(f"unknown patch {plan_id}")
        if not (approved or self.allow_apply):
            raise ApprovalRequired(plan_id)
        target = self._safe(patch.path)
        text = target.read_text(encoding="utf-8")
        target.write_text(apply_search_replace(text, patch))
        return {"plan_id": plan_id, "status": "applied", "path": patch.path}

    def exec_sandboxed(self, code: str) -> dict:
        from app.artifacts.sandbox import run_sandboxed

        return run_sandboxed(code).as_dict()

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
            {
                "name": "collect_run_artifact",
                "description": "ACM-style reproducibility pack for a run id (config, code, log, paper).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
            {
                "name": "propose_artifact",
                "description": "Spec → generate → sandbox. Returns a plan; never writes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "apply_artifact",
                "description": "Write a proposed artifact. Requires approved=true. Refuses failed sandbox runs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "approved": {"type": "boolean", "default": False},
                    },
                    "required": ["plan_id"],
                },
            },
            {
                "name": "propose_patch",
                "description": "SEARCH/REPLACE patch plan. Never writes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["path", "old", "new"],
                },
            },
            {
                "name": "apply_patch",
                "description": "Apply a patch plan. Requires approved=true.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "approved": {"type": "boolean", "default": False},
                    },
                    "required": ["plan_id"],
                },
            },
            {
                "name": "exec_sandboxed",
                "description": "AST-gated Python exec. No os/subprocess/pip.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
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
        if name == "collect_run_artifact":
            return self.collect_run_artifact(str(arguments["run_id"]))
        if name == "propose_artifact":
            return self.propose_artifact(arguments["query"])
        if name == "apply_artifact":
            return self.apply_artifact(
                arguments["plan_id"], approved=bool(arguments.get("approved", False))
            )
        if name == "propose_patch":
            return self.propose_patch(arguments["path"], arguments["old"], arguments["new"])
        if name == "apply_patch":
            return self.apply_patch(
                arguments["plan_id"], approved=bool(arguments.get("approved", False))
            )
        if name == "exec_sandboxed":
            return self.exec_sandboxed(arguments["code"])
        raise KeyError(f"unknown tool {name}")
