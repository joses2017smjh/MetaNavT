"""Filesystem MCP tools.

search_semantic, search_lexical, read_file, list_dir, stat,
propose_move / apply_plan,
collect_run_artifact, propose_artifact / apply_artifact,
propose_patch / apply_patch, exec_sandboxed,
inspect_spreadsheet, propose_visualization / apply_visualization.

Mutations require a short-lived, content-bound operator capability.
Caller-supplied approval booleans never authorize changes.
"""

from __future__ import annotations

import json
import ast
import hashlib
import threading
import tempfile
from functools import wraps
import os
import stat as statmod
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.retrieval.hybrid import InMemoryHybridIndex
from app.mcp.approvals import ApprovalError, ApprovalVerifier, canonical, verifier_from_env
from app.mcp.secure_io import fingerprint, relative, create_file, replace_file, move_file


class ApprovalRequired(Exception):
    def __init__(self, plan_id: str):
        super().__init__(f"Plan {plan_id} requires a trusted operator approval capability")
        self.plan_id = plan_id


def _serialized(method):
    @wraps(method)
    def wrapped(self, plan_id, *args, **kwargs):
        with self._mutation_lock:
            try:
                result = method(self, plan_id, *args, **kwargs)
            except Exception as exc:
                self._audit(plan_id, method.__name__, "rejected", type(exc).__name__)
                raise
            self._audit(plan_id, method.__name__, result.get("status", "applied"))
            return result
    return wrapped


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
    visualizations: dict = field(default_factory=dict)
    allow_apply: bool = False  # retained for constructor compatibility; True is refused
    approval_verifier: ApprovalVerifier | None = None
    audit_events: list[dict] = field(default_factory=list, init=False)
    _reviews: dict[str, str] = field(default_factory=dict, init=False)
    _session: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    _mutation_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if self.allow_apply:
            raise ApprovalError("allow_apply bypass was removed; configure operator capabilities")
        if self.approval_verifier is None:
            self.approval_verifier = verifier_from_env(self.root)

    def _safe(self, path: str) -> Path:
        if path in {"", "."}:
            return self.root
        rel = relative(self.root, path)
        candidate = self.root / rel
        current = self.root
        for component in Path(rel).parts:
            current = current / component
            if current.is_symlink():
                raise PermissionError("symlink paths are refused")
        return candidate

    def _audit(self, plan_id: str, action: str, result: str, detail: str = "") -> None:
        event = {"plan_id": plan_id, "action": action, "result": result,
                 "detail": detail, "timestamp": time.time()}
        # No grant tokens, key material or file content in audit rows.
        if self.approval_verifier is not None:
            self.approval_verifier.audit(plan_id, action, result, detail)
        self.audit_events.append(event)

    def _describe(self, plan_id: str) -> dict:
        if plan_id in self.plans:
            p = self.plans[plan_id]
            action, payload, reads, creates = "apply_plan", {"src": p.src, "dst": p.dst}, [p.src], [p.dst]
        elif plan_id in self.patches:
            p = self.patches[plan_id]
            action, payload, reads, creates = "apply_patch", p.as_dict(), [p.path], []
        elif plan_id in self.artifacts:
            p = self.artifacts[plan_id]
            action = "apply_artifact"
            payload = {"spec": p.spec.as_dict(), "code": p.code, "sandbox_ok": p.exec_result.ok, "kind": p.kind}
            reads, creates = [], [p.spec.file_path]
        elif plan_id in self.visualizations:
            p = self.visualizations[plan_id]
            action, payload, reads, creates = "apply_visualization", p.as_dict(), [p.source_path], [p.script_path, p.chart_path]
            payload.pop("status", None)
        else:
            raise KeyError(f"unknown plan {plan_id}")
        states = [fingerprint(self.root, path) for path in reads]
        if not all(state["exists"] for state in states):
            raise FileNotFoundError("reviewed source is missing")
        destinations = [fingerprint(self.root, path) for path in creates]
        if any(state["exists"] for state in destinations):
            raise FileExistsError("destination already exists; overwrites require a separate reviewed patch")
        st = self.root.stat()
        return {"schema": "metanavit-reviewed-action-v1", "session": self._session,
                "root_identity": hashlib.sha256(f"{self.root}:{st.st_dev}:{st.st_ino}".encode()).hexdigest(),
                "plan_id": plan_id, "action": action, "payload": payload,
                "sources": states, "destinations": destinations}

    def _register_review(self, plan_id: str) -> dict:
        self._reviews[plan_id] = canonical(self._describe(plan_id)).decode()
        return self.review_plan(plan_id)

    def review_plan(self, plan_id: str, **options) -> dict:
        """Read-only host/operator helper, not an approval issuer or MCP tool.

        A returned JSON object is a copy. A reviewer may sign visualization
        options explicitly; these options are covered by the capability hash.
        """
        if plan_id not in self._reviews:
            raise KeyError(f"unknown reviewed plan {plan_id}")
        review = json.loads(self._reviews[plan_id])
        if review["action"] == "apply_visualization":
            defaults = {"chart_type": review["payload"]["recommended_chart"], "execute": True, "backend": "auto"}
            if set(options) - set(defaults):
                raise ValueError("unknown visualization review option")
            defaults.update(options)
            if type(defaults["execute"]) is not bool or defaults["chart_type"] not in {"bar", "line", "dot", "histogram"} or defaults["backend"] not in {"auto", "matlab", "octave"}:
                raise ValueError("invalid visualization options")
            review["options"] = defaults
            # Bind the exact script bytes for the selected options, including
            # an override, so the operator sees precisely what will be written.
            from app.artifacts.visualization import generate_matlab
            p = self.visualizations[plan_id]
            script = generate_matlab(source_path=p.source_path,
                                     headers=[column.name for column in p.columns],
                                     columns=p.columns, group_by=p.group_by, value=p.value,
                                     operation=p.operation, chart_type=defaults["chart_type"],
                                     chart_path=p.chart_path, baseline=p.baseline)
            review["output_content"] = {"script_path": p.script_path, "script": script,
                                        "sha256": hashlib.sha256(script.encode()).hexdigest()}
        elif options:
            raise ValueError("this action has no review options")
        return review

    def _authorize(self, plan_id: str, action: str, approval_token: str | None, **options) -> dict:
        if self.approval_verifier is None or not approval_token:
            raise ApprovalRequired(plan_id)
        review = self.review_plan(plan_id, **options)
        if review["action"] != action:
            raise ApprovalError("capability action mismatch")
        # Content and operation must still match the immutable proposal.
        if canonical(self._describe(plan_id)).decode() != self._reviews[plan_id]:
            raise ApprovalError("reviewed plan or file state is stale; propose again")
        self.approval_verifier.consume(approval_token, review)
        self._audit(plan_id, action, "authorized")  # durable intent before touching files
        return review

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
        review = self._register_review(plan.plan_id)
        return {
            "plan_id": plan.plan_id,
            "src": plan.src,
            "dst": plan.dst,
            "status": "pending_approval",
            "note": "Have an operator inspect and sign the review; approved=true does not authorize writes.",
            "review": review,
        }

    @_serialized
    def apply_plan(self, plan_id: str, approved: bool = False, *, approval_token: str | None = None) -> dict:
        review = self._authorize(plan_id, "apply_plan", approval_token)
        plan = self.plans[plan_id]
        move_file(self.root, plan.src, plan.dst, review["sources"][0])
        plan.approved = plan.applied = True
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
        review = self._register_review(prop.plan_id)
        return {**prop.as_dict(), "review": review}

    @_serialized
    def apply_artifact(self, plan_id: str, approved: bool = False, *, approval_token: str | None = None) -> dict:
        self._authorize(plan_id, "apply_artifact", approval_token)
        prop = self.artifacts[plan_id]
        if not prop.exec_result.ok:
            raise RuntimeError(f"refusing artifact that failed its sandbox: {prop.exec_result.error}")
        create_file(self.root, prop.spec.file_path, prop.code.encode())
        prop.approved = prop.applied = True
        return {"plan_id": plan_id, "status": "applied", "path": prop.spec.file_path, "kind": prop.kind}

    def propose_patch(self, path: str, old: str, new: str) -> dict:
        from app.artifacts.patch import FilePatch, unified_hunk

        target = self._safe(path)
        if not target.exists():
            raise FileNotFoundError(path)
        plan_id = str(uuid.uuid4())
        patch = FilePatch(path=str(target.relative_to(self.root)), old=old, new=new)
        self.patches[plan_id] = patch
        review = self._register_review(plan_id)
        return {
            "plan_id": plan_id,
            "path": patch.path,
            "status": "pending_approval",
            "diff": unified_hunk(patch.path, old, new),
            "note": "An operator must sign the immutable review before applying.",
            "review": review,
        }

    @_serialized
    def apply_patch(self, plan_id: str, approved: bool = False, *, approval_token: str | None = None) -> dict:
        from app.artifacts.patch import apply_search_replace
        review = self._authorize(plan_id, "apply_patch", approval_token)
        patch = self.patches[plan_id]
        text = self._safe(patch.path).read_bytes().decode("utf-8")
        result = apply_search_replace(text, patch)
        replace_file(self.root, patch.path, result.encode(), review["sources"][0])
        return {"plan_id": plan_id, "status": "applied", "path": patch.path}

    def exec_sandboxed(self, code: str) -> dict:
        from app.artifacts.sandbox import run_sandboxed

        # The legacy artifact AST runner permits library file readers. Exposing
        # those as an agent tool would let it read the operator's private key.
        # This MCP tool therefore accepts only a no-import/no-attribute subset.
        # It remains a correctness runner, not a CPU/memory/OS security sandbox.
        try:
            if not isinstance(code, str) or len(code) > 65536:
                raise ValueError("code exceeds the expression-tool limit")
            tree = ast.parse(code)
            if any(isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute)) for node in ast.walk(tree)):
                raise ValueError("imports and attribute access are unavailable in the MCP expression tool")
        except (ValueError, SyntaxError) as exc:
            return {"ok": False, "stdout": "", "stderr": "", "error": str(exc), "timed_out": False}
        return run_sandboxed(code).as_dict()

    def inspect_spreadsheet(self, path: str) -> dict:
        from app.artifacts.visualization import inspect_spreadsheet

        target = self._safe(path)
        report = inspect_spreadsheet(target)
        rows = report.pop("rows", [])
        report["path"] = str(target.relative_to(self.root))
        report["preview"] = rows[:5]
        return report

    def propose_visualization(
        self,
        path: str,
        question: str,
        group_by: str | None = None,
        value: str | None = None,
        operation: str | None = None,
        chart_type: str | None = None,
    ) -> dict:
        from app.artifacts.visualization import propose_visualization

        target = self._safe(path)
        relative = str(target.relative_to(self.root))
        plan = propose_visualization(
            self.root,
            relative,
            question,
            group_by=group_by,
            value=value,
            operation=operation,
            chart_type=chart_type,
        )
        self.visualizations[plan.plan_id] = plan
        review = self._register_review(plan.plan_id)
        return {**plan.as_dict(), "review": review}

    @_serialized
    def apply_visualization(
        self,
        plan_id: str,
        approved: bool = False,
        chart_type: str | None = None,
        execute: bool = True,
        backend: str = "auto",
        approval_token: str | None = None,
    ) -> dict:
        from app.artifacts.visualization import execute_matlab

        plan = self.visualizations.get(plan_id)
        if plan is None:
            raise KeyError(f"unknown visualization {plan_id}")
        selected = chart_type or plan.recommended_chart
        review = self._authorize(plan_id, "apply_visualization", approval_token,
                                 chart_type=selected, execute=execute, backend=backend)
        plan.matlab_code = review["output_content"]["script"]
        script = self._safe(plan.script_path)
        chart = self._safe(plan.chart_path)
        create_file(self.root, plan.script_path, plan.matlab_code.encode())
        # Render in a disposable tree, then publish via exclusive creation. The
        # external backend is not an OS sandbox, and its runtime is not evaluated
        # by the deterministic authorization benchmark.
        if execute:
            with tempfile.TemporaryDirectory(prefix="metanavit-chart-") as temporary:
                staging = Path(temporary)
                create_file(staging, plan.source_path, self._safe(plan.source_path).read_bytes())
                create_file(staging, plan.script_path, plan.matlab_code.encode())
                staged_chart = staging / plan.chart_path
                staged_chart.parent.mkdir(parents=True, exist_ok=True)
                execution = execute_matlab(staging, plan.script_path, chart_path=plan.chart_path, backend=backend)
                if execution["ok"] and staged_chart.is_file():
                    create_file(self.root, plan.chart_path, staged_chart.read_bytes())
        else:
            execution = {
                "ok": None,
                "backend": None,
                "stdout": "",
                "stderr": "execution not requested",
                "returncode": None,
            }
        plan.approved = True
        plan.applied = True
        plan.status = (
            "applied"
            if not execute or (execution["ok"] and chart.is_file())
            else "script_written_chart_failed"
        )
        return {
            "plan_id": plan_id,
            "status": plan.status,
            "selected_chart": selected,
            "script_path": str(script.relative_to(self.root)),
            "chart_path": str(chart.relative_to(self.root)),
            "chart_exists": chart.is_file(),
            "execution": execution,
        }

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
                "description": "Apply a move plan. Requires an operator-issued capability bound to the reviewed action.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "approval_token": {"type": "string"},
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
                "description": "Spec → template generation → AST correctness check. Returns a plan; never writes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "apply_artifact",
                "description": "Write a proposed artifact. Requires an operator-issued capability. Refuses failed sandbox runs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "approval_token": {"type": "string"},
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
                "description": "Apply a patch plan. Requires an operator-issued capability.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "approval_token": {"type": "string"},
                    },
                    "required": ["plan_id"],
                },
            },
            {
                "name": "exec_sandboxed",
                "description": "Run a no-import, no-attribute Python expression subset; no OS isolation or resource guarantees.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
            {
                "name": "inspect_spreadsheet",
                "description": "Profile spreadsheet columns and return a five-row preview. Never writes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "propose_visualization",
                "description": "Inspect, aggregate, and recommend a MATLAB chart. Returns questions; never writes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "question": {"type": "string"},
                        "group_by": {"type": "string"},
                        "value": {"type": "string"},
                        "operation": {
                            "type": "string",
                            "enum": ["mean", "sum", "min", "max", "count"],
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "dot", "histogram"],
                        },
                    },
                    "required": ["path", "question"],
                },
            },
            {
                "name": "apply_visualization",
                "description": "Use a trusted capability covering chart/execute/backend options to write or render.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "approval_token": {"type": "string"},
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "dot", "histogram"],
                        },
                        "execute": {"type": "boolean", "default": True},
                        "backend": {
                            "type": "string",
                            "enum": ["auto", "octave", "matlab"],
                            "default": "auto",
                        },
                    },
                    "required": ["plan_id"],
                },
            },
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name.startswith("apply_") and "approved" in arguments:
            self._audit(str(arguments.get("plan_id", "")), name, "rejected", "untrusted_approval_argument")
            raise ApprovalRequired(str(arguments.get("plan_id", "")))
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
                arguments["plan_id"], approval_token=arguments.get("approval_token")
            )
        if name == "collect_run_artifact":
            return self.collect_run_artifact(str(arguments["run_id"]))
        if name == "propose_artifact":
            return self.propose_artifact(arguments["query"])
        if name == "apply_artifact":
            return self.apply_artifact(
                arguments["plan_id"], approval_token=arguments.get("approval_token")
            )
        if name == "propose_patch":
            return self.propose_patch(arguments["path"], arguments["old"], arguments["new"])
        if name == "apply_patch":
            return self.apply_patch(
                arguments["plan_id"], approval_token=arguments.get("approval_token")
            )
        if name == "exec_sandboxed":
            return self.exec_sandboxed(arguments["code"])
        if name == "inspect_spreadsheet":
            return self.inspect_spreadsheet(arguments["path"])
        if name == "propose_visualization":
            return self.propose_visualization(
                arguments["path"],
                arguments["question"],
                group_by=arguments.get("group_by"),
                value=arguments.get("value"),
                operation=arguments.get("operation"),
                chart_type=arguments.get("chart_type"),
            )
        if name == "apply_visualization":
            return self.apply_visualization(
                arguments["plan_id"],
                approval_token=arguments.get("approval_token"),
                chart_type=arguments.get("chart_type"),
                execute=arguments.get("execute", True),
                backend=arguments.get("backend", "auto"),
            )
        raise KeyError(f"unknown tool {name}")
