"""Restricted in-process sandbox for generated artifacts.

Industry shape: E2B / Daytona / Modal run user code off-host. We cannot
spawn those in CI, so this is an AST allowlist + captured stdout.
No pip. No os/subprocess. Fail loud on banned imports.

Academic shape: SciCode / AutoResearch only count a generation if it
*executes*. Token overlap is not a functional badge.

Timeout is the caller's job (pytest, MCP). SIGALRM is not used — it
interacts badly with the test runner.
"""

from __future__ import annotations

import ast
import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

BANNED_MODULES = {
    "os",
    "subprocess",
    "socket",
    "pathlib",
    "sys",
    "shutil",
    "ctypes",
    "multiprocessing",
    "pickle",
    "importlib",
    "requests",
    "http",
    "urllib",
    "ftplib",
    "pty",
    "signal",
}
BANNED_NAMES = {"eval", "exec", "compile", "__import__", "input", "open", "breakpoint"}
ALLOWED_IMPORT_ROOTS = {
    "math",
    "json",
    "re",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "numpy",
    "pandas",
}

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "print": print,
    "sorted": sorted,
    "reversed": reversed,
    "round": round,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "Exception": Exception,
    "ValueError": ValueError,
    "AssertionError": AssertionError,
}


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    error: str = ""
    timed_out: bool = False

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "timed_out": self.timed_out,
        }


def ast_safe(source: str) -> str | None:
    """Return a reason string if unsafe, else None."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"syntax: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BANNED_MODULES or root not in ALLOWED_IMPORT_ROOTS:
                    return f"banned import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in BANNED_MODULES or root not in ALLOWED_IMPORT_ROOTS:
                return f"banned import {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_NAMES:
                return f"banned call {node.func.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"banned dunder {node.attr}"
    return None


def run_sandboxed(
    source: str,
    *,
    timeout_s: int = 2,
    extra_globals: dict[str, Any] | None = None,
) -> ExecResult:
    del timeout_s  # reserved for an out-of-process runner (E2B / subprocess)
    reason = ast_safe(source)
    if reason:
        return ExecResult(ok=False, stdout="", stderr="", error=reason)
    g: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    if extra_globals:
        g.update(extra_globals)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(source, "<artifact>", "exec"), g, g)  # noqa: S102  — ast-gated
        return ExecResult(ok=True, stdout=stdout.getvalue(), stderr=stderr.getvalue())
    except Exception as exc:
        return ExecResult(
            ok=False,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            error=f"{type(exc).__name__}: {exc}",
        )
