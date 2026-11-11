"""Restricted sandbox for generated artifacts.

Industry shape: E2B / Daytona / Modal run user code off-host. We cannot
spawn those in CI, so this is an AST check + captured stdout in a local
child process. No pip. No os/subprocess. Fail loud on banned imports.

Academic shape: SciCode / AutoResearch only count a generation if it
*executes*. Token overlap is not a functional badge.

The code never runs in the calling process (an API or MCP server). run_sandboxed
checks it with ast_safe and runs it in a child Python process
(sandbox_runner.py): `python -I`, a new session, an empty temporary working
directory, and an environment reduced to PATH, HOME, LANG, LC_ALL, TZ and TMPDIR
plus one-thread settings for the maths libraries (the variables
app/engine/tools/python_exec.py keeps), so API keys, connection strings and the
approval settings are not in it. The child has CPU, address-space and file-size
limits, and its process group is killed at the time limit and after every run.
ast_safe is a block-list, not a boundary, and a child process is not OS
isolation: code that gets past the check runs as the caller's OS user and can
read what that user can read, the caller's start-up environment under /proc
included. SIGALRM is not used.
"""

from __future__ import annotations

import ast
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.artifacts.sandbox_runner import REPORT_MARKER

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


RUNNER = Path(__file__).with_name("sandbox_runner.py")
# What the child keeps of the caller's environment (the list app/engine/tools/python_exec.py
# keeps); everything else is dropped. Thread pools are pinned to one thread.
CHILD_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR")
CHILD_ENV_SET = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
DEFAULT_TIMEOUT_S = 10  # wall clock per run, interpreter start included; also the CPU limit
MEMORY_BYTES = 1024 * 1024 * 1024  # address-space limit of the child
FILE_BYTES = 16 * 1024 * 1024  # largest file the child may write, its stdout included
REPORT_TAIL_BYTES = 8 * 1024 * 1024  # the report is the last line of stdout and its fields are capped


def _child_env() -> dict[str, str]:
    env = {name: os.environ[name] for name in CHILD_ENV_KEEP if name in os.environ}
    env.update(CHILD_ENV_SET)
    return env


def _wait_unreaped(proc: subprocess.Popen, timeout_s: float) -> bool:
    """Wait for the child to exit without reaping it; False if it still runs at the deadline.

    An exited but unreaped child keeps its pid, and so its process group id, reserved,
    which makes the group kill that follows safe in both cases.
    """
    if not hasattr(os, "waitid"):  # macOS: wait() reaps, so the group id is only held by what is left in it
        try:
            proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            return False
    deadline = time.monotonic() + timeout_s
    while os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _kill_session(proc: subprocess.Popen) -> None:
    """SIGKILL the child's process group: the child and anything it started that is still in it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # start_new_session: the group id is the child's pid
    except (ProcessLookupError, PermissionError):
        pass  # nothing left in the group


def _tail(stream, limit: int) -> str:
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - limit))
    return stream.read().decode("utf-8", errors="replace")


def _read_report(stream) -> dict | None:
    for line in reversed(_tail(stream, REPORT_TAIL_BYTES).splitlines()):
        if line.startswith(REPORT_MARKER):
            try:
                return json.loads(line[len(REPORT_MARKER):])
            except ValueError:
                return None
    return None


def _run_child(request: dict, timeout_s: float) -> ExecResult:
    """Run one request ({"code", "builtins", "globals"}) in a child process and return its result.

    The request goes in, and stdout and stderr come back, through temporary files: no
    pipe to block on, and the child's file-size limit bounds what is read back.
    POSIX only (a new session, a process-group kill).
    """
    cpu_seconds = max(1, math.ceil(timeout_s))
    command = [sys.executable, "-I", str(RUNNER), str(cpu_seconds), str(MEMORY_BYTES), str(FILE_BYTES)]
    with tempfile.TemporaryDirectory(prefix="metanavit-sandbox-", ignore_cleanup_errors=True) as workdir, \
            tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        source.write(json.dumps(request).encode("utf-8"))
        source.seek(0)
        proc = subprocess.Popen(command, stdin=source, stdout=out, stderr=err, cwd=workdir,
                                env=_child_env(), start_new_session=True)
        try:
            finished = _wait_unreaped(proc, timeout_s)
        finally:
            _kill_session(proc)  # past the limit the child; in every case what it left running in its group
            proc.wait()
        if not finished:
            return ExecResult(ok=False, stdout="", stderr="", timed_out=True,
                              error=f"timed out after {timeout_s:g} s; the sandbox process and its process group were killed")
        report = _read_report(out)
        if report is not None:
            return ExecResult(ok=report.get("ok") is True, stdout=report.get("stdout", ""),
                              stderr=report.get("stderr", ""), error=report.get("error", ""))
        if proc.returncode < 0:
            try:
                reason = f"was killed by {signal.Signals(-proc.returncode).name}"
            except ValueError:
                reason = f"was killed by signal {-proc.returncode}"
            # The kernel sends SIGXCPU, then SIGKILL, at the CPU limit (soft and hard are equal).
            if -proc.returncode in (getattr(signal, "SIGXCPU", None), signal.SIGKILL):
                return ExecResult(ok=False, stdout="", stderr="", timed_out=True,
                                  error=f"the sandbox process {reason} (its CPU limit is {cpu_seconds} s)")
        else:
            reason = f"exited with status {proc.returncode}"
        # The child's own stderr is not passed on: a traceback there names server paths,
        # and the result can reach an MCP client.
        return ExecResult(ok=False, stdout="", stderr="", error=f"the sandbox process {reason} without a result")


def run_sandboxed(
    source: str,
    *,
    timeout_s: float | None = None,
    extra_globals: dict[str, Any] | None = None,
    builtins: Iterable[str] | None = None,
) -> ExecResult:
    """Check `source` with ast_safe, then run it in a child process (see the module docstring).

    `timeout_s` (default DEFAULT_TIMEOUT_S) is the wall-clock limit, interpreter start
    included, and the child's CPU limit (in whole seconds, rounded up). `extra_globals`
    must hold JSON values: they cross a process boundary. `builtins` narrows SAFE_BUILTINS
    to these names (the MCP expression tool passes its own list); a name outside
    SAFE_BUILTINS is a ValueError.
    """
    timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    names = set(SAFE_BUILTINS if builtins is None else builtins)
    if names - set(SAFE_BUILTINS):
        raise ValueError(f"builtins outside SAFE_BUILTINS: {sorted(names - set(SAFE_BUILTINS))}")
    reason = ast_safe(source)
    if reason:
        return ExecResult(ok=False, stdout="", stderr="", error=reason)
    # The child builds a new builtins dict from these names for every run.
    return _run_child({"code": source, "builtins": sorted(names), "globals": dict(extra_globals or {})}, timeout_s)
