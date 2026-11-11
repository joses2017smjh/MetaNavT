"""
Child-process side of the artifact sandbox (standard library only).

app/artifacts/sandbox.py starts this file as a script with `python -I`, in a new
session and an empty temporary working directory, with a reduced environment. It
sends one JSON request on stdin, {"code", "builtins", "globals"}, and reads one JSON
report back from the last line of stdout that starts with REPORT_MARKER.

Before the code runs, this process lowers its own limits, soft and hard, so the code
cannot raise them again:
    RLIMIT_CPU    CPU seconds (the parent also kills the whole session at the
                  wall-clock limit)
    RLIMIT_AS     address space in bytes; an allocation past it raises MemoryError
    RLIMIT_FSIZE  largest file the code may write, its stdout included
    RLIMIT_CORE   0: no core files
The code then runs in a fresh namespace whose builtins are only the ones the request
names.

Usage: python -I sandbox_runner.py CPU_SECONDS MEMORY_BYTES FILE_BYTES < request.json
"""

import builtins
import contextlib
import io
import json
import sys

REPORT_MARKER = "__artifact_sandbox_report__ "
MAX_FIELD_CHARS = 100_000  # longest stdout, stderr or error sent back to the calling process


def _cap(text: str) -> str:
    if len(text) <= MAX_FIELD_CHARS:
        return text
    return text[:MAX_FIELD_CHARS] + f"\n[truncated: {len(text) - MAX_FIELD_CHARS} more characters]"


def limit_resources(cpu_seconds: int, memory_bytes: int, file_bytes: int) -> None:
    """Lower this process's limits (soft and hard) before the code runs."""
    try:
        import resource
    except ImportError:  # not POSIX: only the parent's wall-clock limit applies
        return
    for name, value in (
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_AS", memory_bytes),
        ("RLIMIT_FSIZE", file_bytes),
        ("RLIMIT_CORE", 0),
    ):
        which = getattr(resource, name, None)
        if which is None:
            continue
        _, hard = resource.getrlimit(which)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)  # an unprivileged process can only lower a hard limit
        resource.setrlimit(which, (value, value))


def run(code: str, builtin_names: list, extra_globals: dict) -> dict:
    """Run the code with only the named builtins; the fields of sandbox.ExecResult except timed_out."""
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in builtin_names}}
    namespace.update(extra_globals)
    stdout, stderr = io.StringIO(), io.StringIO()
    report = {"ok": True, "error": ""}
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(code, "<artifact>", "exec"), namespace, namespace)  # in this child process only
    except BaseException as exc:  # SystemExit and KeyboardInterrupt included: always report
        report = {"ok": False, "error": _cap(f"{type(exc).__name__}: {exc}")}
    report["stdout"] = _cap(stdout.getvalue())
    report["stderr"] = _cap(stderr.getvalue())
    return report


def main(argv: list) -> int:
    cpu_seconds, memory_bytes, file_bytes = (int(value) for value in argv[1:4])
    limit_resources(cpu_seconds, memory_bytes, file_bytes)
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    report = run(request["code"], request["builtins"], request["globals"])
    # Start a new line: code that wrote to file descriptor 1 directly may have left a partial line.
    sys.stdout.write("\n" + REPORT_MARKER + json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
