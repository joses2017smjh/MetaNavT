"""
Child-process side of the Python execution tool (standard library only).

app/engine/tools/python_exec.py starts this file as a script with `python -I`
in a new session, sends the model-written code on stdin and reads one JSON
report back from the last line of stdout that starts with REPORT_MARKER.

Before the code runs, this process lowers its own limits, soft and hard, so the
code cannot raise them again:
    RLIMIT_CPU    CPU seconds (the parent also kills the whole session at the
                  wall-clock limit)
    RLIMIT_AS     address space in bytes; an allocation past it raises MemoryError
    RLIMIT_FSIZE  largest file the code may write, its stdout included
    RLIMIT_CORE   0: no core files

This is a separate process with limits, not OS isolation: the code still runs
as the API's OS user and can read whatever that user can read.

Usage: python -I python_exec_runner.py CPU_SECONDS MEMORY_BYTES FILE_BYTES FIGURE_PATH
"""

import contextlib
import io
import json
import sys

REPORT_MARKER = "__python_exec_report__ "
MAX_FIELD_CHARS = 100_000  # longest output, error or result sent back to the API process


def _cap(text: str) -> str:
    if len(text) <= MAX_FIELD_CHARS:
        return text
    return text[:MAX_FIELD_CHARS] + f"\n[truncated: {len(text) - MAX_FIELD_CHARS} more characters]"


def limit_resources(cpu_seconds: int, memory_bytes: int, file_bytes: int) -> None:
    """Lower this process's limits (soft and hard) before any model-written code runs."""
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


def run(code: str, figure_path: str) -> dict:
    """Run the code in a fresh namespace; the same result fields the tool always returned."""
    stdout_capture = io.StringIO()  # Captures print() output
    stderr_capture = io.StringIO()  # Captures error messages
    report = {"output": "", "error": "", "result": None}
    try:
        namespace = {}
        with contextlib.redirect_stdout(stdout_capture):
            with contextlib.redirect_stderr(stderr_capture):
                exec(code, namespace)  # in this child process, never in the API process
        report["output"] = _cap(stdout_capture.getvalue())
        # If the code defined a variable named "result", include it in the output
        if "result" in namespace:
            report["result"] = _cap(str(namespace["result"]))
        # If the code created a matplotlib figure, save it next to the saved code
        if "plt" in namespace:
            try:
                namespace["plt"].savefig(figure_path)
                report["figure_path"] = figure_path
            except Exception as fig_error:
                report["figure_error"] = _cap(str(fig_error))
    except BaseException as e:  # SystemExit and KeyboardInterrupt included: always report
        report["error"] = _cap(f"Error: {str(e) or type(e).__name__}\n{stderr_capture.getvalue()}")
    return report


def main(argv: list) -> int:
    cpu_seconds, memory_bytes, file_bytes = (int(value) for value in argv[1:4])
    figure_path = argv[4]
    limit_resources(cpu_seconds, memory_bytes, file_bytes)
    code = sys.stdin.buffer.read().decode("utf-8")
    report = run(code, figure_path)
    # Start a new line: the code, or a process it ran, may have written to file descriptor 1
    # directly and left a line without a newline, which would hide the marker.
    sys.stdout.write("\n" + REPORT_MARKER + json.dumps(report) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
