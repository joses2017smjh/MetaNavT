"""
Python Execution Tool

Provides the python engine tool with the capability of executing Python code.
This is used to complete specific tasks like analyzing data and solving problems.
Dynamically generated python code can be better than an LLM models capabilities. 

The tool runs model-written code, so it is off unless switched on explicitly
(see python_exec_enabled):
    ENABLE_PYTHON_EXEC=true         offer the tool to the chat agents
    METANAVIT_API_KEY=<key>         required as well; without it the tool stays off
    PYTHON_EXEC_TIMEOUT_SECONDS     wall-clock (and CPU) limit per call
    PYTHON_EXEC_MAX_MEMORY_MB       address-space limit per call
When it is on, each call runs the code in a child process (python_exec_runner.py),
never in the API process: a new session, its own environment reduced to PATH, HOME,
locale, TZ and TMPDIR plus one-thread settings for the maths libraries (the API key
and the database string are not in it), CPU, memory and file-size limits, and its
process group is killed at the time limit and after every call. A child process is
not OS isolation: it runs as the API's OS user, so it can still read what that user
can read (the API process's start-up environment under /proc, a .env file in the
working directory), and a process it starts in a session of its own leaves the group
and is not killed.
"""

# Standard library imports for file, system and text operations
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path  
from typing import Dict, Any, List, Optional  
from llama_index.core.tools import FunctionTool
from app.api.security import API_KEY_ENV, configured_api_key
from app.engine.tools import create_tool_callback
from app.engine.tools.python_exec_runner import REPORT_MARKER

# Import sandbox environment functions for sandboxed execution
from app.engine.sandbox.sandbox_env import setup_sandbox, teardown_sandbox, get_sandbox_merged_dir

# Initialize logger to record errors and information
logger = logging.getLogger(__name__)

EXEC_FLAG_ENV = "ENABLE_PYTHON_EXEC"
TIMEOUT_ENV = "PYTHON_EXEC_TIMEOUT_SECONDS"
MEMORY_ENV = "PYTHON_EXEC_MAX_MEMORY_MB"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY_MB = 1024
MAX_FILE_BYTES = 64 * 1024 * 1024  # largest file the code may write, its stdout included
RUNNER = Path(__file__).with_name("python_exec_runner.py")
# Variables the child keeps from the API's environment; everything else (API key,
# connection strings, provider keys) is dropped. Thread pools are pinned to one thread.
CHILD_ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR")
CHILD_ENV_SET = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
REPORT_TAIL_BYTES = 8 * 1024 * 1024  # the report is the last line of stdout and its fields are capped
STDERR_TAIL_BYTES = 4096


def python_exec_enabled() -> bool:
    """True only when ENABLE_PYTHON_EXEC is 1/true/yes and the API requires a key.

    The flag alone is not enough: with METANAVIT_API_KEY unset the API is open,
    so the tool stays off and a warning says why.
    """
    if os.getenv(EXEC_FLAG_ENV, "false").strip().lower() not in {"1", "true", "yes"}:
        return False
    if configured_api_key() is None:
        logger.warning(
            f"{EXEC_FLAG_ENV} is set but {API_KEY_ENV} is not: the code-execution tool stays off "
            f"(it runs model-written code, so the API serving it must require the key)"
        )
        return False
    return True


def exec_limits() -> Dict[str, float]:
    """Per-call limits from the environment; an unset, unparsable, non-finite or non-positive value keeps the default."""
    def positive(name: str, default: float) -> float:
        try:
            value = float(os.getenv(name, "") or default)
        except ValueError:
            return default
        return value if math.isfinite(value) and value > 0 else default

    return {
        "timeout_seconds": positive(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS),
        "memory_bytes": int(positive(MEMORY_ENV, DEFAULT_MEMORY_MB) * 1024 * 1024),
    }


def _child_env() -> Dict[str, str]:
    env = {name: os.environ[name] for name in CHILD_ENV_KEEP if name in os.environ}
    env.update(CHILD_ENV_SET)
    return env


def _wait_unreaped(proc: subprocess.Popen, timeout_seconds: float) -> bool:
    """Wait for the child to exit without reaping it; False if it is still running at the deadline.

    An exited but unreaped child keeps its pid, and so its process group id, reserved,
    which makes the group kill that follows safe in both cases.
    """
    if not hasattr(os, "waitid"):  # macOS: wait() reaps, so the group id is only held by what is left in it
        try:
            proc.wait(timeout=timeout_seconds)
            return True
        except subprocess.TimeoutExpired:
            return False
    deadline = time.monotonic() + timeout_seconds
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


def _read_report(stream) -> Optional[Dict[str, Any]]:
    for line in reversed(_tail(stream, REPORT_TAIL_BYTES).splitlines()):
        if line.startswith(REPORT_MARKER):
            try:
                return json.loads(line[len(REPORT_MARKER):])
            except ValueError:
                return None
    return None


def run_in_child(code: str, figure_path: Path, timeout_seconds: float, memory_bytes: int) -> Dict[str, Any]:
    """Run code in a child Python process and return its report (output, error, result, figure).

    The code goes in, and stdout and stderr come back, through temporary files: no pipe
    to block on, and the child's file-size limit bounds what the API process reads back.
    POSIX only (a new session, a process-group kill).
    """
    cpu_seconds = max(1, math.ceil(timeout_seconds))
    command = [sys.executable, "-I", str(RUNNER), str(cpu_seconds), str(memory_bytes), str(MAX_FILE_BYTES), str(figure_path)]
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        source.write(code.encode("utf-8"))
        source.seek(0)
        proc = subprocess.Popen(command, stdin=source, stdout=out, stderr=err, env=_child_env(), start_new_session=True)
        try:
            timed_out = not _wait_unreaped(proc, timeout_seconds)
        finally:
            _kill_session(proc)  # past the limit the child; in every case what it left running in its group
            proc.wait()
        if timed_out:
            return {
                "output": "",
                "error": f"Error: execution exceeded the {timeout_seconds:g} s time limit; "
                         f"the child process and its process group were killed",
                "result": None,
            }
        report = _read_report(out)
        if report is not None:
            return report
        if proc.returncode < 0:
            try:
                reason = f"was killed by {signal.Signals(-proc.returncode).name}"
            except ValueError:
                reason = f"was killed by signal {-proc.returncode}"
            if -proc.returncode in (signal.SIGXCPU, signal.SIGKILL):
                reason += f" (its CPU limit is {cpu_seconds} s)"
        else:
            reason = f"exited with status {proc.returncode}"
        return {
            "output": "",
            "error": f"Error: the child process {reason} without a result\n{_tail(err, STDERR_TAIL_BYTES)}",
            "result": None,
        }


class PythonExecTool:
    """Tool for executing Python code safely with constraints"""
    
    @classmethod
    def setup_workspace(cls, workspace_dir: Optional[str] = None) -> Path:
        """
        Set up and return the workspace directory
        
        This creates a directory where execution artifacts (saved code files,
        generated figures) will be stored.
        
        Args:
            workspace_dir: Custom directory path, or None to use default
            
        Returns:
            Path object pointing to the workspace directory
        """
        # Use provided directory or default to env variable or /tmp
        workspace_path = workspace_dir or os.path.join(os.getenv("STORAGE_DIR", "/tmp"), "dev_workspace")
        workspace_dir_path = Path(workspace_path)
        # Create directory if it doesn't exist
        workspace_dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PythonExecTool initialized with workspace at {workspace_dir_path}")
        return workspace_dir_path
    
    
    @classmethod
    def execute_code(cls, 
                     code: str,
                     workspace_dir: Optional[str] = None,
                     use_sandbox: bool = False) -> Dict[str, Any]:
        """
        Execute Python code safely and return results.
        
        This is the main method that runs user code in a controlled environment,
        captures all output, and handles errors gracefully. The code runs in a
        separate Python process with a time limit and a memory limit.
        
        Args:
            code: Python code to execute
            workspace_dir: Directory for saving execution artifacts
            use_sandbox: Whether to execute the code in a sandboxed OverlayFS environment
            
        Returns:
            Dictionary containing execution results with these keys:
            - output: Standard output captured during execution
            - error: Error message if execution failed
            - result: Value of 'result' variable if defined in the code
            - figure_path: Path to saved matplotlib figure if created
        """
        logger.info('Executing Python code')
        # If sandboxing is enabled, set up the sandbox environment
        if use_sandbox:
            setup_sandbox(dry_run=False)
            workspace_dir_path = Path(get_sandbox_merged_dir())
        else:
            # Set up workspace directory to store files
            workspace_dir_path = cls.setup_workspace(workspace_dir)
        
        result = {"output": "", "error": "", "result": None}  # Initialize result dict
        
        try:
            # Save code to file for debugging purposes
            # Using id(code) to create a unique filename
            temp_file = workspace_dir_path / f"exec_{id(code)}.py"
            with open(temp_file, "w") as f:
                f.write(code)
                
            # Run the code in a child process (python_exec_runner.py). It captures
            # stdout and stderr, reads the 'result' variable and saves a matplotlib
            # figure, as the in-process version did.
            limits = exec_limits()
            report = run_in_child(
                code,
                figure_path=workspace_dir_path / f"figure_{id(code)}.png",
                timeout_seconds=limits["timeout_seconds"],
                memory_bytes=limits["memory_bytes"],
            )
            result["output"] = report.get("output", "")
            result["error"] = report.get("error", "")
            result["result"] = report.get("result")
            if report.get("figure_path"):
                result["figure_path"] = report["figure_path"]
            if report.get("figure_error"):
                # If figure saving fails, log but don't crash
                logger.warning(f"Could not save figure: {report['figure_error']}")
            if result["error"]:
                logger.error(f"Python execution error: {result['error']}")
            return result
            
        except Exception as e:
            # If anything goes wrong around the execution, capture the error
            error_text = f"Error: {str(e)}\n"
            result["error"] = error_text
            logger.error(f"Python execution error: {error_text}")
            return result
        
        finally:
            # Clean up sandboxed environment if one was created
            if use_sandbox:
                teardown_sandbox()


def get_tools(**kwargs):
    """
    Create and return the Python execution tool.
    
    This function is the entry point called by the tool factory system.
    It creates and returns a FunctionTool that the agent can use.
    It returns no tool unless python_exec_enabled() (ENABLE_PYTHON_EXEC and an API key).
        
    Returns:
        List containing the Python execution FunctionTool, or an empty list
    """
    if not python_exec_enabled():
        return []

    # Return the tool in a list, as required by the tool factory system
    return [FunctionTool.from_defaults(PythonExecTool.execute_code, callback=create_tool_callback("execute_code", "Execute Python code safely and return results."))]
