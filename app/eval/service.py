"""Task-local services for a measurement run: PostgreSQL 16 + pgvector through Apptainer, the API under uvicorn.

    with TaskPostgres(container, root) as pg:                 # initdb, postgres on 127.0.0.1:<free port>
        env = {**os.environ, **settings, **pg.env()}         # PG_CONNECTION_STRING, PSYCOPG2_CONNECTION_STRING
        with ApiServer(env, cwd=repo_root, log_path=root / "api.log") as api:
            health, seconds = api.wait_ready()               # uvicorn main:app on 127.0.0.1:<free port>

Used inside a Slurm job (scripts/revamp_http_service.py, app/eval/served_scale.py), where there is no
Docker. The server's data directory, its socket and its log stay under `root`, a directory the caller
chooses (node-local scratch in a job). Apptainer gets exactly that directory, resolved, as its only
bind: no path is written into the code. The connection strings name no user: libpq (and asyncpg)
default to the account that ran initdb, so neither the environment nor a results file needs the
user name. Nothing here is published; a caller that records a value from it passes it through
app/eval/provenance.scrub first.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

POSTGRES_BIN = "/usr/lib/postgresql/16/bin"  # inside the pgvector/pgvector:pg16 image (and its Apptainer build)
DATABASE = "metanavit"


def free_port() -> int:
    """A TCP port on 127.0.0.1 that was free a moment ago."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def apptainer_exec(container: Path, binds: Sequence[Path]) -> list[str]:
    """`apptainer exec --cleanenv --bind <dirs> <image>`: the image sees only the directories passed in."""
    command = ["apptainer", "exec", "--cleanenv"]
    resolved = [str(Path(b).resolve()) for b in binds]
    if resolved:
        command += ["--bind", ",".join(resolved)]
    return [*command, str(Path(container).resolve())]


class TaskPostgres:
    """PostgreSQL with pgvector in an Apptainer image, data and socket under `root`, listening on 127.0.0.1.

    A data directory that already exists under root (a restored snapshot) is started as it is;
    otherwise initdb creates it (UTF-8, no locale, trust authentication on the loopback only).
    """

    def __init__(self, container: Path, root: Path, *, port: int | None = None, database: str = DATABASE, ready_timeout: float = 60.0):
        self.container = Path(container)
        self.root = Path(root)
        self.port = port or free_port()
        self.database = database
        self.ready_timeout = ready_timeout
        self.data = self.root / "pgdata"
        self.socket_dir = self.root / "socket"
        self.log_path = self.root / "postgres.log"
        self.process: subprocess.Popen | None = None
        self.initialized = False

    def command(self, program: str, *args: str) -> list[str]:
        """One server program run inside the image, which sees only `root`."""
        return [*apptainer_exec(self.container, [self.root]), f"{POSTGRES_BIN}/{program}", *args]

    @property
    def dsn(self) -> str:
        return f"postgresql://127.0.0.1:{self.port}/{self.database}"

    @property
    def keyword_dsn(self) -> str:
        return f"dbname={self.database} host=127.0.0.1 port={self.port}"

    def env(self) -> dict[str, str]:
        """The variables the API and VectorStoreManager read their connection from."""
        return {"PG_CONNECTION_STRING": self.dsn, "PSYCOPG2_CONNECTION_STRING": self.keyword_dsn}

    def start(self) -> "TaskPostgres":
        self.root.mkdir(parents=True, exist_ok=True)
        self.socket_dir.mkdir(exist_ok=True)
        with self.log_path.open("a") as log:
            if not self.data.exists():
                subprocess.run(self.command("initdb", "-D", str(self.data), "--auth=trust", "--no-locale", "--encoding=UTF8"), stdout=log, stderr=subprocess.STDOUT, check=True)
                self.initialized = True
            self.process = subprocess.Popen(self.command("postgres", "-D", str(self.data), "-h", "127.0.0.1", "-p", str(self.port), "-k", str(self.socket_dir)), stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.ready_timeout
        while True:
            ready = subprocess.run(self.command("pg_isready", "-h", "127.0.0.1", "-p", str(self.port)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ready.returncode == 0:
                break
            if self.process.poll() is not None:
                raise RuntimeError(f"task-local PostgreSQL exited with {self.process.returncode}; see postgres.log")
            if time.monotonic() > deadline:
                raise RuntimeError(f"task-local PostgreSQL not ready after {self.ready_timeout:g} s")
            time.sleep(0.5)
        if self.initialized:
            subprocess.run(self.command("createdb", "-h", "127.0.0.1", "-p", str(self.port), self.database), check=True)
        return self

    def stop(self) -> None:
        """Fast shutdown (pg_ctl -m fast), then terminate the process if it is still there."""
        if self.process is None:
            return
        if self.process.poll() is None:
            subprocess.run(self.command("pg_ctl", "-D", str(self.data), "-m", "fast", "-w", "stop"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stop_process(self.process)
        self.process = None

    def __enter__(self) -> "TaskPostgres":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def stop_process(process: subprocess.Popen, timeout: float = 10.0) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_health(url: str, process: subprocess.Popen, timeout: float = 300.0) -> dict:
    """Poll GET /health until it reports status ok. Raises if the process exits, the start-up fails or time runs out."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API exited {process.returncode}; see the service log")
        try:
            with urllib.request.urlopen(url + "/health", timeout=10) as response:
                health = json.load(response)
                if health.get("status") == "ok":
                    return health
                last = health
        except urllib.error.HTTPError as exc:
            try:
                last = json.load(exc)
            except (ValueError, OSError):
                last = str(exc)
            if isinstance(last, dict) and last.get("status") == "unavailable" and last.get("error"):
                raise RuntimeError(f"API startup failed: {last['error']}") from exc
        except (urllib.error.URLError, ValueError, TimeoutError, ConnectionError) as exc:
            last = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"API not ready after {timeout:g} s: {last}")


class ApiServer:
    """`python -m uvicorn main:app` on 127.0.0.1:<free port>, with the given environment, logging to `log_path`."""

    def __init__(self, env: Mapping[str, str], *, cwd: Path, log_path: Path, python: str | None = None, port: int | None = None, app: str = "main:app"):
        self.env = dict(env)
        self.cwd = Path(cwd)
        self.log_path = Path(log_path)
        self.python = python or sys.executable
        self.port = port or free_port()
        self.app = app
        self.url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen | None = None
        self.started_at: float | None = None

    def start(self) -> "ApiServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.monotonic()
        with self.log_path.open("a") as log:
            self.process = subprocess.Popen([self.python, "-m", "uvicorn", self.app, "--host", "127.0.0.1", "--port", str(self.port)], cwd=self.cwd, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        return self

    def wait_ready(self, timeout: float = 600.0) -> tuple[dict, float]:
        """(/health payload, seconds from process start to the first ok)."""
        assert self.process is not None and self.started_at is not None, "start() first"
        health = wait_health(self.url, self.process, timeout=timeout)
        return health, time.monotonic() - self.started_at

    def stop(self) -> None:
        if self.process is not None:
            stop_process(self.process)
            self.process = None

    def __enter__(self) -> "ApiServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def base_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The caller's environment without the variables a measured API must not inherit from a shell."""
    env = {k: v for k, v in os.environ.items() if k not in {"FRONTEND_ENDPOINT", "PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "INDEX_ON_START"}}
    env.update(extra or {})
    return env
