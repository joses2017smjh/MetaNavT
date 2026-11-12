"""Task-local services for a measurement run: PostgreSQL 16 + pgvector through Apptainer, the API under uvicorn.

    with TaskPostgres(container, root) as pg:                 # initdb, postgres on 127.0.0.1:<free port>
        env = {**os.environ, **settings, **pg.env()}         # PG_CONNECTION_STRING, PSYCOPG2_CONNECTION_STRING
        with ApiServer(env, cwd=repo_root, log_path=root / "api.log") as api:
            health, seconds = api.wait_ready()               # uvicorn main:app on 127.0.0.1:<free port>

migrate() and index_step() run the deploy order's two steps before the API (`python -m app.database.migrate`, then
`python -m app.engine.ingest`). Used inside a Slurm job (scripts/revamp_http_service.py, app/eval/served_scale.py),
where there is no Docker. The server's data directory, its socket and its log stay under `root`, a directory the caller
chooses (node-local scratch in a job). Apptainer gets exactly that directory, resolved, as its only
bind: no path is written into the code. The connection strings name no user: libpq (and asyncpg)
default to the account that ran initdb, so neither the environment nor a results file needs the
user name. Every connection needs a password (scram-sha-256): a GPU node is shared, and with the trust
authentication this used to set, any account on it could connect to the loopback port as the superuser.
The password is random, one per data directory, kept in a file inside it (initdb makes the directory 0700,
so a restored snapshot still opens), and reaches clients only through PGPASSWORD in env(): no connection
string and no command line carries it. Nothing here is published; a caller that records a value from it
passes it through app/eval/provenance.scrub first.
"""

from __future__ import annotations

import json
import os
import secrets
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
PASSWORD_FILE = "task_password"  # inside the data directory: the superuser's password, written when initdb creates it


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

    A data directory that already exists under root (a restored snapshot) is started as it is, with the
    password stored in it; otherwise initdb creates it (UTF-8, no locale, scram-sha-256 for every connection,
    a random superuser password of its own). A data directory without the file (made before passwords were
    set here) still uses trust authentication, and env() then names no password.
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
        self.password: str | None = None

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
        """The variables the API and VectorStoreManager read their connection from; PGPASSWORD is read by libpq (psycopg2)
        and asyncpg alike, so the connection strings stay without it."""
        env = {"PG_CONNECTION_STRING": self.dsn, "PSYCOPG2_CONNECTION_STRING": self.keyword_dsn}
        if self.password:
            env["PGPASSWORD"] = self.password
        return env

    def client_environment(self) -> dict[str, str]:
        """The environment of a client program run inside the image: --cleanenv passes only APPTAINERENV_ variables."""
        env = dict(os.environ)
        if self.password:
            env["APPTAINERENV_PGPASSWORD"] = self.password
        return env

    def initdb(self, log) -> None:
        """A new data directory whose every connection needs the superuser's password, a random one written to a file
        (0600) that initdb reads and that then moves into the data directory."""
        password = secrets.token_urlsafe(32)
        pwfile = self.root / "initdb.pw"
        descriptor = os.open(pwfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as out:
            out.write(password + "\n")
        try:
            subprocess.run(self.command("initdb", "-D", str(self.data), "--auth=scram-sha-256", f"--pwfile={pwfile}", "--no-locale", "--encoding=UTF8"),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            os.replace(pwfile, self.data / PASSWORD_FILE)
        finally:
            pwfile.unlink(missing_ok=True)
        self.password = password

    def start(self) -> "TaskPostgres":
        self.root.mkdir(parents=True, exist_ok=True)
        self.socket_dir.mkdir(exist_ok=True)
        with self.log_path.open("a") as log:
            if not self.data.exists():
                self.initdb(log)
                self.initialized = True
            else:
                # A restored snapshot: tarfile's data filter ignores a directory's stored mode, so the directory took the umask
                # (and a parent's setgid bit), and postgres refuses a data directory with group write or other access.
                self.data.chmod(0o700)
                stored = self.data / PASSWORD_FILE
                self.password = stored.read_text().strip() if stored.exists() else None
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
            subprocess.run(self.command("createdb", "-h", "127.0.0.1", "-p", str(self.port), self.database), check=True, env=self.client_environment())
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


def migrate(env: Mapping[str, str], *, cwd: Path, log_path: Path | None = None, python: str | None = None) -> float:
    """The deploy step, `python -m app.database.migrate`, against the database `env` names (PG_CONNECTION_STRING;
    the vector table from PGVECTOR_SCHEMA, PGVECTOR_TABLE and EMBEDDING_DIM). It creates the vector table and its
    indexes; the API runs no DDL and refuses to start while a migration is pending. Returns its wall seconds;
    raises CalledProcessError when it fails."""
    command = [python or sys.executable, "-m", "app.database.migrate"]
    started = time.monotonic()
    if log_path is None:
        subprocess.run(command, cwd=cwd, env=dict(env), check=True)
    else:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a") as log:
            subprocess.run(command, cwd=cwd, env=dict(env), check=True, stdout=log, stderr=subprocess.STDOUT)
    return time.monotonic() - started


def index_step(env: Mapping[str, str], *, cwd: Path, log_path: Path, python: str | None = None) -> tuple[dict, float]:
    """The index step, `python -m app.engine.ingest` (make index), against the database and DATA_DIR `env` names: the deploy
    order runs it after the schema migrations and before the API, which does not index (INDEX_ON_START unset). Returns
    (the report it prints as its last line on stdout, its wall seconds); its log (stderr) goes to log_path. Raises
    CalledProcessError when it exits non-zero (it does on any indexing failure)."""
    command = [python or sys.executable, "-m", "app.engine.ingest"]
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with Path(log_path).open("a") as log:
        result = subprocess.run(command, cwd=cwd, env=dict(env), check=True, stdout=subprocess.PIPE, stderr=log, text=True)
    seconds = time.monotonic() - started
    return json.loads(result.stdout.strip().splitlines()[-1]), seconds


def base_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The caller's environment without the variables a measured API must not inherit from a shell."""
    env = {k: v for k, v in os.environ.items() if k not in {"FRONTEND_ENDPOINT", "PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "INDEX_ON_START"}}
    env.update(extra or {})
    return env
