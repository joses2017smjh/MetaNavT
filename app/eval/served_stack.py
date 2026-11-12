"""The default Compose stack's API without Docker, for the agent-path runs (make bench-companion-live and
make chat-route-smoke).

    with ServedAPI(server_dsn, data_dir, model="qwen2.5:7b", ollama_url=..., log_dir=...) as api:
        api.url, api.health

What runs, in the order of docker-compose.yml's api command: the deploy step (DEPLOY_STEP, the schema
migrations; the API runs no DDL and answers 503 until they have run), then `python -m uvicorn main:app` on a free
loopback port, both with docker-compose.yml's api environment (COMPOSE_API_ENV: the hash embedder, no reranker,
the SQL retrieval path, the router), DATA_DIR set to a
task-local copy of fixture v1 (copy_fixture: verified against the MANIFEST, with its mtimes, which the served
staleness filter reads), and a database of its own created on the Postgres server the caller names
(PG_TEST_DSN, as for make ingest-runs) and dropped at the end. Two runs never share a database or a table.

The API's environment is built here, not inherited. PATH, HOME, locale, TZ, TMPDIR and the one-thread settings
come from the caller; every other variable (keys, connection strings, flags) is dropped, and three switches are
set off explicitly, so that a .env file read by main.py (python-dotenv does not override a set variable) cannot
turn them on: ENABLE_PYTHON_EXEC=false (no code-execution agent), METANAVIT_API_KEY empty (no key) and
RATE_LIMIT_PER_MINUTE empty (no request budget). Compose's MODEL is qwen2.5:14b; each run names the model it
used. public_env() is the environment as a results file records it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from app.eval.corpus import apply_manifest_mtimes, load_manifest
from app.eval.provenance import scrub

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "bench" / "corpus" / "files"
MANIFEST = ROOT / "bench" / "corpus" / "MANIFEST.json"

# docker-compose.yml, services.api.environment, minus the variables each run sets (RUN_SET); a test compares.
COMPOSE_API_ENV = {
    "MODEL_PROVIDER": "ollama",
    "EMBEDDING_PROVIDER": "hash",
    "EMBEDDING_DIM": "256",
    "RERANKER_MODEL": "none",
    "RETRIEVAL_MODE": "sql",
    "INDEX_ON_START": "true",
    "INGEST_MAX_REJECTED": "0",
    "CHUNK_SIZE": "512",
    "CHUNK_OVERLAP": "50",
    "RETRIEVE_K": "50",
    "RERANK_TOP_N": "8",
    "ENABLE_ROUTER": "true",
    "LOG_FORMAT": "text",
}
# docker-compose.yml, services.api.command: this runs before the API, on the same database; a test compares.
DEPLOY_STEP = ("-m", "app.database.migrate")
RUN_SET = ("PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "MODEL", "OLLAMA_BASE_URL", "DATA_DIR")
OFF_SWITCHES = {"ENABLE_PYTHON_EXEC": "false", "METANAVIT_API_KEY": "", "RATE_LIMIT_PER_MINUTE": ""}
KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")
_NOT_RECORDED = ("PATH", "HOME", "TMPDIR", "PYTHONPATH")
_REDACTED = {"PG_CONNECTION_STRING": "<task-local database>", "PSYCOPG2_CONNECTION_STRING": "<task-local database>",
             "DB_NAME": "<task-local database>", "DATA_DIR": "<task-local copy of the corpus>"}


def database_dsn(server_dsn: str, name: str) -> str:
    """`server_dsn` (a postgresql:// URI) pointing at database `name` instead."""
    parts = urlsplit(server_dsn)
    if parts.scheme not in {"postgresql", "postgres"}:
        raise ValueError("PG_TEST_DSN must be a postgresql:// URI")
    return urlunsplit(parts._replace(path="/" + name))


def api_env(server_dsn: str, database: str, data_dir: Path, *, model: str, ollama_url: str,
            parent: Mapping[str, str] | None = None, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    parent = os.environ if parent is None else parent
    dsn = database_dsn(server_dsn, database)
    env = {k: parent[k] for k in KEEP if k in parent}
    env.update(COMPOSE_API_ENV)
    env.update({"PG_CONNECTION_STRING": dsn, "PSYCOPG2_CONNECTION_STRING": dsn, "DB_NAME": database,
                "DATA_DIR": str(data_dir), "MODEL": model, "OLLAMA_BASE_URL": ollama_url, "PYTHONPATH": str(ROOT)})
    env.update(extra or {})
    env.update(OFF_SWITCHES)  # last: nothing passed in turns them on
    return env


def public_env(env: Mapping[str, str]) -> dict[str, str]:
    """The API's environment as a results file records it: no connection string, no local path, no user name."""
    return {k: _REDACTED.get(k, scrub(v)) for k, v in sorted(env.items()) if k not in _NOT_RECORDED}


def copy_fixture(dest: Path) -> Path:
    """Copy fixture v1 to `dest` and give every file the MANIFEST's mtime; refuses a copy that differs from it."""
    shutil.copytree(CORPUS, dest)
    apply_manifest_mtimes(dest, load_manifest(MANIFEST))
    return dest


def plant(root: Path, additions: Mapping[str, str]) -> None:
    """Append text to files under `root`, keeping each file's mtime (the staleness filter reads it)."""
    for rel, text in additions.items():
        path = root / rel
        st = path.stat()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
        os.utime(path, (st.st_atime, st.st_mtime))


def tree_sha256(root: Path) -> str:
    rows = [[p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()]
            for p in sorted(Path(root).rglob("*")) if p.is_file()]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _admin(server_dsn: str, statement: str) -> None:
    import psycopg2  # the app extra; only a run that starts the API needs it

    conn = psycopg2.connect(server_dsn)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(statement)
    finally:
        conn.close()


def create_database(server_dsn: str, name: str) -> None:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe database name {name!r}")
    _admin(server_dsn, f'CREATE DATABASE "{name}"')
    _admin(database_dsn(server_dsn, name), "CREATE EXTENSION IF NOT EXISTS vector")


def drop_database(server_dsn: str, name: str) -> None:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe database name {name!r}")
    _admin(server_dsn, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get_json(url: str, timeout: float = 10, payload: Any = None) -> tuple[int, Any]:
    """(status, decoded JSON body or None): a GET, or a POST of `payload` as JSON."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.load(exc)
        except (ValueError, OSError):
            return exc.code, None


class ServedAPI:
    """`uvicorn main:app` over a task-local corpus and database; stopped and the database dropped on exit."""

    def __init__(self, server_dsn: str, data_dir: Path, *, model: str, ollama_url: str, log_dir: Path,
                 label: str = "api", extra_env: Mapping[str, str] | None = None, startup_timeout: float = 300):
        self.server_dsn, self.data_dir, self.log_dir, self.label = server_dsn, Path(data_dir), Path(log_dir), label
        self.database = f"mnv_{label}_{uuid.uuid4().hex[:8]}"
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.env = api_env(server_dsn, self.database, self.data_dir, model=model, ollama_url=ollama_url, extra=extra_env)
        self.startup_timeout = startup_timeout
        self.process: subprocess.Popen | None = None
        self.health: dict | None = None
        self.startup_s: float | None = None
        self._created = False

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"{self.label}.log"

    def __enter__(self) -> "ServedAPI":
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        started = time.monotonic()
        create_database(self.server_dsn, self.database)
        self._created = True
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w") as log:
            deploy = subprocess.run([sys.executable, *DEPLOY_STEP], cwd=ROOT, env=self.env, stdout=log,
                                    stderr=subprocess.STDOUT)
        if deploy.returncode != 0:
            raise RuntimeError(f"{self.label}: the deploy step (schema migrations) exited with {deploy.returncode}; "
                               f"see {self.log_path}\n" + self.log_tail())
        log = self.log_path.open("a")
        self.process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                                         "--port", str(self.port)], cwd=ROOT, env=self.env, stdout=log,
                                        stderr=subprocess.STDOUT)
        log.close()
        self.health = self._wait_healthy()
        self.startup_s = round(time.monotonic() - started, 3)

    def _wait_healthy(self) -> dict:
        deadline, last = time.monotonic() + self.startup_timeout, None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.label}: the API exited with {self.process.returncode}; see {self.log_path}\n"
                                   + self.log_tail())
            try:
                status, body = get_json(self.url + "/health")
            except (urllib.error.URLError, OSError, ValueError) as exc:
                status, body = None, str(exc)
            if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
                return body
            if isinstance(body, dict) and body.get("status") in {"unavailable", "degraded"} and (body.get("error") or body.get("degraded")):
                raise RuntimeError(f"{self.label}: the API is not healthy: {json.dumps(scrub(body))[:2000]}")
            last = body
            time.sleep(1)
        raise RuntimeError(f"{self.label}: no healthy /health after {self.startup_timeout} s: {scrub(str(last))[:500]}")

    def log_tail(self, lines: int = 40) -> str:
        try:
            return scrub("\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:]))
        except OSError:
            return ""

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self._created:
            self._created = False
            drop_database(self.server_dsn, self.database)
