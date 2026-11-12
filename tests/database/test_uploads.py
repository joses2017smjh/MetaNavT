"""Uploads against a live Postgres and real API processes: indexed, kept across a restart, shared by two instances.

POST /api/chat/upload used to save the file to output/uploaded under the receiving process's working directory and
never index it. Each API here is a uvicorn process of its own (main.py reads DATA_DIR when it is imported), with a
working directory of its own; DATA_DIR is one temporary copy of fixture v1 that every process reads, as a shared
volume would be. Needs PG_TEST_DSN (the CI parity service, or a local instance); skipped otherwise. The table is
built by the schema migrations and dropped at the end; the embedder is the hash adapter.
"""

import base64
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

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"
UPLOADED = b"Calibration note: the zylophant sensor offset is 0.0425 after the March recalibration.\n"


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _request(url: str, payload: dict | None = None, timeout: float = 30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"}, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class Api:
    """One API process: `uvicorn main:app` started from a working directory of its own."""

    def __init__(self, env: dict, workdir: Path, **overrides):
        workdir.mkdir(parents=True, exist_ok=True)
        self.url = f"http://127.0.0.1:{_port()}"
        self.log = workdir / "api.log"
        command = [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(ROOT), "--host", "127.0.0.1", "--port", self.url.rsplit(":", 1)[1]]
        with self.log.open("w") as log:
            self.proc = subprocess.Popen(command, cwd=workdir, env={**env, "PYTHONPATH": str(ROOT), **overrides}, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                status, body = _request(self.url + "/health", timeout=5)
                if status == 200:
                    self.health = json.loads(body)
                    return
            except OSError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.3)
        self.stop()
        raise AssertionError(f"API not healthy: {self.log.read_text()[-3000:]}")

    def upload(self, name: str, data: bytes):
        payload = {"name": name, "base64": "data:text/plain;base64," + base64.b64encode(data).decode()}
        status, body = _request(self.url + "/api/chat/upload", payload)
        return status, json.loads(body)

    def paths_for(self, query: str) -> list[str]:
        status, body = _request(self.url + "/api/retrieve/", {"query": query, "k": 8})
        assert status == 200, body
        return [hit["path"] for hit in json.loads(body)["hits"]]

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)


@pytest.fixture
def deployment(tmp_path):
    """A migrated table, DATA_DIR indexed by the index step, and the environment every process gets."""
    from urllib.parse import urlparse

    from app.database.migrate import VectorTable, drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager

    dsn = os.environ["PG_TEST_DSN"]
    data_dir = tmp_path / "shared" / "files"
    shutil.copytree(FIXTURE, data_dir)
    table = "uploads_" + uuid.uuid4().hex[:8]
    env = {k: v for k, v in os.environ.items() if k not in {"PGVECTOR_SCHEMA", "LEXICAL_BACKEND", "INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES",
                                                           "INDEX_ON_START", "FRONTEND_ENDPOINT", "RETRIEVAL_REQUIRE_DEVICE", "METANAVIT_API_KEY",
                                                           "RATE_LIMIT_PER_MINUTE", "FILESERVER_URL_PREFIX", "ENABLE_STALENESS"}}
    env.update({"PG_CONNECTION_STRING": dsn, "PSYCOPG2_CONNECTION_STRING": dsn, "DB_NAME": urlparse(dsn).path.lstrip("/"),
                "PGVECTOR_TABLE": table, "EMBEDDING_DIM": "256", "EMBEDDING_PROVIDER": "hash", "MODEL_PROVIDER": "ollama",
                "RERANKER_MODEL": "none", "DATA_DIR": str(data_dir), "INGEST_MAX_REJECTED": "0", "INDEX_ON_START": "false"})
    vsm = VectorStoreManager(conn_string=dsn)
    with vsm.get_connection() as conn:
        migrate_all(conn, VectorTable(table=table, dim=256))
    step = subprocess.run([sys.executable, "-m", "app.engine.ingest"], cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)},
                          capture_output=True, text=True, timeout=300)
    assert step.returncode == 0, step.stderr[-2000:]
    apis: list[Api] = []
    yield env, data_dir, apis, json.loads(step.stdout.strip().splitlines()[-1])
    for api in apis:
        api.stop()
    with vsm.get_connection() as conn:
        drop_vector_table(conn, VectorTable(table=table, dim=256))
    vsm.close()


def _manifest(env: dict) -> dict:
    from app.database.vector_store_manager import VectorStoreManager

    os.environ["PGVECTOR_TABLE"] = env["PGVECTOR_TABLE"]
    vsm = VectorStoreManager(conn_string=env["PG_CONNECTION_STRING"])
    try:
        return vsm.ingest_manifest()
    finally:
        vsm.close()


def test_an_upload_is_found_through_retrieve_and_still_after_a_restart(deployment, tmp_path, monkeypatch):
    """A29: the uploaded file's chunk comes back from /api/retrieve/, and a restart that indexes DATA_DIR again keeps
    it (the manifest lists it: the removed-file diff does not take it for a deleted file)."""
    env, data_dir, apis, first_step = deployment
    monkeypatch.setenv("PGVECTOR_TABLE", env["PGVECTOR_TABLE"])  # _manifest sets it; restored afterwards
    api = Api(env, tmp_path / "api_1")
    apis.append(api)

    status, stored = api.upload("calibration notes.txt", UPLOADED)

    assert status == 200, stored
    rel = f"uploads/{stored['name']}"
    assert stored["refs"] == [rel] and stored["url"].endswith(f"/data/{rel}")
    assert (data_dir / rel).read_bytes() == UPLOADED
    assert rel in api.paths_for("zylophant sensor offset")
    assert rel in _manifest(env)
    refused, detail = api.upload("copy of the notes.txt", UPLOADED)  # the same bytes again: refused, nothing written
    assert refused == 409 and detail["detail"].startswith("duplicate_content")
    assert sorted(p.name for p in (data_dir / "uploads").iterdir()) == [stored["name"]]
    api.stop()

    restarted = Api(env, tmp_path / "api_2", INDEX_ON_START="true")  # a start that runs the index step itself
    apis.append(restarted)
    indexing = restarted.health["indexing"]
    assert (indexing["files_unchanged"], indexing["files_removed"], indexing["chunks_embedded"]) == (first_step["files_seen"] + 1, 0, 0)
    assert rel in restarted.paths_for("zylophant sensor offset")
    assert rel in _manifest(env)


def test_a_file_uploaded_through_one_instance_is_served_by_the_other(deployment, tmp_path, monkeypatch):
    """C50: two API processes, each with its own working directory, the same DATA_DIR and table. An upload through
    one is retrieved and served as a file by the other, and an index run in another process keeps it."""
    env, data_dir, apis, first_step = deployment
    monkeypatch.setenv("PGVECTOR_TABLE", env["PGVECTOR_TABLE"])
    first, second = Api(env, tmp_path / "instance_a"), Api(env, tmp_path / "instance_b")
    apis.extend([first, second])

    status, stored = first.upload("calibration notes.txt", UPLOADED)
    assert status == 200, stored
    rel = f"uploads/{stored['name']}"

    assert rel in second.paths_for("zylophant sensor offset")
    served, body = _request(f"{second.url}/api/files/data/{rel}")
    assert (served, body) == (200, UPLOADED)
    assert not (tmp_path / "instance_a" / "output" / "uploaded").exists()  # nothing in the receiving process's own folder

    step = subprocess.run([sys.executable, "-m", "app.engine.ingest"], cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT)},
                          capture_output=True, text=True, timeout=300)  # the index step, run beside both instances
    assert step.returncode == 0, step.stderr[-2000:]
    report = json.loads(step.stdout.strip().splitlines()[-1])
    assert (report["files_unchanged"], report["files_removed"], report["chunks_embedded"]) == (first_step["files_seen"] + 1, 0, 0)
    assert rel in second.paths_for("zylophant sensor offset") and rel in first.paths_for("zylophant sensor offset")
