"""Uploads: refused when the index step would reject them, written inside the ingest lock, removed when the run fails.

No database: the manager is a stand-in that keeps a manifest in memory, and index_data_dir is replaced by a fake
that, like the real one, runs before_scan under the lock and then lists what it finds. tests/database/test_uploads.py
runs the real route, indexer and table.
"""

import hashlib

import pytest

from app.engine import uploads
from app.eval.ingest import SERVED_POLICY


class Manager:
    """The two manifest reads save_and_index makes."""

    def __init__(self, listed=None):
        self.listed = dict(listed or {})

    def ingest_manifest(self):
        return dict(self.listed)


def _fake_run(manager, data_dir, fail_after_write=None, listed_too_early=None):
    """index_data_dir as the uploads module calls it: before_scan, then every file under data_dir is listed."""

    def run(vsm, before_scan=None):
        if listed_too_early:  # another upload of the same bytes was indexed while this one waited for the lock
            manager.listed.update(listed_too_early)
        before_scan()
        if fail_after_write:
            raise fail_after_write
        for path in sorted(data_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(data_dir).as_posix()
                manager.listed[rel] = {"content_hash": hashlib.sha256(path.read_bytes()).hexdigest(), "n_chunks": 1, "pipeline": "p"}
        return {"files_indexed": 1, "rejected_sample": []}

    return run


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for name in ("INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES", "FILESERVER_URL_PREFIX"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.parametrize("name,data,status,reason", [
    ("weights.bin", b"\x00\x01", 415, "unsupported_type"),
    ("big.txt", b"x" * (SERVED_POLICY.max_bytes + 1), 413, "too_large"),
    ("latin1.txt", "caf\xe9".encode("latin-1"), 415, "undecodable"),
    ("blank.md", b" \n\t\n", 422, "empty"),
    ("again.yaml", b"run_id: run_047\n", 409, "duplicate_content"),
])
def test_a_file_the_index_step_would_reject_is_refused_and_nothing_is_written(data_dir, monkeypatch, name, data, status, reason):
    manager = Manager({"configs/run_047.yaml": {"content_hash": hashlib.sha256(b"run_id: run_047\n").hexdigest(), "n_chunks": 1}})
    monkeypatch.setattr("app.engine.ingest.index_data_dir", lambda *a, **k: pytest.fail("a refused upload started an indexing run"))
    with pytest.raises(uploads.UploadRejected) as refused:
        uploads.save_and_index(name, data, vsm=manager)
    assert (refused.value.status, refused.value.reason) == (status, reason)
    assert not (data_dir / "uploads").exists()


def test_the_stored_name_is_one_path_component_with_the_uploaded_suffix():
    assert uploads.stored_name("notes.txt", "id1") == "notes_id1.txt"
    assert uploads.stored_name("../../etc/cron.d/job.YAML", "id2") == "job_id2.YAML"  # directories dropped
    assert uploads.stored_name("C:\\Users\\a b\\run 47.md", "id3") == "run_47_id3.md"
    with pytest.raises(uploads.UploadRejected) as refused:
        uploads.stored_name("Makefile", "id4")
    assert refused.value.status == 415


def test_an_upload_is_written_inside_the_run_and_listed_with_a_data_url(data_dir, monkeypatch):
    manager = Manager()
    monkeypatch.setattr("app.engine.ingest.index_data_dir", _fake_run(manager, data_dir))
    monkeypatch.setenv("FILESERVER_URL_PREFIX", "https://files.example/api/files/")

    stored = uploads.save_and_index("run 47 notes.md", b"# Notes\nlearning rate 3e-4\n", vsm=manager)

    rel = f"uploads/{stored['name']}"
    assert (data_dir / rel).read_bytes() == b"# Notes\nlearning rate 3e-4\n"
    assert stored["name"].startswith("run_47_notes_") and stored["name"].endswith(".md") and stored["type"] == "md"
    assert stored["url"] == f"https://files.example/api/files/data/{rel}" and stored["refs"] == [rel]
    assert stored["size"] == len(b"# Notes\nlearning rate 3e-4\n") and rel in manager.listed


def test_a_failed_run_removes_the_upload_again(data_dir, monkeypatch):
    manager = Manager()
    monkeypatch.setattr("app.engine.ingest.index_data_dir", _fake_run(manager, data_dir, fail_after_write=RuntimeError("embedder died")))
    with pytest.raises(RuntimeError, match="embedder died"):
        uploads.save_and_index("notes.txt", b"some text", vsm=manager)
    assert list((data_dir / "uploads").iterdir()) == []


def test_the_same_bytes_listed_while_waiting_for_the_lock_are_refused_before_writing(data_dir, monkeypatch):
    manager = Manager()
    digest = hashlib.sha256(b"some text").hexdigest()
    run = _fake_run(manager, data_dir, listed_too_early={"uploads/first_upload.txt": {"content_hash": digest, "n_chunks": 1}})
    monkeypatch.setattr("app.engine.ingest.index_data_dir", run)
    with pytest.raises(uploads.UploadRejected) as refused:
        uploads.save_and_index("notes.txt", b"some text", vsm=manager)
    assert refused.value.status == 409 and "uploads/first_upload.txt" in str(refused.value)
    assert not (data_dir / "uploads").exists()


def test_the_route_answers_with_the_refusals_status(monkeypatch):
    """POST /api/chat/upload: an UploadRejected is its status and reason, a busy lock is 503 with Retry-After."""
    from fastapi.testclient import TestClient

    from app.api.routers import upload
    from app.database.vector_store_manager import IngestLockTimeout

    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://postgres:password@127.0.0.1:1/metanavit")
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1")
    monkeypatch.setenv("INDEX_ON_START", "false")
    monkeypatch.delenv("FRONTEND_ENDPOINT", raising=False)
    import main

    from app.database import vector_store

    errors = iter([uploads.UploadRejected(409, "duplicate_content", "a file with these bytes is already indexed: a.md"),
                   IngestLockTimeout("another process held the ingest lock of public.data_internal_table for more than 600 s")])

    def refuse(*args):
        raise next(errors)

    monkeypatch.setattr(upload.FileService, "process_private_file", staticmethod(refuse))
    vector_store.get_vector_store_manager.cache_clear()
    body = {"name": "a.md", "base64": "data:text/markdown;base64,IyBh"}
    with TestClient(main.app) as client:
        refused = client.post("/api/chat/upload", json=body)
        busy = client.post("/api/chat/upload", json=body)
    assert refused.status_code == 409 and refused.json()["detail"].startswith("duplicate_content: ")
    assert busy.status_code == 503 and busy.headers["retry-after"] == "10"
    assert "data_internal_table" not in busy.text  # the client is told to retry; the table name stays in the server log


@pytest.mark.parametrize("declared", ["application/x-yaml", "text/yaml", "", "text/plain"])
def test_the_upload_service_decodes_the_body_whatever_type_the_browser_declared(monkeypatch, declared):
    """Browsers send application/x-yaml, text/yaml or nothing for a .yaml file; mimetypes knows none of them, and the
    team's decoder (which only needed the type for a CSV rule that is gone) failed on them with a 500."""
    import base64

    from app.engine.loaders.file import FileService

    seen = {}

    def store(name, data):
        seen.update(name=name, data=data)
        return {"id": "i", "name": "run_047_i.yaml", "type": "yaml", "size": len(data), "path": "p",
                "url": "u", "refs": ["uploads/run_047_i.yaml"]}

    monkeypatch.setattr(uploads, "save_and_index", store)
    body = f"data:{declared};base64," + base64.b64encode(b"run_id: run_047\n").decode()
    stored = FileService.process_private_file("run_047.yaml", body)
    assert seen == {"name": "run_047.yaml", "data": b"run_id: run_047\n"} and stored.refs == ["uploads/run_047_i.yaml"]


def test_a_body_that_is_not_base64_is_refused():
    with pytest.raises(uploads.UploadRejected) as refused:
        uploads.decode_data_url("data:text/plain;base64,not base64!")
    assert (refused.value.status, refused.value.reason) == (422, "bad_base64")
    assert uploads.decode_data_url("aGk=") == b"hi"  # the payload alone is accepted too
