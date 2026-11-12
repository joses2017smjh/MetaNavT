"""Uploaded files: saved under DATA_DIR/uploads and indexed by the one indexing call (app/engine/ingest.py).

POST /api/chat/upload (FileService.process_private_file) used to write the file to output/uploaded, relative to
the process's working directory, and then call an indexing step whose body was empty: the file was never
searchable, and another replica never had it. Now:

  1. the file is checked against the gates of the index step's own scan (app/eval/ingest.scan_directory: type,
     size, UTF-8 text, not empty, not the bytes of a file already indexed). A file the index step would reject
     is refused (UploadRejected, an HTTP status and the reason) and nothing is written: left in DATA_DIR it would
     be counted as rejected by every later run, which fails under INGEST_MAX_REJECTED;
  2. it is written to DATA_DIR/uploads/<name>_<id>.<ext> while the ingest lock is held, before that run's scan
     (index_directory's before_scan), so no run reads it half-written, and the duplicate check is repeated there
     against the manifest as it is under the lock;
  3. index_data_dir indexes it with everything else in DATA_DIR: its chunks are embedded, written and listed in
     the manifest, like any file of DATA_DIR. A restart or another process's index run sees the same file and
     keeps it, provided every process that indexes or serves reads the same DATA_DIR (Compose mounts a volume at
     DATA_DIR/uploads); a process with a DATA_DIR of its own would remove the rows as a deleted file's.

If the run fails, the file is removed again, so a failed upload leaves nothing behind (a run that had indexed it
before failing leaves rows that the next run removes). An upload is visible to every caller of /api/retrieve:
the team's private flag (private: true on uploads, false on the crawler's documents) is not enforced by the
hybrid SQL, and the served rows do not carry it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger("uvicorn")

UPLOADS_SUBDIR = "uploads"  # under DATA_DIR: a file outside DATA_DIR would be removed by the next run's diff
DEFAULT_FILESERVER_URL_PREFIX = "http://localhost:8000/api/files"  # FileService.save_file's fallback as well


class UploadRejected(ValueError):
    """The index step would reject this file (or already has its bytes); nothing was written."""

    def __init__(self, status: int, reason: str, detail: str):
        self.status, self.reason = status, reason
        super().__init__(detail)


def decode_data_url(content: str) -> bytes:
    """The file's bytes from the upload body's base64 field ("data:<type>;base64,<payload>", or the payload alone).

    The declared type is not used: the stored name keeps the uploaded name's extension, which is what the index
    step decides by. (FileService._preprocess_base64_file maps the type to an extension and fails on a type
    mimetypes does not know, such as application/x-yaml or an empty one, which browsers send for YAML files.)
    """
    payload = content.split(",", 1)[1] if content.startswith("data:") and "," in content else content
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise UploadRejected(422, "bad_base64", "the base64 field is not base64 data") from None


def uploads_dir() -> Path:
    """DATA_DIR/uploads, read from the environment at each call (DATA_DIR as the indexer reads it)."""
    return Path(os.getenv("DATA_DIR", "data")) / UPLOADS_SUBDIR


def stored_name(file_name: str, file_id: str) -> str:
    """<sanitized stem>_<id><suffix>: the team's naming (FileService.save_file), one path component, never ".."."""
    from app.engine.loaders.file import _sanitize_file_name

    stem, suffix = os.path.splitext(os.path.basename(file_name.replace("\\", "/")))
    if not suffix or suffix == ".":
        raise UploadRejected(415, "unsupported_type", "the file name needs an extension: the index step decides by suffix")
    return f"{_sanitize_file_name(stem) or 'upload'}_{file_id}{_sanitize_file_name(suffix)}"


def check_upload(name: str, data: bytes, policy, indexed_hashes: dict[str, str]) -> str:
    """The scan's gates, in the scan's order, for one file; returns its SHA-256 or raises UploadRejected.

    indexed_hashes maps every content hash the manifest lists to its path.
    """
    suffix = Path(name).suffix.lower()
    if policy.suffixes is not None and suffix not in policy.suffixes:
        raise UploadRejected(415, "unsupported_type", f"the index step does not index {suffix} files (INGEST_SUFFIXES)")
    if policy.max_bytes is not None and len(data) > policy.max_bytes:
        raise UploadRejected(413, "too_large", f"the index step indexes files up to {policy.max_bytes} bytes (INGEST_MAX_FILE_BYTES)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise UploadRejected(415, "undecodable", "the index step indexes UTF-8 text only") from None
    if not text.strip():
        raise UploadRejected(422, "empty", "the file has no text to index")
    digest = hashlib.sha256(data).hexdigest()
    if policy.dedup and digest in indexed_hashes:
        raise UploadRejected(409, "duplicate_content", f"a file with these bytes is already indexed: {indexed_hashes[digest]}")
    return digest


def save_and_index(file_name: str, data: bytes, vsm=None) -> dict:
    """Check, save under DATA_DIR/uploads inside the ingest lock, index; return the DocumentFile fields.

    Raises UploadRejected (nothing written), IngestLockTimeout (nothing written: another run held the lock), or
    whatever the indexing run raised (the file is removed again).
    """
    from app.database.vector_store import get_vector_store_manager
    from app.engine.ingest import index_data_dir, policy_from_env

    vsm = vsm or get_vector_store_manager()
    policy = policy_from_env()
    file_id = str(uuid.uuid4())
    name = stored_name(file_name, file_id)

    def indexed_hashes() -> dict[str, str]:
        return {row["content_hash"]: path for path, row in vsm.ingest_manifest().items()}

    check_upload(name, data, policy, indexed_hashes())  # before the lock: a refused upload waits for nothing
    target = uploads_dir() / name
    written = {"done": False}

    def write():  # under the lock, before this run's scan
        check_upload(name, data, policy, indexed_hashes())  # again: another upload may have listed these bytes meanwhile
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written["done"] = True

    try:
        report = index_data_dir(vsm, before_scan=write)
    except BaseException:
        if written["done"]:
            target.unlink(missing_ok=True)
            logger.warning(f"Upload {name} removed again: its indexing run failed")
        raise
    rel = f"{UPLOADS_SUBDIR}/{name}"
    listed = vsm.ingest_manifest().get(rel)
    if listed is None:  # every gate was checked first, so this means the scan disagreed: never leave it behind
        target.unlink(missing_ok=True)
        raise RuntimeError(f"the indexing run did not list {rel}; the upload was removed (rejected: {report.get('rejected_sample')})")
    logger.info(f"Upload {rel} indexed: {listed['n_chunks']} chunks ({report['files_indexed']} files indexed in that run)")
    prefix = (os.getenv("FILESERVER_URL_PREFIX") or DEFAULT_FILESERVER_URL_PREFIX).rstrip("/")
    return {
        "id": file_id,
        "name": name,
        "type": Path(name).suffix.lstrip("."),
        "size": len(data),
        "path": str(target),
        "url": f"{prefix}/data/{rel}",  # main.py serves DATA_DIR at /api/files/data
        "refs": [rel],  # the document id of its chunks (their ref_doc_id is the path)
    }
