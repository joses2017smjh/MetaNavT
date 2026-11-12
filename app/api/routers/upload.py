"""
File Upload Router Module

The body is JSON with the file as base64. It is read from the socket with a
byte cap (UPLOAD_MAX_BYTES, the decoded file size) and answers 413 past it,
instead of letting FastAPI buffer a body of any size before validation.

The file is saved under DATA_DIR/uploads and indexed (app/engine/uploads.py).
A file the index step would reject is refused with the reason, nothing written:
415 (type or not UTF-8 text), 413 (over INGEST_MAX_FILE_BYTES), 422 (empty, or
a body that is not base64), 409 (its bytes are already indexed); 503 when
another indexing run holds the ingest lock past INGEST_LOCK_TIMEOUT_SECONDS.
The data URL's declared type is not used: the file name's extension decides.
"""
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ValidationError

from app.api.routers.models import DocumentFile
from app.database.vector_store_manager import IngestLockTimeout
from app.engine.loaders.file import FileService
from app.engine.uploads import UploadRejected

file_upload_router = r = APIRouter()

logger = logging.getLogger("uvicorn")

DEFAULT_UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # decoded file size
_ENVELOPE_BYTES = 64 * 1024  # room for the JSON keys, the file name, params and a data: prefix


def max_upload_bytes() -> int:
    """Largest decoded file accepted, from UPLOAD_MAX_BYTES (read per request so a test can change it)."""
    try:
        value = int(os.getenv("UPLOAD_MAX_BYTES", str(DEFAULT_UPLOAD_MAX_BYTES)))
    except ValueError:
        value = DEFAULT_UPLOAD_MAX_BYTES
    return max(1, value)


def max_body_bytes() -> int:
    """Request-body cap: base64 grows the file by 4/3, plus a fixed envelope."""
    return 4 * ((max_upload_bytes() + 2) // 3) + _ENVELOPE_BYTES


class FileUploadRequest(BaseModel):
    """
    File upload request structure:
    - base64: Encoded file content
    - name: Original filename
    - params: Optional processing parameters
    """
    base64: str
    name: str
    params: Any = None


async def read_bounded_body(request: Request, limit: int) -> bytes:
    """Read at most `limit` bytes of the body; 413 as soon as the cap is known to be exceeded.

    A declared Content-Length over the cap is refused without reading anything.
    A body with no (or a wrong) Content-Length is counted while it streams, so
    the process never holds more than `limit` bytes plus one chunk.
    """
    too_large = HTTPException(
        status_code=413,
        detail=f"upload too large: the request body is limited to {limit} bytes "
               f"(UPLOAD_MAX_BYTES={max_upload_bytes()} bytes of file content, base64-encoded)",
    )
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise too_large
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


@r.post(
    "",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": FileUploadRequest.model_json_schema()}},
        }
    },
)
async def upload_file(request: Request) -> DocumentFile:
    """
    Process private file uploads from chat UI.

    Raises:
        HTTPException: 413 past the size cap, 422 for a malformed body, 500 for processing errors
    """
    body = await read_bounded_body(request, max_body_bytes())
    try:
        upload = FileUploadRequest.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_input=False)) from exc
    try:
        logger.info(f"Processing file: {upload.name}")
        return await run_in_threadpool(
            FileService.process_private_file, upload.name, upload.base64, upload.params
        )
    except UploadRejected as e:  # the index step would reject it: nothing was written
        logger.info(f"Upload refused ({e.reason}): {e}")
        raise HTTPException(status_code=e.status, detail=f"{e.reason}: {e}") from e
    except IngestLockTimeout as e:  # another indexing run held the lock: nothing was written
        logger.warning(f"Upload refused, indexing is busy: {e}")  # names the table; the client is only told to retry
        raise HTTPException(status_code=503, detail="indexing is busy (another indexing run holds the ingest lock), try again",
                            headers={"Retry-After": "10"}) from e
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing file")
