"""What a client is told when the API fails: a fixed text and a request id, never an exception's text.

An exception's text can name the database host, its port and user (a driver error), paths and tables. It goes to
the server log, with the id of the request it happened in (app.observability.report_failure); the client gets a
fixed text written in this repository and the same id.

- RequestIdMiddleware gives every request a new id: the X-Request-ID header of its response, and the id under which
  anything that fails while serving it is logged (a contextvar, so worker threads see it too).
- failure(exc, what, status_code=..., detail=...) logs the exception and returns a PublicError: an HTTPException
  answered as {"detail": <the fixed text>, "request_id": <the id>}. `detail` stays a string, as before.
- An exception no route catches is answered 500 {"detail": "internal error", "request_id": ...} (it used to be
  Starlette's plain-text "Internal Server Error"), and logged with its traceback under that id.
- public_startup_error(): a failed start is reported by its class and the id its attempt was logged under; the
  texts this repository writes for an operator (SHOWN_IN_FULL) are shown as written.

install(app) adds the middleware and both handlers (main.py). An app without them (the mobile fixture backend mounts
retrieve_router alone) still never sends an exception's text: FastAPI answers a PublicError with its fixed detail,
and its header carries the id.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

from app.observability import bind_request_id, current_request_id, new_request_id, public_failure, report_failure, unbind_request_id

REQUEST_ID_HEADER = "X-Request-ID"
INTERNAL_ERROR = "internal error"
# Start-up errors whose text this repository writes for the operator, naming no host, address, user or path (an
# approval configuration error; the pending migrations and the table they are for; the ingest lock's table and the
# wait): /health, /api/retrieve and /api/plans show them as written. Anything else is shown by its class only.
SHOWN_IN_FULL = ("ApprovalError", "MigrationsPending", "IngestLockTimeout")

logger = logging.getLogger("uvicorn")


class PublicError(HTTPException):
    """An HTTP error whose detail is a fixed text; what went wrong is in the server log under `request_id`."""

    def __init__(self, status_code: int, detail: str, request_id: str, headers: dict[str, str] | None = None):
        super().__init__(status_code=status_code, detail=detail, headers={**(headers or {}), REQUEST_ID_HEADER: request_id})
        self.request_id = request_id


def failure(exc: BaseException, what: str, *, status_code: int = 500, detail: str = INTERNAL_ERROR,
            log: logging.Logger = logger, headers: dict[str, str] | None = None, traceback: bool = False) -> PublicError:
    """Log `what` failed with `exc` in full under the request's id; return the fixed answer to raise."""
    request_id, _ = report_failure(log, what, exc, traceback=traceback)
    return PublicError(status_code, detail, request_id, headers)


def public_startup_error(error: str | None, error_id: str | None = None, *, missing: str = "not initialised") -> str:
    """What a client is told of a failed start ("<Class>: <text>", as main.py stores it): a SHOWN_IN_FULL error as
    written, any other by its class and the id the attempt was logged under; `missing` when there is none."""
    if not error:
        return missing
    kind = error.split(":", 1)[0].strip()
    return error if kind in SHOWN_IN_FULL else public_failure(kind, error_id)


class RequestIdMiddleware:
    """Gives every HTTP request an id (a contextvar and request.state.request_id) and sends it as X-Request-ID."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        token = bind_request_id(request_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            unbind_request_id(token)


def _request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", None) or current_request_id() or new_request_id()


async def _public_error(request: Request, exc: PublicError) -> JSONResponse:
    return JSONResponse({"detail": exc.detail, "request_id": exc.request_id}, status_code=exc.status_code, headers=exc.headers)


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Any exception a route lets escape: logged in full with its traceback, answered with a fixed 500. (Starlette
    re-raises it after this answer, so the server also logs it its own way.)"""
    request_id = _request_id_of(request)
    report_failure(logger, f"{request.method} {request.url.path}", exc, error_id=request_id, traceback=True)
    return JSONResponse({"detail": INTERNAL_ERROR, "request_id": request_id}, status_code=500,
                        headers={REQUEST_ID_HEADER: request_id})


def install(app: FastAPI) -> None:
    """The request ids and the two handlers, on the app (main.py)."""
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(PublicError, _public_error)
    app.add_exception_handler(Exception, _unhandled)
