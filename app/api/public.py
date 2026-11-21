"""The public demo's API surface (PUBLIC_DEMO=1): an allow-list, nothing else.

With PUBLIC_DEMO on, main.py mounts this router under /api instead of the full api_router, and leaves out the file
mounts: no chat, agents, code execution, uploads, files, plans or approvals, MCP. Only:

  POST /api/retrieve/   the served hybrid retrieval with byte-range citations (k capped at PUBLIC_MAX_K, query length
                        capped at PUBLIC_MAX_QUERY_CHARS)
  POST /api/answer/     retrieval, then a cited answer from Claude under shared quotas and a daily budget
                        (app/api/routers/answer.py)

GET /health, the OpenAPI docs and the web client's static files stay where main.py puts them. The router carries the
same opt-in dependencies as api_router (app/api/security.py): the API key check (invite keys through
METANAVIT_API_KEYS_FILE) and the per-replica request budget. A body larger than MAX_BODY_BYTES is refused with 413
before it is read (BodyLimit), and with WEB_ORIGIN set only that origin may call the API from a browser.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from app.api.routers.answer import answer_router
from app.api.routers.retrieve import RetrieveRequest, RetrieveResponse, retrieve
from app.api.security import enforce_request_budget, require_api_key

PUBLIC_MAX_K = 10
PUBLIC_MAX_QUERY_CHARS = 500
DEFAULT_MAX_BODY_BYTES = 8192
# Every path the public API serves under /api, with its methods: tests/api/test_public_demo.py compares the app's
# route table with this list.
PUBLIC_API_ROUTES = {("/api/retrieve/", "POST"), ("/api/answer/", "POST")}


def public_demo_enabled() -> bool:
    return os.getenv("PUBLIC_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}


def max_body_bytes() -> int:
    try:
        value = int(os.getenv("MAX_BODY_BYTES", "").strip())
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES
    return value if value > 0 else DEFAULT_MAX_BODY_BYTES


class PublicRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=PUBLIC_MAX_QUERY_CHARS)
    k: int = Field(8, ge=1, le=PUBLIC_MAX_K, description=f"number of hits to return (1-{PUBLIC_MAX_K})")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


_retrieve = APIRouter()


@_retrieve.post("/", response_model=RetrieveResponse)
async def public_retrieve(req: PublicRetrieveRequest, request: Request) -> RetrieveResponse:
    """The served retrieval route with the public caps on k and the query length."""
    return await retrieve(RetrieveRequest(query=req.query, k=req.k), request)


def public_api_router() -> APIRouter:
    router = APIRouter(dependencies=[Depends(require_api_key), Depends(enforce_request_budget)])
    router.include_router(_retrieve, prefix="/retrieve")
    router.include_router(answer_router, prefix="/answer")
    return router


class BodyLimit:
    """ASGI middleware: 413 for a request body over `limit` bytes, by Content-Length before the body is read, and by
    counting the bytes as they arrive when there is none (a chunked upload)."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope.get("headers") or []).get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.limit):
            return await self._refuse(send)
        seen = 0

        async def counted():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise _TooLarge()
            return message

        try:
            return await self.app(scope, counted, send)
        except _TooLarge:
            return await self._refuse(send)

    async def _refuse(self, send):
        body = b'{"detail":"request body too large"}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class _TooLarge(Exception):
    pass
