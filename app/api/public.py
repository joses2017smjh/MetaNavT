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
before it is read (BodyLimit), and with WEB_ORIGIN set only that origin may call the API from a browser. With
ORIGIN_VERIFY_SECRET set (the single host behind CloudFront, doc/deploy.md), a request without CloudFront's
X-Origin-Verify header is refused with 403 before anything else runs (OriginVerify).
"""

from __future__ import annotations

import hmac
import ipaddress
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


# The header CloudFront adds to every request it sends to the single host (infra/modules/single-host); its value is
# ORIGIN_VERIFY_SECRET, a random string Terraform keeps in Parameter Store and the host reads at start.
ORIGIN_VERIFY_HEADER = "x-origin-verify"


def origin_verify_secret() -> str:
    """ORIGIN_VERIFY_SECRET, stripped; empty means no origin check (main.py installs OriginVerify only with a value)."""
    return os.getenv("ORIGIN_VERIFY_SECRET", "").strip()


class OriginVerify:
    """ASGI middleware: 403 for a request that does not carry X-Origin-Verify with the secret, compared in constant time.

    For an origin that CloudFront reaches with no load balancer in between (the single host): its security group
    admits only CloudFront's addresses, which every CloudFront distribution shares, so the header is what tells this
    distribution's requests from another account's. One request passes without it: GET /health from the loopback
    address, the container's own health check, which Compose, the image's HEALTHCHECK and deploy.sh run inside the
    container. A request from outside the container, through the published port, never arrives from the loopback
    address (Docker's port forwarding keeps a remote client's address, and a request from the host itself arrives
    from the bridge's gateway)."""

    def __init__(self, app, secret: str):
        secret = (secret or "").strip()
        if not secret:
            raise ValueError("OriginVerify needs a value (ORIGIN_VERIFY_SECRET); leave the middleware out instead")
        self.app, self._secret = app, secret.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or self._verified(scope) or self._local_health_check(scope):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":  # closed before it is accepted: the server answers 403
            return await send({"type": "websocket.close", "code": 1008})
        body = b'{"detail":"forbidden"}'
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    def _verified(self, scope) -> bool:
        values = [value for name, value in scope.get("headers") or [] if name == ORIGIN_VERIFY_HEADER.encode()]
        # exactly one copy: a second one would be the viewer's own (CloudFront overwrites a viewer's header of this name)
        return len(values) == 1 and hmac.compare_digest(values[0], self._secret)

    @staticmethod
    def _local_health_check(scope) -> bool:
        if scope["type"] != "http" or scope.get("method") != "GET" or scope.get("path") != "/health":
            return False
        client = scope.get("client") or (None,)
        try:
            return ipaddress.ip_address(client[0]).is_loopback
        except (TypeError, ValueError):  # no client, or a name such as "testclient"
            return False
