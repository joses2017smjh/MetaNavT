"""Authentication and a request budget for the /api routes. Both are opt-in.

Authentication (METANAVIT_API_KEY)
    Unset or empty: the API is open, as it always was, and /health reports
    "auth": {"enabled": false}. The Compose stack, the mobile fixture backend
    and the test suites run this way.
    Set: every route of the API router and both static mounts under /api
    require the key, sent as `X-API-Key: <key>` or `Authorization: Bearer <key>`.
    A request without it, or with a wrong one, gets 401. The comparison is
    constant-time. /health is not under /api and stays reachable for probes.
    This is one shared key: it says "a holder of the key", not who. The
    `actor` field of a plan decision is still whatever the client sends.

Request budget (RATE_LIMIT_PER_MINUTE, RATE_LIMIT_BURST)
    Off unless RATE_LIMIT_PER_MINUTE is a positive number. When on, each
    identity gets a token bucket; a request with no token left gets 429 with a
    Retry-After header. The identity is the API key when authentication is on
    (so all holders of the one key share a bucket) and the client address
    otherwise (behind a proxy that is the proxy's address unless uvicorn is
    started with --proxy-headers).

    The limiter is IN-PROCESS, so the budget is PER REPLICA: each API process
    keeps its own buckets, N replicas admit up to N times the configured
    budget, and a restart forgets every bucket. It bounds what one process
    will do for one caller; it is not a global quota. A shared limit belongs
    at the proxy or in a shared store, which this module does not provide.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import threading
import time
from typing import Callable

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

API_KEY_ENV = "METANAVIT_API_KEY"
RATE_ENV = "RATE_LIMIT_PER_MINUTE"
BURST_ENV = "RATE_LIMIT_BURST"
MAX_TRACKED_IDENTITIES = 10_000  # buckets kept in memory before idle ones are dropped


# ---------------------------------------------------------------- authentication


def configured_api_key() -> str | None:
    """The key from the environment, read per request; None means authentication is off."""
    key = os.getenv(API_KEY_ENV, "").strip()
    return key or None


def _presented_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    scheme, _, credential = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and credential.strip():
        return credential.strip()
    return None


def _same_key(presented: str, expected: str) -> bool:
    """Constant-time comparison. Hashing first makes both sides the same length, so the
    comparison time does not depend on how long the presented value is."""
    return hmac.compare_digest(
        hashlib.sha256(presented.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )


async def require_api_key(request: Request) -> str:
    """Router dependency. Returns the caller's identity for the request budget.

    Open (no key configured): identity is the client address.
    Key configured: 401 unless the request carries it; identity is a digest of the key
    (the key itself is never stored, logged or returned).
    """
    expected = configured_api_key()
    if expected is None:
        client = request.client.host if request.client else "unknown"
        return f"addr:{client}"
    presented = _presented_key(request)
    if presented is None or not _same_key(presented, expected):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API key (send X-API-Key or Authorization: Bearer)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return "key:" + hashlib.sha256(expected.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- request budget


class TokenBucketLimiter:
    """One token bucket per identity, held in this process's memory.

    PER REPLICA: the buckets live in one process. Two replicas each admit the full
    budget, and a restart starts every caller with a full bucket. See the module
    docstring; do not read this as a global rate limit.

    `rate_per_minute` tokens are added per minute up to `burst`; each request takes one.
    """

    def __init__(self, rate_per_minute: float, burst: int, clock: Callable[[], float] = time.monotonic,
                 max_identities: int = MAX_TRACKED_IDENTITIES):
        if rate_per_minute <= 0 or burst < 1:
            raise ValueError("rate_per_minute must be > 0 and burst >= 1")
        self.rate_per_second = rate_per_minute / 60.0
        self.burst = burst
        self.clock = clock
        self.max_identities = max_identities
        self._buckets: dict[str, tuple[float, float]] = {}  # identity -> (tokens, last refill time)
        self._lock = threading.Lock()

    def take(self, identity: str) -> float:
        """Take one token. Returns 0.0 when admitted, else the seconds until a token is available."""
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get(identity, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.rate_per_second)
            if tokens >= 1.0:
                self._buckets[identity] = (tokens - 1.0, now)
                wait = 0.0
            else:
                self._buckets[identity] = (tokens, now)
                wait = (1.0 - tokens) / self.rate_per_second
            if len(self._buckets) > self.max_identities:
                self._drop_idle(now)
            return wait

    def _drop_idle(self, now: float) -> None:
        """Forget buckets that have refilled completely: a full bucket carries no information.
        If every tracked identity is active, forget the least recently seen ones instead."""
        full = [key for key, (tokens, last) in self._buckets.items()
                if tokens + (now - last) * self.rate_per_second >= self.burst]
        for key in full:
            del self._buckets[key]
        excess = len(self._buckets) - self.max_identities
        if excess > 0:
            for key in sorted(self._buckets, key=lambda k: self._buckets[k][1])[:excess]:
                del self._buckets[key]


def budget_config() -> tuple[float, int] | None:
    """(tokens per minute, burst) from the environment, or None when the budget is off."""
    try:
        rate = float(os.getenv(RATE_ENV, "0") or 0)
    except ValueError:
        return None
    if rate <= 0:
        return None
    try:
        burst = int(os.getenv(BURST_ENV, "") or math.ceil(rate))
    except ValueError:
        burst = math.ceil(rate)
    return rate, max(1, burst)


_limiter: TokenBucketLimiter | None = None
_limiter_config: tuple[float, int] | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> TokenBucketLimiter | None:
    """The process-wide limiter for the current environment settings (None when off)."""
    global _limiter, _limiter_config
    config = budget_config()
    with _limiter_lock:
        if config != _limiter_config:
            _limiter = TokenBucketLimiter(*config) if config else None
            _limiter_config = config
        return _limiter


def reset_limiter() -> None:
    """Forget every bucket (tests)."""
    global _limiter, _limiter_config
    with _limiter_lock:
        _limiter, _limiter_config = None, None


async def enforce_request_budget(identity: str = Depends(require_api_key)) -> None:
    """Router dependency: 429 with Retry-After when this identity's bucket is empty. Per replica."""
    limiter = get_limiter()
    if limiter is None:
        return
    wait = limiter.take(identity)
    if wait > 0:
        seconds = max(1, math.ceil(wait))
        raise HTTPException(
            status_code=429,
            detail=f"request budget exceeded for this caller; retry in {seconds} s",
            headers={"Retry-After": str(seconds)},
        )


# ---------------------------------------------------------------- static mounts and /health


class GuardedStaticFiles(StaticFiles):
    """StaticFiles behind the same key and budget as the API router.

    A mounted sub-application is not a route of the router, so the router's dependencies
    never run for it. Without this wrapper the corpus files and the uploads under
    /api/files/ would stay public after a key is configured.
    """

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            try:
                await enforce_request_budget(await require_api_key(Request(scope)))
            except HTTPException as exc:
                response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
                await response(scope, receive, send)
                return
        await super().__call__(scope, receive, send)


def security_status() -> dict:
    """What /health reports. Never includes the key."""
    config = budget_config()
    return {
        "auth": {
            "enabled": configured_api_key() is not None,
            "scheme": "X-API-Key header or Authorization: Bearer",
            "scope": "/api routes and /api/files mounts; /health is always open",
        },
        "rate_limit": {
            "enabled": config is not None,
            "per_minute": config[0] if config else None,
            "burst": config[1] if config else None,
            "scope": "per replica (in-process buckets)",
        },
    }
