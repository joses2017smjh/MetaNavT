"""Authentication and a request budget for the /api routes. Both are opt-in.

Authentication (METANAVIT_API_KEY, METANAVIT_API_KEYS_FILE)
    Neither set (or both empty): the API is open, as it always was, and /health
    reports "auth": {"enabled": false}. The Compose stack, the mobile fixture
    backend and the test suites run this way.
    Either set: every route of the API router and both static mounts under /api
    require a key, sent as `X-API-Key: <key>` or `Authorization: Bearer <key>`.
    A request without one, or with a wrong one, gets 401. /health is not under
    /api and stays reachable for probes.
    Each key is an identity (Caller.name): the name of its line in the key file,
    or SHARED_KEY_IDENTITY ("shared-key") for METANAVIT_API_KEY, which every
    holder of that one key shares. The identity is the request budget's key and
    the actor of a plan decision (app/api/routers/plans.py), whatever the body
    of the request says. With authentication off there is no identity: the
    budget counts the client address and a decision's actor is what the client
    sends.
    The key file (METANAVIT_API_KEYS_FILE; `python -m app.api.keys add` writes
    it) holds one "name:<SHA-256 of the key, hex>" line per key, never a key.
    The SHA-256 of a presented key is compared with every configured digest
    (the shared key's and each line's) with hmac.compare_digest, without
    stopping at a match. Keys are never logged or echoed. The file is read again
    when it changes (path, size, mtime, inode), so a line can be removed to
    revoke a key without a restart. A file that cannot be used (missing,
    unreadable, a malformed or duplicate line, the reserved name, a path inside
    DATA_DIR or the output directory, which /api/files serves) refuses every
    request with 503 rather than open the API; the log names the line, not its
    text.

Request budget (RATE_LIMIT_PER_MINUTE, RATE_LIMIT_BURST)
    Off unless RATE_LIMIT_PER_MINUTE is a positive number. When on, each
    identity gets a token bucket; a request with no token left gets 429 with a
    Retry-After header. The identity is the key's name when authentication is
    on (all holders of the shared key share one bucket; each named key has its
    own) and the client address otherwise (behind a proxy that is the proxy's
    address unless uvicorn is started with --proxy-headers).

    The limiter is IN-PROCESS, so the budget is PER REPLICA: each API process
    keeps its own buckets, N replicas admit up to N times the configured
    budget, and a restart forgets every bucket. It bounds what one process
    will do for one caller; it is not a global quota. A shared limit belongs
    at the proxy or in a shared store, which this module does not provide.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

API_KEY_ENV = "METANAVIT_API_KEY"
API_KEYS_FILE_ENV = "METANAVIT_API_KEYS_FILE"
SHARED_KEY_IDENTITY = "shared-key"  # the identity of METANAVIT_API_KEY; reserved in the key file
KEY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")  # a named key's identity: the decision actor, the budget key
KEYS_UNAVAILABLE = "API keys unavailable (the server log has the reason)"
RATE_ENV = "RATE_LIMIT_PER_MINUTE"
BURST_ENV = "RATE_LIMIT_BURST"
MAX_TRACKED_IDENTITIES = 10_000  # buckets kept in memory before idle ones are dropped

logger = logging.getLogger("uvicorn")


# ---------------------------------------------------------------- authentication


def configured_api_key() -> str | None:
    """The shared key from the environment, read per request; None when it is not set."""
    key = os.getenv(API_KEY_ENV, "").strip()
    return key or None


def configured_keys_file() -> Path | None:
    """The named-key file from the environment, read per request; None when it is not set."""
    path = os.getenv(API_KEYS_FILE_ENV, "").strip()
    return Path(path) if path else None


def authentication_enabled() -> bool:
    """True when a shared key or a key file is configured: the one test of "is authentication on" (the API router,
    /health, the code-execution gate). A configured file that cannot be used still counts: it refuses every request."""
    return configured_api_key() is not None or configured_keys_file() is not None


@dataclass(frozen=True)
class Caller:
    """Who sent a request. `name` is the key's identity when authentication is on, None when the API is open;
    `budget_key` is what the request budget counts by ("key:<name>" or "addr:<client address>")."""

    name: str | None
    budget_key: str


class KeyFileError(Exception):
    """The named-key file cannot be used. Its message names the file and a line number, never a line's text."""


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


def parse_key_file(text: str) -> dict[str, bytes]:
    """{name: SHA-256 digest} from a key file's text: "name:<64 hex digits>" per line; blank lines and lines starting
    with # are skipped. Refuses a malformed line, a name used twice, a digest used twice and the reserved name."""
    keys: dict[str, bytes] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, digest = line.partition(":")
        if not sep or not KEY_NAME.fullmatch(name) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise KeyFileError(f"line {number} is not name:<SHA-256 of the key, 64 hex digits>")
        if name == SHARED_KEY_IDENTITY:
            raise KeyFileError(f"line {number} uses the reserved name {SHARED_KEY_IDENTITY} (the identity of {API_KEY_ENV})")
        if name in keys:
            raise KeyFileError(f"line {number} repeats a name")
        value = bytes.fromhex(digest)
        if value in keys.values():
            raise KeyFileError(f"line {number} repeats another line's key")
        keys[name] = value
    return keys


def _served_roots() -> list[Path]:
    """The directories /api/files serves (main.py's mounts): a key file there would be readable, and indexed."""
    return [Path(os.getenv("DATA_DIR", "data")).resolve(), Path("output").resolve()]


_keys_lock = threading.Lock()
_keys_cache: tuple[tuple, dict[str, bytes] | KeyFileError] | None = None


def named_keys(path: Path) -> dict[str, bytes]:
    """The key file's {name: digest}, read again whenever the file changes. Raises KeyFileError (logged once per
    change of the file) when it cannot be used."""
    global _keys_cache
    roots = tuple(str(root) for root in _served_roots())
    try:
        resolved = path.resolve()
        st = resolved.stat()
        stamp = (str(resolved), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, roots)
    except OSError as exc:
        stamp = (str(path), "unreadable", type(exc).__name__, roots)
    with _keys_lock:
        if _keys_cache is not None and _keys_cache[0] == stamp:
            found = _keys_cache[1]
        else:
            found = _load_key_file(path)
            _keys_cache = (stamp, found)
            if isinstance(found, KeyFileError):
                logger.error(f"{API_KEYS_FILE_ENV}={path}: {found}; every /api request is refused (503) until it is fixed")
            else:
                logger.info(f"{API_KEYS_FILE_ENV}: {len(found)} named key(s) read")
    if isinstance(found, KeyFileError):
        raise found
    return found


def _load_key_file(path: Path) -> dict[str, bytes] | KeyFileError:
    try:
        roots, resolved = _served_roots(), path.resolve()
        if any(place == root or root in place.parents for place in (path.absolute(), resolved) for root in roots):
            return KeyFileError("the key file is inside a directory /api/files serves (DATA_DIR or output)")
        return parse_key_file(resolved.read_text(encoding="utf-8"))
    except KeyFileError as exc:
        return exc
    except (OSError, UnicodeDecodeError) as exc:
        return KeyFileError(f"cannot be read ({type(exc).__name__})")


def reset_key_cache() -> None:
    """Forget the key file's last reading (tests)."""
    global _keys_cache
    with _keys_lock:
        _keys_cache = None


def _presented_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    scheme, _, credential = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and credential.strip():
        return credential.strip()
    return None


def _identify(presented: str, shared: str | None, named: dict[str, bytes]) -> str | None:
    """The identity whose key is `presented`, or None. Every configured digest is compared, in constant time, and a
    match does not stop the loop, so the time taken does not say which entry matched or how much of one did."""
    digest = _digest(presented)
    candidates = ([(SHARED_KEY_IDENTITY, _digest(shared))] if shared is not None else []) + list(named.items())
    found = None
    for name, expected in candidates:
        if hmac.compare_digest(digest, expected) and found is None:
            found = name
    return found


async def require_api_key(request: Request) -> Caller:
    """Router dependency: who is calling (Caller), for the request budget and the plan routes.

    Open (no key configured): no identity; the budget counts the client address.
    Keys configured: 401 unless the request carries one of them; 503 when the key file cannot be used. The key itself
    is never stored, logged or returned.
    """
    shared, keys_file = configured_api_key(), configured_keys_file()
    if shared is None and keys_file is None:
        client = request.client.host if request.client else "unknown"
        return Caller(name=None, budget_key=f"addr:{client}")
    try:
        named = named_keys(keys_file) if keys_file is not None else {}
    except KeyFileError:
        raise HTTPException(status_code=503, detail=KEYS_UNAVAILABLE) from None
    presented = _presented_key(request)
    name = _identify(presented, shared, named) if presented is not None else None
    if name is None:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid API key (send X-API-Key or Authorization: Bearer)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Caller(name=name, budget_key=f"key:{name}")


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


async def enforce_request_budget(caller: Caller = Depends(require_api_key)) -> None:
    """Router dependency: 429 with Retry-After when this caller's bucket is empty. Per replica."""
    limiter = get_limiter()
    if limiter is None:
        return
    wait = limiter.take(caller.budget_key)
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
            "enabled": authentication_enabled(),
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
