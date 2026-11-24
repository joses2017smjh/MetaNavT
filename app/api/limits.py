"""Shared limits for the public demo's answer route, in Redis: per-caller quotas, a daily dollar budget, an answer cache.

Unlike the in-process token bucket of app/api/security.py (per replica, forgotten on restart), these live in one Redis
that every API replica uses, so the quotas and the budget hold across replicas and restarts.

Per-caller quotas (ANSWER_LIMIT_PER_MINUTE, ANSWER_LIMIT_PER_DAY)
    Fixed windows: a counter per caller per UTC minute and per UTC day (INCR, then EXPIRE on the first increment).
    The caller is the API key's name when authentication is on, and the client address otherwise (client_address).

Daily budget (ANSWER_DAILY_BUDGET_USD)
    Reserve, then settle. Before a call, the worst case of that call (app.answer.claude.worst_case_usd) is added to the
    day's counter with one INCRBY, in micro-dollars; if the new total is over the budget, the same amount is taken
    back and the call is not made. After the call, the counter is moved by (actual - reserved), so a cheap reply gives
    back most of its reservation. The counter moved is the one the reservation went to (Reservation.key), so a call
    reserved before UTC midnight and settled after it is settled on its own day. A call that fails with a timeout
    keeps its whole reservation (it may have been billed). Concurrent requests cannot overshoot the budget by more than one reservation each, since every
    reservation is counted before its call starts.

Answer cache (ANSWER_CACHE_TTL_SECONDS)
    A delivered answer is stored under the SHA-256 of the model and the normalised question, for the TTL; a repeat is
    served from it, with no call and no cost.

If Redis cannot be reached, the answer route fails closed (no Claude call) and the retrieval route keeps working.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

MICRO = 1_000_000
DEFAULT_PER_MINUTE = 5
DEFAULT_PER_DAY = 50
DEFAULT_DAILY_BUDGET_USD = 2.00
DEFAULT_CACHE_TTL_S = 86_400


def _number(name: str, default: float, kind=float) -> float:
    try:
        value = kind(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value >= 0 else default


@dataclass(frozen=True)
class LimitSettings:
    per_minute: int = DEFAULT_PER_MINUTE
    per_day: int = DEFAULT_PER_DAY
    daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD
    cache_ttl_s: int = DEFAULT_CACHE_TTL_S

    @classmethod
    def from_env(cls) -> "LimitSettings":
        return cls(per_minute=int(_number("ANSWER_LIMIT_PER_MINUTE", DEFAULT_PER_MINUTE, int)),
                   per_day=int(_number("ANSWER_LIMIT_PER_DAY", DEFAULT_PER_DAY, int)),
                   daily_budget_usd=_number("ANSWER_DAILY_BUDGET_USD", DEFAULT_DAILY_BUDGET_USD),
                   cache_ttl_s=int(_number("ANSWER_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_S, int)))


@dataclass(frozen=True)
class Reservation:
    """What reserve() counted: `micros` on the budget counter `key`, the UTC day the call was reserved on."""

    micros: int
    key: str


class Limits:
    """Quotas, budget and cache over a Redis client (redis.Redis, or anything with get/set/incr/incrby/decrby/expire)."""

    def __init__(self, redis, settings: LimitSettings, clock: Callable[[], float] = time.time):
        self.redis, self.settings, self.clock = redis, settings, clock

    def _window(self, key: str, ttl: int) -> int:
        count = int(self.redis.incr(key))
        if count == 1:
            self.redis.expire(key, ttl)
        return count

    def allow(self, caller: str) -> tuple[bool, int | None]:
        """(allowed, seconds until the caller may try again). Counts this request in both windows."""
        now = self.clock()
        minute, day = int(now // 60), time.strftime("%Y-%m-%d", time.gmtime(now))
        tag = hashlib.sha256(caller.encode()).hexdigest()[:16]  # no address or key name stored in Redis
        per_minute = self._window(f"answer:rl:{tag}:m:{minute}", 120)
        per_day = self._window(f"answer:rl:{tag}:d:{day}", 2 * 86_400)
        if self.settings.per_day and per_day > self.settings.per_day:
            return False, int(86_400 - now % 86_400) + 1
        if self.settings.per_minute and per_minute > self.settings.per_minute:
            return False, int(60 - now % 60) + 1
        return True, None

    def _budget_key(self) -> str:
        return "answer:budget:" + time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

    def reserve(self, usd: float) -> Reservation | None:
        """Reserve `usd` against today's budget; the Reservation, or None when it does not fit."""
        micros = max(1, int(usd * MICRO) + 1)
        key = self._budget_key()
        total = int(self.redis.incrby(key, micros))
        if total == micros:
            self.redis.expire(key, 2 * 86_400)
        if total > int(self.settings.daily_budget_usd * MICRO):
            self.redis.decrby(key, micros)
            return None
        return Reservation(micros, key)

    def settle(self, reservation: Reservation, actual_usd: float | None) -> None:
        """Replace a reservation by the actual cost, on the day it was reserved on (its key, not today's: a call can
        cross UTC midnight); None keeps the whole reservation (the cost is unknown)."""
        if actual_usd is None:
            return
        delta = int(actual_usd * MICRO) - reservation.micros
        if delta:
            self.redis.incrby(reservation.key, delta)

    def spent_today_usd(self) -> float:
        return int(self.redis.get(self._budget_key()) or 0) / MICRO

    @staticmethod
    def cache_key(model: str, question: str, corpus: str = "") -> str:
        normalised = " ".join(question.lower().split())
        return "answer:cache:" + hashlib.sha256(f"{model}\n{corpus}\n{normalised}".encode()).hexdigest()

    def cache_get(self, key: str) -> dict[str, Any] | None:
        raw = self.redis.get(key)
        return json.loads(raw) if raw else None

    def cache_set(self, key: str, value: dict[str, Any]) -> None:
        if self.settings.cache_ttl_s:
            self.redis.set(key, json.dumps(value), ex=self.settings.cache_ttl_s)


def connect(url: str | None = None):
    """A redis.Redis client for REDIS_URL, or None when it is unset. Short timeouts: a slow Redis fails closed."""
    url = url if url is not None else os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    import redis  # imported here: only the public demo needs it

    return redis.Redis.from_url(url, socket_timeout=2.0, socket_connect_timeout=2.0)


def trusted_proxy_hops() -> int:
    return int(_number("TRUSTED_PROXY_HOPS", 0, int))


def client_address(peer: str | None, forwarded_for: str | None, hops: int) -> str | None:
    """The client's address behind `hops` trusted proxies. Each proxy appends the address it received the request from
    to X-Forwarded-For, so with hops trusted proxies the client is the hops-th entry from the right; anything a client
    puts in the header itself sits further left and is ignored. With no trusted proxy, or a header shorter than hops
    (a request that did not come through them), the peer address is used."""
    if hops <= 0 or not forwarded_for:
        return peer
    entries = [part.strip() for part in forwarded_for.split(",") if part.strip()]
    if len(entries) < hops:
        return peer
    candidate = entries[-hops]
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return peer
