"""Shared quotas, the daily budget and the answer cache (app/api/limits.py) over an in-memory Redis stand-in, and the
client address behind trusted proxies."""

import pytest

from app.api.limits import MICRO, LimitSettings, Limits, client_address


class FakeRedis:
    """The few commands Limits uses, with redis-py's return types (bytes from get). `down` raises like a lost link."""

    def __init__(self):
        self.data, self.ttl, self.down = {}, {}, False

    def _up(self):
        if self.down:
            raise ConnectionError("redis unreachable")

    def incrby(self, key, amount):
        self._up()
        self.data[key] = int(self.data.get(key, 0)) + amount
        return self.data[key]

    def incr(self, key):
        return self.incrby(key, 1)

    def decrby(self, key, amount):
        return self.incrby(key, -amount)

    def expire(self, key, seconds):
        self._up()
        self.ttl[key] = seconds
        return True

    def get(self, key):
        self._up()
        value = self.data.get(key)
        return None if value is None else (value if isinstance(value, bytes) else str(value).encode())

    def set(self, key, value, ex=None):
        self._up()
        self.data[key], self.ttl[key] = value.encode() if isinstance(value, str) else value, ex
        return True


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


def limits(clock, **kw):
    return Limits(FakeRedis(), LimitSettings(**{"per_minute": 3, "per_day": 5, "daily_budget_usd": 1.0, **kw}), clock=clock)


def test_a_caller_gets_its_quota_per_minute_and_per_day(clock):
    lim = limits(clock)
    assert [lim.allow("ip:1.2.3.4")[0] for _ in range(4)] == [True, True, True, False]
    allowed, retry = lim.allow("ip:1.2.3.4")
    assert not allowed and 1 <= retry <= 61
    assert lim.allow("ip:5.6.7.8") == (True, None)  # another caller has its own counters
    clock.t += 60  # a new minute: the minute window resets, the day window keeps counting
    allowed, retry = lim.allow("ip:1.2.3.4")  # the sixth request today, over the day's five
    assert not allowed and retry > 61  # wait for the next UTC day, not the next minute
    assert all("1.2.3.4" not in key for key in lim.redis.data)  # callers are stored hashed


def test_the_budget_reserves_before_the_call_and_settles_after(clock):
    lim = limits(clock, daily_budget_usd=1.0)
    first = lim.reserve(0.60)
    assert first is not None and lim.spent_today_usd() == pytest.approx(0.60, abs=1e-5)
    assert lim.reserve(0.50) is None  # 0.6 + 0.5 is over 1.0: refused, and nothing is left counted
    assert lim.spent_today_usd() == pytest.approx(0.60, abs=1e-5)
    lim.settle(first, 0.05)  # the reply cost less than its worst case: the rest goes back
    assert lim.spent_today_usd() == pytest.approx(0.05, abs=1e-5)
    second = lim.reserve(0.50)
    lim.settle(second, None)  # unknown cost (a timeout): the whole reservation stays
    assert lim.spent_today_usd() == pytest.approx(0.55, abs=1e-5)
    clock.t += 86_400  # a new UTC day starts from zero
    assert lim.spent_today_usd() == 0 and lim.reserve(0.9) is not None


def test_concurrent_reservations_cannot_overshoot_the_budget(clock):
    lim = limits(clock, daily_budget_usd=1.0)
    granted = [lim.reserve(0.3) for _ in range(5)]  # each one is counted before the next is checked
    assert sum(r is not None for r in granted) == 3 and lim.spent_today_usd() <= 1.0


def test_the_answer_cache_keys_on_model_and_normalised_question(clock):
    lim = limits(clock, cache_ttl_s=60)
    key = Limits.cache_key("claude-opus-5-5", "  What   rate did Run 47 use?")
    assert key == Limits.cache_key("claude-opus-5-5", "what rate did run 47 use?")
    assert key != Limits.cache_key("claude-haiku-5-5", "what rate did run 47 use?")
    assert lim.cache_get(key) is None
    lim.cache_set(key, {"answer": "3e-4"})
    assert lim.cache_get(key) == {"answer": "3e-4"} and lim.redis.ttl[key] == 60


@pytest.mark.parametrize("peer, header, hops, expected", [
    ("10.0.0.2", None, 0, "10.0.0.2"),  # no trusted proxy: the peer
    ("10.0.0.2", "198.51.100.7", 0, "10.0.0.2"),  # a header is ignored when no proxy is trusted
    ("10.0.0.2", "198.51.100.7", 1, "198.51.100.7"),  # behind Caddy: the address it appended
    ("10.0.0.2", "6.6.6.6, 198.51.100.7", 1, "198.51.100.7"),  # a client's own header entry is ignored
    ("10.0.0.3", "6.6.6.6, 198.51.100.7, 54.0.0.9", 2, "198.51.100.7"),  # CloudFront then the ALB
    ("10.0.0.2", "198.51.100.7", 2, "10.0.0.2"),  # fewer entries than trusted hops: did not come through them
    ("10.0.0.2", "not-an-ip", 1, "10.0.0.2"),
])
def test_the_client_address_behind_trusted_proxies(peer, header, hops, expected):
    assert client_address(peer, header, hops) == expected


def test_settings_from_the_environment(monkeypatch):
    for name, value in {"ANSWER_LIMIT_PER_MINUTE": "2", "ANSWER_LIMIT_PER_DAY": "x", "ANSWER_DAILY_BUDGET_USD": "0.5",
                        "ANSWER_CACHE_TTL_SECONDS": "-1"}.items():
        monkeypatch.setenv(name, value)
    s = LimitSettings.from_env()
    assert (s.per_minute, s.per_day, s.daily_budget_usd, s.cache_ttl_s) == (2, 50, 0.5, 86_400)
    assert int(0.5 * MICRO) == 500_000
