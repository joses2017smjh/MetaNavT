"""The public demo (PUBLIC_DEMO=1): exactly the allow-listed routes, and the answer route's order of checks."""

import functools
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.answer import claude
from app.api.limits import LimitSettings, Limits
from app.api.public import PUBLIC_API_ROUTES, BodyLimit, public_api_router

ROOT = Path(__file__).resolve().parents[2]

_ROUTES = r"""
import json, sys
import main
from fastapi.testclient import TestClient
spec = main.app.openapi()
api = sorted([path, method.upper()] for path, ops in spec["paths"].items() for method in ops)
mounts = [r.path for r in main.app.routes if type(r).__name__ == "Mount"]
plain = [r.path for r in main.app.routes if getattr(r, "path", None)]
probe = TestClient(main.app)  # no lifespan: only routing is exercised
hidden = {p: probe.post(p, json={}).status_code for p in sys.argv[1:]}
health = probe.get("/health")  # nothing built without the lifespan: the 503 answer
print(json.dumps({"public": main.PUBLIC_DEMO, "api": api, "mounts": mounts, "paths": plain, "hidden": hidden,
                  "health": {"status_code": health.status_code, "public_demo": health.json().get("public_demo")},
                  "loaded": sorted(m for m in sys.modules if m.startswith("app."))}))
"""
# routes of the full API that the public app must not serve (each answers 404 there)
NOT_PUBLIC = ["/api/chat", "/api/chat/upload", "/api/chat/config", "/api/query/", "/api/plans/", "/api/files/data/x"]


@functools.lru_cache(maxsize=None)  # one interpreter per mode for the tests below; they only read the table
def _app_routes(public: bool) -> dict:
    """main.app's routes, imported in a fresh interpreter with or without PUBLIC_DEMO (main builds at import)."""
    env = {**os.environ, "PUBLIC_DEMO": "1" if public else "", "ENVIRONMENT": "dev", "MODEL_PROVIDER": "ollama",
           "EMBEDDING_PROVIDER": "hash", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("FRONTEND_ENDPOINT", None)
    out = subprocess.run([sys.executable, "-c", _ROUTES, *NOT_PUBLIC], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_public_app_serves_only_the_allow_list():
    table = _app_routes(public=True)
    assert table["public"] is True
    assert {(path, method) for path, method in table["api"] if path.startswith("/api")} == PUBLIC_API_ROUTES
    assert not [path for path in table["mounts"] if path.startswith("/api")]  # no file mounts
    assert {"/health", "/docs", "/openapi.json"} <= set(table["paths"])
    assert all(status == 404 for status in table["hidden"].values()), table["hidden"]
    # the chat stack, agents, uploads and plan routes are never imported, let alone mounted
    assert not [m for m in table["loaded"] if m.startswith(("app.api.routers.chat", "app.api.routers.plans",
                                                              "app.api.routers.upload", "app.engine.agents"))]


def test_without_public_demo_the_full_api_is_mounted():
    table = _app_routes(public=False)
    api = {path for path, _ in table["api"]}
    assert table["public"] is False and {"/api/chat", "/api/retrieve/", "/api/query/"} <= api
    assert "/api/answer/" not in api  # the answer route exists only in the public demo
    assert table["hidden"]["/api/chat"] != 404


def test_health_says_whether_the_api_is_the_public_demo():
    """GET /health names the mode in its 503 answer too (the 200 one: tests/api/test_retrieve_contract.py): the web
    client shows its demo page when it is true, and a starting API answers 503."""
    assert _app_routes(public=True)["health"] == {"status_code": 503, "public_demo": True}
    assert _app_routes(public=False)["health"] == {"status_code": 503, "public_demo": False}


# ---------------------------------------------------------------- the answer route over fakes (no network, no Redis)


class _Outcome:
    def __init__(self):
        from llama_index.core.schema import NodeWithScore, TextNode

        self.nodes = [NodeWithScore(node=TextNode(id_="n1", text="learning_rate: 3e-4",
                                                  metadata={"path": "configs/run_047.yaml", "start_byte": 40, "end_byte": 59}), score=0.03),
                      NodeWithScore(node=TextNode(id_="n2", text="run 47 log", metadata={"path": "logs/run_047.out"}), score=0.01)]
        self.route, self.mode, self.bm25_backend = None, "sql", "ts_rank_cd"
        self.stages_ms, self.counts, self.scores, self.degraded = {"total": 1.0}, {"returned": 2}, {}, []
        self.staleness = {"enabled": False, "applied": False, "dropped": 0}


class _State:
    def __init__(self):
        self.calls = []
        self.retriever = SimpleNamespace(retrieve_detailed=self._retrieve, _reranker=None, reranker_info={})

    def _retrieve(self, query, top_n=None, deadline=None):
        self.calls.append((query, top_n))
        return _Outcome()


class _Client:
    """Stands in for anthropic.Anthropic: records each call and returns a cited reply (the SDK-level request is tested
    in test_answer_claude.py)."""

    def __init__(self):
        self.calls = []
        self.messages = self.beta = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        cited = SimpleNamespace(type="char_location", cited_text="learning_rate: 3e-4", document_index=0,
                                start_char_index=0, end_char_index=19)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Run 47 used ", citations=None),
                                        SimpleNamespace(type="text", text="3e-4", citations=[cited])],
                               model=kwargs["model"], stop_reason="end_turn", stop_details=None,
                               usage=SimpleNamespace(input_tokens=500, output_tokens=40, cache_creation_input_tokens=0,
                                                     cache_read_input_tokens=0))


class _Redis:
    def __init__(self):
        self.data, self.down = {}, False

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

    def get(self, key):
        self._up()
        value = self.data.get(key)
        return None if value is None else str(value).encode() if not isinstance(value, bytes) else value

    def set(self, key, value, ex=None):
        self._up()
        self.data[key] = value.encode()


def _client(*, client="cited", redis="up", **limit_settings):
    app = FastAPI()
    app.include_router(public_api_router(), prefix="/api")
    app.add_middleware(BodyLimit, limit=2048)
    app.state.retrieval = _State()
    fake_redis = _Redis() if redis else None
    if redis == "down":
        fake_redis.down = True
    app.state.answer = {"settings": claude.AnswerSettings(),
                        "client": _Client() if client else None,
                        "limits": Limits(fake_redis, LimitSettings(**{"per_minute": 5, "per_day": 50, "daily_budget_usd": 1.0,
                                                                      **limit_settings})) if fake_redis else None}
    return TestClient(app)


Q = {"question": "What learning rate did run 47 use?"}


def test_a_cited_answer_is_delivered_then_served_from_the_cache():
    tc = _client()
    first = tc.post("/api/answer/", json=Q).json()
    assert first["answer"] == "Run 47 used 3e-4" and first["reason"] is None and first["evaluated"] is False
    assert first["citations"] == [{"chunk_id": "n1", "path": "configs/run_047.yaml", "cited_text": "learning_rate: 3e-4",
                                   "start_char": 0, "end_char": 19, "start_byte": 40, "end_byte": 59}]
    assert first["model"] == "claude-opus-5-5" and first["cost_usd_estimate"] > 0 and len(first["hits"]) == 2
    again = tc.post("/api/answer/", json={"question": "  what learning rate did RUN 47 use?"}).json()
    assert again["cached"] is True and again["answer"] == first["answer"]
    state = tc.app.state.answer
    assert len(state["client"].calls) == 1  # the repeat cost nothing
    assert 0 < state["limits"].spent_today_usd() < 0.01  # settled at the reply's cost, not the worst case


def test_a_reply_after_utc_midnight_is_settled_on_the_day_its_call_reserved():
    tc = _client()
    state = tc.app.state.answer
    limits, client = state["limits"], state["client"]
    now = [(1_800_000_000 // 86_400 + 1) * 86_400 - 1.0]  # one second before UTC midnight
    limits.clock = lambda: now[0]
    reply = client.create

    def late_reply(**kwargs):
        now[0] += 2  # the reply comes back on the next UTC day
        return reply(**kwargs)

    client.create = late_reply
    assert tc.post("/api/answer/", json=Q).json()["answer"] == "Run 47 used 3e-4"
    assert limits.spent_today_usd() == 0  # the new day starts at zero, not below it
    now[0] -= 2
    assert 0 < limits.spent_today_usd() < 0.01  # the old day keeps the reply's cost, not the worst case


@pytest.mark.parametrize("kwargs, reason", [
    ({"client": None}, "answers off"),
    ({"redis": None}, "limits unavailable"),
    ({"redis": "down"}, "limits unavailable"),
    ({"daily_budget_usd": 0.0001}, "daily budget spent"),
])
def test_without_a_key_redis_or_budget_no_call_is_made_and_the_hits_still_come_back(kwargs, reason):
    tc = _client(**kwargs)
    body = tc.post("/api/answer/", json=Q).json()
    assert body["answer"] is None and body["reason"] == reason and len(body["hits"]) == 2
    client = tc.app.state.answer["client"]
    assert client is None or client.calls == []


def _refusing_sdk(kind: str, message: str, calls: list):
    """The real Anthropic SDK over a transport that answers every call with a 400 of this error type and message."""
    import anthropic
    import httpx2

    def handler(request):
        calls.append(str(request.url))
        return httpx2.Response(400, json={"type": "error", "error": {"type": kind, "message": message}})

    return anthropic.Anthropic(api_key="test-key", max_retries=0,
                               http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))


@pytest.mark.parametrize("message", [
    "You have reached your specified workspace API usage limits. You will regain access on 2026-12-01 at 00:00 UTC.",
    "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase "
    "credits.",
])
def test_an_account_that_cannot_pay_gets_the_hits_and_a_reason_and_no_spend_is_recorded(message):
    tc = _client()
    calls = []
    tc.app.state.answer["client"] = _refusing_sdk("invalid_request_error", message, calls)
    limits = tc.app.state.answer["limits"]
    response = tc.post("/api/answer/", json=Q)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] is None and body["reason"] == "answer spend limit reached" and len(body["hits"]) == 2
    assert body["cost_usd_estimate"] == 0.0 and len(calls) == 1
    assert limits.spent_today_usd() == 0  # the call's reservation was given back: nothing was billed
    again = tc.post("/api/answer/", json=Q).json()
    assert again["cached"] is False and len(calls) == 2  # a refusal is not cached: the next ask tries again


def test_any_other_400_is_raised_by_the_route_and_its_reservation_stays():
    tc = _client()
    calls = []
    tc.app.state.answer["client"] = _refusing_sdk("invalid_request_error", "max_tokens: Field required", calls)
    import anthropic

    with pytest.raises(anthropic.BadRequestError):  # a malformed request is a bug: loud, never a reason
        tc.post("/api/answer/", json=Q)
    assert len(calls) == 1 and tc.app.state.answer["limits"].spent_today_usd() > 0  # whether it was billed is unknown


def test_a_caller_over_its_quota_gets_429_before_anything_is_retrieved():
    tc = _client(per_minute=1)
    assert tc.post("/api/answer/", json=Q).status_code == 200
    second = tc.post("/api/answer/", json={"question": "another question"})
    assert second.status_code == 429 and int(second.headers["retry-after"]) >= 1
    assert len(tc.app.state.retrieval.calls) == 1


def test_a_spoofed_forwarded_for_does_not_buy_a_new_quota(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    tc = _client(per_minute=1)
    one = tc.post("/api/answer/", json=Q, headers={"x-forwarded-for": "1.1.1.1, 203.0.113.9"})
    two = tc.post("/api/answer/", json={"question": "again"}, headers={"x-forwarded-for": "2.2.2.2, 203.0.113.9"})
    other = tc.post("/api/answer/", json={"question": "again"}, headers={"x-forwarded-for": "203.0.113.10"})
    assert (one.status_code, two.status_code, other.status_code) == (200, 429, 200)


def test_the_public_caps_and_the_body_limit():
    tc = _client()
    assert tc.post("/api/retrieve/", json={"query": "run 47", "k": 11}).status_code == 422
    assert tc.post("/api/retrieve/", json={"query": "x" * 501}).status_code == 422
    assert tc.post("/api/answer/", json={"question": "q", "k": 11}).status_code == 422
    assert tc.post("/api/answer/", json={"question": "x" * 501}).status_code == 422
    big = tc.post("/api/answer/", content=json.dumps({"question": "q", "pad": "x" * 4000}),
                  headers={"content-type": "application/json"})
    assert big.status_code == 413
    ok = tc.post("/api/retrieve/", json={"query": "run 47", "k": 2})
    assert ok.status_code == 200 and [h["chunk_id"] for h in ok.json()["hits"]] == ["n1", "n2"]


def test_the_web_demo_page_sends_what_the_public_routes_accept():
    """The web client's demo page (.frontend/app/components/demo/api.ts) caps the question at the routes' length cap,
    asks for k within each route's cap, and calls only GET /health and the allow-listed routes."""
    import re

    from app.api.public import PUBLIC_MAX_K, PUBLIC_MAX_QUERY_CHARS
    from app.api.routers import answer as answer_route

    demo = ROOT / ".frontend" / "app" / "components" / "demo"
    api = (demo / "api.ts").read_text()

    def constant(name: str):
        return re.search(rf'^export const {name} = (\d+|"[^"]*");', api, re.M).group(1).strip('"')

    assert int(constant("MAX_QUESTION_CHARS")) == PUBLIC_MAX_QUERY_CHARS == answer_route.MAX_QUESTION_CHARS
    assert 1 <= int(constant("SEARCH_K")) <= PUBLIC_MAX_K and 1 <= int(constant("ANSWER_K")) <= answer_route.MAX_K
    assert {constant("RETRIEVE_PATH"), constant("ANSWER_PATH")} == {path for path, _ in PUBLIC_API_ROUTES}
    assert constant("HEALTH_PATH") == "/health"
    sources = "".join(path.read_text() for path in sorted(demo.glob("*.ts*")))
    assert sorted(set(re.findall(r'"(/(?:api|health)[^"]*)"', sources))) == ["/api/answer/", "/api/retrieve/", "/health"]


def test_the_docs_state_the_allow_list_and_the_deploy_contract_the_code_reads():
    """The reference's Public demo mode section names the allow-list the app serves, and every variable of its deploy
    contract is one the code reads; the README says the demo is deployed on one host behind CloudFront with answers off
    until the owner stores a key, and that its answers are not evaluated."""
    import re

    text = (ROOT / "doc" / "engineering-reference.md").read_text()
    section = text.split("### Public demo mode", 1)[1].split("**Full application locally**", 1)[0]
    flat = " ".join(section.split())
    routes = sorted(f"`{method} {path}`" for path, method in PUBLIC_API_ROUTES)
    assert f"The allow-list is {routes[1]} and {routes[0]}" in flat or f"The allow-list is {routes[0]} and {routes[1]}" in flat
    assert "**Claude's answers are not evaluated here:**" in flat
    contract = section.split("```text", 1)[1].split("```", 1)[0]
    names = set(re.findall(r"^([A-Z][A-Z_]+)=", contract, flags=re.M)) | set(re.findall(r"\s([A-Z][A-Z_]+)=", contract))
    assert names >= {"PUBLIC_DEMO", "REDIS_URL", "ANTHROPIC_API_KEY", "ANSWER_MODEL", "ANSWER_DAILY_BUDGET_USD",
                     "TRUSTED_PROXY_HOPS", "WEB_ORIGIN", "MAX_BODY_BYTES"}
    sources = "".join(p.read_text() for p in [ROOT / "main.py", *(ROOT / "app").rglob("*.py")])
    unread = sorted(n for n in names if f'"{n}"' not in sources)
    assert unread == [], unread
    readme = " ".join((ROOT / "README.md").read_text().split())
    assert ("**Public demo mode (deployed).** It is deployed as a public demo on one host behind CloudFront, with answers "
            "switched off until the owner stores an Anthropic API key") in readme
    assert "not deployed" not in readme
    assert "Claude's answers are not evaluated in this repository" in readme
