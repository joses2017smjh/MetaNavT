"""The origin check of the single host (ORIGIN_VERIFY_SECRET, app/api/public.py OriginVerify).

On the single host CloudFront reaches the API with no load balancer in between, so the API itself refuses a request
that does not carry the header CloudFront adds (X-Origin-Verify, a random value Terraform keeps in Parameter Store).
The one request without it that passes is GET /health from the loopback address: the container's own health check,
which is how Compose and deploy.sh wait for the API. Off unless the secret is set, and only in the public demo.
"""

import hmac
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import public
from app.api.public import ORIGIN_VERIFY_HEADER, BodyLimit, OriginVerify

ROOT = Path(__file__).resolve().parents[2]
SECRET = "s3cretValueFromParameterStore0123456789"
LOOPBACK = ("127.0.0.1", 50000)
OUTSIDE = ("203.0.113.7", 41000)  # a CloudFront edge, or anything else that reaches the published port


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/retrieve/")
    async def retrieve(body: dict):
        return {"hits": [], "echo": body}

    @app.get("/")
    async def page():
        return {"page": "index"}

    app.add_middleware(BodyLimit, limit=64)
    app.add_middleware(OriginVerify, secret=SECRET)  # last added: outermost
    return app


def _client(peer=OUTSIDE) -> TestClient:
    return TestClient(_app(), client=peer)


def test_a_request_without_the_header_is_refused_with_403():
    tc = _client()
    for method, path in (("get", "/"), ("get", "/health"), ("post", "/api/retrieve/")):
        response = getattr(tc, method)(path, **({"json": {"query": "q"}} if method == "post" else {}))
        assert response.status_code == 403, (method, path)
        assert response.json() == {"detail": "forbidden"}
        assert SECRET not in response.text


def test_a_request_with_the_header_reaches_the_api():
    tc = _client()
    assert tc.get("/", headers={ORIGIN_VERIFY_HEADER: SECRET}).json() == {"page": "index"}
    assert tc.get("/health", headers={ORIGIN_VERIFY_HEADER: SECRET}).status_code == 200
    ok = tc.post("/api/retrieve/", json={"query": "q"}, headers={ORIGIN_VERIFY_HEADER: SECRET})
    assert ok.status_code == 200 and ok.json()["echo"] == {"query": "q"}


@pytest.mark.parametrize("headers", [
    {ORIGIN_VERIFY_HEADER: SECRET[:-1]},
    {ORIGIN_VERIFY_HEADER: SECRET + "x"},
    {ORIGIN_VERIFY_HEADER: ""},
    {ORIGIN_VERIFY_HEADER.upper(): "wrong"},
    {"X-Origin-Verified": SECRET},  # another header's name
])
def test_a_wrong_value_is_refused(headers):
    assert _client().get("/", headers=headers).status_code == 403


def test_two_copies_of_the_header_are_refused_even_if_one_is_right():
    tc = _client()
    for pair in ([(ORIGIN_VERIFY_HEADER, "guess"), (ORIGIN_VERIFY_HEADER, SECRET)],
                 [(ORIGIN_VERIFY_HEADER, SECRET), (ORIGIN_VERIFY_HEADER, "guess")]):
        assert tc.get("/", headers=pair).status_code == 403, pair


def test_get_health_from_the_loopback_address_needs_no_header():
    """The container's health check (Compose's and the image's: GET http://127.0.0.1:8000/health inside the
    container) carries no header. Only that request is exempt: another path, another method, or /health from any
    other address still needs it."""
    local = _client(LOOPBACK)
    assert local.get("/health").status_code == 200
    assert _client(("::1", 50000)).get("/health").status_code == 200
    assert local.get("/").status_code == 403
    assert local.post("/api/retrieve/", json={"query": "q"}).status_code == 403
    assert local.post("/health", json={}).status_code == 403
    assert local.head("/health").status_code == 403
    assert local.get("/health/").status_code == 403
    # Docker's bridge gateway: what a request from the host through the published port arrives from
    assert _client(("172.18.0.1", 40000)).get("/health").status_code == 403
    assert _client(("testclient", 50000)).get("/health").status_code == 403  # not an address at all


def test_the_check_runs_before_the_body_limit_and_compares_in_constant_time(monkeypatch):
    compared = []
    real = hmac.compare_digest

    def recording(a, b):
        compared.append((a, b))
        return real(a, b)

    monkeypatch.setattr(public.hmac, "compare_digest", recording)
    tc = _client()
    big = tc.post("/api/retrieve/", content=b"x" * 1000, headers={"content-type": "application/json"})
    assert big.status_code == 403  # refused before the body limit (413) could answer
    allowed = tc.get("/health", headers={ORIGIN_VERIFY_HEADER: SECRET})
    assert allowed.status_code == 200
    assert compared and all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in compared)


def test_an_empty_secret_is_a_configuration_error():
    with pytest.raises(ValueError, match="ORIGIN_VERIFY_SECRET"):
        OriginVerify(FastAPI(), secret=" ")


# ---- main.py: on under PUBLIC_DEMO with the secret, off otherwise (a fresh interpreter: main builds at import) ----

_PROBE = r"""
import json, sys
import main
from fastapi.testclient import TestClient
header = sys.argv[1]
outside = TestClient(main.app, client=("203.0.113.7", 41000))  # no lifespan: /health answers 503 "unavailable"
local = TestClient(main.app, client=("127.0.0.1", 50000))
print(json.dumps({
    "outside": outside.get("/health").status_code,
    "outside_with_header": outside.get("/health", headers={header: "probe-secret"}).status_code,
    "local": local.get("/health").status_code,
    "outside_post": outside.post("/api/retrieve/", content=b"x" * 20000, headers={"content-type": "application/json"}).status_code,
}))
"""


def _main_answers(public_demo: bool, secret: str | None) -> dict:
    env = {**os.environ, "PUBLIC_DEMO": "1" if public_demo else "", "ENVIRONMENT": "dev", "MODEL_PROVIDER": "ollama",
           "EMBEDDING_PROVIDER": "hash", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("FRONTEND_ENDPOINT", None)
    env.pop("ORIGIN_VERIFY_SECRET", None)
    if secret is not None:
        env["ORIGIN_VERIFY_SECRET"] = secret
    out = subprocess.run([sys.executable, "-c", _PROBE, ORIGIN_VERIFY_HEADER], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_main_installs_the_check_in_the_public_demo_when_the_secret_is_set():
    answers = _main_answers(public_demo=True, secret="probe-secret")
    assert answers["outside"] == 403
    assert answers["outside_with_header"] == 503  # through to the app, which has not started (no lifespan here)
    assert answers["local"] == 503  # the health check's request: exempt
    assert answers["outside_post"] == 403  # outermost: before the body limit's 413


@pytest.mark.parametrize("public_demo,secret", [(True, None), (True, ""), (False, "probe-secret")])
def test_main_leaves_the_check_off_without_the_secret_or_outside_the_public_demo(public_demo, secret):
    answers = _main_answers(public_demo=public_demo, secret=secret)
    assert answers["outside"] == 503 and answers["outside_with_header"] == 503
