"""Real demo retrieval and bounded agent protocol tests; live LLM is simulated."""
from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.agent.companion import AskRequest, ResearchTools, run_agent
from app.agent.companion_api import AgentConfig, create_app
from app.api.routers.retrieve import RetrieveResponse


def retrieved(text="learning_rate: 0.0003\n", chunk="config", path="configs/run_047.yaml"):
    return RetrieveResponse.model_validate({
        "query": "learning rate", "k": 5, "route": "lexical_path", "retrieval_mode": "test",
        "bm25_backend": "test", "embedding_provider": "test", "reranker_loaded": False,
        "reranker": {}, "degraded": [], "staleness": {"enabled": False, "applied": False, "dropped": 0},
        "counts": {"returned": 1}, "latency_ms": {}, "hits": [
            {"rank": 1, "chunk_id": chunk, "path": path, "start_byte": 0,
             "end_byte": len(text.encode()), "text": text, "score": 1, "scores": {}}]})


class Backend:
    def __init__(self):
        self.calls = 0

    async def search(self, query, k):
        self.calls += 1
        return retrieved()


class SequenceModel:
    name = "test-model-not-a-real-LLM"

    def __init__(self, messages):
        self.messages = iter(messages)
        self.calls = 0

    async def next(self, messages):
        self.calls += 1
        return next(self.messages)


def tool(name, **arguments):
    return {"role": "assistant", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def final(ids=None, text="The learning rate is 0.0003."):
    return {"role": "assistant", "content": json.dumps({"claims": [
        {"text": text, "source_ids": ids or ["S1"]}], "abstain": False, "reason": ""})}


def run(messages, backend=None):
    return asyncio.run(run_agent(AskRequest(question="What is the learning rate?"),
                                 SequenceModel(messages), backend or Backend()))


@pytest.mark.parametrize("question,expected", [
    ("What is the learning rate in run_047.yaml?", "0.0003"),
    ("What checkpoint belongs to run_047.ckpt.meta.json?", "run_047.pt"),
    ("What GPU resources does run_047.sbatch request?", "gpu:1"),
])
def test_keyless_demo_executes_existing_retrieval_and_inspection(question, expected):
    with TestClient(create_app(AgentConfig())) as client:
        result = client.post("/agent/ask", json={"question": question}).json()
    assert result["status"] == "complete"
    assert expected in result["answer"]
    assert result["model"] == "scripted-extractive-demo-v1"
    assert result["synthetic"] and "No LLM inference" in result["notice"]
    assert [event["tool"] for event in result["trace"]] == ["search_research", "inspect_source"]
    assert result["limits"]["model_rounds"] == 3
    assert result["citation_check"]["passed"]
    for citation in result["citations"]:
        source = next(x for x in result["sources"] if x["source_id"] == citation["source_id"])
        assert source["inspected"]
        assert hashlib.sha256(source["text"].encode()).hexdigest() == citation["excerpt_sha256"]


def test_demo_abstains_instead_of_inventing_unknown_measurement():
    with TestClient(create_app(AgentConfig())) as client:
        result = client.post("/agent/ask", json={"question": "What is the weather in Nairobi?"}).json()
    assert result["status"] == "abstained"
    assert not result["citations"]


@pytest.mark.parametrize("body", [{"question": " "}, {"question": "x" * 1001},
                                  {"question": "test", "mode": "unexpected"},
                                  {"question": "test", "provider_url": "https://example.com"}])
def test_bad_mobile_requests_fail_validation(body):
    with TestClient(create_app(AgentConfig())) as client:
        assert client.post("/agent/ask", json=body).status_code == 422


def test_live_requires_authentication_before_any_network_request():
    requests = []
    transport = httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(500))
    config = AgentConfig(api_key="operator-secret", retrieval_url="http://retrieval", model="qwen3:4b")
    with TestClient(create_app(config, transport=transport)) as client:
        for key in (None, "wrong"):
            response = client.post("/agent/ask", json={"question": "test", "mode": "live"},
                                   headers={"X-API-Key": key} if key else {})
            assert response.status_code == 401
    assert not requests


def test_configured_does_not_claim_model_readiness_or_expose_credentials():
    config = AgentConfig(api_key="private-operator-token", retrieval_url="http://retrieval", model="qwen3:4b",
                         model_key="private-provider-key", retrieval_key="private-retrieval-key")
    with TestClient(create_app(config)) as client:
        result = client.get("/agent/health").json()
    assert result["live_configured"] is True and result["live_readiness"] == "not_probed"
    assert "private-" not in json.dumps(result)


def test_authenticated_live_calls_real_provider_protocol_and_fixed_retrieval_endpoint():
    observed = []
    replies = iter([tool("search_research", query="learning rate"),
                    tool("inspect_source", source_id="S1"), final()])

    def respond(request):
        observed.append(request)
        if request.url.path == "/api/retrieve/":
            return httpx.Response(200, json=retrieved().model_dump())
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["stream"] is False and payload["model"] == "qwen3:4b"
        assert {spec["function"]["name"] for spec in payload["tools"]} == {"search_research", "inspect_source"}
        return httpx.Response(200, json={"message": next(replies)})

    config = AgentConfig(api_key="operator-token", retrieval_url="http://fixed-retrieval", model="qwen3:4b",
                         model_url="http://fixed-model", model_key="provider-only-key", retrieval_key="retrieval-only-key")
    with TestClient(create_app(config, transport=httpx.MockTransport(respond))) as client:
        response = client.post("/agent/ask", json={"question": "learning rate", "mode": "live"},
                               headers={"X-API-Key": "operator-token"})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "complete" and result["synthetic"] is False
    assert result["answer"] == "The learning rate is 0.0003. [S1]"
    assert len(observed) == 4
    assert set(str(request.url.host) for request in observed) == {"fixed-retrieval", "fixed-model"}
    assert observed[0].headers["Authorization"] == "Bearer provider-only-key"
    assert observed[1].headers["X-API-Key"] == "retrieval-only-key"
    assert "only-key" not in json.dumps(result)


@pytest.mark.parametrize("provider_response", [httpx.Response(302, headers={"Location": "http://unexpected"}),
                                              httpx.Response(401, text="secret credentials"),
                                              httpx.Response(200, text="not JSON"),
                                              httpx.Response(200, json={"message": {"role": "user"}}),
                                              httpx.Response(200, content=b"x" * 300_000)])
def test_provider_failures_are_sanitized_and_redirects_are_not_followed(provider_response):
    requests = []

    def respond(request):
        requests.append(request)
        return provider_response

    config = AgentConfig(api_key="key", retrieval_url="http://retrieval", model="qwen3:4b")
    with TestClient(create_app(config, transport=httpx.MockTransport(respond))) as client:
        response = client.post("/agent/ask", json={"question": "test", "mode": "live"}, headers={"X-API-Key": "key"})
    assert response.status_code == 503
    assert len(requests) == 1
    assert "secret" not in response.text and "unexpected" not in response.text


@pytest.mark.parametrize("messages,reason", [
    ([final(["S1"])], "invalid_citations"),
    ([tool("search_research", query="rate"), final(["S1"])], "invalid_citations"),
    ([tool("search_research", query="rate"), tool("inspect_source", source_id="S1"), final(["S9"])], "invalid_citations"),
    ([tool("search_research", query="rate"), tool("inspect_source", source_id="S1"), final(text="Bad marker [S9]")], "invalid_final_answer"),
    ([{"content": "An unsupported answer"}], "invalid_final_answer"),
])
def test_citation_integrity_fails_closed(messages, reason):
    result = run(messages)
    assert result.status == "blocked" and result.stop_reason == reason
    assert not result.citations and not result.citation_check.passed


def test_valid_citation_is_bound_to_an_inspected_exact_returned_excerpt():
    result = run([tool("search_research", query="rate"), tool("inspect_source", source_id="S1"), final()])
    assert result.status == "complete" and result.citation_check.passed
    assert result.citations[0].excerpt_sha256 == hashlib.sha256(b"learning_rate: 0.0003\n").hexdigest()


def test_unpermitted_tools_and_argument_extensions_cannot_reach_backend():
    backend = Backend()
    result = run([tool("delete_file", path="configs/run_047.yaml"),
                  tool("search_research", query="rate", provider_url="http://unexpected"),
                  {"content": '{"claims":[],"abstain":true}'}], backend)
    assert backend.calls == 0
    assert all(event.status == "error" and not event.arguments for event in result.trace)


def test_model_round_limit_stops_repeated_tool_loop():
    backend = Backend()
    result = run([tool("search_research", query="rate")] * 4, backend)
    assert result.status == "limit_reached" and result.stop_reason == "model_round_limit"
    assert backend.calls == 4 and result.limits.model_rounds == 4


def test_over_budget_parallel_calls_are_not_partially_executed():
    backend = Backend()
    calls = tool("search_research", query="rate")["tool_calls"] * 7
    result = run([{"tool_calls": calls}], backend)
    assert result.status == "limit_reached" and result.stop_reason == "tool_call_limit"
    assert backend.calls == 0 and result.limits.tool_calls == 0


def test_sources_cannot_be_inspected_across_requests():
    first = ResearchTools(Backend())
    second = ResearchTools(Backend())
    asyncio.run(first.call("search_research", {"query": "rate"}))
    with pytest.raises(ValueError):
        asyncio.run(second.call("inspect_source", {"source_id": "S1"}))


def test_request_retained_source_limit():
    class ManySources:
        n = 0

        async def search(self, query, k):
            result = retrieved(chunk=f"chunk-{self.n}")
            self.n += 1
            return result

    async def execute():
        tools = ResearchTools(ManySources())
        for _ in range(12):
            await tools.call("search_research", {"query": "rate"})
        return tools

    assert len(asyncio.run(execute()).sources) == 8


def test_changed_path_metadata_does_not_reuse_an_existing_source_identity():
    class ChangedPath:
        n = 0

        async def search(self, query, k):
            result = retrieved(path=f"configs/run_{self.n}.yaml")
            self.n += 1
            return result

    async def execute():
        tools = ResearchTools(ChangedPath())
        await tools.call("search_research", {"query": "rate"})
        await tools.call("search_research", {"query": "rate"})
        return tools

    sources = asyncio.run(execute()).sources
    assert len(sources) == 2 and sources["S1"].path != sources["S2"].path


def test_active_request_limit_returns_429_then_recovers():
    async def execute():
        config = AgentConfig(max_concurrent=1, api_key="key", retrieval_url="http://retrieval", model="test")
        app = create_app(config)
        entered = asyncio.Event()
        release = asyncio.Event()

        class DelayedModel:
            name = "simulated-delayed-model"

            async def next(self, messages):
                entered.set()
                await release.wait()
                return {"content": '{"claims":[],"abstain":true}'}

        async with app.router.lifespan_context(app):
            app.state.live_model = DelayedModel()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as client:
                kwargs = {"json": {"question": "test", "mode": "live"}, "headers": {"X-API-Key": "key"}}
                first = asyncio.create_task(client.post("/agent/ask", **kwargs))
                await entered.wait()
                second = await client.post("/agent/ask", **kwargs)
                assert second.status_code == 429
                release.set()
                assert (await first).status_code == 200
                assert app.state.active_requests == 0
                assert (await client.post("/agent/ask", **kwargs)).status_code == 200

    asyncio.run(execute())


def test_explicit_cors_and_mutation_routes_absent():
    config = AgentConfig(allowed_origins=("http://localhost:8081",))
    with TestClient(create_app(config)) as client:
        allowed = client.options("/agent/ask", headers={"Origin": "http://localhost:8081",
                                 "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-API-Key"})
        denied = client.options("/agent/ask", headers={"Origin": "https://unconfigured.example",
                                "Access-Control-Request-Method": "POST"})
        assert allowed.headers["Access-Control-Allow-Origin"] == "http://localhost:8081"
        assert "Access-Control-Allow-Origin" not in denied.headers
        assert client.post("/agent/approve", json={}).status_code == 404
        assert client.post("/agent/execute", json={}).status_code == 404


@pytest.mark.parametrize("kwargs", [dict(allowed_origins=("*",)),
                                    dict(model_url="http://user:password@localhost"),
                                    dict(retrieval_url="http://localhost?token=secret"),
                                    dict(timeout_seconds=121), dict(max_concurrent=0)])
def test_invalid_operator_configuration_fails_startup(kwargs):
    with pytest.raises(ValueError):
        AgentConfig(**kwargs)
