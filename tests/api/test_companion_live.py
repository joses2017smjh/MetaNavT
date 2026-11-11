"""The live-mode runner of the research companion (A25 / C13): app/eval/companion_live.py, without a model.

The companion's own app answers /agent/ask; its two upstreams (Ollama /api/chat and the retrieval API) are an
httpx.MockTransport behind the runner's recording transport, as in tests/eval/test_companion_agent.py.
"""

import asyncio
import json

import httpx
import pytest

pytest.importorskip("fastapi")

from app.agent import companion
from app.eval import companion_live as live
from app.eval import companion_score as cs

TEXT = "learning_rate: 0.0003\n"


def retrieved(query):
    return {"query": query, "k": 5, "route": "lexical_path", "retrieval_mode": "test", "bm25_backend": "test",
            "embedding_provider": "test", "reranker_loaded": False, "reranker": {}, "degraded": [],
            "staleness": {"enabled": False, "applied": False, "dropped": 0}, "counts": {"returned": 1},
            "latency_ms": {}, "hits": [{"rank": 1, "chunk_id": "c1", "path": "configs/run_042.yaml", "start_byte": 0,
                                        "end_byte": len(TEXT), "text": TEXT, "score": 1, "scores": {}}]}


def scripted(request):
    """Ollama: search, inspect S1, answer; a question containing "fail" gets a provider error."""
    if request.url.path == "/api/retrieve/":
        return httpx.Response(200, json=retrieved(json.loads(request.content)["query"]))
    body = json.loads(request.content)
    if "fail" in body["messages"][1]["content"]:
        return httpx.Response(500, text="provider exploded at /home/someone")
    tools = [m for m in body["messages"] if m["role"] == "tool"]
    if not tools:
        message = {"role": "assistant", "tool_calls": [{"function": {"name": "search_research", "arguments": {"query": "lr"}}}]}
    elif len(tools) == 1:
        message = {"role": "assistant", "tool_calls": [{"function": {"name": "inspect_source", "arguments": {"source_id": "S1"}}}]}
    else:
        message = {"role": "assistant", "content": json.dumps({"claims": [{"text": "learning_rate: 0.0003", "source_ids": ["S1"]}],
                                                              "abstain": False, "reason": ""})}
    return httpx.Response(200, json={"message": message})


def test_recorded_limits_are_the_companions():
    assert live.limits() == {"max_model_rounds": companion.MAX_MODEL_ROUNDS, "max_tool_calls": companion.MAX_TOOL_CALLS,
                             "max_sources": companion.MAX_SOURCES, "max_excerpt_chars": companion.MAX_EXCERPT_CHARS,
                             "preview_chars": live.PREVIEW_CHARS}

    class Backend:
        async def search(self, query, k):
            from app.api.routers.retrieve import RetrieveResponse

            payload = retrieved(query)
            payload["hits"][0]["text"] = "y" * 700
            return RetrieveResponse.model_validate(payload)

    result = asyncio.run(companion.ResearchTools(Backend()).call("search_research", {"query": "q"}))
    assert len(result["sources"][0]["preview"]) == live.PREVIEW_CHARS


def test_the_hijack_tasks_are_planted_once_each_and_match_what_the_scorer_rebuilds():
    tasks = live.hijack_tasks()
    assert len(tasks) == 10 and {t["task_id"].split("-")[1] for t in tasks} == {"hijack"}
    additions = live.planted_additions(tasks)
    assert len(additions) == 20
    clean = cs.corpus_files()
    planted = cs.planted_files(clean, tasks)
    assert {p for p in planted if planted[p] != clean[p]} == set(additions)
    with pytest.raises(ValueError):
        live.planted_additions(tasks + tasks[:1])


def test_each_question_goes_through_the_live_route_and_failures_are_rows():
    transport = live.RecordingTransport(httpx.MockTransport(scripted))
    questions = [{"id": "q011", "kind": "gold", "category": "simple_factual", "question": "what learning rate did run 42 use"},
                 {"id": "qx", "kind": "gold", "category": "simple_factual", "question": "fail please"}]
    rows, gateway = asyncio.run(live.ask_all(questions, retrieval_url="http://retrieval", model="qwen2.5:7b",
                                             ollama_url="http://ollama", timeout=120, transport=transport))
    ok, failed = rows
    assert ok["http_status"] == 200 and ok["response"]["status"] == "complete"
    assert [c["upstream"] for c in ok["upstream_calls"]] == ["model", "retrieval", "model", "model"]
    assert ok["response"]["limits"]["model_rounds"] == 3
    assert failed["http_status"] == 503 and "home" not in json.dumps(failed)
    assert gateway == {"request_timeout_s": 120, "upstream_timeout_s": 30, "max_concurrent": 1}
    assert transport.model_request["think"] is False and transport.model_request["tools"] == ["search_research", "inspect_source"]
    assert "messages" not in transport.model_request


def test_preflight_stops_when_the_model_refuses_the_companions_request():
    refuse = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "does not support thinking"}))
    with pytest.raises(SystemExit, match="HTTP 400"):
        asyncio.run(live.preflight_model("qwen2.5:7b", "http://ollama", refuse))
    answer = httpx.MockTransport(lambda request: httpx.Response(200, json={"message": {"role": "assistant", "content": "{}"}}))
    checked = asyncio.run(live.preflight_model("qwen2.5:7b", "http://ollama", answer))
    assert checked["model_request"]["options"] == {"temperature": 0, "num_predict": 1500}
    assert checked["returned_fields"] == ["content", "role"] and checked["returned_tool_calls"] is False


def test_publishable_rows_drop_local_paths_and_say_so():
    rows = live._publishable([{"id": "q1", "response": {"answer": "see /home/someone/x.txt"}},
                              {"id": "q2", "response": {"answer": "configs/run_047.yaml"}}])
    assert rows[0]["scrubbed"] and "/home/someone" not in json.dumps(rows[0])
    assert rows[1]["scrubbed"] is False and rows[1]["response"]["answer"] == "configs/run_047.yaml"


def test_recorded_command_names_the_dsn_instead_of_recording_it():
    for argv in (["app/eval/companion_live.py", "--pg-dsn", "postgresql://u:pw@127.0.0.1:1/x", "--limit", "2"],
                 ["app/eval/companion_live.py", "--pg-dsn=postgresql://u:pw@127.0.0.1:1/x"]):
        command = live.recorded_command(argv)
        assert "pw" not in command and "<PG_TEST_DSN>" in command and command.startswith("python -m app.eval.companion_live")


def test_a_limited_run_does_not_write_the_committed_file(tmp_path):
    with pytest.raises(SystemExit):
        live.main(["--limit", "2", "--pg-dsn", "postgresql://u@127.0.0.1:1/x"])
