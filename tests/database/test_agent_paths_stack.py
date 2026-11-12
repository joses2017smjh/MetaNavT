"""The agent-path runs end to end against a live Postgres, with a scripted stand-in for Ollama (no model).

make bench-companion-live and make chat-route-smoke start `uvicorn main:app` over task-local fixture copies and
databases of their own (app/eval/served_stack.py) and talk to Ollama over HTTP. This runs the same code with
--limit, against FakeOllama: a loopback HTTP server that answers /api/chat from a script, enough for the
companion's adapter and for llama-index's Ollama client. It checks the plumbing (the index step and the APIs' start, the
retrieve path form, the byte ranges the scorer checks, the databases are dropped), not any model.

Needs PG_TEST_DSN (a server where the test may create and drop databases). Skipped otherwise.
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

MODEL = "fake-tools:0b"


def _companion_reply(messages):
    tools = [m for m in messages if m["role"] == "tool"]
    if not tools:
        return {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search_research", "arguments": {"query": messages[1]["content"]}}}]}
    last = json.loads(tools[-1]["content"])
    if "sources" in last:
        if not last["sources"]:
            return {"role": "assistant", "content": json.dumps({"claims": [], "abstain": True, "reason": "none"})}
        return {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "inspect_source", "arguments": {"source_id": last["sources"][0]["source_id"]}}}]}
    line = next((x for x in last["text"].splitlines() if x.strip() and not x.startswith("#")), "no text")
    return {"role": "assistant", "content": json.dumps({"claims": [{"text": line.strip(), "source_ids": [last["source_id"]]}],
                                                        "abstain": False, "reason": ""})}


def _chat_reply(messages):
    tools = [m for m in messages if m["role"] == "tool"]
    if not tools:
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        return {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "query_index", "arguments": {"input": user}}}]}
    return {"role": "assistant", "content": "Scripted answer from the index result."}


class FakeOllama(ThreadingHTTPServer):
    """Ollama's HTTP API as the two clients use it; replies come from _companion_reply and _chat_reply."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests = []
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload, ndjson=False):
        body = ("\n".join(json.dumps(p) for p in payload) + "\n") if ndjson else json.dumps(payload)
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson" if ndjson else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send({"version": "fake"})
        if self.path == "/api/tags":
            return self._send({"models": [{"name": MODEL, "model": MODEL, "digest": "0" * 64, "size": 1,
                                           "details": {"family": "fake", "parameter_size": "0B",
                                                       "quantization_level": "none", "context_length": 8192},
                                           "capabilities": ["completion", "tools"]}]})
        if self.path == "/api/ps":
            return self._send({"models": [{"name": MODEL, "digest": "0" * 64, "size": 1, "size_vram": 0,
                                           "context_length": 8192}]})
        self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append({"path": self.path, "body": body})
        if self.path == "/api/generate":  # an empty prompt loads the model
            return self._send({"model": MODEL, "response": "", "done": True, "done_reason": "load"})
        if self.path == "/api/show":
            return self._send({"model_info": {"general.architecture": "fake", "fake.context_length": 8192},
                               "details": {"family": "fake"}, "capabilities": ["completion", "tools"]})
        if self.path != "/api/chat":
            return self.send_error(404)
        names = {t["function"]["name"] for t in body.get("tools") or []}
        if "search_research" in names:
            message = _companion_reply(body["messages"])
        elif names:
            message = _chat_reply(body["messages"])
        else:  # the query engine's synthesis call: no tools
            message = {"role": "assistant", "content": "Scripted synthesis."}
        done = {"model": MODEL, "created_at": "2026-01-01T00:00:00Z", "message": message, "done": True,
                "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}
        if body.get("stream"):
            return self._send([{**done, "done": False, "message": message},
                               {**done, "message": {"role": "assistant", "content": ""}}], ndjson=True)
        return self._send(done)


def _names_this_account(text):
    """The text names the account running the test (provenance's names: generic CI accounts are left out)."""
    import re

    from app.eval.provenance import user_names

    return any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) for name in user_names())


def _databases(dsn):
    import psycopg2

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT datname FROM pg_database WHERE datname LIKE 'mnv_%'")
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def test_companion_live_run_end_to_end(tmp_path):
    from app.eval import companion_live as live
    from app.eval import companion_score as cs

    dsn = os.environ["PG_TEST_DSN"]
    before = _databases(dsn)
    with FakeOllama() as ollama:
        out = tmp_path / "companion_live.json"
        validation = tmp_path / "validation.json"
        validation.write_text(json.dumps({"schema_version": 1, "real_LLM_inference_tested": False}))
        assert live.main(["--model", MODEL, "--ollama-url", ollama.url, "--pg-dsn", dsn, "--limit", "3",
                          "--out", str(out), "--validation", str(validation), "--log-dir", str(tmp_path / "logs")]) == 0
    report = json.loads(out.read_text())
    assert _databases(dsn) == before  # both task-local databases dropped
    assert [r["kind"] for r in report["rows"]] == ["gold"] * 3 + ["planted"] * 3
    assert all(r["http_status"] == 200 and r["response"]["status"] == "complete" for r in report["rows"])
    # the served hit text is the cited bytes of the task-local file, for the clean and the planted copy
    assert all(r["scores"]["all_citations_valid"] for r in report["rows"]), [r["scores"]["citations"] for r in report["rows"]]
    assert report["served_api"]["clean"]["corpus_sha256"] != report["served_api"]["planted"]["corpus_sha256"]
    assert report["served_api"]["clean"]["env"]["ENABLE_PYTHON_EXEC"] == "false"
    assert report["companion"]["model_request"]["think"] is False
    assert report["ollama"]["models"][MODEL]["digest"] == "0" * 64
    assert cs.rescore(report) == report
    assert json.loads(validation.read_text())["real_LLM_inference_tested"] is True
    assert not _names_this_account(out.read_text())


def test_chat_route_smoke_run_end_to_end(tmp_path):
    from app.eval import chat_smoke as smoke

    dsn = os.environ["PG_TEST_DSN"]
    before = _databases(dsn)
    with FakeOllama() as ollama:
        out = tmp_path / "chat_route_smoke.json"
        assert smoke.main(["--model", MODEL, "--ollama-url", ollama.url, "--pg-dsn", dsn, "--limit", "2",
                           "--out", str(out), "--log-dir", str(tmp_path / "logs")]) == 0
        chat_requests = [r["body"] for r in ollama.requests if r["path"] == "/api/chat"]
    report = json.loads(out.read_text())
    assert _databases(dsn) == before
    rows = report["rows"]
    assert [(r["id"], r["route"]) for r in rows] == [(q, route) for q in ("q001", "q081") for route in smoke.ROUTES]
    assert all(r["http_status"] == 200 and r["answered"] and r["total_s"] >= r["ttfb_s"] for r in rows)
    assert all([t["name"] for t in r["tools"]] == ["query_index"] for r in rows if r["route"] == "/api/chat/request")
    # the size of the text the model got back, read from the route's JSON (text blocks), and no failure flagged
    calls = [t for r in rows if r["route"] == "/api/chat/request" for t in r["tools"]]
    assert all(t["output_chars"] > 0 and not t["is_error"] and not t["returned_error_text"] for t in calls), calls
    assert report["summary"]["/api/chat/request"]["tool_calls_raised"] == {}
    assert report["summary"]["/api/chat"]["code_tool_calls"] is None  # tool calls are not visible on that route
    assert all(r["source_nodes"] >= 1 for r in rows)
    # code execution off: the model is offered the file-access agent's tools only (no code tool, no handoff)
    offered = {t["function"]["name"] for body in chat_requests for t in body.get("tools") or []}
    assert offered == {"artifact", "get_file_content", "list_available_files", "query_index"}
    assert report["code_execution"]["ENABLE_PYTHON_EXEC"] == "false" and not report["code_execution"]["METANAVIT_API_KEY_set"]
    assert report["chat_llm"]["model"] == MODEL and report["summary"]["/api/chat/request"]["code_tool_calls"] == 0
    assert report["working_tree_dirty"] == bool(report["working_tree_changes"])
    assert not _names_this_account(out.read_text())
