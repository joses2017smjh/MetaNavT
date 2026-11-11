"""The chat-route smoke run (C23): app/eval/chat_smoke.py, without a model or a server.

The end-to-end path (uvicorn main:app, both chat routes, llama-index's Ollama client against a scripted stand-in)
is tests/database/test_agent_paths_stack.py.
"""

import json
import os

import pytest

pytest.importorskip("fastapi")

from app.eval import chat_smoke as smoke


def test_the_question_rule_takes_the_first_question_of_each_category():
    chosen = smoke.questions()
    assert [q["category"] for q in chosen] == ["simple_factual", "conditional", "comparative", "aggregation",
                                                "multi_hop", "staleness", "exact_path", "semantic"]
    assert [q["id"] for q in chosen] == ["q001", "q081", "q089", "q097", "q107", "q115", "q123", "q131"]


def test_the_vercel_stream_is_split_into_text_data_and_errors():
    body = "\n".join(['0:""', '8:[{"type": "sources", "data": {"nodes": [{"text": "a"}, {"text": "b"}]}}]',
                      '0:"learning_rate "', '0:"is 1e-4"', '3:"An unexpected error occurred"'])
    stream = smoke.parse_stream(body)
    assert stream["text"] == "learning_rate is 1e-4"
    assert stream["errors"] == ["An unexpected error occurred"]
    assert smoke._source_nodes(stream["data"]) == 2


def test_tool_calls_come_from_the_request_route_result():
    result = {"result": {"role": "assistant", "content": "x"}, "nodes": [],
              "tools": [{"tool_name": "query_index", "raw_input": {"args": [], "kwargs": {"input": "lr of run 40"}},
                         "content": "abc", "is_error": False},
                        {"tool_name": "get_file_content", "raw_input": {"kwargs": {"path": "x"}}, "content": "",
                         "is_error": True}]}
    assert smoke.tool_calls(result) == [
        {"name": "query_index", "arguments": {"input": "lr of run 40"}, "is_error": False, "output_chars": 3},
        {"name": "get_file_content", "arguments": {"path": "x"}, "is_error": True, "output_chars": 0}]


def test_summary_counts_answers_tools_and_code_tool_calls():
    rows = [{"route": "/api/chat", "http_status": 200, "answered": True, "ttfb_s": 3.0, "total_s": 4.0, "tools": None,
             "gold_value_stated": True},
            {"route": "/api/chat", "http_status": 500, "answered": False, "ttfb_s": 1.0, "total_s": 1.0, "tools": None,
             "gold_value_stated": None},
            {"route": "/api/chat/request", "http_status": 200, "answered": True, "ttfb_s": 2.0, "total_s": 2.0,
             "tools": [{"name": "query_index"}, {"name": "execute_code"}], "gold_value_stated": False}]
    summary = smoke.summarize(rows)
    assert summary["/api/chat"]["answered"] == 1 and summary["/api/chat"]["http_status"] == {"200": 1, "500": 1}
    assert summary["/api/chat"]["states_gold_value_aside"] == {"k": 1, "n": 1}
    request = summary["/api/chat/request"]
    assert request["tool_calls"] == {"execute_code": 1, "query_index": 1} and request["code_tool_calls"] == 1


def test_chat_llm_settings_are_read_back_and_nothing_leaks_into_the_process(monkeypatch):
    from llama_index.core.settings import Settings
    from llama_index.llms.ollama.base import DEFAULT_REQUEST_TIMEOUT

    monkeypatch.delenv("MODEL", raising=False)
    before = Settings._llm
    settings = smoke.chat_llm_settings({"MODEL": "qwen2.5:7b", "OLLAMA_BASE_URL": "http://127.0.0.1:9",
                                        "EMBEDDING_PROVIDER": "hash", "EMBEDDING_DIM": "256"})
    assert settings["class"] == "Ollama" and settings["model"] == "qwen2.5:7b"
    assert settings["request_timeout_s"] == DEFAULT_REQUEST_TIMEOUT and settings["set_in_api_environment"] == []
    assert isinstance(settings["temperature"], float)
    assert Settings._llm is before and "MODEL" not in os.environ


def test_a_limited_run_does_not_write_the_committed_file():
    with pytest.raises(SystemExit):
        smoke.main(["--limit", "1", "--pg-dsn", "postgresql://u@127.0.0.1:1/x"])
    assert json.dumps(smoke.ROUTES) == '["/api/chat", "/api/chat/request"]'
