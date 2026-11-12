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


def _served_result(tools):
    """The JSON POST /api/chat/request sends for these llama-index ToolOutputs: the route's own Result model
    through FastAPI, so the test sees the serialized fields (text blocks), not the Python object."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llama_index.core.llms import MessageRole

    from app.api.routers.models import Message, Result

    app = FastAPI()

    @app.post("/request")
    async def request() -> Result:
        return Result(result=Message(role=MessageRole.ASSISTANT, content="x"), nodes=[], tools=tools)

    return TestClient(app).post("/request").json()


def test_tool_calls_come_from_the_request_route_result():
    from llama_index.core.tools.types import ToolOutput

    raised = "'FieldInfo' object has no attribute 'replace'"
    caught = "Error: 'dict' object has no attribute 'condition'"
    missing = "Not found: the index holds no file at that path. list_available_files lists the indexed paths."
    tools = [ToolOutput(content="Run 40 used 0.0001.", tool_name="query_index", raw_input={"input": "lr of run 40"},
                        raw_output="Run 40 used 0.0001."),
             # how the agent workflow records a tool that raised (llama_index.core.agent.workflow.base_agent)
             ToolOutput(content=raised, tool_name="list_available_files", raw_input={}, raw_output=raised, is_error=True),
             # get_file_by_path catches its failure and returns it as text: no error flag
             ToolOutput(content=caught, tool_name="get_file_content",
                        raw_input={"args": [], "kwargs": {"file_path": "configs/run_047.yaml"}}, raw_output=caught),
             # a path the index does not hold: an answer from a working tool, not a failure
             ToolOutput(content=missing, tool_name="get_file_content",
                        raw_input={"args": [], "kwargs": {"file_path": "run_47_config.yaml"}}, raw_output=missing)]
    result = _served_result(tools)
    assert "content" not in result["tools"][0]  # the text travels as blocks
    assert smoke.tool_calls(result) == [
        {"name": "query_index", "arguments": {"input": "lr of run 40"}, "is_error": False, "returned_error_text": False,
         "returned_not_found": False, "output_chars": len("Run 40 used 0.0001.")},
        {"name": "list_available_files", "arguments": {}, "is_error": True, "returned_error_text": False,
         "returned_not_found": False, "output_chars": len(raised), "error": raised},
        {"name": "get_file_content", "arguments": {"file_path": "configs/run_047.yaml"}, "is_error": False,
         "returned_error_text": True, "returned_not_found": False, "output_chars": len(caught), "error": caught},
        {"name": "get_file_content", "arguments": {"file_path": "run_47_config.yaml"}, "is_error": False,
         "returned_error_text": False, "returned_not_found": True, "output_chars": len(missing)}]


def test_the_runner_reads_the_file_tools_own_reply_texts():
    # The runner classifies a reply by its first word; the words come from the tool module, which it does not import.
    from app.engine.tools import index_file_access as files

    assert files.NOT_FOUND.startswith(smoke.NOT_FOUND_PREFIX)
    assert not files.NOT_FOUND.startswith(smoke.ERROR_TEXT_PREFIX)
    assert files.LIST_FAILED.startswith(smoke.ERROR_TEXT_PREFIX) and files.READ_FAILED.startswith(smoke.ERROR_TEXT_PREFIX)
    # the results file records the hash of the code the file tools run
    assert {"app/engine/tools/index_file_access.py", "app/database/vector_store_manager.py"} <= set(smoke.SOURCES)


def test_summary_counts_answers_tools_failed_calls_and_code_tool_calls():
    rows = [{"route": "/api/chat", "http_status": 200, "answered": True, "ttfb_s": 3.0, "total_s": 4.0, "tools": None,
             "gold_value_stated": True},
            {"route": "/api/chat", "http_status": 500, "answered": False, "ttfb_s": 1.0, "total_s": 1.0, "tools": None,
             "gold_value_stated": None, "gold_value_not_scored": "the question states the gold value"},
            {"route": "/api/chat/request", "http_status": 200, "answered": True, "ttfb_s": 2.0, "total_s": 2.0,
             "tools": [{"name": "query_index"}, {"name": "execute_code"},
                       {"name": "list_available_files", "is_error": True},
                       {"name": "get_file_content", "returned_error_text": True},
                       {"name": "get_file_content", "returned_not_found": True}], "gold_value_stated": False}]
    summary = smoke.summarize(rows)
    stream = summary["/api/chat"]
    assert stream["answered"] == 1 and stream["http_status"] == {"200": 1, "500": 1}
    assert stream["states_gold_value_aside"] == {"k": 1, "n": 1, "not_scored": {"the question states the gold value": 1}}
    # tool calls are not visible on the streaming route: no count there, not a zero
    assert stream["tool_calls"] is stream["code_tool_calls"] is stream["tool_calls_raised"] is None
    assert stream["tool_calls_returned_not_found"] is None
    request = summary["/api/chat/request"]
    assert request["tool_calls"] == {"execute_code": 1, "get_file_content": 2, "list_available_files": 1, "query_index": 1}
    assert request["code_tool_calls"] == 1
    assert request["tool_calls_raised"] == {"list_available_files": 1}
    assert request["tool_calls_returned_error_text"] == {"get_file_content": 1}
    assert request["tool_calls_returned_not_found"] == {"get_file_content": 1}


def test_the_aside_does_not_score_a_gold_value_the_question_states():
    gold = {"question": "open configs/run_047.yaml", "answer": "file configs/run_047.yaml"}
    row = {"answered": True, "answer": "There was an error retrieving configs/run_047.yaml."}
    assert smoke.gold_value_stated(row, gold) == (None, "the question states the gold value")
    lr = {"question": "what learning rate did run 40 use", "answer": "1e-04"}
    assert smoke.gold_value_stated({"answered": True, "answer": "Run 40 used 0.0001."}, lr) == (True, None)
    assert smoke.gold_value_stated({"answered": False, "answer": ""}, lr) == (False, None)
    assert smoke.gold_value_stated({"answered": True, "answer": "yes"}, {"question": "q", "answer": "yes"})[0] is None


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
