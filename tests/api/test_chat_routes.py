"""Request-level tests of the team's chat and query routes, with mock models (A20 / C22).

POST /api/chat runs the real route, get_chat_engine(), the team's AgentWorkflow and
FunctionAgents, the tools from config/tools.yaml and VercelStreamResponse; GET
/api/query/ runs the real query engine. Only the models and the index are stand-ins:
a scripted function-calling MockLLM (MockLLM itself reports no function calling and
FunctionAgent refuses it), MockEmbedding, and an in-memory VectorStoreIndex in place of
the Postgres one. No GPU and no database (PG_CONNECTION_STRING points at a closed port,
as in the other tests/api modules), and a guard fails the test on any connection to a
host other than this one.
"""

import json
import os
import re
import socket

import pytest
from fastapi.testclient import TestClient
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, LLMMetadata, MessageRole
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.tools import ToolSelection

import app.engine.agents.file_access_agent as file_access_agent
from app.api import security
from app.api.routers import query as query_router

FACT = "learning_rate: 3e-4"
QUESTION = "What is the learning rate of run 47?"
KEY = "test-key-chat-9e2a-do-not-log"
# Tools in the order the model "wants" them; each agent calls the ones it holds.
PLAN = [
    ("query_index", {"input": "learning rate of run 47"}),
    ("handoff", {"to_agent": "PythonCodeAgent", "reason": "the question needs code"}),
    ("execute_code", {"code": "import os\nprint(os.getpid())"}),
]


def answer_from(tool_text: str) -> str:
    """The scripted final answer, a few words (the route streams one word per 0.1 s)."""
    pid = re.search(r"'output': '(\d+)", tool_text)
    if pid:
        return f"child pid {pid.group(1)}"
    return "learning_rate is 3e-4" if FACT in tool_text else "no context found"


class ScriptedToolLLM(MockLLM):
    """MockLLM that calls tools, which is what the team's FunctionAgents need.

    Each agent step calls the first planned tool that the current agent holds and has
    not called yet (a tool result carries the id of its call). With nothing left to
    call it answers from the latest tool result. Query synthesis is plain MockLLM,
    which echoes its prompt.
    """

    _plan: list = PrivateAttr(default_factory=list)
    _offered: list = PrivateAttr(default_factory=list)  # tool names offered at each agent step

    def __init__(self, plan):
        super().__init__()
        self._plan = list(plan)

    @property
    def offered(self) -> list:
        return self._offered

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True, model_name="scripted-tool-llm")

    def _reply(self, tools, history) -> ChatResponse:
        names = {tool.metadata.name for tool in tools}
        self._offered.append(sorted(names))
        called = {m.additional_kwargs.get("tool_call_id") for m in history if m.role == MessageRole.TOOL}
        for name, kwargs in self._plan:
            if name in names and f"call-{name}" not in called:
                selection = ToolSelection(tool_id=f"call-{name}", tool_name=name, tool_kwargs=kwargs)
                message = ChatMessage(role=MessageRole.ASSISTANT, content="", additional_kwargs={"tool_calls": [selection]})
                return ChatResponse(message=message, delta="")
        results = [m.content or "" for m in history if m.role == MessageRole.TOOL]
        text = answer_from(results[-1] if results else "")
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=text)

    async def astream_chat_with_tools(self, tools, user_msg=None, chat_history=None, verbose=False,
                                      allow_parallel_tool_calls=False, **kwargs):
        reply = self._reply(tools, list(chat_history or []))

        async def stream():
            yield reply

        return stream()

    def get_tool_calls_from_response(self, response, error_on_no_tool_call=True, **kwargs):
        calls = list(response.message.additional_kwargs.get("tool_calls", []))
        if not calls and error_on_no_tool_call:
            raise ValueError("the scripted reply has no tool call")
        return calls


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail on any TCP connection to a host other than this one (the closed-port database is local)."""
    local = {"127.0.0.1", "::1", "localhost", "testserver"}
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def check(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in local:
            raise AssertionError(f"network access to {address!r} in a no-network test")

    monkeypatch.setattr(socket.socket, "connect", lambda sock, address: (check(sock, address), real_connect(sock, address))[1])
    monkeypatch.setattr(socket.socket, "connect_ex", lambda sock, address: (check(sock, address), real_connect_ex(sock, address))[1])


@pytest.fixture
def scripted_llm():
    return ScriptedToolLLM(PLAN)


@pytest.fixture
def client(monkeypatch, tmp_path, scripted_llm):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://postgres:password@127.0.0.1:1/metanavit")
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1")
    monkeypatch.setenv("INDEX_ON_START", "false")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))  # where the exec tool saves the code it runs
    monkeypatch.delenv("FRONTEND_ENDPOINT", raising=False)
    for name in ("ENABLE_PYTHON_EXEC", security.API_KEY_ENV, security.RATE_ENV, security.BURST_ENV):
        monkeypatch.delenv(name, raising=False)
    security.reset_limiter()
    import main

    from app.database import vector_store

    vector_store.get_vector_store_manager.cache_clear()
    with TestClient(main.app) as c:
        # The lifespan has run init_settings(); the stand-ins go in after it and are
        # restored when the test ends (Settings is shared by the whole test process).
        embed = MockEmbedding(embed_dim=8)
        index = VectorStoreIndex(
            [
                TextNode(id_="n1", text=FACT, metadata={"file_path": "configs/run_047.yaml"}),
                TextNode(id_="n2", text="encoder: bge-small", metadata={"file_path": "configs/run_040.yaml"}),
            ],
            embed_model=embed,
        )
        monkeypatch.setattr(Settings, "_embed_model", embed)
        monkeypatch.setattr(Settings, "_llm", scripted_llm)
        monkeypatch.setattr(file_access_agent, "get_index", lambda config=None: index)
        monkeypatch.setattr(query_router, "get_index", lambda config=None: index)
        yield c
    security.reset_limiter()


def _stream(resp) -> tuple[str, list[str]]:
    """(the text tokens joined, the error lines) of a Vercel data stream."""
    text, errors = "", []
    for line in resp.text.splitlines():
        if line.startswith("0:"):
            text += json.loads(line[2:])
        elif line.startswith("3:"):
            errors.append(line)
    return text, errors


def _chat(client, **kwargs):
    return client.post("/api/chat", json={"messages": [{"role": "user", "content": QUESTION}]}, **kwargs)


# ---------------------------------------------------------------- POST /api/chat


def test_chat_streams_an_answer_built_from_a_tool_call(client):
    resp = _chat(client)
    assert resp.status_code == 200, resp.text
    text, errors = _stream(resp)
    assert errors == []  # the stream reports a failure as a 3: line under HTTP 200
    assert text.strip() == "learning_rate is 3e-4"  # query_index ran over the stub index


def test_chat_with_python_exec_on_needs_the_key_and_runs_code_in_another_process(client, monkeypatch):
    monkeypatch.setenv("ENABLE_PYTHON_EXEC", "true")
    monkeypatch.setenv(security.API_KEY_ENV, KEY)

    assert _chat(client).status_code == 401
    resp = _chat(client, headers={"X-API-Key": KEY})
    assert resp.status_code == 200, resp.text
    text, errors = _stream(resp)
    assert errors == []
    # handoff to the code agent, then execute_code, whose output is the pid that ran it
    match = re.fullmatch(r"child pid (\d+)", text.strip())
    assert match, text
    assert int(match.group(1)) != os.getpid()  # TestClient serves the app in this process


def test_chat_started_from_another_directory_still_gets_the_configured_tools(client, scripted_llm, monkeypatch, tmp_path):
    # ToolFactory used to read config/tools.yaml relative to the working directory, so an
    # API started elsewhere gave the agent no configured tools and raised nothing.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    resp = _chat(client)
    assert resp.status_code == 200, resp.text
    first_step = scripted_llm.offered[0]
    assert {"artifact", "get_file_content", "list_available_files", "query_index"} <= set(first_step), first_step


def test_each_chat_request_records_its_tool_calls_in_its_own_registry(client, monkeypatch):
    # The tool-call registry used to be one module-level list shared by every request.
    from app.api.routers import chat as chat_router
    from app.engine.tools import get_tool_call_registry

    monkeypatch.setenv("ENABLE_PYTHON_EXEC", "true")  # execute_code records through the tool callback
    monkeypatch.setenv(security.API_KEY_ENV, KEY)
    registries = []
    build = chat_router.get_chat_engine

    def build_and_capture(*args, **kwargs):
        engine = build(*args, **kwargs)
        registries.append(get_tool_call_registry())  # the registry this request's tools record into
        return engine

    monkeypatch.setattr(chat_router, "get_chat_engine", build_and_capture)
    for _ in range(2):
        assert _chat(client, headers={"X-API-Key": KEY}).status_code == 200
    assert len(registries) == 2 and registries[0] is not registries[1]
    assert [[call["name"] for call in registry] for registry in registries] == [["execute_code"], ["execute_code"]]


def test_non_streaming_chat_route_returns_the_answer_and_the_tool_calls(client):
    resp = client.post("/api/chat/request", json={"messages": [{"role": "user", "content": QUESTION}]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"]["content"] == "learning_rate is 3e-4"
    assert [tool["tool_name"] for tool in body["tools"]] == ["query_index"]


# ---------------------------------------------------------------- GET /api/query/


def test_query_returns_an_answer_from_the_index(client):
    resp = client.get("/api/query/", params={"query": "learning rate of run 47"})
    assert resp.status_code == 200, resp.text
    answer = resp.json()
    # MockLLM echoes the synthesis prompt: the retrieved text and the question are in it
    assert isinstance(answer, str) and FACT in answer and "learning rate of run 47" in answer
