"""C47: an exception's text never reaches a client; the client gets a fixed text and an id the server log files it under.

A driver error names the database host, its address, the port and the user. Each test injects such an exception at
the point where the code catches it (or lets it escape) on chat, retrieve, plans and /health, plus the upload and
query routes and the start-up, and checks that no response body carries any part of it, that the body's request id
is the response's X-Request-ID header, and that the server log has the full text under that id. Texts this repository
wrote (the plans' fixed 503 texts, "weights not cached", an approval configuration error) still reach the client.
No database: the same closed-port setup as tests/api/test_retrieve_contract.py.
"""

import logging
import re

import psycopg2
import pytest
from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

HOST, USER, ADDRESS, PORT = "db-7.internal.example", "metanavit_admin", "10.20.30.40", "6432"
SECRET = f'connection to server at "{HOST}" ({ADDRESS}), port {PORT} failed: FATAL: password authentication failed for user "{USER}"'
NEEDLES = (HOST, USER, ADDRESS, PORT)
BAD_DB = "postgresql://postgres:password@127.0.0.1:1/metanavit"
ID = re.compile(r"[0-9a-f]{16}")


def leaked(text: str) -> list[str]:
    return [needle for needle in NEEDLES if needle in text]


def _raise(*args, **kwargs):
    raise psycopg2.OperationalError(SECRET)


def logged_under(caplog, error_id: str) -> None:
    """The server log has the exception's full text on a line that names `error_id`."""
    lines = [record.getMessage() for record in caplog.records if error_id in record.getMessage()]
    assert any(HOST in line and USER in line for line in lines), caplog.text


def assert_public(resp, status: int, detail: str, caplog) -> str:
    """The answer: `status`, the fixed `detail`, the request id in the body and the header, nothing of SECRET."""
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["detail"] == detail and not leaked(resp.text), resp.text
    assert ID.fullmatch(body["request_id"]) and resp.headers["x-request-id"] == body["request_id"]
    logged_under(caplog, body["request_id"])
    return body["request_id"]


def _env(monkeypatch):
    for name, value in {"MODEL_PROVIDER": "ollama", "EMBEDDING_PROVIDER": "hash", "PG_CONNECTION_STRING": BAD_DB,
                        "PSYCOPG2_CONNECTION_STRING": "dbname=metanavit user=postgres password=password host=127.0.0.1 port=1",
                        "INDEX_ON_START": "false"}.items():
        monkeypatch.setenv(name, value)
    for name in ("FRONTEND_ENDPOINT", "RETRIEVAL_REQUIRE_DEVICE", "METANAVIT_API_KEY", "METANAVIT_API_KEYS_FILE",
                 "RATE_LIMIT_PER_MINUTE", "STARTUP_RETRY_ATTEMPTS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(monkeypatch, caplog):
    _env(monkeypatch)
    caplog.set_level(logging.INFO)
    import main  # noqa: WPS433 - the FastAPI app module

    from app.database import vector_store

    vector_store.get_vector_store_manager.cache_clear()
    with TestClient(main.app, raise_server_exceptions=False) as c:  # an escaped exception is answered, as in uvicorn
        yield c
        c.app.state.retrieval = None
        c.app.state.plans = None


# ---------------------------------------------------------------- the id


def test_every_response_carries_its_own_request_id(client):
    answers = [client.get("/health"), client.post("/api/retrieve/", json={"query": "q"}), client.get("/api/plans/"),
               client.get("/openapi.json")]
    ids = [resp.headers.get("x-request-id") for resp in answers]
    assert all(value and ID.fullmatch(value) for value in ids) and len(set(ids)) == len(ids), ids


# ---------------------------------------------------------------- chat


def _chat(client, path="/api/chat"):
    return client.post(path, json={"messages": [{"role": "user", "content": "what learning rate did run 47 use"}]})


def test_chat_engine_failure_is_a_fixed_text_and_an_id(client, monkeypatch, caplog):
    from app.api.routers import chat as chat_router

    monkeypatch.setattr(chat_router, "get_chat_engine", _raise)
    assert_public(_chat(client), 500, "Error in chat engine", caplog)  # was "Error in chat engine: <the driver's text>"


def test_a_failure_while_the_chat_answer_streams_names_the_id_not_the_error(client, monkeypatch, caplog):
    from app.api.routers import chat as chat_router

    class _Answer:
        source_nodes = []

        async def async_response_gen(self):
            raise psycopg2.OperationalError(SECRET)
            yield  # pragma: no cover - an async generator

    class _Engine:
        async def achat(self, message, history):
            return _Answer()

    monkeypatch.setattr(chat_router, "get_chat_engine", lambda **kwargs: _Engine())
    resp = _chat(client)
    assert resp.status_code == 200 and not leaked(resp.text), resp.text  # the stream reports a failure under HTTP 200
    errors = [line for line in resp.text.splitlines() if line.startswith("3:")]
    request_id = resp.headers["x-request-id"]
    assert len(errors) == 1 and "Please try again" in errors[0] and request_id in errors[0], errors
    logged_under(caplog, request_id)


def test_an_exception_no_route_catches_is_a_fixed_500_with_an_id(client, monkeypatch, caplog):
    from app.api.routers import chat as chat_router
    from app.api.routers import query as query_router

    monkeypatch.setattr(chat_router, "get_chat_engine", _raise)
    monkeypatch.setattr(query_router, "get_query_engine", _raise)
    assert_public(_chat(client, "/api/chat/request"), 500, "internal error", caplog)  # was Starlette's bare text
    assert_public(client.get("/api/query/", params={"query": "q"}), 500, "internal error", caplog)


def test_an_upload_that_fails_is_its_fixed_text_and_an_id(client, monkeypatch, caplog):
    import base64

    from app.engine.loaders.file import FileService

    monkeypatch.setattr(FileService, "process_private_file", staticmethod(_raise))
    body = {"name": "notes.md", "base64": "data:text/markdown;base64," + base64.b64encode(b"# notes\n").decode()}
    assert_public(client.post("/api/chat/upload", json=body), 500, "Error processing file", caplog)


def test_an_upload_name_cannot_start_a_log_line_of_its_own(client, monkeypatch, caplog):
    """The route logs the client's file name when it starts and again when the upload is refused; a name holding a
    newline, a carriage return or a forged record must stay inside the one line the route writes (as report_failure's
    lines do since M35 C4)."""
    import base64

    from app.engine.loaders.file import FileService
    from app.engine.uploads import UploadRejected

    forged = "notes.md\nERROR forged record id=deadbeefdeadbeef\rmore"
    caplog.set_level(logging.INFO, logger="uvicorn")  # the route's lines are INFO; an earlier test may raise this logger's level

    def refuse(name, *args, **kwargs):
        raise UploadRejected(415, "unsupported_type", f"{name} is not a text file")

    monkeypatch.setattr(FileService, "process_private_file", staticmethod(refuse))
    body = {"name": forged, "base64": "data:text/markdown;base64," + base64.b64encode(b"# notes\n").decode()}
    resp = client.post("/api/chat/upload", json=body)
    assert resp.status_code == 415
    upload_lines = [r.getMessage() for r in caplog.records
                    if r.getMessage().startswith(("Processing file: ", "Upload refused"))]
    assert any(line.startswith("Processing file: ") for line in upload_lines), upload_lines
    assert any(line.startswith("Upload refused (unsupported_type): ") for line in upload_lines), upload_lines
    for line in upload_lines:
        assert "\n" not in line and "\r" not in line, line
        assert "forged record" not in line or "\\n" in line, line  # the forged text is still there, escaped


# ---------------------------------------------------------------- retrieve


class _VectorRetriever:
    def retrieve(self, query):
        return [NodeWithScore(node=TextNode(id_="a", text="learning_rate: 3e-4", metadata={"path": "configs/run_047.yaml"}), score=0.9),
                NodeWithScore(node=TextNode(id_="b", text="lr 3e-4", metadata={"path": "notes/lr.md"}), score=0.5)]


class _LexicalDown:
    """The database manager's lexical search on a database that refuses the connection."""

    hybrid_backend = "ts_rank_cd"
    pgvector_version = (0, 8, 6)

    def search_bm25(self, query, limit):
        raise psycopg2.OperationalError(SECRET)

    def count_nodes(self):
        return 2

    def search_settings(self, k):
        return {}


class _CrossEncoderDown:
    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        raise RuntimeError(SECRET)


class _State:
    def __init__(self, retriever, vsm):
        self.retriever, self.vsm, self.index, self.indexing = retriever, vsm, None, {"indexed": False}


def test_a_retrieval_failure_is_a_fixed_503_and_an_id(client, caplog):
    class _Broken:
        _reranker, mode, depth, reranker_info = None, "sql", 50, {}

        def retrieve_detailed(self, query, top_n=None, deadline=None):
            raise psycopg2.OperationalError(SECRET)

    client.app.state.retrieval = _State(_Broken(), _LexicalDown())
    assert_public(client.post("/api/retrieve/", json={"query": "q"}), 503, "retrieval backend unavailable", caplog)


def test_degraded_lexical_and_rerank_legs_name_the_class_and_the_id_not_the_error(client, caplog):
    """The real HybridRetriever: its BM25 leg and its reranker fail on this request; the 200 lists both as degraded."""
    from app.engine.retriever import HybridRetriever

    retriever = HybridRetriever(_VectorRetriever(), _LexicalDown(), reranker=_CrossEncoderDown(), rerank_top_n=8,
                                enable_router=False, reranker_info={"configured": True, "loaded": True, "model": "x"})
    client.app.state.retrieval = _State(retriever, _LexicalDown())
    resp = client.post("/api/retrieve/", json={"query": "learning rate"})
    assert resp.status_code == 200 and not leaked(resp.text), resp.text
    request_id = resp.headers["x-request-id"]
    degraded = {entry["component"]: entry["error"] for entry in resp.json()["degraded"]}
    assert degraded == {"bm25": f"OperationalError (the server log has the reason, id {request_id})",
                        "rerank": f"RuntimeError (the server log has the reason, id {request_id})"}
    logged_under(caplog, request_id)


def test_a_reranker_that_fails_to_load_is_reported_by_class_and_id(client, monkeypatch, caplog):
    from app.engine import retriever as mod

    def cannot_load(*args, **kwargs):
        raise OSError(f"{SECRET}: /home/{USER}/hf_cache/models--x not found")

    mod._reranker_cache.clear()
    monkeypatch.setattr("app.retrieval.rerank.get_cross_encoder", cannot_load)
    settings = {"model": "x/reranker", "precision": "fp32", "max_length": 512, "depth": 20, "required": True, "device": "cpu"}
    with pytest.raises(mod.RerankerUnavailable) as refused:
        mod.load_reranker(settings)
    assert not leaked(str(refused.value))  # RerankerUnavailable carries the loader's error into the start-up error
    mod._reranker_cache.clear()
    _, info = mod.load_reranker({**settings, "required": False})
    mod._reranker_cache.clear()
    error_id = ID.search(info["error"]).group(0)
    assert info["error"] == f"OSError (the server log has the reason, id {error_id})"
    logged_under(caplog, error_id)  # logged once, when it failed to load; every answer then names that id

    retriever = mod.HybridRetriever(_VectorRetriever(), _LexicalDown(), reranker=None, enable_router=False, reranker_info=info)
    client.app.state.retrieval = _State(retriever, _LexicalDown())
    for resp in (client.post("/api/retrieve/", json={"query": "q"}), client.get("/health")):
        assert not leaked(resp.text) and error_id in resp.text, resp.text


# ---------------------------------------------------------------- /health


def test_health_reports_an_unreadable_data_table_by_class_and_id(client, caplog):
    class _CountFails(_LexicalDown):
        def count_nodes(self):
            raise psycopg2.OperationalError(SECRET)

    client.app.state.retrieval = _State(type("_R", (), {"_reranker": None, "mode": "sql", "depth": 50, "reranker_info": {}})(), _CountFails())
    resp = client.get("/health")
    assert resp.status_code == 503 and not leaked(resp.text), resp.text
    request_id = resp.headers["x-request-id"]
    assert resp.json()["degraded"] == [{"component": "database", "error": f"OperationalError (the server log has the reason, id {request_id})"}]
    logged_under(caplog, request_id)


def test_a_failed_start_is_shown_by_class_and_the_id_it_was_logged_under(monkeypatch, caplog):
    """/health, /api/retrieve and /api/plans used to send the stored start-up error in full (/api/plans by class since
    M21). The start-up logs it once under an id; every answer names the class and that id."""
    from app.api.routers import plans
    from app.engine import bootstrap

    _env(monkeypatch)
    monkeypatch.setenv("STARTUP_RETRY_ATTEMPTS", "1")  # one attempt, no retry
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(bootstrap, "build_retrieval_state", _raise)
    monkeypatch.setattr(plans, "build_plan_store", lambda vsm, root=None: pytest.fail("never reached"))
    import main  # noqa: WPS433 - the FastAPI app module

    with TestClient(main.app) as client:
        health = client.get("/health")
        error_id = client.app.state.startup_error_id
        shown = f"OperationalError (the server log has the reason, id {error_id})"
        assert health.status_code == 503 and health.json()["error"] == shown and not leaked(health.text)
        retrieve = client.post("/api/retrieve/", json={"query": "q"})
        assert (retrieve.status_code, retrieve.json()["detail"]) == (503, f"retrieval backend unavailable: {shown}")
        listed = client.get("/api/plans/")
        assert (listed.status_code, listed.json()["detail"]) == (503, f"plans unavailable: {shown}")
        assert not leaked(retrieve.text + listed.text)
        assert HOST in client.app.state.startup_error  # kept in full for the server, never sent
    logged_under(caplog, error_id)


# ---------------------------------------------------------------- plans


def test_a_plan_store_failure_keeps_its_fixed_text_and_gains_the_id(client, caplog, tmp_path):
    from app.api.routers.plans import PlanStore
    from app.mcp.filesystem import FilesystemTools

    class _TableDown:
        def _fail(self, *args, **kwargs):
            raise psycopg2.OperationalError(SECRET)

        add = get = list = deciding = _fail

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "run_047.yaml").write_text("learning_rate: 3e-4\n")
    client.app.state.plans = PlanStore(FilesystemTools(root=tmp_path), None, _TableDown())
    assert_public(client.get("/api/plans/"), 503, "plan store unavailable", caplog)
    assert_public(client.post("/api/plans/x/reject"), 503, "plan store unavailable", caplog)


def test_a_failure_is_one_log_line_whatever_the_client_put_in_the_url(client, caplog, tmp_path):
    """The plan id comes from the URL, where %0A and %0D decode to a newline and a carriage return, and the failure is
    logged with it twice: in what failed, and in the driver's error, which quotes the value it refused. Each such
    character is written as its escape, so one failure is one log line and the client starts no line of its own, in
    the text format and in the JSON one (LOG_FORMAT=json), which stays one valid object per line."""
    import io
    import json
    from urllib.parse import quote

    from app.api.routers.plans import PlanStore
    from app.mcp.filesystem import FilesystemTools
    from app.observability import JsonFormatter

    class _TableRefuses:
        def deciding(self, plan_id, root_identity):
            raise psycopg2.DataError(f'invalid input syntax for type uuid: "{plan_id}"')

    forged = "x\nERROR forged by a newline [id deadbeefdeadbeef]\rERROR forged by a carriage return [id feedfacefeedface]"
    client.app.state.plans = PlanStore(FilesystemTools(root=tmp_path), None, _TableRefuses())
    json_log = io.StringIO()
    handler = logging.StreamHandler(json_log)
    handler.setFormatter(JsonFormatter())
    logging.getLogger("uvicorn").addHandler(handler)  # the logger app/api/errors.py and the plans router write to
    try:
        resp = client.post(f"/api/plans/{quote(forged, safe='')}/reject")
    finally:
        logging.getLogger("uvicorn").removeHandler(handler)
    assert resp.status_code == 503 and resp.json()["detail"] == "plan store unavailable", resp.text
    request_id = resp.json()["request_id"]
    escaped = forged.replace("\n", "\\n").replace("\r", "\\r")
    [message] = [record.getMessage() for record in caplog.records if request_id in record.getMessage()]
    assert message == (f"plan store: deciding plan {escaped} failed [id {request_id}]: DataError: "
                       f'invalid input syntax for type uuid: "{escaped}"')
    assert message.splitlines() == [message] and not any(line.startswith("ERROR forged") for line in caplog.text.splitlines())
    lines = json_log.getvalue().splitlines()
    assert [json.loads(line)["msg"] for line in lines if request_id in line] == [message]
    assert not any(line.startswith("ERROR forged") for line in lines)


def test_an_unexpected_failure_in_a_plans_route_is_a_fixed_500_with_an_id(client, caplog, tmp_path, monkeypatch):
    from app.api.routers.plans import PlanStore
    from app.mcp.filesystem import FilesystemTools

    store = PlanStore(FilesystemTools(root=tmp_path), None)
    monkeypatch.setattr(store.tools, "propose_move", _raise)
    client.app.state.plans = store
    assert_public(client.post("/api/plans/", json={"src": "a.yaml", "dst": "b.yaml"}), 500, "internal error", caplog)
