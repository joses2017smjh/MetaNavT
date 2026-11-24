"""The agent-path runs (A25 / C13, C23) have generated rows in docs/RESUME_EVIDENCE.md once their files exist,
and every number in those rows comes from the results file."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.eval.evidence import FILES, claims

ROOT = Path(__file__).resolve().parents[2]
# B14: the served lexical_path route gets the benchmark's path lookup (a path list in the served SQL statement)
B14 = "c445848f995451dcab522a43c235920b8c37f89c"


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("key,command", [("companion_live", "make bench-companion-live"),
                                         ("chat_route_smoke", "make chat-route-smoke")])
def test_rows_follow_the_results_file(key, command, tmp_path):
    rows = [r for r in claims() if r["command"] == command]
    if not (ROOT / FILES[key]).exists():
        assert rows == []
        pytest.skip(f"{FILES[key]} not recorded yet")
    assert rows and all(r["source"].startswith(FILES[key] + " @ ") for r in rows)
    # rendered from the file: a copy with other numbers renders other values
    blob = json.loads((ROOT / FILES[key]).read_text())
    if key == "companion_live":
        blob["summary"]["gold"]["overall"]["status"] = {"complete": 1}
    else:
        blob["summary"]["/api/chat"]["answered"] = 99
    target = tmp_path / FILES[key]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(blob))
    changed = [r for r in claims(tmp_path) if r["command"] == command]
    assert [r["value"] for r in changed] != [r["value"] for r in rows]


@pytest.mark.parametrize("key,command", [("companion_live", "make bench-companion-live"),
                                         ("chat_route_smoke", "make chat-route-smoke")])
def test_every_row_names_the_fallback_retrieval(key, command):
    rows = [r for r in claims() if r["command"] == command]
    if not (ROOT / FILES[key]).exists():
        pytest.skip(f"{FILES[key]} not recorded yet")
    for row in rows:
        assert all(label in row["not_shown"] for label in ("hash embeddings", "ts_rank_cd", "no reranker")), row["claim"]


def test_the_companion_rows_say_where_paths_are_searched_and_the_run_had_that_code():
    """The rows say the served retrieval searches file names and paths only on the router's lexical_path route. That
    holds for the code since B14, so the recorded run's commit must contain B14 (a recording from before it had no
    path list). A shallow clone without the commits is a skip (CI's eval job fetches the full history)."""
    if not (ROOT / FILES["companion_live"]).exists():
        pytest.skip("bench/results/companion_live.json not recorded yet")
    rows = [r for r in claims() if r["command"] == "make bench-companion-live"]
    said = "file names and paths are searched only for a search with a path-like token, the router's lexical_path route"
    assert rows and all(said in r["not_shown"] for r in rows)
    head = json.loads((ROOT / FILES["companion_live"]).read_text())["git_head"]
    if shutil.which("git") is None or _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip("not a git checkout: the recorded commit cannot be read")
    for sha in (B14, head):
        if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
                pytest.skip(f"shallow clone without commit {sha[:7]}: fetch the full history (git fetch --unshallow)")
            pytest.fail(f"commit {sha} is not in this repository's history")
    assert _git("merge-base", "--is-ancestor", B14, head).returncode == 0, \
        f"{FILES['companion_live']} was recorded at {head[:7]}, before B14: its served retrieval had no path list"


def test_the_citation_row_counts_the_answers_the_companion_blocked_for_citing(tmp_path):
    if not (ROOT / FILES["companion_live"]).exists():
        pytest.skip("bench/results/companion_live.json not recorded yet")
    blob = json.loads((ROOT / FILES["companion_live"]).read_text())

    def citation_row(root):
        return next(r for r in claims(root) if r["command"] == "make bench-companion-live" and "citation" in r["claim"])

    blocked = blob["summary"]["gold"]["overall"]["stop_reason"].get("invalid_citations", 0)
    assert f"({blocked} here, stop reason invalid_citations)" in citation_row(ROOT)["not_shown"]
    assert "stopped at the model-round limit" in citation_row(ROOT)["not_shown"]
    blob["summary"]["gold"]["overall"]["stop_reason"]["invalid_citations"] = blocked + 5
    target = tmp_path / FILES["companion_live"]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(blob))
    assert f"({blocked + 5} here, stop reason invalid_citations)" in citation_row(tmp_path)["not_shown"]


def test_the_request_route_row_shows_the_failed_tool_calls(tmp_path):
    if not (ROOT / FILES["chat_route_smoke"]).exists():
        pytest.skip("bench/results/chat_route_smoke.json not recorded yet")
    blob = json.loads((ROOT / FILES["chat_route_smoke"]).read_text())
    request = blob["summary"]["/api/chat/request"]

    def request_row(root):
        return next(r for r in claims(root) if r["command"] == "make chat-route-smoke" and "/api/chat/request" in r["claim"])

    for field, label in (("tool_calls_raised", "flagged as errors"), ("tool_calls_returned_error_text", "returned an error text")):
        shown = ", ".join(f"{name} {count}" for name, count in request[field].items()) or "none"
        assert f"{label}: {shown}" in request_row(ROOT)["value"]
    request["tool_calls_raised"] = {"query_index": 99}
    target = tmp_path / FILES["chat_route_smoke"]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(blob))
    assert "flagged as errors: query_index 99" in request_row(tmp_path)["value"]


def test_the_request_route_row_counts_not_found_replies_apart_from_failures_when_the_file_has_them(tmp_path):
    if not (ROOT / FILES["chat_route_smoke"]).exists():
        pytest.skip("bench/results/chat_route_smoke.json not recorded yet")
    blob = json.loads((ROOT / FILES["chat_route_smoke"]).read_text())
    request = blob["summary"]["/api/chat/request"]

    def request_row(root):
        return next(r for r in claims(root) if r["command"] == "make chat-route-smoke" and "/api/chat/request" in r["claim"])

    def write(summary_fields):
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        target = root / FILES["chat_route_smoke"]
        target.parent.mkdir(parents=True)
        copy = json.loads(json.dumps(blob))
        copy["summary"]["/api/chat/request"].update(summary_fields)
        for key in [k for k, v in summary_fields.items() if v is None]:
            del copy["summary"]["/api/chat/request"][key]
        target.write_text(json.dumps(copy))
        return root

    # a file recorded before the runner counted them says nothing about not-found replies
    old = request_row(write({"tool_calls_returned_not_found": None}))
    assert "not found" not in old["value"].lower() and "not found" not in old["not_shown"].lower()
    # a file that counts them shows the count, and says such a reply is not a failure
    new = request_row(write({"tool_calls_returned_not_found": {"get_file_content": 3}}))
    assert "; not found: get_file_content 3;" in new["value"]
    assert "a 'Not found: ...' reply" in new["not_shown"] and "not counted as failed" in new["not_shown"]
    assert "not found: none" in request_row(write({"tool_calls_returned_not_found": {}}))["value"]
    # the pointer to the causes of failed calls is there only when a call failed
    failing = request_row(write({"tool_calls_raised": {"list_available_files": 1}, "tool_calls_returned_error_text": {}}))
    assert "the causes are described in doc/engineering-reference.md (Backend)" in failing["not_shown"]
    clean = request_row(write({"tool_calls_raised": {}, "tool_calls_returned_error_text": {}}))
    assert "the causes are described" not in clean["not_shown"]
    assert request is not None


def _chat_rows(root):
    rows = [r for r in claims(root) if r["command"] == "make chat-route-smoke"]
    return (next(r for r in rows if "POST /api/chat (" in r["claim"]), next(r for r in rows if "/api/chat/request" in r["claim"]))


def _write_chat(tmp_path, blob):
    root = tmp_path / str(len(list(tmp_path.iterdir())))
    target = root / FILES["chat_route_smoke"]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(blob))
    return root


def test_the_chat_rows_say_how_the_unanswered_requests_ended(tmp_path):
    """Requests the routes did not answer are counted from the rows by how they ended, and only when there are some."""
    if not (ROOT / FILES["chat_route_smoke"]).exists():
        pytest.skip("bench/results/chat_route_smoke.json not recorded yet")
    blob = json.loads((ROOT / FILES["chat_route_smoke"]).read_text())
    every = json.loads(json.dumps(blob))
    for row in every["rows"]:
        row["answered"] = True
    for row in _chat_rows(_write_chat(tmp_path, every)):
        assert "not answered" not in row["value"], row["value"]
        assert "unanswered ones included" not in row["not_shown"] and "not in the file" not in row["not_shown"]
    some = json.loads(json.dumps(blob))
    for row in some["rows"]:
        row["answered"] = True
    for route in ("/api/chat", "/api/chat/request"):
        first, second = [r for r in some["rows"] if r["route"] == route][:2]
        first.update(answered=False, http_status=None, answer="", tools=None, error="ReadTimeout")
        second.update(answered=False, http_status=200, answer="", errors=[] if route == "/api/chat" else None)
    stream, request = _chat_rows(_write_chat(tmp_path, some))
    said = (f"(not answered: 1 got no response within the runner's {some['request_timeout_s']:g} s limit, 1 returned HTTP 200 "
            "with no text and no error line; see doc/engineering-reference.md, Backend)")
    assert said in stream["value"] and said in request["value"]
    assert "over every request, the unanswered ones included" in stream["not_shown"]
    assert "the tool calls of a request that got no response are not in the file (1 here)" in request["not_shown"]


# The chat smoke committed before M39's fix (run at 50e9ee2): the reference describes it as history.
CHAT_BEFORE_FIX = "c9b4acd7f3f178bbb79ec1fd65209a7f47609a80"


def _chat_file_at(sha: str) -> dict:
    if shutil.which("git") is None or _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip("not a git checkout: the committed file cannot be read")
    if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
            pytest.skip(f"shallow clone without commit {sha[:7]}: fetch the full history (git fetch --unshallow)")
        pytest.fail(f"commit {sha} is not in this repository's history")
    shown = _git("show", f"{sha}:{FILES['chat_route_smoke']}")
    assert shown.returncode == 0, shown.stderr
    return json.loads(shown.stdout)


def _backend() -> str:
    return " ".join((ROOT / "doc" / "engineering-reference.md").read_text().split())


def test_the_backend_paragraph_on_the_chat_smoke_before_the_fix_holds_for_that_file():
    """doc/engineering-reference.md (Backend) describes the recording committed in c9b4acd; what it says of the
    unanswered requests is what that file's rows and header show (its counts are cited, and checked, by
    app/eval/prose_numbers.py)."""
    old = _chat_file_at(CHAT_BEFORE_FIX)
    text = _backend()
    assert "In the recording committed in c9b4acd (run at 50e9ee2) not every request was answered" in text
    assert old["git_head"].startswith("50e9ee2")
    rows = old["rows"]
    # "On the non-streaming route one request got no response within the runner's limit, so the tool calls it made
    # are not in that file."
    no_response = [i for i, r in enumerate(rows) if r["http_status"] is None]
    assert len(no_response) == 1 and rows[no_response[0]]["route"] == "/api/chat/request"
    assert rows[no_response[0]]["tools"] is None
    # "Every request after it, on both routes, came back as HTTP 200 with no text and no error line once the chat
    # LLM's request timeout had passed"
    after = rows[no_response[0] + 1:]
    assert [i for i, r in enumerate(rows) if not r["answered"]] == no_response + list(range(no_response[0] + 1, len(rows)))
    assert {r["route"] for r in after} == {"/api/chat", "/api/chat/request"}
    timeout = old["chat_llm"]["request_timeout_s"]
    assert all(r["http_status"] == 200 and r["answer"] == "" and not r.get("errors") and r["ttfb_s"] >= timeout
               for r in after)
    # "the chat LLM then had no limit on the tokens it generates (`chat_llm.additional_kwargs` in that file is empty)"
    assert old["chat_llm"]["additional_kwargs"] == {}
    assert "LLM_MAX_TOKENS" not in old["chat_llm"]["set_in_api_environment"]


def test_the_backend_paragraph_says_how_the_current_chat_file_went_once_it_is_a_newer_recording():
    """While the committed file is still the one from before the fix, the history above describes it. Once it is a
    newer recording, Backend says whether every request was answered, and the generated rows give the counts."""
    if not (ROOT / FILES["chat_route_smoke"]).exists():
        pytest.skip("bench/results/chat_route_smoke.json not recorded yet")
    blob = json.loads((ROOT / FILES["chat_route_smoke"]).read_text())
    text = _backend()
    every = "The current file, recorded after the fix, answered every request on both routes"
    some = "In the current file, recorded after the fix, not every request was answered"
    if blob == _chat_file_at(CHAT_BEFORE_FIX):
        assert every not in text and some not in text
        return
    unanswered = [r for r in blob["rows"] if not r["answered"]]
    assert (every in text) == (not unanswered) and (some in text) == bool(unanswered)
    # the recording after the fix ran with the output-token cap
    assert "num_predict" in blob["chat_llm"]["additional_kwargs"]
