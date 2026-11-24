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
