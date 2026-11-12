"""Drive app/mcp/server.py (the hand-rolled JSON-RPC server) over stdio as a subprocess.

The server indexes its root at start, so search_semantic, search_lexical and
propose_artifact answer over stdio. The root is a copy of fixture v1, so
nothing here can touch bench/corpus.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from app.mcp.approvals import REFUSAL_CHECKS

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"
# numpy in the server process stays single-threaded on shared hosts.
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
# What a driver or operating system error can name (synthetic: no real host, address, user or path).
HOST, ADDRESS, USER = "db-7.internal.example", "10.20.30.40", "metanavit_ops"
PRIVATE = f"/home/{USER}/private/metanavit/operator.key"
INJECTED_ERROR = (f'connection to server at "{HOST}" ({ADDRESS}), port 5432 failed: password authentication failed '
                  f'for user "{USER}"; key file {PRIVATE}')
# The server's own main() and stdio loop, with the stat tool replaced by one that raises INJECTED_ERROR.
FAILING_STAT = (
    "import sys\n"
    "from app.mcp import filesystem, server\n"
    "def stat(self, paths):\n"
    f"    raise ConnectionError({INJECTED_ERROR!r})\n"
    "filesystem.FilesystemTools.stat = stat\n"
    "server.main(sys.argv[1:])\n"
)
# The answer to a call that raised something other than a refusal: the tool, the exception's class and an id.
FAILURE = re.compile(r"Error executing tool (\S+): (\w+) \(the server log has the reason, id ([0-9a-f]{16})\)")


def tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        rows.append([str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"])
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def fixture_copy(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)
    return corpus


def rpc(root: Path, calls: list[tuple[str, dict]], *flags: str, code: str | None = None) -> tuple[dict, str]:
    """Send initialize plus one tools/call per entry, close stdin, and return the responses by id and stderr.
    `code` is run with `python -c` instead of `-m app.mcp.server` (it calls the server's main itself)."""
    requests = [{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"}]
    requests += [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
                 for i, (name, arguments) in enumerate(calls, start=1)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("METANAVIT_APPROVAL_")}
    env.update({"PYTHONPATH": str(ROOT), **THREADS})
    launcher = ["-c", code] if code else ["-m", "app.mcp.server"]
    proc = subprocess.run([sys.executable, *launcher, str(root), *flags], cwd=ROOT, env=env,
                          input="".join(json.dumps(r) + "\n" for r in requests), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    responses = {}
    for line in proc.stdout.splitlines():  # every stdout line must be a JSON-RPC message
        message = json.loads(line)
        responses[message["id"]] = message
    return responses, proc.stderr


def result(response: dict):
    assert response["result"]["isError"] is False, response
    return json.loads(response["result"]["content"][0]["text"])


def error_text(response: dict) -> str:
    assert response["result"]["isError"] is True, response
    return response["result"]["content"][0]["text"]


def failure(response: dict) -> tuple[str, str, str]:
    """The answer to a call that raised (not a refusal): (the tool, the exception's class, the id it was logged under)."""
    match = FAILURE.fullmatch(error_text(response))
    assert match, response
    return match.groups()


def logged(stderr: str, error_id: str) -> str:
    """The server's log line for the failure with this id: stderr, since stdout carries the protocol."""
    lines = [line for line in stderr.splitlines() if f"[id {error_id}]" in line]
    assert len(lines) == 1, stderr
    return lines[0]


def test_search_and_propose_artifact_over_jsonrpc_stdio_on_the_fixture(tmp_path):
    corpus = fixture_copy(tmp_path)
    before = tree_digest(corpus)
    responses, _ = rpc(corpus, [("search_semantic", {"query": "learning rate of run_047", "k": 5}),
                                ("search_lexical", {"query": "run_047", "k": 5}),
                                ("propose_artifact", {"query": "write a test for the fusion module"}),
                                ("collect_run_artifact", {"run_id": "47"})])
    assert responses[0]["result"]["serverInfo"]["name"] == "metanavit-filesystem"
    semantic, lexical, proposal, bundle = (result(responses[i]) for i in (1, 2, 3, 4))
    assert bundle["entity"] == "run:47" and bundle["files"]
    for hits in (semantic, lexical):
        assert 0 < len(hits) <= 5
        assert all((corpus / hit["path"]).is_file() for hit in hits)
        assert any("run_047" in hit["path"] or "run_047" in hit["text"] for hit in hits)
    assert proposal["status"] == "pending_approval" and proposal["exec"]["ok"]
    assert proposal["review"]["action"] == "apply_artifact"
    assert proposal["spec"]["citations"] and all((corpus / c["path"]).is_file() for c in proposal["spec"]["citations"])
    assert not (corpus / proposal["spec"]["file_path"]).exists()  # a proposal never writes
    assert tree_digest(corpus) == before


def test_server_starts_without_an_index_on_an_empty_root_or_when_asked(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    responses, stderr = rpc(empty, [("list_dir", {"path": "."}), ("search_lexical", {"query": "run_047"})])
    assert result(responses[1]) == []
    tool, kind, error_id = failure(responses[2])  # the reason is in the server's log, not in the answer
    assert (tool, kind) == ("search_lexical", "RuntimeError") and "requires an index" in logged(stderr, error_id)
    assert "no index" in stderr

    corpus = fixture_copy(tmp_path)
    responses, stderr = rpc(corpus, [("search_semantic", {"query": "run_047"})], "--no-index")
    tool, kind, error_id = failure(responses[1])
    assert (tool, kind) == ("search_semantic", "RuntimeError") and "requires an index" in logged(stderr, error_id)
    assert "no index" in stderr


def test_an_answer_names_no_server_path_host_or_user_only_a_refusals_check_or_an_errors_class_and_id(tmp_path):
    """An exception's text never reaches the client: an operating system error names the root's absolute path, and an
    error from below the tools can name a host, an address and a user. A refusal is answered with its check's name and
    fixed description (as app/mcp/server_sdk.py answers it); anything else with its class and the id under which its
    text goes to stderr. A path in an answer is relative to the root."""
    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "run_047.yaml").write_text("learning_rate: 3e-4\n")
    outside.mkdir()
    (outside / "secret.txt").write_text("outside the root\n")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    calls = [("read_file", {"path": "missing.txt"}),                  # 1 FileNotFoundError: names the absolute root
             ("read_file", {"path": "."}),                             # 2 IsADirectoryError: names it too
             ("read_file", {"path": str(outside / "secret.txt")}),     # 3 a path outside the root
             ("read_file", {"path": "../outside/secret.txt"}),         # 4 the same, relative
             ("stat", {"paths": ["configs/run_047.yaml"]}),            # 5 INJECTED_ERROR: a host, an address, a user, a path
             ("read_file", {"path": "link.txt"}),                      # 6 refused by the symlink check
             ("apply_plan", {"plan_id": "p", "approved": True}),       # 7 refused: a caller's approval flag
             ("apply_plan", {"plan_id": "p", "approval_token": "x"}),  # 8 refused: no verifier, so no capability counts
             ("read_file", {"path": "configs/run_047.yaml"}),          # 9 answered
             ("apply_visualization", {"plan_id": "p"})]                # 10 refused as 8, before the plan is looked up
    responses, stderr = rpc(root, calls, "--no-index", code=FAILING_STAT)

    answers = json.dumps(responses)
    for needle in (str(tmp_path), str(tmp_path.resolve()), str(root.resolve()), str(outside), HOST, ADDRESS, USER, PRIVATE):
        assert needle not in answers, needle
    failures = {i: failure(responses[i]) for i in (1, 2, 3, 4, 5)}
    assert {i: (tool, kind) for i, (tool, kind, _) in failures.items()} == {
        1: ("read_file", "FileNotFoundError"), 2: ("read_file", "IsADirectoryError"), 3: ("read_file", "PermissionError"),
        4: ("read_file", "PermissionError"), 5: ("stat", "ConnectionError")}
    assert len({error_id for _, _, error_id in failures.values()}) == 5  # an id per failure
    # The server's log (stderr) has what the answers leave out, each under its answer's id.
    assert str(root.resolve() / "missing.txt") in logged(stderr, failures[1][2])
    assert str(root.resolve()) in logged(stderr, failures[2][2])
    assert INJECTED_ERROR in logged(stderr, failures[5][2])
    assert error_text(responses[6]) == f"Error executing tool read_file: refused by the symlink check: {REFUSAL_CHECKS['symlink']}"
    assert error_text(responses[7]) == ("Error executing tool apply_plan: refused by the approval_argument check: "
                                        + REFUSAL_CHECKS["approval_argument"])
    assert error_text(responses[8]) == ("Error executing tool apply_plan: refused by the capability_required check: "
                                        + REFUSAL_CHECKS["capability_required"])
    assert error_text(responses[10]) == error_text(responses[8]).replace("apply_plan", "apply_visualization")
    assert stderr.count("[id ") == 5  # a refusal is answered, not logged as a failure
    assert result(responses[9])["path"] == "configs/run_047.yaml"


def test_index_leaves_out_files_reached_through_symlinks(tmp_path):
    """The shared scanner rejects a link out of the root (outside_root) and the tools refuse
    such paths, so its text must not come back through search or a proposal either."""
    root, outside = tmp_path / "root", tmp_path / "outside"
    (outside / "dir").mkdir(parents=True)
    root.mkdir()
    (root / "inside.txt").write_text("calibration notes for the fusion test\n")
    (outside / "secret.txt").write_text("operator secret zqxjfilemarker\n")
    (outside / "dir" / "note.txt").write_text("outside directory zqxjdirmarker\n")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "linkdir").symlink_to(outside / "dir", target_is_directory=True)
    responses, stderr = rpc(root, [("search_lexical", {"query": "zqxjfilemarker"}),
                                   ("search_semantic", {"query": "zqxjfilemarker zqxjdirmarker", "k": 8}),
                                   ("propose_artifact", {"query": "write a test for zqxjfilemarker calibration"}),
                                   ("read_file", {"path": "link.txt"})])
    assert result(responses[1]) == []
    semantic = result(responses[2])
    assert semantic and {hit["path"] for hit in semantic} == {"inside.txt"}
    proposal = result(responses[3])  # the query itself is echoed; the outside file's text must not be
    assert {c["path"] for c in proposal["spec"]["citations"]} == {"inside.txt"}
    assert "operator secret" not in json.dumps(proposal)
    assert responses[4]["result"]["isError"]  # read_file refuses the same path
    assert "1 file reached through a symlink left out" in stderr


def test_the_start_line_counts_links_out_of_the_root_and_links_inside_it(tmp_path):
    """Links out of the root never reach the index (the shared scanner rejects them); a link
    that stays inside the root is indexed by the scanner and left out by the server, whose
    tools refuse every path through a symlink. The start-up line counts both."""
    from app.mcp.filesystem import build_root_index, index_summary

    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "inside.txt").write_text("calibration notes for the fusion test\n")
    (outside / "secret.txt").write_text("operator secret zqxjfilemarker\n")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "alias.txt").symlink_to(root / "inside.txt")

    index = build_root_index(root)

    assert {chunk.path for chunk in index.chunks} == {"inside.txt"}
    assert index.symlinked_files_left_out == 2
    assert "2 files reached through a symlink left out" in index_summary(index)
