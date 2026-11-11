"""Drive app/mcp/server.py (the hand-rolled JSON-RPC server) over stdio as a subprocess.

The server indexes its root at start, so search_semantic, search_lexical and
propose_artifact answer over stdio. The root is a copy of fixture v1, so
nothing here can touch bench/corpus.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"
# numpy in the server process stays single-threaded on shared hosts.
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        rows.append([str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"])
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def fixture_copy(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)
    return corpus


def rpc(root: Path, calls: list[tuple[str, dict]], *flags: str) -> tuple[dict, str]:
    """Send initialize plus one tools/call per entry, close stdin, and return the responses by id and stderr."""
    requests = [{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"}]
    requests += [{"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
                 for i, (name, arguments) in enumerate(calls, start=1)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("METANAVIT_APPROVAL_")}
    env.update({"PYTHONPATH": str(ROOT), **THREADS})
    proc = subprocess.run([sys.executable, "-m", "app.mcp.server", str(root), *flags], cwd=ROOT, env=env,
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
    assert responses[2]["result"]["isError"] and "requires an index" in responses[2]["result"]["content"][0]["text"]
    assert "no index" in stderr

    corpus = fixture_copy(tmp_path)
    responses, stderr = rpc(corpus, [("search_semantic", {"query": "run_047"})], "--no-index")
    assert responses[1]["result"]["isError"] and "requires an index" in responses[1]["result"]["content"][0]["text"]
    assert "no index" in stderr


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
