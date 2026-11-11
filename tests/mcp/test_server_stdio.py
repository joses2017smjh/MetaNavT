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
                                ("propose_artifact", {"query": "write a test for the fusion module"})])
    assert responses[0]["result"]["serverInfo"]["name"] == "metanavit-filesystem"
    semantic, lexical, proposal = (result(responses[i]) for i in (1, 2, 3))
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
