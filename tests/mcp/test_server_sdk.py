"""Spawn app/mcp/server_sdk.py over stdio with the official SDK client; list and call tools."""

import asyncio
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from app.mcp.approvals import create_key, sign_review

import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"
# numpy in the server process stays single-threaded on shared hosts.
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _text(result) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


def _payload(result):
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if structured:
        return structured.get("result", structured) if isinstance(structured, dict) and set(structured) == {"result"} else structured
    return json.loads(_text(result))


def _blocks(result) -> list:
    """A list result: the SDK sends one text block per item."""
    return [json.loads(c.text) for c in result.content if getattr(c, "text", None)]


async def _drive(corpus: Path, private: Path) -> dict:
    key = private / "key"
    create_key(key)
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp.server_sdk", "--root", str(corpus)], cwd=str(ROOT), env={"PYTHONPATH": str(ROOT), "METANAVIT_APPROVAL_KEY_FILE": str(key), "METANAVIT_APPROVAL_LEDGER_FILE": str(private / "ledger.sqlite")})
    out: dict = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "serverInfo", None) or getattr(init, "server_info", None)
            out["server_name"] = info.name
            tools = await session.list_tools()
            out["tools"] = {t.name: (getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)) for t in tools.tools}
            listed = await session.call_tool("list_dir", {"path": "."})
            out["list_dir"] = _payload(listed)
            plan = await session.call_tool("propose_move", {"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"})
            out["plan"] = _payload(plan)
            refused = await session.call_tool("apply_plan", {"plan_id": out["plan"]["plan_id"]})
            out["refused"] = _payload(refused)
            token = sign_review(out["plan"]["review"], key, actor="sdk-test-operator")
            applied = await session.call_tool("apply_plan", {"plan_id": out["plan"]["plan_id"], "approval_token": token})
            out["applied"] = _payload(applied)
    return out


def test_sdk_server_lists_and_calls_tools(tmp_path, tmp_path_factory):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "run_047.yaml").write_text("learning_rate: 3e-4\n")
    out = asyncio.run(_drive(tmp_path, tmp_path_factory.mktemp("sdk-operator-private")))

    assert out["server_name"] == "metanavit-filesystem"
    assert {"search_lexical", "read_file", "list_dir", "propose_move", "apply_plan"} <= set(out["tools"])
    assert "approval_token" in out["tools"]["apply_plan"]["properties"]
    assert "approved" not in out["tools"]["apply_plan"]["properties"]
    assert "src" in out["tools"]["propose_move"]["properties"] and "src" in out["tools"]["propose_move"].get("required", [])
    listed = out["list_dir"]
    entries = listed if isinstance(listed, list) else listed.get("result", listed)
    assert any("configs" in json.dumps(e) for e in (entries if isinstance(entries, list) else [entries]))
    assert out["plan"]["status"] == "pending_approval"
    assert out["refused"].get("error") == "approval_required"  # the gate survives the SDK port
    assert out["applied"]["status"] == "applied" and (tmp_path / "configs" / "archive" / "run_047.yaml").exists()


def _tree_digest(root: Path) -> str:
    rows = [[str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"] for p in sorted(root.rglob("*"))]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


async def _search_and_propose(corpus: Path) -> dict:
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp.server_sdk", "--root", str(corpus)], cwd=str(ROOT), env={"PYTHONPATH": str(ROOT), **THREADS})
    out: dict = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for name, arguments in (("search_semantic", {"query": "learning rate of run_047", "k": 5}),
                                    ("search_lexical", {"query": "run_047", "k": 5}),
                                    ("propose_artifact", {"query": "write a test for the fusion module"})):
                called = await session.call_tool(name, arguments)
                assert not called.is_error, _text(called)
                out[name] = _payload(called) if name == "propose_artifact" else _blocks(called)
    return out


def test_sdk_server_searches_and_proposes_on_the_fixture(tmp_path):
    corpus = tmp_path / "corpus"
    shutil.copytree(FIXTURE, corpus)  # a copy: nothing here can touch bench/corpus
    before = _tree_digest(corpus)
    out = asyncio.run(_search_and_propose(corpus))

    for name in ("search_semantic", "search_lexical"):
        hits = out[name]
        assert 0 < len(hits) <= 5
        assert all((corpus / hit["path"]).is_file() for hit in hits)
        assert any("run_047" in hit["path"] or "run_047" in hit["text"] for hit in hits)
    proposal = out["propose_artifact"]
    assert proposal["status"] == "pending_approval" and proposal["exec"]["ok"]
    assert proposal["review"]["action"] == "apply_artifact"
    assert proposal["spec"]["citations"] and all((corpus / c["path"]).is_file() for c in proposal["spec"]["citations"])
    assert not (corpus / proposal["spec"]["file_path"]).exists()  # a proposal never writes
    assert _tree_digest(corpus) == before


async def _refusals(corpus: Path, private: Path) -> dict:
    """Forged, expired and replayed capabilities against one SDK server process."""
    key = private / "key"
    create_key(key)
    ledger = private / "ledger.sqlite"
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp.server_sdk", "--root", str(corpus)], cwd=str(ROOT),
                                   env={"PYTHONPATH": str(ROOT), "METANAVIT_APPROVAL_KEY_FILE": str(key), "METANAVIT_APPROVAL_LEDGER_FILE": str(ledger), **THREADS})
    out: dict = {"trees": {}, "texts": {}, "alive": {}}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def refused(label: str, tool: str, arguments: dict) -> None:
                result = await session.call_tool(tool, arguments)
                assert result.is_error, (label, _text(result))
                out["texts"][label] = _text(result)
                out["trees"][label] = _tree_digest(corpus)
                out["alive"][label] = bool(_blocks(await session.call_tool("list_dir", {"path": "."})))

            out["trees"]["start"] = _tree_digest(corpus)
            move = _payload(await session.call_tool("propose_move", {"src": "configs/run_047.yaml", "dst": "configs/archive/run_047.yaml"}))
            token = sign_review(move["review"], key, actor="sdk-test-operator")
            encoded, signature = token.split(".")
            forged = encoded + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
            await refused("forged", "apply_plan", {"plan_id": move["plan_id"], "approval_token": forged})
            expired = sign_review(move["review"], key, actor="sdk-test-operator", ttl_seconds=1, now=time.time() - 200)
            await refused("expired", "apply_plan", {"plan_id": move["plan_id"], "approval_token": expired})

            # A grant burnt by a failed apply (the SEARCH block is absent): the plan and tree still match the review.
            patch = _payload(await session.call_tool("propose_patch", {"path": "configs/run_047.yaml", "old": "absent: 1", "new": "absent: 2"}))
            burnt = sign_review(patch["review"], key, actor="sdk-test-operator")
            first = await session.call_tool("apply_patch", {"plan_id": patch["plan_id"], "approval_token": burnt})
            assert first.is_error and _tree_digest(corpus) == out["trees"]["start"]
            await refused("replay_burnt", "apply_patch", {"plan_id": patch["plan_id"], "approval_token": burnt})

            # A grant that applied: the replay finds the reviewed source gone.
            applied = await session.call_tool("apply_plan", {"plan_id": move["plan_id"], "approval_token": token})
            assert _payload(applied)["status"] == "applied"
            out["trees"]["applied"] = _tree_digest(corpus)
            await refused("replay_applied", "apply_plan", {"plan_id": move["plan_id"], "approval_token": token})
    with sqlite3.connect(ledger) as db:
        out["grants"] = db.execute("SELECT count(*) FROM grants").fetchone()[0]
    return out


def test_sdk_server_refuses_forged_expired_and_replayed_capabilities(tmp_path, tmp_path_factory):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "run_047.yaml").write_text("learning_rate: 3e-4\n")
    out = asyncio.run(_refusals(tmp_path, tmp_path_factory.mktemp("sdk-operator-private")))

    expected = {"forged": "signature", "expired": "expiry", "replay_burnt": "ledger", "replay_applied": "staleness"}
    for label, check in expected.items():
        assert f"refused by the {check} check" in out["texts"][label], out["texts"][label]
        assert out["alive"][label]  # the server answers the next call
    for label in ("forged", "expired", "replay_burnt"):
        assert out["trees"][label] == out["trees"]["start"]
    assert out["trees"]["applied"] != out["trees"]["start"]
    assert out["trees"]["replay_applied"] == out["trees"]["applied"]
    assert out["grants"] == 2  # the burnt patch grant and the applied move grant; forged and expired consume nothing
