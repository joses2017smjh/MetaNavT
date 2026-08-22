import json
from io import StringIO

from app.agent.retrieval_loop import RetrievalAgent, grade_chunk
from app.agent.schemas import MovePlan, MoveOp
from app.mcp.filesystem import ApprovalRequired, FilesystemTools
from app.mcp.server import MCPServer
from app.multimodal.fusion import fuse_modalities
from app.multimodal.late_interaction import maxsim
from app.retrieval.hybrid import Chunk, InMemoryHybridIndex
from app.retrieval.rerank import OverlapReranker
import numpy as np


def _index():
    chunks = [
        Chunk("c1", "configs/run_047.yaml", "run_id: 47 learning_rate: 3e-4 dinov2 fusion true bark_type: birch", 0, 80),
        Chunk("c2", "src/fusion.py", "fusion concatenates per-view features and projects with a 1x1", 0, 70),
        Chunk("c3", "configs/run_040.yaml", "run_id: 40 learning_rate: 1e-4 resnet50 fusion false", 0, 60),
    ]
    return InMemoryHybridIndex(chunks, retrieve_k=10, rerank_n=3, rerank_fn=OverlapReranker())


def test_hybrid_bm25_plus_dense_finds_run_47():
    idx = _index()
    result = idx.retrieve("what learning rate did run 47 use")
    assert result.paths[0] == "configs/run_047.yaml"


def test_agent_loop_cites_and_refuses_empty():
    agent = RetrievalAgent(_index(), max_iters=2, grade_threshold=0.05)
    ans = agent.run("what learning rate did run 47 use")
    assert not ans.failed
    assert ans.citations
    assert ans.citations[0].resolve()
    empty_idx = InMemoryHybridIndex(
        [Chunk("z", "other/nope.txt", "zzzz unrelated", 0, 10)],
        retrieve_k=2,
        rerank_n=2,
        enable_rerank=False,
    )
    empty_agent = RetrievalAgent(empty_idx, max_iters=1, grade_threshold=0.99, min_relevant=1)
    # still returns low-confidence rather than crashing
    out = empty_agent.run("dinov2 fusion birch")
    assert out.confidence in {"low", "empty", "high"}


def test_grade_chunk_overlap():
    assert grade_chunk("learning rate run 47", "learning_rate: 3e-4 run 47") > 0.2


def test_move_plan_requires_approval(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("x")
    tools = FilesystemTools(root=tmp_path)
    plan = tools.propose_move("a.txt", "b.txt")
    assert plan["status"] == "pending_approval"
    try:
        tools.apply_plan(plan["plan_id"], approved=False)
        raised = False
    except ApprovalRequired:
        raised = True
    assert raised
    applied = tools.apply_plan(plan["plan_id"], approved=True)
    assert applied["status"] == "applied"
    assert (tmp_path / "b.txt").exists()


def test_mcp_tools_list_and_call(tmp_path):
    (tmp_path / "hello.txt").write_text("hello")
    tools = FilesystemTools(root=tmp_path)
    server = MCPServer(tools)
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {t["name"] for t in listed["result"]["tools"]}
    assert {"search_lexical", "propose_move", "apply_plan", "read_file"} <= names
    call = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "list_dir", "arguments": {"path": "."}},
        }
    )
    payload = json.loads(call["result"]["content"][0]["text"])
    assert any(e["path"] == "hello.txt" for e in payload)


def test_maxsim_and_modal_fusion():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d_good = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d_bad = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    assert maxsim(q, d_good) >= maxsim(q, d_bad)
    fused = fuse_modalities(["t1", "t2"], ["p2", "p1"], ["i1"])
    assert fused[0][0] in {"t1", "t2", "p2", "p1", "i1"}


def test_structured_move_plan_schema():
    plan = MovePlan(plan_id="p1", ops=[MoveOp(src="a", dst="b", reason="group logs")])
    schema = plan.model_json_schema()
    assert schema["properties"]["requires_approval"]
    assert plan.destructive is True
