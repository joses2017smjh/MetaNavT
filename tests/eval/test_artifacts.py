"""Phase 11: research artifacts + production code (academia and industry)."""

import json

from app.agent.schemas import ArtifactSpec
from app.artifacts.manifest import collect_run_artifact, score_badges
from app.artifacts.paper2code import paper2code
from app.artifacts.patch import apply_search_replace, parse_search_replace
from app.artifacts.pipeline import ArtifactAgent
from app.artifacts.sandbox import ast_safe, run_sandboxed
from app.artifacts.spec import spec_from_query
from app.mcp.filesystem import ApprovalRequired, FilesystemTools
from app.mcp.server import MCPServer
from app.retrieval.hybrid import InMemoryHybridIndex
from app.retrieval.rerank import OverlapReranker
from app.retrieval.router import QueryRouter, RouteType
from app.retrieval.types import Chunk, RetrievalHit


def _chunks():
    return [
        Chunk(
            "c1",
            "configs/run_047.yaml",
            "run_id: 47 learning_rate: 3e-4 encoder: dinov2 fusion: true bark_type: birch",
            0,
            90,
        ),
        Chunk(
            "c2",
            "src/fusion.py",
            "fusion concatenates per-view features and projects with a 1x1",
            0,
            70,
        ),
        Chunk(
            "c3",
            "paper/draft_v2.md",
            "Stereo fusion does help. DINOv2 run 47 val RMSE 0.0521 learning_rate 3e-4.",
            0,
            90,
        ),
    ]


def _index():
    return InMemoryHybridIndex(_chunks(), retrieve_k=8, rerank_n=4, rerank_fn=OverlapReranker())


def test_sandbox_bans_os_and_runs_safe_code():
    blocked = run_sandboxed("import os\nos.system('echo hi')")
    assert not blocked.ok
    assert "banned" in blocked.error
    assert ast_safe("eval('1')")
    good = run_sandboxed("print(1 + 1)\nassert 1 + 1 == 2")
    assert good.ok
    assert "2" in good.stdout


def test_acm_bundle_for_run(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "slurm").mkdir()
    (tmp_path / "paper").mkdir()
    (tmp_path / "configs" / "run_047.yaml").write_text("run_id: 47 learning_rate: 3e-4")
    (tmp_path / "src" / "fusion.py").write_text("def project(x): return x  # run 47")
    (tmp_path / "slurm" / "run_047.sbatch").write_text("#SBATCH run 47")
    (tmp_path / "paper" / "draft_v2.md").write_text(
        "Stereo fusion does help on run 47 with DINOv2.\n"
    )
    bundle = collect_run_artifact(tmp_path, 47)
    roles = {f.role for f in bundle.files}
    assert {"config", "code", "env", "paper"} <= roles
    names = {b.name: b.earned for b in bundle.badges}
    assert names["available"] is True
    scored = score_badges(bundle, tests_passed=True)
    assert any(b.name == "functional" and b.earned for b in scored)


def test_paper2code_cites_and_executes():
    hits = [
        RetrievalHit(_chunks()[0], 1.0, 1),
        RetrievalHit(_chunks()[1], 0.8, 2),
        RetrievalHit(_chunks()[2], 0.7, 3),
    ]
    out = paper2code("reproduce run 47 from the paper", hits)
    assert out.plan["files"]
    assert "dinov2" in out.code.lower()
    assert not out.unsupported_claims
    ran = run_sandboxed(out.code)
    assert ran.ok, ran.error


def test_research_evidence_prefers_live_config_current_paper_and_source():
    chunks = [
        Chunk(
            "live",
            "configs/run_047.yaml",
            "run_id: 47 learning_rate: 3e-4 encoder: dinov2 fusion: true",
            0,
            70,
            mtime=20,
        ),
        Chunk(
            "old-config",
            "configs/archive/run_047_v1.yaml",
            "run_id: 47 learning_rate: 1e-5 encoder: dinov2 fusion: false",
            0,
            70,
            mtime=10,
        ),
        Chunk(
            "paper-current",
            "paper/draft_v2.md",
            "Run 47 uses DINOv2. Stereo fusion **does** help.",
            0,
            60,
            mtime=20,
        ),
        Chunk(
            "paper-old",
            "paper/draft_v1.md",
            "Run 47 uses DINOv2. Stereo fusion does not help.",
            0,
            60,
            mtime=10,
        ),
        Chunk(
            "source",
            "src/fusion.py",
            "fusion concatenates per-view features",
            0,
            40,
        ),
    ]
    idx = InMemoryHybridIndex(
        chunks,
        retrieve_k=10,
        rerank_n=5,
        rerank_fn=OverlapReranker(),
    )
    agent = ArtifactAgent(idx)
    evidence = agent.retrieve_research_evidence("reproduce run 47 from the paper")
    paths = [hit.chunk.path for hit in evidence]
    assert paths[0] == "configs/run_047.yaml"
    assert "paper/draft_v2.md" in paths
    assert "src/fusion.py" in paths
    assert "configs/archive/run_047_v1.yaml" not in paths
    assert "paper/draft_v1.md" not in paths
    prop = agent.produce("reproduce run 47 from the paper")
    cited = [c["path"] for c in prop.spec.citations]
    assert "configs/run_047.yaml" in cited
    assert "paper/draft_v2.md" in cited
    assert "src/fusion.py" in cited
    assert prop.exec_result.ok, prop.exec_result.error


def test_spec_required_citations_and_agent_hitl(tmp_path):
    idx = _index()
    spec = spec_from_query("write a test for fusion", idx.retrieve("fusion").hits)
    assert spec.citations
    assert spec.template == "pytest"
    agent = ArtifactAgent(idx)
    prop = agent.produce("write a test for the fusion module")
    assert prop.exec_result.ok, prop.exec_result.error
    assert prop.spec.ready() or prop.spec.citations
    tools = FilesystemTools(root=tmp_path, index=idx)
    pending = tools.propose_artifact("write a test for fusion")
    assert pending["status"] == "pending_approval"
    try:
        tools.apply_artifact(pending["plan_id"], approved=False)
        raised = False
    except ApprovalRequired:
        raised = True
    assert raised
    applied = tools.apply_artifact(pending["plan_id"], approved=True)
    assert applied["status"] == "applied"
    written = tmp_path / pending["spec"]["file_path"]
    assert written.exists()
    assert "citations:" in written.read_text()


def test_patch_search_replace_requires_approval(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "fusion.py").write_text("def project(features, fusion_on):\n    return features[0]\n")
    tools = FilesystemTools(root=tmp_path)
    pending = tools.propose_patch(
        "src/fusion.py",
        "    return features[0]",
        "    return features[0]  # reference view only",
    )
    assert "diff" in pending
    try:
        tools.apply_patch(pending["plan_id"], approved=False)
        raised = False
    except ApprovalRequired:
        raised = True
    assert raised
    tools.apply_patch(pending["plan_id"], approved=True)
    text = (src / "fusion.py").read_text()
    assert "reference view only" in text
    parsed = parse_search_replace(
        "path: src/fusion.py\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"
    )
    assert parsed[0].new == "new"
    assert apply_search_replace("abc old xyz", parsed[0]) == "abc new xyz"


def test_mcp_lists_artifact_tools(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    tools = FilesystemTools(root=tmp_path, index=_index())
    names = {t["name"] for t in tools.tool_specs()}
    assert {
        "collect_run_artifact",
        "propose_artifact",
        "apply_artifact",
        "propose_patch",
        "apply_patch",
        "exec_sandboxed",
    } <= names
    server = MCPServer(tools)
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    listed_names = {t["name"] for t in listed["result"]["tools"]}
    assert "propose_artifact" in listed_names
    execed = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "exec_sandboxed", "arguments": {"code": "print(3)"}},
        }
    )
    payload = json.loads(execed["result"]["content"][0]["text"])
    assert payload["ok"] is True


def test_artifact_spec_schema_requires_approval():
    spec = ArtifactSpec(
        plan_id="p1",
        goal="reproduce fusion",
        template="research-repro",
        file_path="artifacts/reproduce.py",
        citations=["src/fusion.py"],
    )
    schema = spec.model_json_schema()
    assert schema["properties"]["requires_approval"]
    assert spec.writes_tree is False


def test_router_does_not_steal_semantic_fusion_question():
    r = QueryRouter()
    assert r.route("what does the fusion module do").route == RouteType.SEMANTIC
