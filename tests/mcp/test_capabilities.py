"""Capability integrity, replay transactions and actual mutation boundaries."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import time

import pytest
from app.eval.trusted_actions import ATTACK_FAMILIES, BENIGN_FAMILIES, action_case
from app.mcp.approvals import ApprovalError, ApprovalVerifier, create_key, sign_review
from app.mcp.filesystem import ApprovalRequired, FilesystemTools


@pytest.mark.parametrize("family", ATTACK_FAMILIES + BENIGN_FAMILIES)
def test_tool_boundary_fixture(family):
    row = action_case(family, 2)
    assert row["passed"], row


def _setup(tmp_path):
    root, private = tmp_path / "corpus", tmp_path / "private"
    root.mkdir(); private.mkdir(mode=0o700)
    key = private / "key"
    create_key(key)
    verifier = ApprovalVerifier(root, key, private / "ledger.sqlite")
    (root / "source").write_text("original")
    tools = FilesystemTools(root=root, approval_verifier=verifier)
    plan = tools.propose_move("source", "destination")
    return tools, key, verifier, plan


def test_single_use_is_transactional_across_independent_verifiers(tmp_path):
    tools, key, verifier, plan = _setup(tmp_path)
    review = plan["review"]
    token = sign_review(review, key, actor="operator")
    other = ApprovalVerifier(tools.root, key, verifier.ledger)

    def consume(candidate):
        try:
            candidate.consume(token, review)
            return "consumed"
        except ApprovalError:
            return "refused"
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(consume, [verifier, other]))
    assert sorted(outcomes) == ["consumed", "refused"]
    with sqlite3.connect(verifier.ledger) as db:
        assert db.execute("SELECT count(*) FROM grants").fetchone()[0] == 1


def test_review_copy_does_not_mutate_immutable_manifest_and_audit_excludes_token(tmp_path):
    tools, key, verifier, plan = _setup(tmp_path)
    copied = tools.review_plan(plan["plan_id"])
    copied["payload"]["dst"] = "unreviewed"
    assert tools.review_plan(plan["plan_id"])["payload"]["dst"] == "destination"
    token = sign_review(plan["review"], key, actor="operator")
    assert tools.apply_plan(plan["plan_id"], approval_token=token)["status"] == "applied"
    with sqlite3.connect(verifier.ledger) as db:
        events = db.execute("SELECT action,result,detail FROM audit").fetchall()
    assert [(row[0], row[1]) for row in events] == [("apply_plan", "authorized"), ("apply_plan", "applied")]
    assert token not in json.dumps(events)


def test_no_configuration_never_creates_a_key_or_enables_boolean_bypass(tmp_path, monkeypatch):
    for name in ("METANAVIT_APPROVAL_KEY_FILE", "METANAVIT_APPROVAL_LEDGER_FILE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "source").write_text("original")
    tools = FilesystemTools(root=tmp_path)
    plan = tools.propose_move("source", "destination")
    with pytest.raises(ApprovalRequired):
        tools.apply_plan(plan["plan_id"], approved=True)
    with pytest.raises(ApprovalError):
        FilesystemTools(root=tmp_path, allow_apply=True)
    assert {path.name for path in tmp_path.iterdir()} == {"source"}


def test_key_inside_corpus_and_public_key_permissions_are_refused(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    key = root / "key"; create_key(key)
    with pytest.raises(ApprovalError):
        ApprovalVerifier(root, key, tmp_path / "ledger")
    key.rename(tmp_path / "key")
    key = tmp_path / "key"; key.chmod(0o644)
    with pytest.raises(ApprovalError):
        ApprovalVerifier(root, key, tmp_path / "ledger")


def test_expiry_boundary_and_future_issuance_are_checked(tmp_path):
    tools, key, verifier, plan = _setup(tmp_path)
    review = plan["review"]
    token = sign_review(review, key, actor="operator", ttl_seconds=1, now=100)
    with pytest.raises(ApprovalError):
        verifier.consume(token, review, now=101)
    future = sign_review(review, key, actor="operator", now=200)
    with pytest.raises(ApprovalError):
        verifier.consume(future, review, now=100)


def test_wrong_visualization_boolean_type_cannot_be_signed(tmp_path):
    tools, key, verifier, _ = _setup(tmp_path)
    (tools.root / "data.csv").write_text("group,value\na,1\nb,2\n")
    plan = tools.propose_visualization("data.csv", "average value by group")
    with pytest.raises(ValueError):
        tools.review_plan(plan["plan_id"], execute="false")


def test_directory_move_and_symlink_reads_are_refused(tmp_path):
    (tmp_path / "folder").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "folder", target_is_directory=True)
    tools = FilesystemTools(root=tmp_path)
    with pytest.raises(PermissionError):
        tools.propose_move("folder", "new-folder")
    with pytest.raises(PermissionError):
        tools.read_file("alias/file")


def test_generated_goal_and_citation_newlines_cannot_execute_library_key_reader(tmp_path):
    from app.artifacts.templates import render
    from app.artifacts.spec import SpecCard
    from app.artifacts.sandbox import run_sandboxed
    key = tmp_path / "private-key"
    key.write_text("must-never-leak")
    injected = f"safe goal\nimport numpy as np\nprint(np.fromfile({str(key)!r}, dtype='uint8'))"
    spec = SpecCard(goal=injected, template="python-lib", citations=[{"path": injected}])
    code = render(spec)
    result = run_sandboxed(code)
    assert result.ok and result.stdout == ""
    assert "\n# import numpy as np" in code


def test_visualization_writes_exact_operator_reviewed_bytes(tmp_path):
    tools, key, verifier, _ = _setup(tmp_path)
    (tools.root / "data.csv").write_text("group,value\na,1\nb,2\n")
    plan = tools.propose_visualization("data.csv", "average value by group")
    options = {"chart_type": "bar", "execute": False, "backend": "auto"}
    review = tools.review_plan(plan["plan_id"], **options)
    token = sign_review(review, key, actor="operator")
    applied = tools.apply_visualization(plan["plan_id"], approval_token=token, **options)
    assert (tools.root / applied["script_path"]).read_text() == review["output_content"]["script"]


def test_authorized_patch_preserves_unrelated_crlf_bytes(tmp_path):
    tools, key, verifier, _ = _setup(tmp_path)
    target = tools.root / "source"
    target.write_bytes(b"first\r\nsecond\r\n")
    plan = tools.propose_patch("source", "first", "changed")
    token = sign_review(plan["review"], key, actor="operator")
    tools.apply_patch(plan["plan_id"], approval_token=token)
    assert target.read_bytes() == b"changed\r\nsecond\r\n"
