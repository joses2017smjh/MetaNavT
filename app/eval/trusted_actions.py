"""Deterministic action-capability regression benchmark using temporary fixtures.

This measures enforcement at FilesystemTools through the JSON-RPC adapter,
not an LLM's resistance to prompt injection, an OS sandbox, or production safety.
Ten content/path variants per attack family are reported as variants, not as
independent attack strategies. Operator keys/ledgers are outside the corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import uuid

from app.artifacts.pipeline import ProposedArtifact
from app.artifacts.sandbox import ExecResult
from app.artifacts.spec import SpecCard
from app.mcp.approvals import ApprovalVerifier, create_key, sign_review
from app.mcp.filesystem import FilesystemTools
from app.mcp.server import MCPServer
from app.eval.m4_provenance import provenance

ATTACK_FAMILIES = (
    "missing_capability", "boolean_true", "boolean_false", "truthy_string_false",
    "numeric_approval", "malformed_token", "forged_signature", "different_plan",
    "different_session", "expired_token", "replay_move", "replay_patch",
    "changed_source_content", "replaced_source_inode", "mutated_move_payload",
    "source_symlink", "parent_symlink", "destination_exists", "mutated_patch_payload",
    "visualization_options_changed", "mutated_artifact_code", "library_key_reader",
)
BENIGN_FAMILIES = ("authorized_move", "authorized_patch", "authorized_artifact", "authorized_visualization_script", "unapproved_reads")


def _snapshot(root: Path, outside: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            rows.append([rel, "symlink", os.readlink(path)])
        elif path.is_file():
            rows.append([rel, "file", hashlib.sha256(path.read_bytes()).hexdigest()])
        elif path.is_dir():
            rows.append([rel, "directory"])
    rows.append(["outside_sentinel", hashlib.sha256(outside.read_bytes()).hexdigest()])
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def _request(tools, action, arguments):
    response = MCPServer(tools).handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": action, "arguments": arguments}})["result"]
    return response


def _artifact(tools, variant):
    plan_id = str(uuid.uuid4())
    prop = ProposedArtifact(plan_id, SpecCard(goal="temporary fixture", template="python-lib", citations=[{"path": f"inputs/source_{variant}.txt"}], file_path=f"generated/fixture_{variant}.py"), f"VALUE = {variant}\n", ExecResult(True, "", ""), "code")
    tools.artifacts[plan_id] = prop
    tools._register_review(plan_id)
    return plan_id


def action_case(family: str, variant: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="metanavit-capability-eval-") as temporary:
        base = Path(temporary)
        root = base / "corpus"
        private = base / "operator-private"
        root.mkdir(); private.mkdir(mode=0o700)
        (root / "inputs").mkdir()
        source = root / f"inputs/source_{variant}.txt"
        content = f"VALUE = {variant}\n" + ("robot α μ calibration\n" * (variant + 1))
        source.write_text(content)
        outside = base / "outside.txt"
        outside.write_text("outside sentinel must remain unchanged")
        key = private / "key"
        create_key(key)
        verifier = ApprovalVerifier(root, key, private / "ledger.sqlite")
        tools = FilesystemTools(root=root, approval_verifier=verifier)
        src, dst = str(source.relative_to(root)), f"outputs/moved_{variant}.txt"
        action = "apply_plan"
        plan_id = tools.propose_move(src, dst)["plan_id"]
        token = sign_review(tools.review_plan(plan_id), key, actor="benchmark-operator")
        args = {"plan_id": plan_id, "approval_token": token}
        is_attack = family in ATTACK_FAMILIES
        expected = None
        if family == "missing_capability":
            args.pop("approval_token")
        elif family in {"boolean_true", "boolean_false", "truthy_string_false", "numeric_approval"}:
            args.pop("approval_token")
            args["approved"] = {"boolean_true": True, "boolean_false": False, "truthy_string_false": "false", "numeric_approval": variant + 1}[family]
        elif family == "malformed_token":
            args["approval_token"] = ["", "broken", "!!!.bad", "a.b.c"][variant % 4]
        elif family == "forged_signature":
            encoded, signature = token.split(".")
            args["approval_token"] = encoded + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
        elif family == "different_plan":
            second = tools.propose_move(src, f"outputs/other_{variant}.txt")["plan_id"]
            args["plan_id"] = second
        elif family == "different_session":
            other = FilesystemTools(root=root, approval_verifier=verifier)
            args["plan_id"] = other.propose_move(src, dst)["plan_id"]
            tools = other
        elif family == "expired_token":
            args["approval_token"] = sign_review(tools.review_plan(plan_id), key, actor="benchmark-operator", now=time.time()-200, ttl_seconds=1)
        elif family == "replay_move":
            tools.apply_plan(plan_id, approval_token=token)
        elif family in {"replay_patch", "mutated_patch_payload", "authorized_patch"}:
            action = "apply_patch"
            plan_id = tools.propose_patch(src, f"VALUE = {variant}", f"VALUE = {variant + 100}")["plan_id"]
            args = {"plan_id": plan_id, "approval_token": sign_review(tools.review_plan(plan_id), key, actor="benchmark-operator")}
            if family == "replay_patch":
                tools.apply_patch(**args)
            elif family == "mutated_patch_payload":
                tools.patches[plan_id].new = "UNREVIEWED"
            else:
                expected = content.replace(f"VALUE = {variant}", f"VALUE = {variant + 100}", 1)
        elif family == "changed_source_content":
            source.write_text(content + "changed after review\n")
        elif family == "replaced_source_inode":
            replacement = root / "replacement"
            replacement.write_text(content)
            os.replace(replacement, source)
        elif family == "mutated_move_payload":
            tools.plans[plan_id].dst = f"outputs/unreviewed_{variant}.txt"
        elif family == "source_symlink":
            source.unlink(); source.symlink_to(outside)
        elif family == "parent_symlink":
            source.unlink(); (root / "inputs").rmdir()
            remote = base / "remote"
            remote.mkdir(); (remote / source.name).write_text(content)
            (root / "inputs").symlink_to(remote, target_is_directory=True)
        elif family == "destination_exists":
            (root / "outputs").mkdir()
            (root / dst).write_text("existing destination must survive")
        elif family in {"visualization_options_changed", "authorized_visualization_script"}:
            (root / "table.csv").write_text(f"group,value\narm,{variant+1}\nleg,{variant+2}\n")
            action = "apply_visualization"
            plan_id = tools.propose_visualization("table.csv", "compare average value by group", group_by="group", value="value")["plan_id"]
            review = tools.review_plan(plan_id, execute=False, chart_type="bar", backend="auto")
            args = {"plan_id": plan_id, "approval_token": sign_review(review, key, actor="benchmark-operator"), "execute": False, "chart_type": "bar", "backend": "auto"}
            if is_attack:
                args["chart_type"] = "line"
            else:
                expected = tools.visualizations[plan_id].script_path
        elif family in {"mutated_artifact_code", "authorized_artifact"}:
            action = "apply_artifact"
            plan_id = _artifact(tools, variant)
            args = {"plan_id": plan_id, "approval_token": sign_review(tools.review_plan(plan_id), key, actor="benchmark-operator")}
            if is_attack:
                tools.artifacts[plan_id].code += "UNREVIEWED = True\n"
            else:
                expected = tools.artifacts[plan_id].spec.file_path
        elif family == "library_key_reader":
            action = "exec_sandboxed"
            args = {"code": f"import numpy as np\nprint(np.fromfile({str(key)!r}, dtype='uint8'))"}
        elif family == "unapproved_reads":
            action = "read_file"
            args = {"path": src}
        elif family != "authorized_move":
            raise ValueError(f"unknown benchmark family {family}")
        before = _snapshot(root, outside)
        response = _request(tools, action, args)
        after = _snapshot(root, outside)
        if family == "library_key_reader":
            denied = not json.loads(response["content"][0]["text"])["ok"]
        else:
            denied = response["isError"]
        if is_attack:
            passed = denied and before == after
        else:
            result = json.loads(response["content"][0]["text"]) if not response["isError"] else {}
            if family == "authorized_move":
                passed = not denied and not source.exists() and (root / dst).read_text() == content
            elif family == "authorized_patch":
                passed = not denied and source.read_text() == expected
            elif family == "authorized_artifact":
                passed = not denied and (root / expected).read_text() == f"VALUE = {variant}\n"
            elif family == "authorized_visualization_script":
                passed = not denied and result["status"] == "applied" and (root / expected).is_file() and result["execution"]["ok"] is None
            else:
                passed = not denied and result["text"] == content and before == after
        return {"family": family, "variant": variant, "kind": "attack" if is_attack else "benign", "passed": bool(passed), "denied": bool(denied), "corpus_and_outside_unchanged": before == after, "fixture_content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def run(attack_variants=10, benign_variants=20):
    started = time.time()
    rows = [action_case(family, variant) for family in ATTACK_FAMILIES for variant in range(attack_variants)]
    rows += [action_case(family, variant) for family in BENIGN_FAMILIES for variant in range(benign_variants)]
    attacks = [row for row in rows if row["kind"] == "attack"]
    benign = [row for row in rows if row["kind"] == "benign"]
    return {**provenance(), "benchmark": "trusted_action_capabilities_v1", "scope": "deterministic temporary-fixture tool-boundary regression; no LLM prompt-injection or OS-isolation evaluation", "attack_families": len(ATTACK_FAMILIES), "variants_per_attack_family": attack_variants, "benign_families": len(BENIGN_FAMILIES), "variants_per_benign_family": benign_variants, "attack_cases": len(attacks), "attack_cases_denied_without_mutation": sum(row["passed"] for row in attacks), "benign_cases": len(benign), "benign_cases_succeeded": sum(row["passed"] for row in benign), "unexpected_mutation_cases": sum(not row["corpus_and_outside_unchanged"] for row in attacks), "failed_cases": [row for row in rows if not row["passed"]], "family_counts": dict(Counter(row["family"] for row in rows)), "elapsed_seconds": round(time.time()-started, 3), "python": platform.python_version(), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/results/trusted_actions.json"))
    parser.add_argument("--attack-variants", type=int, default=10)
    parser.add_argument("--benign-variants", type=int, default=20)
    args = parser.parse_args()
    if args.attack_variants < 1 or args.benign_variants < 1:
        parser.error("variant counts must be positive")
    result = run(args.attack_variants, args.benign_variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "family_counts"}}, indent=2))
    return bool(result["failed_cases"])


if __name__ == "__main__":
    raise SystemExit(main())
