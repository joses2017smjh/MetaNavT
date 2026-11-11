"""Capability integrity, replay transactions and actual mutation boundaries."""
import base64
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import hmac
import json
from pathlib import Path
import socket
import sqlite3
import time
import uuid

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from app.artifacts.pipeline import ProposedArtifact
from app.artifacts.sandbox import ExecResult
from app.artifacts.spec import SpecCard
from app.eval.trusted_actions import ATTACK_FAMILIES, BENIGN_FAMILIES, action_case
from app.eval.trusted_actions import main as write_benchmark
from app.mcp import approve
from app.mcp.approvals import REFUSAL_CHECKS, ApprovalError, ApprovalVerifier, canonical, create_key, review_hash, sign_review, verifier_from_env
from app.mcp.filesystem import ApprovalRequired, FilesystemTools, refusal_check

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("family", ATTACK_FAMILIES + BENIGN_FAMILIES)
def test_tool_boundary_fixture(family):
    row = action_case(family, 2)
    assert row["passed"], row


def _setup(tmp_path):
    root, private = tmp_path / "corpus", tmp_path / "private"
    root.mkdir(); private.mkdir(mode=0o700)
    key = private / "key"
    create_key(key, private / "key.pub")
    verifier = ApprovalVerifier(root, private / "key.pub", private / "ledger.sqlite")
    (root / "source").write_text("original")
    tools = FilesystemTools(root=root, approval_verifier=verifier)
    plan = tools.propose_move("source", "destination")
    return tools, key, verifier, plan


def test_single_use_is_transactional_across_independent_verifiers(tmp_path):
    tools, key, verifier, plan = _setup(tmp_path)
    review = plan["review"]
    token = sign_review(review, key, actor="operator")
    other = ApprovalVerifier(tools.root, key.parent / "key.pub", verifier.ledger)

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
    for name in ("METANAVIT_APPROVAL_KEY_FILE", "METANAVIT_APPROVAL_PUBLIC_KEY_FILE", "METANAVIT_APPROVAL_LEDGER_FILE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "source").write_text("original")
    tools = FilesystemTools(root=tmp_path)
    plan = tools.propose_move("source", "destination")
    with pytest.raises(ApprovalRequired):
        tools.apply_plan(plan["plan_id"], approved=True)
    with pytest.raises(ApprovalError):
        FilesystemTools(root=tmp_path, allow_apply=True)
    assert {path.name for path in tmp_path.iterdir()} == {"source"}


def test_key_files_inside_the_corpus_replaceable_or_readable_by_others_are_refused(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    create_key(root / "key", root / "key.pub")
    with pytest.raises(ApprovalError):  # inside the corpus root
        ApprovalVerifier(root, root / "key.pub", tmp_path / "ledger")
    key, public = tmp_path / "key", tmp_path / "key.pub"
    create_key(key, public)
    assert (key.stat().st_mode & 0o777) == 0o600 and not public.stat().st_mode & 0o022
    (tmp_path / "alias.pub").symlink_to(public)
    with pytest.raises(ApprovalError):
        ApprovalVerifier(root, tmp_path / "alias.pub", tmp_path / "ledger")
    public.chmod(0o664)  # whoever can replace the public key can mint tokens
    with pytest.raises(ApprovalError):
        ApprovalVerifier(root, public, tmp_path / "ledger")
    public.chmod(0o644)  # readable by anyone is fine: it is public
    ApprovalVerifier(root, public, tmp_path / "ledger")
    key.chmod(0o640)
    with pytest.raises(ApprovalError):  # the signer refuses a private key others can read
        sign_review({"schema": "metanavit-reviewed-action-v1"}, key, actor="operator")
    with pytest.raises(FileExistsError):  # never overwrites, never leaves half a pair
        create_key(tmp_path / "fresh", public)
    assert not (tmp_path / "fresh").exists()


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


def test_collect_run_artifact_leaves_out_files_reached_through_a_symlink(tmp_path):
    from app.mcp.filesystem import build_root_index
    root, outside = tmp_path / "corpus", tmp_path / "outside"
    (root / "configs").mkdir(parents=True)
    (outside / "dir").mkdir(parents=True)
    (root / "configs" / "run_047.yaml").write_text("run_id: 47 learning_rate: 3e-4\n")
    (outside / "secret.md").write_text("Run 47 outside claim zqxjfilemarker, long enough to be taken as a claim.\n")
    (outside / "dir" / "run_047.log").write_text("run 47 outside log zqxjdirmarker\n")
    (root / "paper_run_047.md").symlink_to(outside / "secret.md")
    (root / "linkdir").symlink_to(outside / "dir", target_is_directory=True)
    for index in (None, build_root_index(root)):  # both paths of the tool
        bundle = FilesystemTools(root=root, index=index).call("collect_run_artifact", {"run_id": "47"})
        assert [f["path"] for f in bundle["files"]] == ["configs/run_047.yaml"], bundle["files"]
        text = json.dumps(bundle)
        assert "zqxj" not in text and "paper_run_047" not in text and "linkdir" not in text


def test_list_dir_lists_a_symlink_without_its_target_metadata(tmp_path):
    root, outside = tmp_path / "corpus", tmp_path / "outside"
    (root / "sub").mkdir(parents=True)
    (outside / "dir").mkdir(parents=True)
    (root / "plain.txt").write_bytes(b"x" * 7)
    (outside / "secret.bin").write_bytes(b"s" * 1234)
    (root / "link.txt").symlink_to(outside / "secret.bin")
    (root / "linkdir").symlink_to(outside / "dir", target_is_directory=True)
    tools = FilesystemTools(root=root)
    listed = {entry["path"]: entry for entry in tools.call("list_dir", {"path": "."})}
    assert listed["plain.txt"] == {"path": "plain.txt", "is_dir": False, "size": 7}  # regular entries as before
    assert listed["sub"] == {"path": "sub", "is_dir": True, "size": None}
    assert listed["link.txt"] == {"path": "link.txt", "is_dir": False, "size": None, "symlink": True}
    assert listed["linkdir"] == {"path": "linkdir", "is_dir": False, "size": None, "symlink": True}
    for link in ("link.txt", "linkdir"):  # stat refuses the link, as the engineering reference says
        with pytest.raises(PermissionError):
            tools.stat([link])
    assert tools.stat(["plain.txt"])[0]["size"] == 7


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


@pytest.mark.parametrize("code", ["import math\nprint(math.pi)", "from math import pi\nprint(pi)", "print('a'.upper())",
                                  "print((1).real)", "print(print.__self__)", "import os as _\nprint(1)"])
def test_expression_tool_refuses_imports_and_attribute_expressions(tmp_path, code):
    result = FilesystemTools(root=tmp_path).exec_sandboxed(code)
    assert result["ok"] is False and result["stdout"] == ""
    assert result["error"] == "imports and attribute access are unavailable in the MCP expression tool"


@pytest.mark.parametrize("code", ['print(__import__("os"))', 'print(getattr("a", "upper")())'])
def test_expression_tool_builtins_offer_no_import_or_getattr(tmp_path, code):
    result = FilesystemTools(root=tmp_path).exec_sandboxed(code)
    assert result["ok"] is False and result["stdout"] == ""


def test_expression_tool_runs_plain_expressions(tmp_path):
    result = FilesystemTools(root=tmp_path).exec_sandboxed("print(sum(range(4)))")
    assert result["ok"] and result["stdout"].strip() == "6"


# The reviewer's probe (2026-11-11). On the earlier gate a class pattern bound a template
# string's format method, whose field lookups then read the server's environment. It
# prints only booleans, and the assertion below compares booleans only.
ENVIRONMENT_PROBE = (
    'match "{0.__self__.__loader__.find_spec.__func__.__globals__[sys].modules[os].environ}":\n'
    "    case str(format=f):\n"
    "        text = f(print)\n"
    '        print("PATH" in text, "HOME" in text)\n'
)


def test_expression_tool_refuses_the_environment_probe(tmp_path):
    result = FilesystemTools(root=tmp_path).exec_sandboxed(ENVIRONMENT_PROBE)
    refused, silent, ran = result.get("refused_by") == "expression_gate", result["stdout"] == "", result["ok"] is True
    assert (refused, silent, ran) == (True, True, False)


# One case per construct the allow-list refuses. Each must be refused by the gate itself,
# before anything runs: refused_by is set and nothing is printed.
REFUSED_CONSTRUCTS = {
    "match class keyword pattern": 'match "{0}":\n    case str(format=f):\n        print(f(1))',
    "match class positional pattern": "match 5:\n    case int(x):\n        print(x)",
    "match mapping pattern": 'match {"a": 1}:\n    case {"a": x}:\n        print(x)',
    "match sequence pattern": "match [1, 2]:\n    case [a, *rest]:\n        print(a)",
    "str.format": 'print("{0}".format(1))',
    "str.format_map": 'print("{x}".format_map({"x": 1}))',
    "getattr": 'print(getattr("a", "upper")())',
    "vars": "print(vars())",
    "dir": "print(dir())",
    "type": "print(type(1))",
    "globals": "print(globals())",
    "locals": "print(locals())",
    "format builtin": 'print(format(1, "d"))',
    "open": 'print(open("x"))',
    "__builtins__ read": 'print("len" in __builtins__)',
    "__builtins__ subscript": 'print(__builtins__["len"]([1]))',
    "dunder walrus target": "print((__builtins__ := 1))",
    "dunder assignment": "__x = 1\nprint(__x)",
    "dunder parameter": "def f(__x):\n    return 1\nprint(f(1))",
    "dunder function name": "def __init__():\n    return 1\nprint(__init__())",
    "dunder except name": "try:\n    int('x')\nexcept ValueError as __e:\n    print(1)",
    "dunder comprehension target": "print([1 for __x in range(2)])",
    "dunder keyword argument": "print(dict(__class__=1))",
    "decorator naming a builtin outside the list": "@vars\ndef f():\n    return 1",
    "lambda default naming a builtin outside the list": "f = lambda g=getattr: g\nprint(f)",
    "comprehension naming a builtin outside the list": "print([dir for _ in range(1)])",
    "walrus binding a builtin outside the list": "print((g := type))",
    "f-string format spec naming a builtin outside the list": 'print(f"{1:{vars}}")',
    "class definition": "class A:\n    pass\nprint(1)",
    "with": "x = 1\nwith x:\n    pass",
    "global": "def f():\n    global y\n    y = 1\nf()\nprint(y)",
    "nonlocal": "def f():\n    y = 1\n    def g():\n        nonlocal y\n        y = 2\n    g()\n    return y\nprint(f())",
    "yield": "def g():\n    yield 1\nprint(sum(g()))",
    "async def": "async def f():\n    return 1\nprint(1)",
    "annotated assignment": "x: int = 1\nprint(x)",
    "except*": "try:\n    pass\nexcept* ValueError:\n    pass",
    "nested too deeply to parse": "-" * 60000 + "1",
}


@pytest.mark.parametrize("code", list(REFUSED_CONSTRUCTS.values()), ids=list(REFUSED_CONSTRUCTS))
def test_expression_tool_refuses_constructs_outside_its_allow_list(tmp_path, code):
    result = FilesystemTools(root=tmp_path).exec_sandboxed(code)
    assert result["ok"] is False and result["stdout"] == "" and result.get("refused_by") == "expression_gate"
    assert result["error"] == REFUSAL_CHECKS["expression_gate"] and result["detail"]


# What the tool is for: plain Python over the allow-listed builtins.
ALLOWED_PROGRAMS = {
    "sum": ("print(sum(range(4)))", "6"),
    "arithmetic": ("x = 2\ny = x ** 10\nprint(y)", "1024"),
    "sorted with a lambda key": ("print(sorted([3, 1, 2], key=lambda v: -v))", "[3, 2, 1]"),
    "recursive function": ("def fact(n):\n    return 1 if n < 2 else n * fact(n - 1)\nprint(fact(5))", "120"),
    "try and except": ('try:\n    int("x")\nexcept ValueError:\n    print("bad")', "bad"),
    "raise and catch": ('try:\n    raise ValueError("no")\nexcept Exception as e:\n    print(str(e))', "no"),
    "f-string with a nested format spec": ('w = 3\nprint(f"{3.14159:.{w}f}")', "3.142"),
    "walrus in a while loop": ("n = 0\nwhile (n := n + 1) < 3:\n    pass\nprint(n)", "3"),
    "dict, zip and enumerate": ('d = dict(zip("ab", range(2)))\nprint(d, list(enumerate("xy")))', "{'a': 0, 'b': 1} [(0, 'x'), (1, 'y')]"),
    "comprehensions": ("print([i * i for i in range(4)], {i % 2 for i in range(4)}, {i: i for i in range(2)})", "[0, 1, 4, 9] {0, 1} {0: 0, 1: 1}"),
    "generator expression": ("print(sum(i * i for i in range(4)))", "14"),
    "slices and subscripts": ("xs = list(range(5))\nxs[0] = 9\nprint(xs[1:3], xs[-1], xs[0])", "[1, 2] 4 9"),
    "percent formatting": ('print("%d-%s" % (1, "a"))', "1-a"),
    "assert, isinstance and del": ("x = 1\nassert isinstance(x, int)\ndel x\nprint(all([1, 2]), any([]))", "True False"),
}


@pytest.mark.parametrize("code,expected", list(ALLOWED_PROGRAMS.values()), ids=list(ALLOWED_PROGRAMS))
def test_expression_tool_runs_its_language(tmp_path, code, expected):
    result = FilesystemTools(root=tmp_path).exec_sandboxed(code)
    assert result["ok"] and "refused_by" not in result, result
    assert result["stdout"].strip() == expected


def test_expression_tool_code_runs_under_the_sandbox_time_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("app.artifacts.sandbox.DEFAULT_TIMEOUT_S", 1)
    started = time.monotonic()
    result = FilesystemTools(root=tmp_path).exec_sandboxed("n = 0\nwhile n < 10**8:\n    n += 1\nprint(n)")
    assert result["timed_out"] is True and result["ok"] is False and result["stdout"] == "", result
    assert time.monotonic() - started < 10


def test_expression_builtins_are_a_subset_and_each_run_gets_its_own_copy():
    from app.artifacts.sandbox import SAFE_BUILTINS, run_sandboxed
    from app.mcp.filesystem import EXPRESSION_BUILTINS
    assert EXPRESSION_BUILTINS <= set(SAFE_BUILTINS)
    narrowed = run_sandboxed("print(len([1]))", builtins={"print"})
    assert not narrowed.ok and "NameError" in narrowed.error
    with pytest.raises(ValueError):
        run_sandboxed("print(1)", builtins={"print", "getattr"})
    saved = dict(SAFE_BUILTINS)
    try:  # the artifact sandbox lets code name __builtins__; changing it must not reach the next run
        assert run_sandboxed('__builtins__["len"] = None').ok
        later = run_sandboxed("print(len([1]))")
        assert later.ok and later.stdout.strip() == "1", later
    finally:
        SAFE_BUILTINS.clear()
        SAFE_BUILTINS.update(saved)



# The check that refuses each attack family, read from what the request raised.
# A tuple is indexed by variant modulo its length.
EXPECTED_REFUSAL = {
    "missing_capability": "capability_required",
    "boolean_true": "approval_argument", "boolean_false": "approval_argument",
    "truthy_string_false": "approval_argument", "numeric_approval": "approval_argument",
    "malformed_token": ("capability_required", "malformed", "signature", "malformed"),  # "", "broken", "!!!.bad", "a.b.c"
    "forged_signature": "signature",
    "different_plan": "binding", "different_session": "binding",
    "expired_token": "expiry",
    "replay_move": "staleness",  # the applied move took the reviewed source away
    "replay_patch": ("staleness", "ledger"),  # an applied grant; a grant burnt by a failed apply
    "changed_source_content": "staleness", "replaced_source_inode": "staleness", "mutated_move_payload": "staleness",
    "source_symlink": "symlink", "parent_symlink": "symlink",
    "destination_exists": "staleness", "mutated_patch_payload": "staleness",
    "visualization_options_changed": "binding",
    "mutated_artifact_code": "staleness",
    "library_key_reader": "expression_gate",
}


@pytest.mark.parametrize("family", ATTACK_FAMILIES)
def test_benchmark_rows_name_the_check_that_refused_them(family):
    assert set(EXPECTED_REFUSAL) == set(ATTACK_FAMILIES)
    expected = EXPECTED_REFUSAL[family]
    for variant in range(4):
        row = action_case(family, variant)
        check = expected if isinstance(expected, str) else expected[variant % len(expected)]
        assert row["passed"] and row["refused_by"] == check, row
        assert row["refusal_error"] is not None or family == "library_key_reader"


def _digest(root: Path) -> str:
    rows = [[str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"] for p in sorted(root.rglob("*"))]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def _grants(verifier) -> int:
    with sqlite3.connect(verifier.ledger) as db:
        return db.execute("SELECT count(*) FROM grants").fetchone()[0]


def _signed(tools, key, kind, move_plan, *, burnt=False):
    """(apply, plan_id, token) for one reviewed proposal; with `burnt`, the first apply is set up to fail after authorization."""
    if kind == "move":
        return tools.apply_plan, move_plan["plan_id"], sign_review(move_plan["review"], key, actor="operator")
    if kind == "patch":  # a burnt patch: its SEARCH block is absent from the file
        plan = tools.propose_patch("source", "absent text" if burnt else "original", "changed")
        return tools.apply_patch, plan["plan_id"], sign_review(plan["review"], key, actor="operator")
    if kind == "artifact":  # a burnt artifact: its sandbox run failed, which apply_artifact refuses after authorization
        plan_id = str(uuid.uuid4())
        spec = SpecCard(goal="fixture", template="python-lib", citations=[{"path": "source"}], file_path="generated/artifact.py")
        tools.artifacts[plan_id] = ProposedArtifact(plan_id, spec, "VALUE = 1\n", ExecResult(not burnt, "", "", "failed" if burnt else ""), "code")
        return tools.apply_artifact, plan_id, sign_review(tools._register_review(plan_id), key, actor="operator")
    (tools.root / "data.csv").write_text("group,value\na,1\nb,2\n")
    plan = tools.propose_visualization("data.csv", "average value by group")
    options = {"chart_type": "bar", "execute": False, "backend": "auto"}
    token = sign_review(tools.review_plan(plan["plan_id"], **options), key, actor="operator")
    return (lambda plan_id, approval_token: tools.apply_visualization(plan_id, approval_token=approval_token, **options)), plan["plan_id"], token


@pytest.mark.parametrize("kind,error", [("move", FileNotFoundError), ("patch", ApprovalError),
                                        ("artifact", FileExistsError), ("visualization", FileExistsError)])
def test_replay_of_an_applied_capability_is_refused_by_the_staleness_check(tmp_path, kind, error):
    tools, key, verifier, plan = _setup(tmp_path)
    apply, plan_id, token = _signed(tools, key, kind, plan)
    assert apply(plan_id, approval_token=token)["status"] == "applied"
    applied = _digest(tools.root)
    with pytest.raises(error) as refused:
        apply(plan_id, approval_token=token)
    assert refusal_check(refused.value) == "staleness"
    assert tools.audit_events[-1]["result"] == "rejected" and tools.audit_events[-1]["check"] == "staleness"
    assert _digest(tools.root) == applied and _grants(verifier) == 1


@pytest.mark.parametrize("kind", ["move", "patch", "artifact", "visualization"])
def test_replay_of_a_burnt_capability_is_refused_by_the_ledger(tmp_path, kind, monkeypatch):
    tools, key, verifier, plan = _setup(tmp_path)
    apply, plan_id, token = _signed(tools, key, kind, plan, burnt=True)

    def fail(*args, **kwargs):
        raise OSError("simulated failure after authorization")
    if kind == "move":
        monkeypatch.setattr("app.mcp.filesystem.move_file", fail)
    elif kind == "visualization":
        monkeypatch.setattr("app.mcp.filesystem.create_file", fail)
    before = _digest(tools.root)
    with pytest.raises(Exception) as failed:
        apply(plan_id, approval_token=token)
    assert refusal_check(failed.value) is None  # authorized and consumed; the action itself failed
    assert _digest(tools.root) == before and _grants(verifier) == 1
    with pytest.raises(ApprovalError) as refused:
        apply(plan_id, approval_token=token)
    assert refusal_check(refused.value) == "ledger"
    assert tools.audit_events[-1]["check"] == "ledger"
    assert _digest(tools.root) == before and _grants(verifier) == 1
    with sqlite3.connect(verifier.ledger) as db:
        assert db.execute("SELECT detail FROM audit ORDER BY id DESC LIMIT 1").fetchone()[0] == "ApprovalError (ledger)"


def test_benchmark_writer_records_the_refusing_check_and_no_host_user_or_path(tmp_path):
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    for output in outputs:
        assert write_benchmark(["--output", str(output), "--attack-variants", "2", "--benign-variants", "1"]) is False
    first, second = (json.loads(output.read_text()) for output in outputs)
    assert first["benchmark"] == "trusted_action_capabilities_v2" and first["refusal_checks"] == REFUSAL_CHECKS
    assert "python -m " in first["command"]  # the sanitized command (app/eval/provenance.py), checked for leaks below
    assert first["rows"] == second["rows"]  # deterministic rows
    attacks = [row for row in first["rows"] if row["kind"] == "attack"]
    assert len(attacks) == 2 * len(ATTACK_FAMILIES) and all(row["passed"] for row in first["rows"])
    assert all(row["refused_by"] in REFUSAL_CHECKS for row in attacks)
    assert first["attack_cases_by_refusing_check"]["ledger"] == 1  # replay_patch variant 1
    text = outputs[0].read_text()
    for needle in (str(tmp_path), str(ROOT), str(Path.home())):
        if len(needle) >= 5:  # a home of "/" would match every relative path
            assert needle not in text
    for name in (socket.gethostname(), getpass.getuser()):
        if len(name) >= 4 and name not in {"root", "runner", "user", "admin", "ubuntu", "nobody", "jovyan"}:
            assert name not in text



def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def test_the_verifier_cannot_mint_an_accepted_token_from_what_it_loads(tmp_path):
    tools, key, verifier, plan = _setup(tmp_path)
    review, public = plan["review"], key.parent / "key.pub"
    # Everything the verifier holds: an Ed25519 public key object (no private key, no shared secret).
    held = [value for value in vars(verifier).values() if not isinstance(value, (Path, str, int, float, type(None)))]
    assert len(held) == 1 and isinstance(held[0], Ed25519PublicKey)
    raw_public = verifier.public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    body = canonical({"review_sha256": review_hash(review), "grant_id": "ab" * 24, "issued_at": time.time(),
                      "expires_at": time.time() + 60, "actor": "verifier-side"})
    attempts = {
        "hmac keyed with the raw public key (the old construction)": hmac.new(raw_public, body, hashlib.sha256).digest(),
        "hmac keyed with the public key file": hmac.new(public.read_bytes(), body, hashlib.sha256).digest(),
        "Ed25519 with the public key bytes as a private seed": Ed25519PrivateKey.from_private_bytes(raw_public).sign(body),
        "Ed25519 with an unrelated key": Ed25519PrivateKey.generate().sign(body),
    }
    before = _digest(tools.root)
    for name, signature in attempts.items():
        with pytest.raises(ApprovalError) as refused:
            tools.apply_plan(plan["plan_id"], approval_token=_b64(body) + "." + _b64(signature))
        assert refusal_check(refused.value) == "signature", name
    with pytest.raises(ApprovalError):  # the signing path refuses the public key file
        sign_review(review, public, actor="verifier-side")
    assert _digest(tools.root) == before and _grants(verifier) == 0
    # Control: the operator's private key, which the verifier never loads, mints an accepted token.
    assert tools.apply_plan(plan["plan_id"], approval_token=sign_review(review, key, actor="operator"))["status"] == "applied"


def test_the_verifier_refuses_a_private_key_or_an_old_hmac_key(tmp_path, monkeypatch):
    root = tmp_path / "corpus"; root.mkdir()
    private = tmp_path / "private"; private.mkdir(mode=0o700)
    create_key(private / "key", private / "key.pub")
    old = private / "old.key"
    old.write_bytes(bytes(range(32))); old.chmod(0o600)  # the HMAC key format before Ed25519
    for wrong in (private / "key", old):
        with pytest.raises(ApprovalError):
            ApprovalVerifier(root, wrong, private / "ledger.sqlite")
    with pytest.raises(ApprovalError):  # the old key cannot sign either
        sign_review({"schema": "metanavit-reviewed-action-v1"}, old, actor="operator")
    monkeypatch.delenv("METANAVIT_APPROVAL_KEY_FILE", raising=False)
    monkeypatch.setenv("METANAVIT_APPROVAL_PUBLIC_KEY_FILE", str(private / "key"))  # the private key by mistake
    monkeypatch.setenv("METANAVIT_APPROVAL_LEDGER_FILE", str(private / "ledger.sqlite"))
    with pytest.raises(ApprovalError):
        verifier_from_env(root)
    monkeypatch.setenv("METANAVIT_APPROVAL_PUBLIC_KEY_FILE", str(private / "key.pub"))
    assert isinstance(verifier_from_env(root).public_key, Ed25519PublicKey)
    # The pre-Ed25519 setting fails loudly, even beside a valid new one, so the
    # message must say to unset it as well as where the public key goes.
    monkeypatch.setenv("METANAVIT_APPROVAL_KEY_FILE", str(old))
    with pytest.raises(ApprovalError, match="unset METANAVIT_APPROVAL_KEY_FILE.*METANAVIT_APPROVAL_PUBLIC_KEY_FILE"):
        verifier_from_env(root)


def test_operator_cli_creates_a_key_pair_and_signs_a_review_the_server_accepts(tmp_path, capsys):
    root = tmp_path / "corpus"; root.mkdir()
    private = tmp_path / "private"; private.mkdir(mode=0o700)
    (root / "source").write_text("original")
    assert approve.main(["init", "--key", str(private / "operator.key"), "--public-key", str(tmp_path / "operator.pub")]) == 0
    tools = FilesystemTools(root=root, approval_verifier=ApprovalVerifier(root, tmp_path / "operator.pub", private / "ledger.sqlite"))
    plan = tools.propose_move("source", "destination")
    (tmp_path / "review.json").write_text(json.dumps(plan["review"]))
    capsys.readouterr()
    assert approve.main(["sign", "--key", str(private / "operator.key"), "--review", str(tmp_path / "review.json"), "--actor", "operator"]) == 0
    token = capsys.readouterr().out.strip()
    assert tools.apply_plan(plan["plan_id"], approval_token=token)["status"] == "applied"
