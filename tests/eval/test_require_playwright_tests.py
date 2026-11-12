"""scripts/require_playwright_tests.py: the mobile workflow's guard that a Playwright test ran and passed, not skipped.

A test that calls test.skip() is reported as skipped and Playwright still exits 0, so the browser step alone cannot
tell whether connected.spec.ts reached the companion service. The reports below have the shape Playwright 1.56.1's
JSON reporter writes (checked against a probe run of that version: one spec entry per project, status and
expectedStatus per test, config.projects naming the projects).
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("require_playwright_tests", ROOT / "scripts" / "require_playwright_tests.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)

TITLE = "actual research-agent demo service returns inspected citations through HTTP"
SOCCER = "connected app searches MetaNavT and runs the actual soccer workflow and review boundary"
PROJECTS = ["iphone-390x844", "android-412x915"]


def _test(project: str, outcome: str) -> dict:
    expected, status, results = {
        "passed": ("passed", "expected", ["passed"]),
        "skipped": ("skipped", "skipped", ["skipped"]),
        "failed": ("passed", "unexpected", ["failed", "failed"]),
        "flaky": ("passed", "flaky", ["failed", "passed"]),
    }[outcome]
    return {"projectName": project, "expectedStatus": expected, "status": status,
            "results": [{"status": result, "retry": retry} for retry, result in enumerate(results)]}


def _report(agent: dict[str, str], soccer: str = "skipped", file: str = "connected.spec.ts") -> dict:
    specs = []
    for project in PROJECTS:
        if project in agent:
            specs.append({"title": TITLE, "file": file, "ok": True, "tests": [_test(project, agent[project])]})
        specs.append({"title": SOCCER, "file": file, "ok": True, "tests": [_test(project, soccer)]})
    companion = {"title": "companion.spec.ts", "file": "companion.spec.ts", "specs": [], "suites": [
        {"title": "group", "file": "companion.spec.ts",
         "specs": [{"title": "inner", "file": "companion.spec.ts", "tests": [_test(p, "passed") for p in PROJECTS]}]}]}
    return {"config": {"projects": [{"name": project} for project in PROJECTS]},
            "suites": [companion, {"title": file, "file": file, "specs": specs}], "errors": [], "stats": {}}


def _run(tmp_path, report, capsys, title=TITLE):
    path = tmp_path / "e2e-report.json"
    path.write_text(json.dumps(report))
    code = guard.main([str(path), "--spec", "connected.spec.ts", "--title", title])
    return code, capsys.readouterr().out


def test_passes_when_the_test_passed_in_every_project(tmp_path, capsys):
    code, out = _run(tmp_path, _report({project: "passed" for project in PROJECTS}), capsys)
    assert code == 0, out
    for project in PROJECTS:
        assert f"{TITLE} [{project}]: passed" in out
        assert f"{SOCCER} [{project}]: skipped (not required)" in out  # the soccer test needs the other repository


def test_fails_when_the_test_was_skipped(tmp_path, capsys):
    """What every run of mobile-app.yml reported before M33: no AGENT_E2E_URL, so the test skipped itself."""
    code, out = _run(tmp_path, _report({project: "skipped" for project in PROJECTS}), capsys)
    assert code == 1
    assert f"{TITLE} [{PROJECTS[0]}]: skipped" in out


@pytest.mark.parametrize("other", ["skipped", "failed", "missing"])
def test_fails_unless_every_project_passed(tmp_path, capsys, other):
    agent = {PROJECTS[0]: "passed"} if other == "missing" else {PROJECTS[0]: "passed", PROJECTS[1]: other}
    code, out = _run(tmp_path, _report(agent), capsys)
    assert code == 1
    assert PROJECTS[1] in out.split("FAIL", 1)[1]


def test_a_pass_on_retry_counts(tmp_path, capsys):
    code, out = _run(tmp_path, _report({PROJECTS[0]: "flaky", PROJECTS[1]: "passed"}), capsys)
    assert code == 0, out
    assert f"[{PROJECTS[0]}]: passed on retry" in out


def test_fails_when_no_test_has_the_title(tmp_path, capsys):
    code, out = _run(tmp_path, _report({project: "passed" for project in PROJECTS}), capsys, title="a renamed test")
    assert code == 1
    assert "no test" in out


def test_the_same_title_in_another_spec_file_does_not_count(tmp_path, capsys):
    code, _ = _run(tmp_path, _report({project: "passed" for project in PROJECTS}, file="other.spec.ts"), capsys)
    assert code == 1


def test_fails_without_a_report(tmp_path, capsys):
    code = guard.main([str(tmp_path / "missing.json"), "--spec", "connected.spec.ts", "--title", TITLE])
    assert code == 1
    assert "missing.json" in capsys.readouterr().out
