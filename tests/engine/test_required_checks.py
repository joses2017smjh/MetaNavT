"""scripts/required_checks.py: the checks main's branch protection requires, derived from the workflow files.

A required check that never reports blocks every merge, so the derivation must leave out path-filtered workflows and
jobs that skip pull requests, and must name matrix jobs as GitHub does. GitHub itself is not called here.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("required_checks", ROOT / "scripts" / "required_checks.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def test_the_repository_requires_every_always_run_job_and_nothing_path_filtered():
    names = rc.required_checks()
    assert names == sorted([
        "lint (ruff) + lock check",
        "eval (py3.11)",
        "eval (py3.12)",
        "unit + engine + api (app extra, no database)",
        "parity (API over Postgres vs in-memory bench)",
        "docker compose smoke",
        "image (build + smoke; push to ghcr.io on main)",
        "web client (npm ci, npm test, next build)",
    ])
    assert not any("ml smoke" in n for n in names)  # weekly + manual only


def test_the_cpu_model_image_job_is_not_required():
    """image-ml runs on a push to main and by hand, never on a pull request: required, it would block every merge."""
    ci = rc.yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    job = ci["jobs"]["image-ml"]
    assert rc.CONDITIONS[job["if"]] is False
    assert job["name"] not in rc.required_checks()


def _write(root: Path, name: str, text: str) -> None:
    (root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (root / ".github" / "workflows" / name).write_text(text)


def test_a_path_filter_or_another_target_branch_leaves_a_workflow_out(tmp_path):
    _write(tmp_path, "a.yml", "on:\n  pull_request:\n    paths: ['x/**']\njobs:\n  j: {runs-on: x, steps: []}\n")
    _write(tmp_path, "b.yml", "on:\n  pull_request:\n    branches: [dev]\njobs:\n  j: {runs-on: x, steps: []}\n")
    _write(tmp_path, "c.yml", "on: [push]\njobs:\n  j: {runs-on: x, steps: []}\n")
    _write(tmp_path, "d.yml", "on: pull_request\njobs:\n  kept: {runs-on: x, steps: []}\n")
    assert rc.required_checks(tmp_path) == ["kept"]


def test_matrix_jobs_are_named_as_github_names_them(tmp_path):
    _write(tmp_path, "m.yml", "on:\n  pull_request:\n    branches: [main]\njobs:\n"
           "  t:\n    name: test (py${{ matrix.py }})\n    strategy: {matrix: {py: ['3.11', '3.12']}}\n    runs-on: x\n    steps: []\n"
           "  u:\n    strategy: {matrix: {os: [a, b]}}\n    runs-on: x\n    steps: []\n")
    assert rc.required_checks(tmp_path) == ["test (py3.11)", "test (py3.12)", "u (a)", "u (b)"]


def test_an_unknown_job_condition_stops_the_script(tmp_path):
    _write(tmp_path, "w.yml", "on: pull_request\njobs:\n  j:\n    if: ${{ github.actor == 'x' }}\n    runs-on: x\n    steps: []\n")
    with pytest.raises(ValueError, match="CONDITIONS"):
        rc.required_checks(tmp_path)


def test_the_live_settings_are_compared_name_by_name_and_by_source():
    names = ["a", "b"]
    protection = {"required_status_checks": {"checks": [{"context": "a", "app_id": rc.GITHUB_ACTIONS_APP_ID},
                                                        {"context": "gone", "app_id": rc.GITHUB_ACTIONS_APP_ID}]}}
    assert rc.differences(names, protection, {"allow_auto_merge": False}) == [
        "required but no job reports it: gone", "not required: b", "auto-merge is off (allow_auto_merge)"]
    ok = {"required_status_checks": {"checks": [{"context": n, "app_id": rc.GITHUB_ACTIONS_APP_ID} for n in names]}}
    assert rc.differences(names, ok, {"allow_auto_merge": True}) == []
    spoofable = {"required_status_checks": {"checks": [{"context": "a", "app_id": None}, {"context": "b", "app_id": 1}]}}
    assert rc.differences(names, spoofable, {"allow_auto_merge": True}) == [
        "not tied to GitHub Actions: a", "not tied to GitHub Actions: b"]


def test_the_payload_requires_the_checks_from_github_actions_and_nothing_else():
    payload = rc.protection_payload(["a"])
    assert payload["required_status_checks"] == {"strict": False, "checks": [{"context": "a", "app_id": 15368}]}
    assert payload["required_pull_request_reviews"] is None and payload["restrictions"] is None
