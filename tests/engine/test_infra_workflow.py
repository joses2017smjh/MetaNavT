"""The Terraform checks in CI (.github/workflows/infra.yml), read from the workflow file: it runs only on GitHub.

Path-filtered on purpose: a required check that does not report on every pull request would block merges, so this
workflow stays out of main's required checks (scripts/required_checks.py) and runs when infra/ or the deploy
workflows change. Everything it runs is offline except the plan job, which waits for the AWS account to exist.
"""

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
_spec = importlib.util.spec_from_file_location("required_checks", ROOT / "scripts" / "required_checks.py")
required_checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(required_checks)

PATHS = ["infra/**", "deploy/single-host/**", ".github/workflows/infra.yml", ".github/workflows/deploy.yml"]
PINNED_ACTION = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+")


def _workflow(name: str) -> tuple[dict, dict]:
    data = yaml.safe_load((WORKFLOWS / name).read_text())
    return data, data.get("on", data.get(True))  # YAML 1.1 reads the bare key `on` as true


def _uses_lines(name: str) -> list[str]:
    return [line.split("uses:", 1)[1].strip() for line in (WORKFLOWS / name).read_text().splitlines()
            if line.strip().startswith(("uses:", "- uses:"))]


def _steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def test_it_runs_on_infra_changes_only_and_is_never_a_required_check():
    workflow, on = _workflow("infra.yml")
    for event in ("pull_request", "push"):
        assert on[event] == {"branches": ["main"], "paths": PATHS}, event
    assert workflow["permissions"] == {"contents": "read"}
    names = required_checks.required_checks()
    assert len(names) == 8
    for job_key, job in workflow["jobs"].items():
        assert job.get("name", job_key) not in names
    assert not required_checks.runs_on_every_pull_request(workflow)


def test_every_action_is_pinned_by_commit_sha_with_its_release_beside_it():
    for name in ("infra.yml", "deploy.yml"):
        if not (WORKFLOWS / name).exists():
            continue
        uses = _uses_lines(name)
        assert uses, name
        assert [u for u in uses if not PINNED_ACTION.fullmatch(u)] == [], name


def test_terraform_and_tflint_are_pinned_releases():
    workflow, _ = _workflow("infra.yml")
    version = workflow["env"]["TERRAFORM_VERSION"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert tuple(map(int, version.split("."))) >= (1, 11, 0)  # write-only arguments (infra/ requires >= 1.11.0)
    (setup,) = [s for s in _steps(workflow["jobs"]["check"]) if s.get("uses", "").startswith("terraform-linters/setup-tflint@")]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", setup["with"]["tflint_version"])
    assert re.fullmatch(r"[0-9a-f]{64}", setup["with"]["checksums"])
    config = (ROOT / "infra" / ".tflint.hcl").read_text()
    assert re.search(r'plugin "aws" \{[^}]*version = "\d+\.\d+\.\d+"', config)


def test_every_root_module_is_initialised_from_its_lock_file_and_validated():
    workflow, _ = _workflow("infra.yml")
    runs = [s.get("run", "") for s in _steps(workflow["jobs"]["check"])]
    (validate,) = [run for run in runs if "validate" in run]
    roots = sorted(p.parent.relative_to(ROOT).as_posix() for p in (ROOT / "infra").rglob(".terraform.lock.hcl")
                   if ".terraform" not in p.parent.parts)
    assert re.search(r"for root in ([^;]+); do", validate).group(1).split() == roots
    assert "init -backend=false -input=false -lockfile=readonly" in validate  # no credentials; the lock is not rewritten
    assert any(run == "terraform fmt -check -recursive -diff infra" for run in runs)


def test_tflint_reads_the_config_by_absolute_path():
    """`tflint --recursive` loads .tflint.hcl from each module's own directory unless --config is absolute: the AWS
    rules were silently off in every module below infra/ without it (found while writing this workflow)."""
    workflow, _ = _workflow("infra.yml")
    (step,) = [s for s in _steps(workflow["jobs"]["check"]) if "tflint --recursive" in s.get("run", "")]
    assert step["working-directory"] == "infra"
    assert step["run"].splitlines() == ['tflint --init --config "$PWD/.tflint.hcl"',
                                        'tflint --recursive --config "$PWD/.tflint.hcl" --format compact']


def test_the_plan_job_is_read_only_and_waits_for_the_aws_account():
    workflow, _ = _workflow("infra.yml")
    job = workflow["jobs"]["plan"]
    assert job["if"] == "${{ github.event_name == 'pull_request' }}" and job["needs"] == "check"
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    steps = _steps(job)
    gate = steps[0]
    assert "::notice title=Plan skipped::" in gate["run"] and "enabled=false" in gate["run"]
    assert all(s["if"] == "${{ steps.variables.outputs.enabled == 'true' }}" for s in steps[1:])
    (credentials,) = [s for s in steps if s.get("uses", "").startswith("aws-actions/configure-aws-credentials@")]
    assert credentials["with"]["role-to-assume"] == "${{ vars.AWS_PLAN_ROLE_ARN }}"
    (plan,) = [s for s in steps if "terraform plan" in s.get("run", "")]
    assert "terraform plan -input=false -lock=false -no-color" in plan["run"]  # the plan role writes nothing, not even a lock
    # the digest the state holds (the last apply's), not the one the service runs now: a deploy since the last apply
    # would otherwise show both task definitions replaced in every plan
    assert "if ! digest=$(terraform output -raw image_digest 2>/dev/null); then" in plan["run"]
    assert 'export TF_VAR_image_digest="$digest"' in plan["run"] and "describe-services" not in plan["run"]
    for name in ("infra.yml", "deploy.yml"):
        if (WORKFLOWS / name).exists():
            text = (WORKFLOWS / name).read_text()
            assert "terraform apply" not in text and "terraform destroy" not in text, name


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_every_run_script_of_the_two_workflows_parses_as_bash():
    for name in ("infra.yml", "deploy.yml"):
        if not (WORKFLOWS / name).exists():
            continue
        workflow, _ = _workflow(name)
        for key, job in workflow["jobs"].items():
            for step in _steps(job):
                if "run" in step:
                    assert "${{" not in step["run"], (name, key, step.get("name"))  # values reach scripts through env
                    done = subprocess.run(["bash", "-n"], input=step["run"], capture_output=True, text=True)
                    assert done.returncode == 0, (name, key, step.get("name"), done.stderr)
