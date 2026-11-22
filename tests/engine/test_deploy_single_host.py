"""The single host's deploy (.github/workflows/deploy-single-host.yml, scripts/deploy_single_host.py) and the
DEPLOY_TARGET switch shared with deploy.yml and infra.yml, checked without AWS.

The workflows are read from their files, and the shell of their gate steps runs here for each value of DEPLOY_TARGET;
the deploy script runs against a fake aws CLI that answers like EC2 and Systems Manager, so the order of the steps
and what stops a deploy are exercised, not just read. No call leaves this process.
"""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


single = _load("deploy_single_host", ROOT / "scripts" / "deploy_single_host.py")
hcl = _load("infra_hcl", Path(__file__).with_name("infra_hcl.py"))

DIGEST = "sha256:" + "ab" * 32


def _workflow(name: str) -> tuple[dict, dict]:
    data = yaml.safe_load((WORKFLOWS / name).read_text())
    return data, data.get("on", data.get(True))  # YAML 1.1 reads the bare key `on` as true


def _step(job: dict, step_id: str) -> dict:
    (step,) = [s for s in job["steps"] if s.get("id") == step_id]
    return step


# ---- the workflow -------------------------------------------------------------------------------------------------

def test_it_deploys_after_a_successful_ci_run_on_main_or_by_hand_with_a_digest_like_deploy_yml():
    workflow, on = _workflow("deploy-single-host.yml")
    ecs, ecs_on = _workflow("deploy.yml")
    assert on == ecs_on  # the same triggers: CI completed on main, or by hand with image_digest
    assert workflow["jobs"]["gate"]["if"] == ecs["jobs"]["gate"]["if"]  # never a pull request's run, nor a fork's main
    # never cancelled halfway, and in a group of its own: in deploy.yml's, the skipping run could cancel a pending deploy
    assert workflow["concurrency"] == {"group": "deploy-single-host", "cancel-in-progress": False}
    assert ecs["concurrency"]["group"] != workflow["concurrency"]["group"]
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"gate", "deploy"}


def test_the_digest_is_resolved_and_checked_with_deploy_yml_s_own_commands():
    gate = _workflow("deploy-single-host.yml")[0]["jobs"]["gate"]
    ecs_gate = _workflow("deploy.yml")[0]["jobs"]["gate"]
    mine, theirs = _step(gate, "image"), _step(ecs_gate, "image")
    assert mine["env"] == theirs["env"] and mine["if"] == theirs["if"]
    shared = [line for line in theirs["run"].splitlines() if not line.strip().startswith("echo \"image=")
              and "GITHUB_STEP_SUMMARY" not in line]
    lines = mine["run"].splitlines()
    assert [line for line in lines if line in shared] == shared  # every line of deploy.yml's step, in its order
    assert 'echo "digest=$digest" >> "$GITHUB_OUTPUT"' in lines
    assert gate["outputs"] == {"enabled": "${{ steps.variables.outputs.enabled }}",
                               "digest": "${{ steps.image.outputs.digest }}"}


def test_the_deploy_waits_for_the_production_environment_and_assumes_the_deploy_role():
    job = _workflow("deploy-single-host.yml")[0]["jobs"]["deploy"]
    assert job["environment"]["name"] == "production"  # the owner's approval: required reviewers on it
    assert job["needs"] == "gate" and job["if"] == "${{ needs.gate.outputs.enabled == 'true' }}"
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    (credentials,) = [s for s in job["steps"] if s.get("uses", "").startswith("aws-actions/configure-aws-credentials@")]
    assert credentials["with"]["role-to-assume"] == "${{ vars.AWS_DEPLOY_ROLE_ARN }}"
    assert credentials["with"]["aws-region"] == "${{ vars.AWS_REGION }}"
    deploy = _step(job, "deploy")
    assert deploy["run"] == "python3 scripts/deploy_single_host.py"
    assert deploy["env"]["IMAGE_DIGEST"] == "${{ needs.gate.outputs.digest }}"
    text = (WORKFLOWS / "deploy-single-host.yml").read_text()
    assert "aws-access-key-id" not in text and "AWS_SECRET_ACCESS_KEY" not in text
    assert ":main-ml" not in text  # a tag moves; only digests are deployed


def test_the_deploy_reads_exactly_the_variables_terraform_outputs_for_the_single_host():
    module = hcl.read_dir(ROOT / "infra" / "modules" / "single-host")
    exported = hcl.object_keys(hcl.attr(hcl.block(module, "output", "github_environment_variables"), "value"))
    deploy = _step(_workflow("deploy-single-host.yml")[0]["jobs"]["deploy"], "deploy")
    read = {key: value for key, value in deploy["env"].items() if key != "IMAGE_DIGEST"}
    assert read == {key: f"${{{{ vars.{key} }}}}" for key in exported}
    assert set(single.REQUIRED) == {"IMAGE_DIGEST", *exported}


# ---- the DEPLOY_TARGET switch: ecs, single-host, unset -------------------------------------------------------------

def _gate_outputs(workflow: str, job: str, step_id: str, env: dict) -> tuple[dict, str]:
    """Run one gate step's shell with `env`; (its GITHUB_OUTPUT as a dict, its stdout)."""
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    step = _step(_workflow(workflow)[0]["jobs"][job], step_id)
    return _run_step(step["run"], {name: env.get(name, "") for name in step["env"]})


def _run_step(script: str, env: dict) -> tuple[dict, str]:
    import tempfile

    with tempfile.NamedTemporaryFile("r", suffix=".out") as output:
        done = subprocess.run(["bash", "-e", "-c", script], env={"PATH": os.environ["PATH"], "GITHUB_OUTPUT": output.name, **env},
                              capture_output=True, text=True, timeout=30)
        assert done.returncode == 0, done.stderr
        return dict(line.split("=", 1) for line in Path(output.name).read_text().splitlines() if line), done.stdout


ACCOUNT = {"AWS_REGION": "us-west-2", "AWS_DEPLOY_ROLE_ARN": "arn:aws:iam::0:role/x", "AWS_PLAN_ROLE_ARN": "arn:aws:iam::0:role/y",
           "TF_STATE_BUCKET": "metanavit-tfstate-0"}


@pytest.mark.parametrize("target, ecs, single_host", [("ecs", "true", "false"), ("single-host", "false", "true"),
                                                      ("", "false", "false"), ("Single-Host", "false", "false")])
def test_deploy_target_picks_one_deploy_workflow_and_unset_picks_none(target, ecs, single_host):
    env = {**ACCOUNT, "DEPLOY_TARGET": target}
    ecs_out, ecs_log = _gate_outputs("deploy.yml", "gate", "variables", env)
    host_out, host_log = _gate_outputs("deploy-single-host.yml", "gate", "variables", env)
    assert (ecs_out["enabled"], host_out["enabled"]) == (ecs, single_host)
    for out, log in ((ecs_out, ecs_log), (host_out, host_log)):
        if out["enabled"] == "false":
            assert "::notice title=Deploy skipped::" in log  # skipped with a notice: the run stays green


@pytest.mark.parametrize("target", ["ecs", "single-host", ""])
def test_without_the_account_variables_neither_deploys_whatever_the_target(target):
    env = {"DEPLOY_TARGET": target}
    for workflow in ("deploy.yml", "deploy-single-host.yml"):
        out, log = _gate_outputs(workflow, "gate", "variables", env)
        assert out["enabled"] == "false" and "AWS_REGION and AWS_DEPLOY_ROLE_ARN are not set" in log


@pytest.mark.parametrize("target, root", [("ecs", "infra/envs/production"), ("single-host", "infra/envs/single-host"),
                                          ("", None)])
def test_the_plan_job_plans_the_targets_root_and_skips_when_it_is_unset(target, root):
    out, log = _gate_outputs("infra.yml", "plan", "variables", {**ACCOUNT, "DEPLOY_TARGET": target})
    if root is None:
        assert out == {"enabled": "false"} and "::notice title=Plan skipped::" in log
    else:
        assert out == {"enabled": "true", "root": root}
    plan = [s for s in _workflow("infra.yml")[0]["jobs"]["plan"]["steps"] if "terraform plan" in s.get("run", "")]
    assert plan[0]["working-directory"] == "${{ steps.variables.outputs.root }}"
    for directory in ("infra/envs/production", "infra/envs/single-host"):  # both roots hand the plan their last digest
        main = hcl.read_dir(ROOT / directory)
        assert hcl.attr(hcl.block(main, "output", "image_digest"), "value") == "var.image_digest"


# ---- the deploy script, against a fake aws CLI ----------------------------------------------------------------------

ENV = {"IMAGE_DIGEST": DIGEST, "SINGLE_HOST_NAME": "metanavit-single-host", "PUBLIC_URL": "https://d111.cloudfront.net/",
       "SERVE_RERANKER": "false"}


class FakeAws:
    """Answers like the aws CLI: describe-instances returns `instances`; the invocation walks `statuses` (None: not
    registered yet). Every call is recorded as (service, command, args)."""

    def __init__(self, instances=("i-0abc",), statuses=(None, "Pending", "InProgress", "Success"), output="DEPLOYED x",
                 response_code=0):
        self.calls, self.instances, self.statuses = [], list(instances), list(statuses)
        self.output, self.response_code = output, response_code

    def __call__(self, *args):
        service, command, rest = args[0], args[1], list(args[2:])
        self.calls.append((service, command, rest))
        if command == "describe-instances":
            return {"Reservations": [{"Instances": [{"InstanceId": i} for i in self.instances]}]}
        if command == "send-command":
            return {"Command": {"CommandId": "cmd-1"}}
        status = self.statuses[0]
        if command == "list-command-invocations":
            if status is None and len(self.statuses) > 1:
                self.statuses.pop(0)
                return {"CommandInvocations": []}
            return {"CommandInvocations": [{"Status": status}]}
        if command == "get-command-invocation":
            if len(self.statuses) > 1:
                self.statuses.pop(0)
            return {"Status": status, "StatusDetails": status, "ResponseCode": self.response_code if status in single.DONE else -1,
                    "StandardOutputContent": self.output, "StandardErrorContent": "" if status == "Success" else "ERROR: x"}
        raise AssertionError(f"unexpected aws call {args}")

    def commands(self) -> list[str]:
        return [command for _, command, _ in self.calls]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.now > 86400:  # a day of polling: the deploy would never give up
            raise AssertionError("still polling after a day: no deadline")


def _deploy(aws, env=None, timeout=3000.0, smoke_exit=0, page=(200, "text/html")):
    smoked = []

    def run(cmd, **_):
        smoked.append(cmd)
        return type("Done", (), {"returncode": smoke_exit})()

    clock = Clock()
    single.deploy(single.environment({**ENV, **(env or {})}), timeout, aws=aws, run=run, get=lambda url: page,
                  clock=clock, sleep=clock.sleep)
    return smoked


def test_a_deploy_finds_the_host_runs_deploy_sh_through_run_command_follows_it_then_smokes(capsys):
    aws = FakeAws()
    smoked = _deploy(aws)
    commands = aws.commands()
    assert commands[:2] == ["describe-instances", "send-command"] and commands[-1] == "get-command-invocation"
    (_, _, describe) = aws.calls[0]
    assert describe[describe.index("--filters") + 1:] == ["Name=tag:Name,Values=metanavit-single-host",
                                                          "Name=tag:Target,Values=single-host",
                                                          "Name=instance-state-name,Values=running"]
    (_, _, sent) = aws.calls[1]
    assert sent[sent.index("--document-name") + 1] == "AWS-RunShellScript"  # the one document the role may send
    assert sent[sent.index("--instance-ids") + 1] == "i-0abc"
    parameters = json.loads(sent[sent.index("--parameters") + 1])
    assert parameters["commands"] == [f"/opt/metanavit/deploy.sh {DIGEST}"]
    assert 60 <= int(parameters["executionTimeout"][0]) <= 3000
    assert "DEPLOYED x" in capsys.readouterr().out  # the host's own log, in the workflow's
    (command,) = smoked
    assert command[1].endswith("scripts/smoke_retrieve.py") and command[2:4] == ["--url", "https://d111.cloudfront.net"]
    assert {"--answer", "--require-bm25"} <= set(command) and "--require-reranker" not in command


def test_the_smoke_requires_the_reranker_only_when_the_host_serves_it():
    (command,) = _deploy(FakeAws(), env={"SERVE_RERANKER": "true"})
    assert "--require-reranker" in command


@pytest.mark.parametrize("status", ["Failed", "TimedOut", "Cancelled"])
def test_a_failed_run_command_fails_the_deploy_before_the_smoke(status, capsys):
    aws = FakeAws(statuses=("InProgress", status), response_code=1)
    with pytest.raises(single.DeployError, match=f"deploy.sh on i-0abc: {status}"):
        _deploy(aws)
    assert "ERROR: x" in capsys.readouterr().err


def test_a_command_still_running_at_the_deadline_fails_the_deploy():
    aws = FakeAws(statuses=("InProgress",))
    with pytest.raises(single.DeployError, match="still running at the deadline"):
        _deploy(aws, timeout=120)


@pytest.mark.parametrize("instances", [(), ("i-1", "i-2")])
def test_exactly_one_running_instance_is_required(instances):
    aws = FakeAws(instances=instances)
    with pytest.raises(single.DeployError, match="expected one running instance"):
        _deploy(aws)
    assert aws.commands() == ["describe-instances"]


def test_the_page_or_the_smoke_failing_fails_the_deploy():
    with pytest.raises(single.DeployError, match="smoke against"):
        _deploy(FakeAws(), smoke_exit=1)
    with pytest.raises(single.DeployError, match="the web client is not served"):
        _deploy(FakeAws(), page=(502, "text/html"))


def test_the_inputs_are_checked_before_any_call():
    with pytest.raises(single.DeployError, match="not a tag"):
        single.environment({**ENV, "IMAGE_DIGEST": "main-ml"})
    with pytest.raises(single.DeployError, match="not set: SINGLE_HOST_NAME, SERVE_RERANKER"):
        single.environment({**ENV, "SINGLE_HOST_NAME": "", "SERVE_RERANKER": " "})
    with pytest.raises(single.DeployError, match="SERVE_RERANKER must be true or false"):
        single.environment({**ENV, "SERVE_RERANKER": "yes"})
    assert single.environment(ENV)["PUBLIC_URL"] == "https://d111.cloudfront.net"
