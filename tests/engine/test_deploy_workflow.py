"""The approval-gated deploy (.github/workflows/deploy.yml and scripts/deploy_ecs.py), checked without AWS.

The workflow is read from its file (it runs only on GitHub); the deploy script runs here against a fake aws CLI that
records every call and answers like ECS, S3 and CloudFront would, so the order of the steps and what stops a deploy
are exercised, not just read. No call leaves this process.
"""

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deploy_ecs = _load("deploy_ecs", ROOT / "scripts" / "deploy_ecs.py")
smoke = _load("smoke_retrieve", ROOT / "scripts" / "smoke_retrieve.py")
hcl = _load("infra_hcl", Path(__file__).with_name("infra_hcl.py"))

DIGEST = "sha256:" + "ab" * 32
IMAGE = f"ghcr.io/joses2017smjh/metanavt@{DIGEST}"


def _workflow() -> tuple[dict, dict]:
    data = yaml.safe_load(WORKFLOW.read_text())
    return data, data.get("on", data.get(True))  # YAML 1.1 reads the bare key `on` as true


# ---- the workflow -------------------------------------------------------------------------------------------------

def test_it_deploys_after_a_successful_ci_run_on_main_or_by_hand_with_a_digest():
    workflow, on = _workflow()
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    assert on["workflow_run"] == {"workflows": [ci["name"]], "types": ["completed"], "branches": ["main"]}
    assert set(on["workflow_dispatch"]["inputs"]) == {"image_digest"}
    gate = workflow["jobs"]["gate"]["if"]
    for condition in ("github.event.workflow_run.conclusion == 'success'", "github.event.workflow_run.event == 'push'",
                      "github.event.workflow_run.head_branch == 'main'",
                      "github.event.workflow_run.head_repository.full_name == github.repository"):
        assert condition in gate  # never a pull request's run, nor a fork's branch named main
    assert workflow["concurrency"] == {"group": "deploy", "cancel-in-progress": False}  # never cancelled halfway


def test_production_waits_for_its_environment_and_staging_is_opt_in():
    jobs = _workflow()[0]["jobs"]
    assert set(jobs) == {"gate", "staging", "production"}
    production, staging = jobs["production"], jobs["staging"]
    assert production["environment"]["name"] == "production"  # the owner's approval: required reviewers on it
    assert staging["environment"]["name"] == "staging"
    assert production["needs"] == ["gate", "staging"] and staging["needs"] == "gate"
    assert staging["if"] == "${{ needs.gate.outputs.enabled == 'true' && needs.gate.outputs.staging == 'true' }}"
    assert production["if"] == ("${{ !cancelled() && needs.gate.outputs.enabled == 'true' && "
                                "(needs.staging.result == 'success' || needs.staging.result == 'skipped') }}")
    (variables,) = [s for s in jobs["gate"]["steps"] if s.get("id") == "variables"]
    assert variables["env"]["DEPLOY_STAGING"] == "${{ vars.DEPLOY_STAGING }}"
    assert 'if [ "$DEPLOY_STAGING" = "true" ]; then echo "staging=true"' in variables["run"]


def test_nothing_runs_until_the_owner_sets_the_repository_variables():
    jobs = _workflow()[0]["jobs"]
    (variables,) = [s for s in jobs["gate"]["steps"] if s.get("id") == "variables"]
    assert variables["env"]["AWS_REGION"] == "${{ vars.AWS_REGION }}"
    assert variables["env"]["AWS_DEPLOY_ROLE_ARN"] == "${{ vars.AWS_DEPLOY_ROLE_ARN }}"
    assert '[ -z "$AWS_REGION" ] || [ -z "$AWS_DEPLOY_ROLE_ARN" ]' in variables["run"]
    assert "::notice title=Deploy skipped::" in variables["run"] and "enabled=false" in variables["run"]
    for step in jobs["gate"]["steps"][1:]:
        assert step["if"] == "${{ steps.variables.outputs.enabled == 'true' }}"
    for name in ("staging", "production"):
        assert "needs.gate.outputs.enabled == 'true'" in jobs[name]["if"]


def test_only_the_deploy_jobs_get_an_oidc_token_and_they_assume_the_deploy_role():
    workflow, _ = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert jobs["gate"]["permissions"] == {"contents": "read", "packages": "read"}
    for name in ("staging", "production"):
        assert jobs[name]["permissions"] == {"contents": "read", "id-token": "write"}
        (credentials,) = [s for s in jobs[name]["steps"] if s.get("uses", "").startswith("aws-actions/configure-aws-credentials@")]
        assert credentials["with"]["role-to-assume"] == "${{ vars.AWS_DEPLOY_ROLE_ARN }}"
        assert credentials["with"]["aws-region"] == "${{ vars.AWS_REGION }}"
    assert "aws-access-key-id" not in WORKFLOW.read_text() and "AWS_SECRET_ACCESS_KEY" not in WORKFLOW.read_text()


def test_the_image_is_this_commits_ml_build_by_digest():
    jobs = _workflow()[0]["jobs"]
    (image,) = [s for s in jobs["gate"]["steps"] if s.get("id") == "image"]
    assert image["env"]["COMMIT"] == "${{ github.event.workflow_run.head_sha || github.sha }}"  # not main's tip
    assert '"$repository:sha-$COMMIT-ml"' in image["run"]
    assert "grep -Eq '^sha256:[0-9a-f]{64}$'" in image["run"]
    assert 'image="$repository@$digest"' in image["run"]
    assert '[ "$revision" != "$COMMIT" ]' in image["run"]  # the image says which commit it was built from
    assert jobs["gate"]["outputs"]["image_ref"] == "${{ steps.image.outputs.image }}"
    for name in ("staging", "production"):
        (deploy,) = [s for s in jobs[name]["steps"] if s.get("id") == "deploy"]
        assert deploy["env"]["IMAGE"] == "${{ needs.gate.outputs.image_ref }}"
        assert deploy["run"] == "python3 scripts/deploy_ecs.py --web-dir web"
    assert ":main-ml" not in WORKFLOW.read_text()  # a tag moves; only digests are deployed


def test_the_deploy_reads_exactly_the_variables_terraform_outputs_for_its_environment():
    outputs = hcl.read_dir(ROOT / "infra" / "modules" / "demo")
    exported = hcl.object_keys(hcl.attr(hcl.block(outputs, "output", "github_environment_variables"), "value"))
    jobs = _workflow()[0]["jobs"]
    for name in ("staging", "production"):
        (deploy,) = [s for s in jobs[name]["steps"] if s.get("id") == "deploy"]
        read = {key: value for key, value in deploy["env"].items() if key != "IMAGE"}
        assert read == {key: f"${{{{ vars.{key} }}}}" for key in exported}, name
    assert set(deploy_ecs.REQUIRED) == {"IMAGE", *exported}


# ---- the deploy script, against a fake aws CLI ------------------------------------------------------------------

ENV = {"IMAGE": IMAGE, "ECS_CLUSTER": "metanavit-production", "ECS_SERVICE": "metanavit-production-api",
       "API_TASK_FAMILY": "metanavit-production-api", "MIGRATE_TASK_FAMILY": "metanavit-production-migrate",
       "WEB_BUCKET": "metanavit-production-web-x", "CLOUDFRONT_DISTRIBUTION_ID": "E2EXAMPLE",
       "PUBLIC_URL": "https://d111.cloudfront.net/"}
SECRET_ARN = "arn:aws:secretsmanager:us-west-2:0:secret:metanavit-production/db-password-Ab12"


def _definition(family: str, container: str, revision: int) -> dict:
    return {"taskDefinitionArn": f"arn:aws:ecs:us-west-2:0:task-definition/{family}:{revision}", "family": family,
            "revision": revision, "status": "ACTIVE", "taskRoleArn": "role/task", "executionRoleArn": "role/execution",
            "networkMode": "awsvpc", "requiresCompatibilities": ["FARGATE"], "cpu": "1024", "memory": "4096",
            "runtimePlatform": {"operatingSystemFamily": "LINUX", "cpuArchitecture": "X86_64"},
            "requiresAttributes": [{"name": "x"}], "compatibilities": ["EC2", "FARGATE"], "registeredAt": "t",
            "containerDefinitions": [
                {"name": container, "image": "ghcr.io/joses2017smjh/metanavt@sha256:" + "00" * 32,
                 "environment": [{"name": "PUBLIC_DEMO", "value": "1"}],
                 "secrets": [{"name": "PGPASSWORD", "valueFrom": SECRET_ARN}]},
                *([{"name": "redis", "image": "public.ecr.aws/docker/library/redis:7.2-alpine@sha256:" + "11" * 32}]
                  if container == "api" else []),
            ]}


class FakeAws:
    """Answers like the aws CLI: describe-tasks walks `task_states`, describe-services after the update walks
    `rollouts`. Every call is recorded as (service, command, args)."""

    def __init__(self, exit_code=0, task_states=("PROVISIONING", "RUNNING", "STOPPED"), rollouts=("IN_PROGRESS", "COMPLETED"),
                 stored=True, rollback=False):
        self.calls, self.exit_code, self.stored, self.rollback = [], exit_code, stored, rollback
        self.task_states, self.rollouts = list(task_states), list(rollouts)
        self.definitions = {"metanavit-production-migrate": _definition("metanavit-production-migrate", "migrate", 7),
                            "metanavit-production-api": _definition("metanavit-production-api", "api", 12)}
        self.registered, self.updated_to = [], None

    def __call__(self, *args):
        service, command, rest = args[0], args[1], list(args[2:])
        self.calls.append((service, command, rest))

        def option(name):
            return rest[rest.index(name) + 1]

        if command == "describe-task-definition":
            return {"taskDefinition": copy.deepcopy(self.definitions[option("--task-definition")])}
        if command == "register-task-definition":
            spec = json.loads(option("--cli-input-json"))
            self.registered.append(spec)
            current = self.definitions[spec["family"]]
            new = {**spec, "revision": current["revision"] + 1,
                   "taskDefinitionArn": f"arn:aws:ecs:us-west-2:0:task-definition/{spec['family']}:{current['revision'] + 1}"}
            self.definitions[spec["family"]] = new
            return {"taskDefinition": new}
        if command == "describe-secret":
            return {"Name": "metanavit-production/db-password",
                    "VersionIdsToStages": {"v1": ["AWSCURRENT"]} if self.stored else {}}
        if command == "describe-services":
            if self.updated_to is None:
                return {"services": [{"status": "ACTIVE", "desiredCount": 1, "networkConfiguration": {
                    "awsvpcConfiguration": {"subnets": ["subnet-a"], "securityGroups": ["sg-t"], "assignPublicIp": "ENABLED"}}}]}
            state = self.rollouts.pop(0) if len(self.rollouts) > 1 else self.rollouts[0]
            primary = {"status": "PRIMARY", "taskDefinition": "arn:old" if self.rollback else self.updated_to,
                       "rolloutState": state, "rolloutStateReason": "r", "desiredCount": 1,
                       "runningCount": 1 if state == "COMPLETED" else 0}
            others = [] if state == "COMPLETED" else [{"status": "ACTIVE", "taskDefinition": "arn:old"}]
            return {"services": [{"status": "ACTIVE", "desiredCount": 1, "deployments": [primary, *others]}]}
        if command == "run-task":
            return {"tasks": [{"taskArn": "arn:aws:ecs:us-west-2:0:task/metanavit-production/abc123"}], "failures": []}
        if command == "describe-tasks":
            state = self.task_states.pop(0) if len(self.task_states) > 1 else self.task_states[0]
            return {"tasks": [{"lastStatus": state, "stoppedReason": "Essential container in task exited",
                               "containers": [{"name": "migrate", "exitCode": self.exit_code if state == "STOPPED" else None}]}]}
        if command == "update-service":
            self.updated_to = option("--task-definition")
            return {}
        if command in ("stop-task", "sync", "create-invalidation"):
            return {"Invalidation": {"Id": "I1"}} if command == "create-invalidation" else {}
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


def _deploy(tmp_path, aws, timeout=3000.0, smoke_exit=0, page=(200, "text/html")):
    web = tmp_path / "web"
    web.mkdir(exist_ok=True)
    (web / "index.html").write_text("<html></html>")
    smoked = []

    def run(cmd, **_):
        smoked.append(cmd)
        return type("Done", (), {"returncode": smoke_exit})()

    clock = Clock()
    deploy_ecs.deploy(deploy_ecs.environment(ENV), web, timeout, aws=aws, run=run, get=lambda url: page, clock=clock,
                      sleep=clock.sleep, started_by="deploy-1")
    return smoked


def test_a_deploy_migrates_and_indexes_first_then_moves_the_service_then_uploads_and_smokes(tmp_path):
    aws = FakeAws()
    smoked = _deploy(tmp_path, aws)
    commands = aws.commands()
    order = [commands.index(c) for c in ("run-task", "update-service", "sync", "create-invalidation")]
    assert order == sorted(order)
    assert commands.index("describe-secret") < commands.index("run-task")  # a missing value stops it before anything runs
    # the migrate task stopped (exit 0) before the API revision was registered
    assert commands.index("register-task-definition", commands.index("describe-tasks")) > max(
        i for i, c in enumerate(commands) if c == "describe-tasks")
    migrate, api = aws.registered
    assert [c["image"] for c in migrate["containerDefinitions"]] == [IMAGE]
    assert [c["image"] for c in api["containerDefinitions"]][0] == IMAGE
    assert api["containerDefinitions"][1]["image"].startswith("public.ecr.aws/docker/library/redis:7.2-alpine@sha256:")  # the sidecar untouched
    for spec in (migrate, api):  # ECS's bookkeeping is not sent back, nor tags (no ecs:TagResource)
        assert not {"taskDefinitionArn", "revision", "status", "requiresAttributes", "compatibilities", "registeredAt", "tags"} & set(spec)
    (_, _, run_args) = next(call for call in aws.calls if call[1] == "run-task")
    assert run_args[run_args.index("--task-definition") + 1].endswith("metanavit-production-migrate:8")
    assert json.loads(run_args[run_args.index("--network-configuration") + 1])["awsvpcConfiguration"]["subnets"] == ["subnet-a"]
    (_, _, update_args) = next(call for call in aws.calls if call[1] == "update-service")
    assert update_args[update_args.index("--task-definition") + 1].endswith("metanavit-production-api:13")
    (_, _, sync_args) = next(call for call in aws.calls if call[1] == "sync")
    assert sync_args[1] == "s3://metanavit-production-web-x" and "--delete" not in sync_args
    (command,) = smoked
    assert command[1].endswith("scripts/smoke_retrieve.py") and command[2:4] == ["--url", "https://d111.cloudfront.net"]
    assert {"--answer", "--require-reranker", "--require-bm25"} <= set(command)


def test_a_failed_migration_stops_the_deploy_before_the_service_moves(tmp_path):
    aws = FakeAws(exit_code=1)
    with pytest.raises(deploy_ecs.DeployError, match="migrate/index task exited 1"):
        _deploy(tmp_path, aws)
    assert "update-service" not in aws.commands() and "sync" not in aws.commands()


def test_a_secret_with_no_stored_value_stops_the_deploy_before_anything_runs(tmp_path):
    aws = FakeAws(stored=False)
    with pytest.raises(deploy_ecs.DeployError, match="has no stored value"):
        _deploy(tmp_path, aws)
    assert "run-task" not in aws.commands()


def test_a_migration_past_the_deadline_is_stopped(tmp_path):
    aws = FakeAws(task_states=("RUNNING",))
    with pytest.raises(deploy_ecs.DeployError, match="at the deadline"):
        _deploy(tmp_path, aws, timeout=60)
    assert "stop-task" in aws.commands() and "update-service" not in aws.commands()


@pytest.mark.parametrize("fake,message", [
    (dict(rollouts=("IN_PROGRESS", "FAILED")), "rollout failed"),
    (dict(rollback=True), "rolled back"),
    (dict(rollouts=("IN_PROGRESS",)), "not stable at the deadline"),
])
def test_a_rollout_that_fails_rolls_back_or_never_completes_fails_the_deploy(tmp_path, fake, message):
    aws = FakeAws(**fake)
    with pytest.raises(deploy_ecs.DeployError, match=message):
        _deploy(tmp_path, aws, timeout=600)
    assert "sync" not in aws.commands()


def test_the_smoke_or_the_page_failing_fails_the_deploy(tmp_path):
    with pytest.raises(deploy_ecs.DeployError, match="smoke against"):
        _deploy(tmp_path, FakeAws(), smoke_exit=1)
    with pytest.raises(deploy_ecs.DeployError, match="the web client is not served"):
        _deploy(tmp_path, FakeAws(), page=(403, "application/xml"))


def test_the_deploy_takes_an_image_only_by_digest_and_names_what_is_missing():
    with pytest.raises(deploy_ecs.DeployError, match="not a tag"):
        deploy_ecs.environment({**ENV, "IMAGE": "ghcr.io/joses2017smjh/metanavt:main-ml"})
    with pytest.raises(deploy_ecs.DeployError, match="not set: ECS_SERVICE, PUBLIC_URL"):
        deploy_ecs.environment({**ENV, "ECS_SERVICE": "", "PUBLIC_URL": " "})
    assert deploy_ecs.environment(ENV)["PUBLIC_URL"] == "https://d111.cloudfront.net"


def test_the_aws_cli_wrapper_raises_on_a_failed_call_and_parses_json():
    ok = deploy_ecs.Aws(run=lambda cmd, **_: type("D", (), {"returncode": 0, "stdout": '{"a": 1}', "stderr": ""})())
    assert ok("ecs", "describe-services") == {"a": 1}
    empty = deploy_ecs.Aws(run=lambda cmd, **_: type("D", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    assert empty("s3", "sync") == {}
    failed = deploy_ecs.Aws(run=lambda cmd, **_: type("D", (), {"returncode": 254, "stdout": "", "stderr": "AccessDenied"})())
    with pytest.raises(deploy_ecs.DeployError, match="AccessDenied"):
        failed("ecs", "update-service")


# ---- the answer smoke (scripts/smoke_retrieve.py --answer) --------------------------------------------------------

HIT = {"score": 0.5, "path": "configs/run_047.yaml", "chunk_id": "n1"}


@pytest.mark.parametrize("answer,passes", [
    ({"answer": "Run 47 uses 3e-4.", "reason": None, "citations": [{"chunk_id": "n1"}], "hits": [HIT]}, True),
    ({"answer": None, "reason": "answers off", "citations": [], "hits": [HIT]}, True),  # no key: still a pass
    ({"answer": None, "reason": "daily budget spent", "citations": [], "hits": [HIT]}, True),
    ({"answer": None, "reason": None, "citations": [], "hits": [HIT]}, False),
    ({"answer": "uncited", "reason": None, "citations": [], "hits": [HIT]}, False),
    ({"answer": None, "reason": "answers off", "citations": [], "hits": []}, False),
])
def test_the_answer_smoke_wants_an_answer_or_a_reason_never_that_claude_answered(monkeypatch, capsys, answer, passes):
    posted = []

    def post(url, payload):
        posted.append((url, payload))
        if url.endswith("/api/answer/"):
            return 200, answer
        return 200, {"hits": [HIT], "counts": {"bm25": 1}}

    monkeypatch.setattr(smoke, "wait_for_health", lambda base, timeout: {"status": "ok", "n_nodes": 1, "indexing": {}})
    monkeypatch.setattr(smoke, "_post", post)
    monkeypatch.setattr(sys, "argv", ["smoke_retrieve.py", "--answer"])
    assert smoke.main() == (0 if passes else 1)
    assert posted[1] == ("http://localhost:8000/api/answer/", {"question": "current learning rate for run 47", "k": 3})


def test_without_the_flag_the_smoke_posts_no_question_to_the_answer_route(monkeypatch):
    posted = []
    monkeypatch.setattr(smoke, "wait_for_health", lambda base, timeout: {"status": "ok", "n_nodes": 1, "indexing": {}})
    monkeypatch.setattr(smoke, "_post", lambda url, payload: posted.append(url) or (200, {"hits": [HIT], "counts": {"bm25": 1}}))
    monkeypatch.setattr(sys, "argv", ["smoke_retrieve.py"])
    assert smoke.main() == 0
    assert [re.sub(r"^http://localhost:8000", "", url) for url in posted] == ["/api/retrieve/"]
