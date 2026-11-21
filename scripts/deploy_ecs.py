"""Deploy one environment of the public demo on AWS: the steps of .github/workflows/deploy.yml's staging and production
jobs (doc/deploy.md). Standard library only; AWS through the aws CLI on the runner, with the credentials the job
assumed through OIDC (the deploy role of infra/bootstrap, whose permissions are exactly these steps).

    IMAGE=ghcr.io/<owner>/metanavt@sha256:<64 hex>  ECS_CLUSTER=...  ECS_SERVICE=...  API_TASK_FAMILY=...
    MIGRATE_TASK_FAMILY=...  WEB_BUCKET=...  CLOUDFRONT_DISTRIBUTION_ID=...  PUBLIC_URL=https://...
    python3 scripts/deploy_ecs.py --web-dir web [--timeout 3000]

The environment's values are the outputs of its Terraform root (github_environment_variables), set once as variables
of its GitHub environment. In order, each step ending the deploy if it fails:

  1. register a revision of the migrate/index task family with IMAGE (its newest revision, only the image changed),
     and check that every secret the task reads has a stored value (a secret without one stops a task's start);
  2. run that task once, with the service's own network configuration, until it stops, and require exit 0 from its
     container: the schema migrations, then the index step (docker-compose.yml's order);
  3. register a revision of the API family with IMAGE and move the service to it;
  4. wait until the service is stable: its one deployment rolled out, the desired count running. The deployment
     circuit breaker's rollback, or no rollout before the deadline, fails the deploy;
  5. upload the web client's export (taken from the same image by the workflow) to the web bucket, nothing deleted,
     and invalidate CloudFront's cache;
  6. smoke the public URL: the web client's page, then scripts/smoke_retrieve.py (GET /health, POST /api/retrieve/
     with hits, POST /api/answer/ with an answer or a reason: no assertion that Claude answered).

A rollback is the same run with the previous digest (the workflow's image_digest input).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
IMAGE = re.compile(r"[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}")  # a repository and a digest, never a tag
REQUIRED = ("IMAGE", "ECS_CLUSTER", "ECS_SERVICE", "API_TASK_FAMILY", "MIGRATE_TASK_FAMILY", "WEB_BUCKET",
            "CLOUDFRONT_DISTRIBUTION_ID", "PUBLIC_URL")
# What register-task-definition takes back from describe-task-definition's answer; the rest (ARN, revision, status,
# registration times, requiresAttributes, compatibilities) is ECS's own bookkeeping. Tags are not copied: tagging
# would need ecs:TagResource, which the deploy role does not have.
REGISTER_FIELDS = ("family", "taskRoleArn", "executionRoleArn", "networkMode", "containerDefinitions", "volumes",
                   "placementConstraints", "requiresCompatibilities", "cpu", "memory", "pidMode", "ipcMode",
                   "proxyConfiguration", "inferenceAccelerators", "ephemeralStorage", "runtimePlatform")
POLL_SECONDS = 15
SMOKE_ARGS = ["--require-bm25", "--require-reranker", "--answer", "--timeout", "900", "--post-timeout", "120"]


class DeployError(RuntimeError):
    pass


class Aws:
    """The aws CLI: arguments in, the parsed JSON answer out (an empty answer is {})."""

    def __init__(self, run: Callable = subprocess.run):
        self._run = run

    def __call__(self, *args: str) -> dict:
        done = self._run(["aws", *args, "--output", "json"], capture_output=True, text=True)
        if done.returncode != 0:
            raise DeployError(f"aws {' '.join(args[:2])} failed: {(done.stderr or done.stdout).strip()}")
        return json.loads(done.stdout) if done.stdout.strip() else {}


def environment(env: dict) -> dict:
    """The deploy's inputs from the environment; every name required, IMAGE by digest."""
    missing = [name for name in REQUIRED if not env.get(name, "").strip()]
    if missing:
        raise DeployError(f"not set: {', '.join(missing)} (doc/deploy.md: the GitHub environment's variables, from "
                          "terraform output github_environment_variables)")
    values = {name: env[name].strip() for name in REQUIRED}
    if not IMAGE.fullmatch(values["IMAGE"]):
        raise DeployError(f"IMAGE must be <repository>@sha256:<64 hex>, not a tag: {values['IMAGE']!r}")
    values["PUBLIC_URL"] = values["PUBLIC_URL"].rstrip("/")
    return values


def revision_with_image(task_definition: dict, container: str, image: str) -> dict:
    """register-task-definition's input: `task_definition` (describe-task-definition's) with only `container`'s image
    replaced."""
    spec = {key: copy.deepcopy(task_definition[key]) for key in REGISTER_FIELDS if task_definition.get(key) is not None}
    names = [c["name"] for c in spec.get("containerDefinitions", [])]
    if names.count(container) != 1:
        raise DeployError(f"{task_definition.get('family')}: expected one container named {container!r}, found {names}")
    for definition in spec["containerDefinitions"]:
        if definition["name"] == container:
            definition["image"] = image
    return spec


def register(aws: Aws, family: str, container: str, image: str) -> tuple[str, dict]:
    """A new revision of `family` (from its newest one) running `image` in `container`: (its ARN, its definition)."""
    current = aws("ecs", "describe-task-definition", "--task-definition", family)["taskDefinition"]
    spec = revision_with_image(current, container, image)
    registered = aws("ecs", "register-task-definition", "--cli-input-json", json.dumps(spec))["taskDefinition"]
    print(f"registered {registered['taskDefinitionArn']} ({container}: {image})")
    return registered["taskDefinitionArn"], registered


def check_secrets_stored(aws: Aws, task_definition: dict) -> None:
    """Every Secrets Manager secret the task reads has a current value (DescribeSecret: metadata, never the value)."""
    for container in task_definition.get("containerDefinitions", []):
        for secret in container.get("secrets", []):
            arn = secret["valueFrom"]
            if ":secretsmanager:" not in arn:
                continue
            described = aws("secretsmanager", "describe-secret", "--secret-id", arn)
            stages = described.get("VersionIdsToStages") or {}
            if not any("AWSCURRENT" in labels for labels in stages.values()):
                raise DeployError(f"{secret['name']}: the secret {described.get('Name', arn)} has no stored value "
                                  "(doc/deploy.md: store it, or leave its reference off in Terraform)")


def network_configuration(aws: Aws, cluster: str, service: str) -> dict:
    services = aws("ecs", "describe-services", "--cluster", cluster, "--services", service).get("services") or []
    if len(services) != 1 or services[0].get("status") != "ACTIVE":
        raise DeployError(f"service {service} not found (active) in cluster {cluster}")
    return services[0]["networkConfiguration"]


def run_migrate(aws: Aws, cluster: str, task_definition_arn: str, network: dict, started_by: str, deadline: float,
                clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
    """Run the migrate/index task once and require exit 0 from its container `migrate`."""
    started = aws("ecs", "run-task", "--cluster", cluster, "--task-definition", task_definition_arn,
                  "--launch-type", "FARGATE", "--count", "1", "--started-by", started_by[:36],
                  "--network-configuration", json.dumps(network))
    if started.get("failures") or not started.get("tasks"):
        raise DeployError(f"run-task failed: {started.get('failures')}")
    task_arn = started["tasks"][0]["taskArn"]
    print(f"migrate/index task {task_arn} started")
    while True:
        tasks = aws("ecs", "describe-tasks", "--cluster", cluster, "--tasks", task_arn).get("tasks") or []
        if len(tasks) != 1:
            raise DeployError(f"migrate/index task {task_arn} not found")
        task = tasks[0]
        if task.get("lastStatus") == "STOPPED":
            break
        if clock() > deadline:
            aws("ecs", "stop-task", "--cluster", cluster, "--task", task_arn, "--reason", "deploy deadline")
            raise DeployError(f"migrate/index task still {task.get('lastStatus')} at the deadline: stopped")
        sleep(POLL_SECONDS)
    containers = {c["name"]: c for c in task.get("containers", [])}
    exit_code = (containers.get("migrate") or {}).get("exitCode")
    task_id = task_arn.rsplit("/", 1)[-1]
    if exit_code != 0:
        raise DeployError(f"migrate/index task exited {exit_code} ({task.get('stoppedReason')}); its log: stream "
                          f"migrate/migrate/{task_id} of the cluster's log group")
    print(f"migrate/index task {task_id} exited 0")


def update_and_wait(aws: Aws, cluster: str, service: str, task_definition_arn: str, deadline: float,
                    clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
    """Move the service to `task_definition_arn` and wait for a completed rollout of it."""
    aws("ecs", "update-service", "--cluster", cluster, "--service", service, "--task-definition", task_definition_arn)
    print(f"service {service} -> {task_definition_arn}")
    while True:
        services = aws("ecs", "describe-services", "--cluster", cluster, "--services", service).get("services") or []
        if len(services) != 1:
            raise DeployError(f"service {service} not found in cluster {cluster}")
        current = services[0]
        deployments = current.get("deployments") or []
        primary = next((d for d in deployments if d.get("status") == "PRIMARY"), {})
        if primary.get("taskDefinition") != task_definition_arn:
            raise DeployError(f"the service's deployment is {primary.get('taskDefinition')}, not this revision: rolled back "
                              f"or replaced ({primary.get('rolloutStateReason')})")
        state = primary.get("rolloutState")
        if state == "FAILED":
            raise DeployError(f"rollout failed: {primary.get('rolloutStateReason')}")
        if (state == "COMPLETED" and len(deployments) == 1
                and primary.get("runningCount") == primary.get("desiredCount") == current.get("desiredCount")):
            print(f"service stable: {primary.get('runningCount')} task(s) of {task_definition_arn}")
            return
        if clock() > deadline:
            raise DeployError(f"service not stable at the deadline (rollout {state}, running {primary.get('runningCount')} "
                              f"of {primary.get('desiredCount')})")
        sleep(POLL_SECONDS)


def upload_web(aws: Aws, web_dir: Path, bucket: str, distribution: str) -> None:
    if not (web_dir / "index.html").is_file():
        raise DeployError(f"{web_dir}/index.html is missing: no web client export to upload")
    aws("s3", "sync", str(web_dir), f"s3://{bucket}", "--only-show-errors", "--no-progress")
    invalidation = aws("cloudfront", "create-invalidation", "--distribution-id", distribution, "--paths", "/*")
    print(f"web client uploaded to s3://{bucket}; invalidation {invalidation.get('Invalidation', {}).get('Id')}")


def _get(url: str) -> tuple[int, str]:
    """(status, content type) of a GET; an HTTP error status is an answer too."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, response.headers.get("content-type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("content-type", "")
    except urllib.error.URLError as error:
        raise DeployError(f"GET {url}: {error.reason}") from error


def smoke(public_url: str, run: Callable = subprocess.run, get: Callable = _get) -> None:
    status, content_type = get(f"{public_url}/")
    if status != 200 or not content_type.startswith("text/html"):
        raise DeployError(f"GET {public_url}/ -> {status} {content_type}: the web client is not served")
    print(f"GET {public_url}/ -> {status} {content_type}")
    done = run([sys.executable, str(ROOT / "scripts" / "smoke_retrieve.py"), "--url", public_url, *SMOKE_ARGS])
    if done.returncode != 0:
        raise DeployError(f"smoke against {public_url} failed (exit {done.returncode})")


def deploy(values: dict, web_dir: Path, timeout: float, *, aws: Aws, run: Callable = subprocess.run,
           get: Callable = _get, clock: Callable[[], float] = time.monotonic,
           sleep: Callable[[float], None] = time.sleep, started_by: str = "deploy") -> None:
    deadline = clock() + timeout
    cluster, service, image = values["ECS_CLUSTER"], values["ECS_SERVICE"], values["IMAGE"]

    _group("1. migrate/index revision, secrets")
    migrate_arn, migrate = register(aws, values["MIGRATE_TASK_FAMILY"], "migrate", image)
    check_secrets_stored(aws, migrate)
    _group("2. migrate/index task")
    run_migrate(aws, cluster, migrate_arn, network_configuration(aws, cluster, service), started_by, deadline, clock, sleep)
    _group("3. API revision")
    api_arn, api = register(aws, values["API_TASK_FAMILY"], "api", image)
    check_secrets_stored(aws, api)
    _group("4. service update, until stable")
    update_and_wait(aws, cluster, service, api_arn, deadline, clock, sleep)
    _group("5. web client")
    upload_web(aws, web_dir, values["WEB_BUCKET"], values["CLOUDFRONT_DISTRIBUTION_ID"])
    _group("6. smoke")
    smoke(values["PUBLIC_URL"], run=run, get=get)
    _group(None)


def _group(title: str | None) -> None:
    """GitHub's log groups (plain lines elsewhere)."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::endgroup::")
        if title:
            print(f"::group::{title}")
    elif title:
        print(f"== {title}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--web-dir", type=Path, default=Path("web"), help="the web client's export to upload")
    parser.add_argument("--timeout", type=float, default=3000, help="seconds for the whole deploy before it gives up")
    args = parser.parse_args(argv)
    try:
        values = environment(dict(os.environ))
        deploy(values, args.web_dir, args.timeout, aws=Aws(), started_by=f"deploy-{os.environ.get('GITHUB_RUN_ID', 'manual')}")
    except DeployError as error:
        print(f"::error::{error}" if os.environ.get("GITHUB_ACTIONS") == "true" else f"DEPLOY FAILED: {error}", file=sys.stderr)
        return 1
    print(f"DEPLOYED {values['IMAGE']} to {values['PUBLIC_URL']}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"url={values['PUBLIC_URL']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
