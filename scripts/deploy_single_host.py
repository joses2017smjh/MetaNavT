"""Deploy the live demo's single host: the deploy job of .github/workflows/deploy-single-host.yml (doc/deploy.md, "The
live demo: one host on the AWS Free plan"). Standard library only; AWS through the aws CLI on the runner, with the
credentials the job assumed through OIDC: infra/bootstrap's deploy role, whose Run Command permission covers only the
AWS-RunShellScript document on an instance tagged with this project and Target=single-host.

    IMAGE_DIGEST=sha256:<64 hex>  SINGLE_HOST_NAME=metanavit-single-host  PUBLIC_URL=https://...  SERVE_RERANKER=false
    python3 scripts/deploy_single_host.py [--timeout 3000]

SINGLE_HOST_NAME, PUBLIC_URL and SERVE_RERANKER are the outputs of infra/envs/single-host
(github_environment_variables), set once as variables of the GitHub environment production. In order, each step ending
the deploy if it fails:

  1. find the one running instance tagged Name=SINGLE_HOST_NAME and Target=single-host (ec2 describe-instances);
  2. run /opt/metanavit/deploy.sh <digest> on it through SSM Run Command and wait for the invocation to end, printing
     its output (deploy/single-host/deploy.sh: pull, stop the api, the migrations and the index step, the stack on the
     new image, its health checks; it starts the previous image again when one of those fails);
  3. smoke the public URL, through CloudFront: the web client's page, then scripts/smoke_retrieve.py (GET /health,
     POST /api/retrieve/ with hits, POST /api/answer/ with an answer or a reason), requiring a loaded reranker only
     when the host serves one (SERVE_RERANKER=true).

A rollback is the same run with the previous digest (the workflow's image_digest input).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_ecs import Aws, DeployError, _get, _group, smoke  # noqa: E402  the ECS deploy's CLI wrapper and smoke

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
REQUIRED = ("IMAGE_DIGEST", "SINGLE_HOST_NAME", "PUBLIC_URL", "SERVE_RERANKER")
DOCUMENT = "AWS-RunShellScript"
DEPLOY_SCRIPT = "/opt/metanavit/deploy.sh"
POLL_SECONDS = 15
DELIVERY_SECONDS = 600  # send-command --timeout-seconds: how long the instance may take to start the command
DONE = {"Success", "Cancelled", "TimedOut", "Failed"}  # get-command-invocation's final statuses
SMOKE_ARGS = ["--require-bm25", "--answer", "--timeout", "900", "--post-timeout", "120"]


def environment(env: dict) -> dict:
    """The deploy's inputs from the environment; every name required, the image by digest only."""
    missing = [name for name in REQUIRED if not env.get(name, "").strip()]
    if missing:
        raise DeployError(f"not set: {', '.join(missing)} (doc/deploy.md: the GitHub environment's variables, from "
                          "terraform output github_environment_variables of infra/envs/single-host)")
    values = {name: env[name].strip() for name in REQUIRED}
    if not DIGEST.fullmatch(values["IMAGE_DIGEST"]):
        raise DeployError(f"IMAGE_DIGEST must be sha256:<64 hex>, not a tag: {values['IMAGE_DIGEST']!r}")
    if values["SERVE_RERANKER"] not in {"true", "false"}:
        raise DeployError(f"SERVE_RERANKER must be true or false: {values['SERVE_RERANKER']!r}")
    values["PUBLIC_URL"] = values["PUBLIC_URL"].rstrip("/")
    return values


def find_instance(aws: Aws, name: str) -> str:
    """The id of the one running instance tagged Name=<name> and Target=single-host."""
    answer = aws("ec2", "describe-instances", "--filters", f"Name=tag:Name,Values={name}",
                 "Name=tag:Target,Values=single-host", "Name=instance-state-name,Values=running")
    ids = [instance["InstanceId"] for reservation in answer.get("Reservations") or []
           for instance in reservation.get("Instances") or []]
    if len(ids) != 1:
        raise DeployError(f"expected one running instance tagged Name={name} and Target=single-host, found {ids}")
    print(f"instance {ids[0]}")
    return ids[0]


def run_deploy_script(aws: Aws, instance: str, digest: str, deadline: float, clock: Callable[[], float],
                      sleep: Callable[[float], None]) -> None:
    """deploy.sh <digest> on the instance through Run Command, followed until it ends; exit 0 required."""
    execution = max(int(deadline - clock()), 60)
    parameters = {"commands": [f"{DEPLOY_SCRIPT} {digest}"], "executionTimeout": [str(execution)]}
    sent = aws("ssm", "send-command", "--document-name", DOCUMENT, "--instance-ids", instance,
               "--comment", f"deploy {digest[:19]}", "--timeout-seconds", str(DELIVERY_SECONDS),
               "--parameters", json.dumps(parameters))
    command = sent["Command"]["CommandId"]
    print(f"Run Command {command}: {DEPLOY_SCRIPT} {digest}")
    invocation: dict = {}
    status = None
    while True:
        listed = aws("ssm", "list-command-invocations", "--command-id", command, "--instance-id", instance)
        if listed.get("CommandInvocations"):  # registered: get-command-invocation can answer
            invocation = aws("ssm", "get-command-invocation", "--command-id", command, "--instance-id", instance)
            status = invocation.get("Status")
            if status in DONE:
                break
        if clock() > deadline:
            raise DeployError(f"Run Command {command} still running at the deadline: see it in Systems Manager, "
                              "Run Command, command history (the host's deploy.sh goes on)")
        sleep(POLL_SECONDS)
    output = (invocation.get("StandardOutputContent") or "").rstrip()
    errors = (invocation.get("StandardErrorContent") or "").rstrip()
    if output:
        print(output)
    if errors:
        print(errors, file=sys.stderr)
    if status != "Success":
        raise DeployError(f"deploy.sh on {instance}: {status} ({invocation.get('StatusDetails')}), exit "
                          f"{invocation.get('ResponseCode')}")
    print(f"deploy.sh on {instance}: Success")


def deploy(values: dict, timeout: float, *, aws: Aws, run: Callable = subprocess.run, get: Callable = _get,
           clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
    deadline = clock() + timeout
    _group("1. the instance")
    instance = find_instance(aws, values["SINGLE_HOST_NAME"])
    _group("2. deploy.sh through Run Command")
    run_deploy_script(aws, instance, values["IMAGE_DIGEST"], deadline, clock, sleep)
    _group("3. smoke")
    args = SMOKE_ARGS + (["--require-reranker"] if values["SERVE_RERANKER"] == "true" else [])
    smoke(values["PUBLIC_URL"], run=run, get=get, args=args)
    _group(None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--timeout", type=float, default=3000, help="seconds for the whole deploy before it gives up")
    args = parser.parse_args(argv)
    try:
        values = environment(dict(os.environ))
        deploy(values, args.timeout, aws=Aws())
    except DeployError as error:
        print(f"::error::{error}" if os.environ.get("GITHUB_ACTIONS") == "true" else f"DEPLOY FAILED: {error}", file=sys.stderr)
        return 1
    print(f"DEPLOYED {values['IMAGE_DIGEST']} to {values['PUBLIC_URL']}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"url={values['PUBLIC_URL']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
