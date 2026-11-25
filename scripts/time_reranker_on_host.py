"""Time one reranker on the live demo's single host: the serving bar's measurement (M43). Standard library only; AWS
through the aws CLI with the caller's credentials, like scripts/deploy_single_host.py.

    SINGLE_HOST_NAME=metanavit-single-host AWS_REGION=us-west-2 \\
    python3 scripts/time_reranker_on_host.py --model cross-encoder/ms-marco-MiniLM-L6-v2 --stop-api
        [--depths 20] [--max-length 512] [--repeats 30] [--memory <the api's limit>] [--beside-api] [--print-remote-script]

Nothing in the repository runs this: it sends a command to a live host, so the owner does, on purpose. What it does:

  1. finds the one running instance tagged Name=SINGLE_HOST_NAME and Target=single-host (ec2 describe-instances);
  2. sends one AWS-RunShellScript command through SSM Run Command (the document the deploy role may run) and follows it.
     On the host that script starts a one-off container of the image the stack is running (/opt/metanavit/image.env)
     with the stack's model volume, 2 CPUs and the api container's memory limit (app.env's API_MEMORY, no swap), and
     pipes scripts/rerank_timing_payload.py into `python -` there: import main, the embedder, then the reranker as the
     API loads it, then `repeats` rerank calls per depth, each over passages cut at max_length. That is the API's own
     process with this reranker in it, so its peak is what the api container would need;
  3. prints the result: p50 and p95 per rerank call at each depth, the process's peak resident set and the container's
     own peak, and whether the container was killed for memory (exit 137: the model does not fit that limit).

Two ways to run beside a live demo, and one must be chosen:
  --stop-api     stop the api container for the measurement and start the stack again afterwards (deploy.sh --start),
                 also when the measurement fails. The demo is down for those minutes. The host has 2 GiB: only this
                 way does the one-off container get the memory the api would have, so only this is the serving bar.
  --beside-api   leave the api running. No downtime, but the two containers and the database share 2 GiB, so the host
                 may swap and the latencies are then an upper bound; an indication, not the bar.

A model is named by its Hugging Face id: the first run downloads it into the stack's model volume (the image carries
no weights). A distilled student is not published, so time the public model it was trained from: same architecture,
same shapes, same cost (cross-encoder/ms-marco-MiniLM-L6-v2 for student-minilm-l6, -L12-v2 for student-minilm-l12).
--model-dir <directory on the host> mounts a checkpoint the owner copied there instead.

--print-remote-script prints the script the host would run and exits without calling AWS.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_ecs import Aws, DeployError  # noqa: E402  the deploy scripts' aws CLI wrapper
from deploy_single_host import DOCUMENT, DONE, find_instance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "scripts" / "rerank_timing_payload.py"
MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")  # a Hugging Face id: nothing a shell could read otherwise
HOST_PATH = re.compile(r"/[A-Za-z0-9._/-]+")
MEMORY = re.compile(r"[1-9][0-9]*m")
CONTAINER = "metanavit-rerank-timing"
POLL_SECONDS = 10
DELIVERY_SECONDS = 600

# The host's side. __NAMES__ are filled in by remote_script(); every value is validated before it gets here.
REMOTE = r"""#!/usr/bin/env bash
# Sent by scripts/time_reranker_on_host.py: time one reranker in a one-off container of the deployed image.
set -euo pipefail
HOME_DIR=/opt/metanavit
RUN_DIR=/run/metanavit
setting() { sed -n "s/^$1=//p" "$2" | tail -n 1; }
IMAGE=$(setting METANAVIT_IMAGE "$HOME_DIR/image.env")
[ -n "$IMAGE" ] || { echo "no image deployed ($HOME_DIR/image.env)" >&2; exit 1; }
MEMORY=__MEMORY__
[ -n "$MEMORY" ] || MEMORY=$(setting API_MEMORY "$HOME_DIR/app.env")
[ -n "$MEMORY" ] || { echo "API_MEMORY is not in $HOME_DIR/app.env; pass --memory" >&2; exit 1; }
mkdir -p "$RUN_DIR"
echo "image $IMAGE"
echo "limits: 2 CPUs, memory $MEMORY (no swap for the container)"
echo "host: $(nproc) CPUs, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
free -m | sed 's/^/host memory (MiB): /'
api_container() { docker ps -q --filter label=com.docker.compose.project=metanavit --filter label=com.docker.compose.service=api; }
STOPPED=""
restore() {
  status=$?
  docker rm -f __CONTAINER__ > /dev/null 2>&1 || true
  if [ -n "$STOPPED" ]; then
    echo "starting the stack again (deploy.sh --start)"
    "$HOME_DIR/deploy.sh" --start || echo "deploy.sh --start FAILED: start the stack by hand" >&2
  fi
  exit $status
}
trap restore EXIT
if [ "__STOP_API__" = "1" ]; then
  # not while a deploy runs: its lock, released before the measurement (deploy.sh --start takes it again at the end)
  exec 9> "$RUN_DIR/deploy.lock"
  flock --wait 600 9 || { echo "a deploy still holds $RUN_DIR/deploy.lock" >&2; exit 1; }
  API=$(api_container)
  if [ -n "$API" ]; then
    echo "stopping the api container for the measurement"
    docker stop $API > /dev/null
    STOPPED=1
  fi
  flock --unlock 9
else
  echo "the api keeps running beside the measurement: the host may swap, so the latencies are an upper bound"
fi
cat > "$RUN_DIR/rerank_timing_payload.py" <<'PAYLOAD_B64'
__PAYLOAD_B64__
PAYLOAD_B64
base64 -d "$RUN_DIR/rerank_timing_payload.py" > "$RUN_DIR/rerank_timing_payload.decoded.py"
set +e
docker run --rm -i --name __CONTAINER__ --cpus 2 --memory "$MEMORY" --memory-swap "$MEMORY" \
  -v metanavit_models:/root/.cache/huggingface __MODEL_MOUNT__ \
  -e BGE_ALLOW_DOWNLOAD=1 -e TIMING_MODEL=__TIMING_MODEL__ -e TIMING_MAX_LENGTH=__MAX_LENGTH__ -e TIMING_DEPTHS=__DEPTHS__ \
  -e TIMING_THREADS=2 -e TIMING_REPEATS=__REPEATS__ -e TIMING_MIN_REPEATS=__MIN_REPEATS__ -e TIMING_BUDGET_SECONDS=__BUDGET__ \
  "$IMAGE" python - < "$RUN_DIR/rerank_timing_payload.decoded.py" > "$RUN_DIR/rerank_timing.out" 2> "$RUN_DIR/rerank_timing.err"
code=$?
set -e
rm -f "$RUN_DIR/rerank_timing_payload.py" "$RUN_DIR/rerank_timing_payload.decoded.py"
if [ $code -eq 137 ]; then
  echo "TIMING_EXIT 137: the container was killed, almost certainly for memory: this model does not fit $MEMORY beside the app and the embedder"
elif [ $code -ne 0 ]; then
  echo "TIMING_EXIT $code"
  tail -n 30 "$RUN_DIR/rerank_timing.err" >&2
fi
grep '^RESULT ' "$RUN_DIR/rerank_timing.out" || true
free -m | sed 's/^/host memory after (MiB): /'
exit $code
"""


def remote_script(*, model: str | None, model_dir: str | None, depths: list[int], max_length: int, repeats: int, min_repeats: int,
                  budget_seconds: int, memory: str | None, stop_api: bool, payload: str | None = None) -> str:
    """The bash script the host runs. Every value that reaches it is validated here: a model is a Hugging Face id, a
    directory an absolute path of plain characters, the numbers are integers, the memory is <n>m."""
    if (model is None) == (model_dir is None):
        raise DeployError("name the model with exactly one of --model <Hugging Face id> and --model-dir <directory on the host>")
    if model is not None and not MODEL.fullmatch(model):
        raise DeployError(f"--model must be a Hugging Face id like cross-encoder/ms-marco-MiniLM-L6-v2, not {model!r}")
    if model_dir is not None and not HOST_PATH.fullmatch(model_dir):
        raise DeployError(f"--model-dir must be an absolute path of letters, digits and . _ - /, not {model_dir!r}")
    if memory is not None and not MEMORY.fullmatch(memory):
        raise DeployError(f"--memory must be a number of MiB like 1024m, not {memory!r}")
    if not depths or any(d < 1 or d > 100 for d in depths) or not 16 <= max_length <= 8192 or repeats < 1 or min_repeats < 1 or budget_seconds < 1:
        raise DeployError("depths 1..100, max_length 16..8192, repeats, min_repeats and budget_seconds 1 or more")
    text = payload if payload is not None else PAYLOAD.read_text()
    compile(text, str(PAYLOAD), "exec")  # the program piped into `python -` must at least parse
    encoded = base64.encodebytes(text.encode()).decode().strip()
    values = {
        "__MEMORY__": shlex.quote(memory or ""), "__CONTAINER__": CONTAINER, "__STOP_API__": "1" if stop_api else "0", "__PAYLOAD_B64__": encoded,
        "__MODEL_MOUNT__": f"-v {shlex.quote(model_dir)}:/models/candidate:ro" if model_dir else "",
        "__TIMING_MODEL__": shlex.quote(model) if model else "/models/candidate", "__MAX_LENGTH__": str(max_length),
        "__DEPTHS__": ",".join(str(d) for d in sorted(depths)), "__REPEATS__": str(repeats), "__MIN_REPEATS__": str(min_repeats),
        "__BUDGET__": str(budget_seconds),
    }
    script = REMOTE
    for name, value in values.items():
        script = script.replace(name, value)
    return script


def commands(script: str) -> list[str]:
    """Run Command's `commands`: the script travels base64-encoded, so nothing in it meets a second shell's quoting."""
    encoded = base64.b64encode(script.encode()).decode()
    return [f"echo {encoded} | base64 -d > /run/metanavit-rerank-timing.sh", "bash /run/metanavit-rerank-timing.sh"]


def parse_result(output: str) -> dict | None:
    """The payload's one RESULT line, or None (the container was killed, or failed before printing it)."""
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return None


def report(result: dict, model: str) -> str:
    """The serving bar's numbers as lines: per depth p50 / p95 of one rerank call, and the peaks."""
    steps = result["memory_mib"]
    lines = [f"{model} at max_length {result['max_length']} on {result.get('cpu_count')} CPUs, {result['threads']} torch threads (torch {result.get('torch')}):",
             f"  memory before the reranker: {steps['after_embedder']['VmRSS']} MiB resident (peak {steps['after_embedder']['VmHWM']} MiB); "
             f"after loading it: {steps['after_reranker_load']['VmRSS']} MiB"]
    for shape in result["shapes"]:
        peak = shape.get("cgroup_peak_mib")
        lines.append(f"  depth {shape['depth']:>3}: p50 {shape['p50_ms']} ms, p95 {shape['p95_ms']} ms per rerank call over {shape['n']} calls; "
                     f"process peak {shape['memory_mib']['VmHWM']} MiB" + (f", container peak {peak} MiB" if peak is not None else ""))
    return "\n".join(lines)


def run_on_host(aws: Aws, instance: str, script: str, timeout: float, clock: Callable[[], float] = time.monotonic,
                sleep: Callable[[float], None] = time.sleep) -> tuple[str, str, str]:
    """Send the script through Run Command and follow it: (status, stdout, stderr)."""
    parameters = {"commands": commands(script), "executionTimeout": [str(int(timeout))]}
    sent = aws("ssm", "send-command", "--document-name", DOCUMENT, "--instance-ids", instance, "--comment", "time a reranker (M43)",
               "--timeout-seconds", str(DELIVERY_SECONDS), "--parameters", json.dumps(parameters))
    command = sent["Command"]["CommandId"]
    print(f"Run Command {command} on {instance}")
    deadline = clock() + timeout + DELIVERY_SECONDS
    invocation: dict = {}
    while True:
        listed = aws("ssm", "list-command-invocations", "--command-id", command, "--instance-id", instance)
        if listed.get("CommandInvocations"):
            invocation = aws("ssm", "get-command-invocation", "--command-id", command, "--instance-id", instance)
            if invocation.get("Status") in DONE:
                break
        if clock() > deadline:
            raise DeployError(f"Run Command {command} still running at the deadline: see it in Systems Manager, Run Command, command history")
        sleep(POLL_SECONDS)
    return invocation.get("Status") or "", invocation.get("StandardOutputContent") or "", invocation.get("StandardErrorContent") or ""


def main(argv: list[str] | None = None, *, aws: Aws | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="a Hugging Face id, downloaded into the stack's model volume on first use")
    parser.add_argument("--model-dir", help="instead: a checkpoint directory on the host, mounted read-only")
    parser.add_argument("--depths", default="20", help="comma-separated rerank depths (default: the served RERANK_DEPTH, 20)")
    parser.add_argument("--max-length", type=int, default=512, help="the token cap (default: the served RERANKER_MAX_LENGTH, 512)")
    parser.add_argument("--repeats", type=int, default=30, help="timed rerank calls per depth at most")
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--budget-seconds", type=int, default=240, help="per depth: a slow model gets fewer timed calls, never under --min-repeats")
    parser.add_argument("--memory", default=None, help="the one-off container's memory limit, like 1024m (default: the api container's, API_MEMORY in app.env)")
    parser.add_argument("--timeout", type=float, default=3000, help="seconds the command may run on the host")
    how = parser.add_mutually_exclusive_group()
    how.add_argument("--stop-api", action="store_true", help="stop the api for the measurement and start the stack again afterwards (the demo is down meanwhile)")
    how.add_argument("--beside-api", action="store_true", help="leave the api running (no downtime; the host may swap: an upper bound)")
    parser.add_argument("--print-remote-script", action="store_true", help="print the script the host would run and exit; AWS is not called")
    args = parser.parse_args(argv)
    try:
        script = remote_script(model=args.model, model_dir=args.model_dir, depths=[int(d) for d in args.depths.split(",") if d.strip()],
                               max_length=args.max_length, repeats=args.repeats, min_repeats=args.min_repeats, budget_seconds=args.budget_seconds,
                               memory=args.memory, stop_api=args.stop_api)
        if args.print_remote_script:
            print(script)
            return 0
        if not (args.stop_api or args.beside_api):
            raise DeployError("choose --stop-api (the demo is down for the measurement; the serving bar) or --beside-api (no downtime; an "
                              "upper bound): see this script's docstring")
        name = os.environ.get("SINGLE_HOST_NAME", "").strip()
        if not name:
            raise DeployError("SINGLE_HOST_NAME is not set (terraform output github_environment_variables of infra/envs/single-host)")
        aws = aws or Aws()
        instance = find_instance(aws, name)
        status, out, err = run_on_host(aws, instance, script, args.timeout)
    except DeployError as error:
        print(f"TIMING FAILED: {error}", file=sys.stderr)
        return 1
    print(out.rstrip())
    if err.strip():
        print(err.rstrip(), file=sys.stderr)
    result = parse_result(out)
    if result is None:
        print(f"TIMING FAILED: no result ({status}); a TIMING_EXIT 137 above means the model does not fit the memory limit", file=sys.stderr)
        return 1
    print(report(result, args.model or args.model_dir))
    return 0 if status == "Success" else 1


if __name__ == "__main__":
    sys.exit(main())
