"""The status checks main's branch protection requires, derived from the workflow files.

Branch protection (a repository setting on GitHub, not a file in the tree) blocks a merge into main until every
required check has passed. A required check that never reports blocks every merge, so only jobs that report on every
pull request into main are required:

  * the workflow runs on pull requests into main with no `paths` / `paths-ignore` filter (the mobile and research
    agent workflows run only when their files change, so their jobs cannot be required);
  * the job's `if` lets a pull request through. Each condition a workflow uses is classified in CONDITIONS below; an
    unknown one stops this script instead of being guessed.

A matrix job reports one check per combination, named as GitHub names it. Renaming a job renames its check, so after
a rename run `--apply` again (a stale required name blocks every merge until then).

  python scripts/required_checks.py            # the check names, one per line
  python scripts/required_checks.py --check    # exit 1 unless main's protection requires exactly these checks
                                               # and auto-merge is on (reads GitHub through `gh api`)
  python scripts/required_checks.py --apply    # require exactly these checks on main (GitHub Actions as their
                                               # source), not up-to-date branches, no reviews; turn on auto-merge
                                               # (`gh api`, repository admin)
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "main"
GITHUB_ACTIONS_APP_ID = 15368  # the app that reports Actions jobs; a status with the same name from elsewhere does not count

# A job's `if`, as written in the workflow -> does the job run on a pull request into main.
CONDITIONS: dict[str, bool] = {
    "${{ github.event_name != 'schedule' }}": True,
    "${{ github.event_name == 'schedule' || (github.event_name == 'workflow_dispatch' && inputs.ml_smoke) }}": False,
    # ci.yml's image-ml (the CPU model image): a push to main or a manual run
    "${{ (github.event_name == 'push' && github.ref == 'refs/heads/main') || github.event_name == 'workflow_dispatch' }}": False,
}
MATRIX_REF = re.compile(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}")


def _on(workflow: dict) -> dict:
    on = workflow.get("on", workflow.get(True))  # YAML 1.1 reads the bare key `on` as true
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {event: None for event in on}
    return on or {}


def runs_on_every_pull_request(workflow: dict, branch: str = BRANCH) -> bool:
    on = _on(workflow)
    if "pull_request" not in on:
        return False
    trigger = on["pull_request"] or {}
    if "paths" in trigger or "paths-ignore" in trigger:
        return False
    if "branches" in trigger and branch not in trigger["branches"]:
        return False
    return branch not in trigger.get("branches-ignore", [])


def _job_runs_on_pull_requests(workflow_name: str, key: str, job: dict) -> bool:
    condition = job.get("if")
    if condition is None:
        return True
    if condition not in CONDITIONS:
        raise ValueError(f"{workflow_name}: job {key!r} has a condition this script does not know: {condition!r}; "
                         f"add it to CONDITIONS in {Path(__file__).name} with whether it runs on a pull request")
    return CONDITIONS[condition]


def check_names(key: str, job: dict) -> list[str]:
    """The check name(s) a job reports: its `name` (or key), once per matrix combination."""
    name = job.get("name", key)
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    axes = {axis: values for axis, values in matrix.items() if axis not in ("include", "exclude")}
    if matrix.keys() - axes.keys():
        raise ValueError(f"job {key!r}: matrix include/exclude is not supported here; list the checks by hand")
    if not axes:
        return [name]
    names = []
    for combination in itertools.product(*axes.values()):
        values = dict(zip(axes, combination))
        if MATRIX_REF.search(name):
            names.append(MATRIX_REF.sub(lambda m: str(values[m.group(1)]), name))
        else:  # GitHub appends the combination when the name does not use it
            names.append(f"{name} ({', '.join(str(v) for v in combination)})")
    return names


def required_checks(root: Path = ROOT, branch: str = BRANCH) -> list[str]:
    names = []
    for path in sorted((root / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        if not runs_on_every_pull_request(workflow, branch):
            continue
        for key, job in workflow["jobs"].items():
            if _job_runs_on_pull_requests(path.name, key, job):
                names += check_names(key, job)
    if len(names) != len(set(names)):
        raise ValueError(f"two jobs report the same check name: {sorted(names)}")
    return sorted(names)


def _gh(*args: str, payload: dict | None = None) -> dict:
    done = subprocess.run(["gh", "api", *args, *(["--input", "-"] if payload is not None else [])],
                          input=json.dumps(payload) if payload is not None else None,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"gh api {' '.join(args)}: {done.stderr.strip() or done.stdout.strip()}")
    return json.loads(done.stdout or "{}")


def protection_payload(names: list[str]) -> dict:
    return {
        "required_status_checks": {"strict": False,
                                   "checks": [{"context": n, "app_id": GITHUB_ACTIONS_APP_ID} for n in names]},
        "enforce_admins": False,
        "required_pull_request_reviews": None,
        "restrictions": None,
    }


def differences(names: list[str], protection: dict, repo: dict) -> list[str]:
    """What main's live protection and the repository settings lack, compared with the derived checks."""
    status = protection.get("required_status_checks") or {}
    live = {c["context"]: c.get("app_id") for c in status.get("checks", [])}
    problems = [f"required but no job reports it: {n}" for n in sorted(live.keys() - set(names))]
    problems += [f"not required: {n}" for n in names if n not in live]
    problems += [f"not tied to GitHub Actions: {n}" for n in names if n in live and live[n] != GITHUB_ACTIONS_APP_ID]
    if not repo.get("allow_auto_merge"):
        problems.append("auto-merge is off (allow_auto_merge)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    names = required_checks()
    if args.apply:
        _gh("-X", "PUT", f"repos/{{owner}}/{{repo}}/branches/{BRANCH}/protection", payload=protection_payload(names))
        _gh("-X", "PATCH", "repos/{owner}/{repo}", payload={"allow_auto_merge": True})
    if args.check or args.apply:
        try:
            protection = _gh(f"repos/{{owner}}/{{repo}}/branches/{BRANCH}/protection")
        except RuntimeError as error:
            if "Branch not protected" not in str(error):
                raise
            protection = {}
        problems = differences(names, protection, _gh("repos/{owner}/{repo}"))
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            return 1
        print(f"{BRANCH} requires {len(names)} checks, auto-merge on: OK")
        return 0
    print("\n".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
