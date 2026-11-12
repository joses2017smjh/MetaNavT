"""What the GitHub workflows run and when, checked from the workflow files (they run only on GitHub; no Docker here).

CI's weekly run of the model-backed Compose smoke (owner decision OD18) and the mobile workflow's connected browser
test are configuration; a mistake in either shows up only after a scheduled run or a pull request, so the parts that
can be read from the files are pinned here.
"""

import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _workflow(name: str) -> tuple[dict, dict]:
    data = yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())
    return data, data.get("on", data.get(True))  # YAML 1.1 reads the bare key `on` as true


def test_ci_runs_the_ml_smoke_once_a_week_off_the_hour():
    ci, on = _workflow("ci.yml")
    (entry,) = on["schedule"]
    minute, hour, day_of_month, month, day_of_week = entry["cron"].split()
    assert minute.isdigit() and int(minute) != 0  # GitHub delays scheduled runs most at the top of the hour
    assert hour.isdigit() and day_of_month == "*" and month == "*" and day_of_week.isdigit()  # one day a week
    condition = ci["jobs"]["docker-ml-smoke"]["if"]
    assert "github.event_name == 'schedule'" in condition
    assert "github.event_name == 'workflow_dispatch' && inputs.ml_smoke" in condition  # the manual run stays


def test_a_scheduled_run_runs_only_the_ml_smoke():
    ci, _ = _workflow("ci.yml")
    others = {name: job.get("if") for name, job in ci["jobs"].items() if name != "docker-ml-smoke"}
    assert others and set(others.values()) == {"${{ github.event_name != 'schedule' }}"}, others


def test_a_scheduled_run_and_a_push_to_main_do_not_cancel_each_other():
    """Both run on refs/heads/main; with one concurrency group and cancel-in-progress, a merge would cancel the weekly
    smoke and the weekly smoke a push run."""
    ci, _ = _workflow("ci.yml")
    assert ci["concurrency"]["cancel-in-progress"] is True
    assert "github.event_name == 'schedule'" in ci["concurrency"]["group"]


def test_the_mobile_browser_job_runs_connected_spec_against_the_companion():
    """connected.spec.ts's research-agent test skips itself without AGENT_E2E_URL, and every run before M33 skipped
    it. The web job now starts the companion (demo mode: no model, no key), points the test at it, and then requires
    the test to have passed in every project, read from Playwright's JSON report."""
    workflow, _ = _workflow("mobile-app.yml")
    steps = workflow["jobs"]["web"]["steps"]
    runs = [step.get("run", "") for step in steps]
    start = next(i for i, run in enumerate(runs) if "app.agent.companion_api:app" in run)
    e2e = next(i for i, run in enumerate(runs) if "test:e2e" in run)
    guard = next(i for i, run in enumerate(runs) if "require_playwright_tests.py" in run)
    assert start < e2e < guard
    port = re.search(r"--port (\d+)", runs[start]).group(1)
    assert f"http://127.0.0.1:{port}/agent/health" in runs[start]  # ready before the browser starts
    vite = re.search(r"baseURL: '([^']+)'", (ROOT / "mobile" / "playwright.config.ts").read_text()).group(1)
    assert vite in steps[start]["env"]["AGENT_ALLOWED_ORIGINS"].split(",")  # the page's origin may call the service
    env = steps[e2e]["env"]
    assert env["AGENT_E2E_URL"] == f"http://127.0.0.1:{port}"
    reporters = runs[e2e].split("--reporter=")[1].split()[0].split(",")
    assert "json" in reporters and "html" in reporters  # html: the report the artifact step uploads
    args = shlex.split(runs[guard])
    assert args[2] == env["PLAYWRIGHT_JSON_OUTPUT_FILE"]
    assert args[args.index("--spec") + 1] == "connected.spec.ts"
    title = args[args.index("--title") + 1]
    assert f"test('{title}'" in (ROOT / "mobile" / "e2e" / "connected.spec.ts").read_text()


def test_the_mobile_workflow_runs_when_the_companion_or_the_guard_changes():
    _, on = _workflow("mobile-app.yml")
    for event in ("push", "pull_request"):
        paths = on[event]["paths"]
        for path in ("app/agent/**", "app/mobile/**", "app/api/routers/retrieve.py", "requirements/agent.txt",
                     "scripts/require_playwright_tests.py"):
            assert path in paths, (event, path)


def test_the_mobile_browser_job_runs_on_an_image_its_playwright_supports():
    """ubuntu-latest became 26.04, where the pinned Playwright cannot install Chromium, and the web job failed at
    the install step. The job names an Ubuntu release Playwright supports instead of following ubuntu-latest."""
    workflow, _ = _workflow("mobile-app.yml")
    runner = workflow["jobs"]["web"]["runs-on"]
    assert runner != "ubuntu-latest"
    assert re.fullmatch(r"ubuntu-(22|24)\.04", runner), runner
