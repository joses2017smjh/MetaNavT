"""What the GitHub workflows run and when, checked from the workflow files (they run only on GitHub; no Docker here).

CI's weekly run of the model-backed Compose smoke (owner decision OD18) and the mobile workflow's connected browser
test are configuration; a mistake in either shows up only after a scheduled run or a pull request, so the parts that
can be read from the files are pinned here.
"""

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
