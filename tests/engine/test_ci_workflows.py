"""What the GitHub workflows run and when, checked from the workflow files (they run only on GitHub; no Docker here).

CI's weekly run of the model-backed Compose smoke (owner decision OD18), the two image jobs that publish to ghcr.io and
the mobile workflow's connected browser test are configuration; a mistake in any of them shows up only after a
scheduled run, a push to main or a pull request, so the parts that can be read from the files are pinned here.
"""

import ast
import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

NOT_ON_A_SCHEDULE = "${{ github.event_name != 'schedule' }}"
# image-ml: a push to main or a manual run; never a pull request (so never a required check) and never a schedule
ML_IMAGE_RUNS = "${{ (github.event_name == 'push' && github.ref == 'refs/heads/main') || github.event_name == 'workflow_dispatch' }}"
IMAGE_JOBS = ("image", "image-ml")  # the only jobs that log in to ghcr.io, push, or hold packages: write


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
    assert others.pop("image-ml") == ML_IMAGE_RUNS and "schedule" not in ML_IMAGE_RUNS  # push to main or by hand only
    assert others and set(others.values()) == {NOT_ON_A_SCHEDULE}, others


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


# ---- delivery: the image job publishes to ghcr.io only what CI passed and the smoke ran against

PUSH_TO_MAIN = "${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}"


def test_the_image_job_starts_only_after_every_other_job_a_push_to_main_runs():
    ci, on = _workflow("ci.yml")
    assert on["push"] == {"branches": ["main"]}
    job = ci["jobs"]["image"]
    pushed = {name for name, other in ci["jobs"].items() if name != "image" and other.get("if") == job["if"]}
    assert pushed and set(job["needs"]) == pushed  # a new always-run job must gate the image too


def test_only_a_push_to_main_logs_in_or_pushes():
    ci, _ = _workflow("ci.yml")
    for name, job in ci["jobs"].items():
        registry = [step for step in job["steps"]
                    if "docker login" in step.get("run", "") or "docker push" in step.get("run", "")]
        if name not in IMAGE_JOBS:
            assert registry == [], name
            continue
        assert len(registry) == 2 and all(step["if"] == PUSH_TO_MAIN for step in registry), name
        (login,) = [step for step in registry if "docker login" in step["run"]]
        assert "--password-stdin" in login["run"] and "secrets.GITHUB_TOKEN" not in login["run"]  # never on a command line


def test_the_pushed_image_is_the_one_compose_built_and_the_smoke_ran_against():
    """No separate `docker build`: the image is built through docker-compose.yml (the build arguments docker-smoke
    uses), under the name the push step pushes, and the smoke runs before any push."""
    ci, _ = _workflow("ci.yml")
    steps = ci["jobs"]["image"]["steps"]
    runs = [step.get("run", "") for step in steps]
    assert not any("docker build" in run or "buildx" in run for run in runs)
    (name_step,) = [i for i, run in enumerate(runs) if "IMAGE_REF=" in run]
    assert "COMPOSE_FILE=docker-compose.yml:" in runs[name_step]
    assert re.search(r"image: \\\"%s\\\"\\n    pull_policy: build", runs[name_step])  # built here, never pulled
    assert '"$image:sha-$GITHUB_SHA"' in runs[name_step] and "IMAGE_REF=$image:sha-$GITHUB_SHA" in runs[name_step]
    up = next(i for i, run in enumerate(runs) if "docker compose up --build" in run)
    smoke = next(i for i, run in enumerate(runs) if "scripts/smoke_retrieve.py" in run)
    push = next(i for i, run in enumerate(runs) if "docker push" in run)
    assert name_step < up < smoke < push
    assert runs[push].splitlines()[0] == 'docker push "$IMAGE_REF"'
    smoke_in_docker_smoke = [s["run"] for s in ci["jobs"]["docker-smoke"]["steps"] if "smoke_retrieve.py" in s.get("run", "")]
    assert smoke_in_docker_smoke == [runs[smoke]]  # the same check as the Compose smoke


def test_only_the_image_jobs_may_write_packages():
    ci, _ = _workflow("ci.yml")
    assert ci["permissions"] == {"contents": "read"}
    assert [name for name, job in ci["jobs"].items() if "permissions" in job] == list(IMAGE_JOBS)
    for name in IMAGE_JOBS:
        assert ci["jobs"][name]["permissions"] == {"contents": "read", "packages": "write"}, name


# ---- the CPU model image (image-ml): docker-compose.ml.yml's build, the image docker-compose.prod.yml runs

def test_the_ml_image_job_runs_after_the_image_job_on_a_push_to_main_or_by_hand_never_on_a_pull_request():
    """It takes about ten minutes (CPU torch, both models, the corpus embedded on CPU), so it does not run on pull
    requests and is not a required check; it waits for the image job, so it builds only a commit every other job
    passed. A scheduled run stays docker-ml-smoke's alone (test_a_scheduled_run_runs_only_the_ml_smoke)."""
    ci, on = _workflow("ci.yml")
    job = ci["jobs"]["image-ml"]
    assert job["needs"] == ["image"]
    assert job["if"] == ML_IMAGE_RUNS
    assert "pull_request" not in job["if"] and "workflow_dispatch" in on
    assert job["timeout-minutes"] >= ci["jobs"]["docker-ml-smoke"]["timeout-minutes"]  # the same build and first start


def test_the_pushed_ml_image_is_the_ml_variant_compose_built_and_the_smoke_ran_against():
    """No separate `docker build`: the image is built through docker-compose.yml + docker-compose.ml.yml (EXTRAS=app,ml,
    the Next.js export) under the name the push step pushes, and the smoke and the import check run against it before
    any push. The tags differ from the image job's, so neither job overwrites the other's image."""
    ci, _ = _workflow("ci.yml")
    steps = ci["jobs"]["image-ml"]["steps"]
    runs = [step.get("run", "") for step in steps]
    assert not any("docker build" in run or "buildx" in run for run in runs)
    (name_step,) = [i for i, run in enumerate(runs) if "IMAGE_REF=" in run]
    assert "COMPOSE_FILE=docker-compose.yml:docker-compose.ml.yml:" in runs[name_step]
    assert re.search(r"image: \\\"%s\\\"\\n    pull_policy: build", runs[name_step])  # built here, never pulled
    assert '"$image:sha-$GITHUB_SHA-ml"' in runs[name_step] and "IMAGE_REF=$image:sha-$GITHUB_SHA-ml" in runs[name_step]
    up = next(i for i, run in enumerate(runs) if "docker compose up --build" in run)
    smoke = next(i for i, run in enumerate(runs) if "scripts/smoke_retrieve.py" in run)
    imports = next(i for i, run in enumerate(runs) if "docker compose exec -T api python -c" in run)
    push = next(i for i, run in enumerate(runs) if "docker push" in run)
    assert name_step < up < smoke < imports < push
    assert runs[push].splitlines()[:3] == ['docker push "$IMAGE_REF"', 'docker tag "$IMAGE_REF" "$IMAGE:main-ml"',
                                           'docker push "$IMAGE:main-ml"']
    ml = ci["jobs"]["docker-ml-smoke"]["steps"]
    (ml_up,) = [s["run"] for s in ml if "up --build" in s.get("run", "")]
    assert ml_up.split(" up ")[1] == runs[up].strip().split(" up ")[1]  # the same wait for the first start
    (ml_smoke,) = [s["run"] for s in ml if "smoke_retrieve.py" in s.get("run", "")]
    args = shlex.split(runs[smoke])
    assert {"--require-bm25", "--require-reranker"} <= set(args) <= set(shlex.split(ml_smoke))  # its smoke, no latency
    compose_ml = yaml.safe_load((ROOT / "docker-compose.ml.yml").read_text())["services"]["api"]
    assert compose_ml["build"]["args"]["EXTRAS"] == "app,ml" and compose_ml["build"]["args"]["WITH_FRONTEND"] == "1"


def test_the_ml_image_imports_the_served_app_the_ml_extra_on_cpu_and_the_answer_path_clients():
    """Imported inside the running container: main (every router of the full API: the smoke stack leaves PUBLIC_DEMO
    off), the public demo's surface with the answer route and the shared limits (main imports them only under
    PUBLIC_DEMO), the deploy and index steps, the reranker and the HuggingFace embedder, torch's CPU build, and the
    anthropic and redis clients, which the answer path imports only inside its functions. A walk of app/ would fail on
    a module no image installs for (app/engine/chunkers/semantic.py imports chonkie), so the list is named."""
    ci, _ = _workflow("ci.yml")
    (run,) = [s["run"] for s in ci["jobs"]["image-ml"]["steps"] if "docker compose exec -T api python -c" in s.get("run", "")]
    code = shlex.split(run.strip())[-1]
    tree = ast.parse(code)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    app_modules = {"main", "app.api.public", "app.api.routers.answer", "app.answer.claude", "app.api.limits",
                   "app.database.migrate", "app.engine.ingest", "app.retrieval.rerank"}
    assert imported == app_modules | {"llama_index.embeddings.huggingface", "sentence_transformers", "torch",
                                      "anthropic", "redis"}
    for module in app_modules:  # a renamed module fails here, not on main
        assert (ROOT / (module.replace(".", "/") + ".py")).is_file(), module
    pinned = (ROOT / "requirements" / "app.txt").read_text().splitlines()
    for package in ("anthropic", "redis"):  # the image installs requirements/app.txt, so the import can succeed
        assert any(line.startswith(f"{package}==") for line in pinned), package
    asserts = [ast.unparse(node.test) for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert asserts == ["torch.version.cuda is None"]  # the CPU wheel requirements/ml.txt pins
