"""The web client's workflow (.github/workflows/web.yml) and its pins, read from the files: the workflow runs only on
GitHub, and no Node is installed for this suite, so what can be read from the files is pinned here."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXACT = re.compile(r"\d+\.\d+\.\d+")
TEST_TOOLS = ("jsdom", "@testing-library/react", "@testing-library/dom")


def _workflow() -> tuple[dict, dict]:
    data = yaml.safe_load((ROOT / ".github" / "workflows" / "web.yml").read_text())
    return data, data.get("on", data.get(True))  # YAML 1.1 reads the bare key `on` as true


def test_the_web_client_is_built_on_every_push_to_main_and_every_pull_request():
    _, on = _workflow()
    assert on["push"] == {"branches": ["main"]}  # no paths filter
    assert on["pull_request"] == {"branches": ["main"]}


def test_the_job_installs_from_the_lockfile_then_tests_then_builds_on_a_pinned_node_and_runner():
    workflow, _ = _workflow()
    (job,) = workflow["jobs"].values()
    assert job["runs-on"] == "ubuntu-24.04"  # a named release; ubuntu-latest moves
    assert job["defaults"]["run"]["working-directory"] == ".frontend"
    (setup,) = [step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-node@")]
    assert EXACT.fullmatch(setup["with"]["node-version"])  # one release, not "22"
    assert setup["with"]["cache-dependency-path"] == ".frontend/package-lock.json"
    assert [step["run"] for step in job["steps"] if "run" in step] == ["npm ci", "npm test", "npm run build"]


def test_the_web_tests_and_their_tools_are_pinned():
    package = json.loads((ROOT / ".frontend" / "package.json").read_text())
    lock = json.loads((ROOT / ".frontend" / "package-lock.json").read_text())
    for name in TEST_TOOLS:
        version = package["devDependencies"][name]
        assert EXACT.fullmatch(version), (name, version)
        assert lock["packages"][f"node_modules/{name}"]["version"] == version
    assert lock["packages"][""]["devDependencies"] == package["devDependencies"]  # npm ci refuses a stale lock
    test = package["scripts"]["test"]
    assert test.startswith("tsx ") and "--test" in test
    tests = sorted(path.name for path in (ROOT / ".frontend" / "tests").glob("*.test.ts*"))
    assert "plans-panel.test.tsx" in tests and "requests.test.ts" in tests, tests
