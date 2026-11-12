"""Fail unless a named Playwright test ran and passed in every project. No dependencies; reads the JSON report.

A Playwright test that calls test.skip() is reported as skipped and the run still exits 0, so a CI job whose backing
service never started would stay green with the test never run. mobile-app.yml runs this after the browser tests,
on the report Playwright's JSON reporter wrote (PLAYWRIGHT_JSON_OUTPUT_FILE):

    python scripts/require_playwright_tests.py mobile/test-results/e2e-report.json --spec connected.spec.ts \\
        --title "actual research-agent demo service returns inspected citations through HTTP"

Exit 0 when the test passed (a pass on a retry counts) in every project the report's configuration lists; exit 1 when
it was skipped, failed or did not run in one of them, or when the report is missing. Prints one line per project for
the test, then the other tests of the same spec file with their status, which are not required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _specs(suite: dict):
    yield from suite.get("specs", [])
    for child in suite.get("suites", []):
        yield from _specs(child)


def _outcome(test: dict) -> tuple[bool, str]:
    results = [result.get("status") for result in test.get("results", [])]
    if test.get("expectedStatus") == "passed" and test.get("status") in ("expected", "flaky") and results[-1:] == ["passed"]:
        return True, "passed on retry" if test.get("status") == "flaky" else "passed"
    if test.get("status") == "skipped":
        return False, "skipped"
    return False, f"{test.get('status')} (results: {', '.join(map(str, results)) or 'none'})"


def check(report: dict, spec_file: str, title: str) -> tuple[bool, list[str]]:
    """(ok, lines): ok when `title` in `spec_file` passed in every project of the report's configuration."""
    projects = [project.get("name", "") for project in report.get("config", {}).get("projects", [])]
    required: dict[str, tuple[bool, str]] = {}
    lines, others = [], []
    for suite in report.get("suites", []):
        if Path(suite.get("file", "")).name != spec_file:
            continue
        for spec in _specs(suite):
            for test in spec.get("tests", []):
                project = test.get("projectName", "")
                passed, status = _outcome(test)
                if spec.get("title") == title:
                    required[project] = (passed, status)
                    lines.append(f"{spec_file} > {title} [{project}]: {status}")
                else:
                    others.append(f"{spec_file} > {spec.get('title')} [{project}]: {status} (not required)")
    lines += others
    if not required:
        return False, lines + [f"FAIL: no test titled {title!r} in {spec_file}"]
    missing = [project for project in projects if project not in required]
    not_passed = [project for project, (passed, _) in required.items() if not passed]
    if missing or not_passed:
        reasons = [f"{project}: did not run" for project in missing] + [
            f"{project}: {required[project][1]}" for project in not_passed]
        return False, lines + ["FAIL: " + "; ".join(reasons)]
    return True, lines + [f"OK: {title!r} passed in every project ({len(required)})"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="Playwright JSON report")
    parser.add_argument("--spec", required=True, help="spec file name, e.g. connected.spec.ts")
    parser.add_argument("--title", required=True, help="the test's title as written in the spec")
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text())
    except (OSError, ValueError) as error:
        print(f"FAIL: cannot read the Playwright report {args.report}: {error}")
        return 1
    ok, lines = check(report, args.spec, args.title)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
