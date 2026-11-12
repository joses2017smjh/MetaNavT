"""The lint gate (`make lint`, CI's lint job): the ruff rule set in pyproject.toml and its one exemption.

Unused imports (F401) stayed out of the gate until M33b: the tree had many, some of them in the frozen
fixture (bench/corpus), which is never edited. The others were removed; the fixture keeps its own and
is the only place F401 is waived. Read with tomllib, so this runs where ruff is not installed (only the
lint job installs it).
"""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Every ruff setting in use. Several others take files or rules out of the gate (include, exclude, ignore, extend,
# lint.pyflakes.allowed-unused-imports, ...), so any new one fails here until it is checked and listed.
SETTINGS = {"[tool.ruff]": {"target-version", "lint"}, "[tool.ruff.lint]": {"select", "per-file-ignores"}}

# A ruff or flake8 noqa directive in a comment waives a whole file, a ruff disable[...] directive the lines after it.
# Either waives F401 unless it names its codes and F401 is not one of them (main.py's flake8 one names E402 only).
# Matched anywhere on a line and in any case, so it catches at least what ruff honours.
DIRECTIVE = re.compile(r"#[ \t]*(?:ruff|flake8)[ \t]*:[ \t]*(?:noqa|disable)\b(.*)$", re.IGNORECASE | re.MULTILINE)


def _ruff() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]


def _tracked(*patterns: str) -> list[str]:
    """The tracked files matching `patterns`: what a clean checkout, and so CI's lint job, gives ruff."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout to list the tracked files")
    listed = subprocess.run(["git", "ls-files", "-z", "--", *patterns], cwd=ROOT, capture_output=True, text=True, check=True)
    return [name for name in listed.stdout.split("\0") if name]


def test_the_lint_gate_flags_unused_imports():
    assert "F401" in _ruff()["lint"]["select"]


def test_the_frozen_fixture_is_the_only_exemption_and_only_from_unused_imports():
    """Every file is linted: bench/corpus is waived from F401 alone, so the defect rules still run over it."""
    ruff = _ruff()
    assert ruff["lint"]["per-file-ignores"] == {"bench/corpus/**": ["F401"]}
    assert {"[tool.ruff]": set(ruff), "[tool.ruff.lint]": set(ruff["lint"])} == SETTINGS


def test_pyproject_toml_is_the_only_ruff_configuration():
    """A ruff.toml, .ruff.toml, or pyproject.toml with a [tool.ruff] table further down replaces these settings there."""
    others = [
        name for name in _tracked("*ruff.toml", "*pyproject.toml")
        if name != "pyproject.toml"
        and (not name.endswith("pyproject.toml") or "ruff" in tomllib.loads((ROOT / name).read_text()).get("tool", {}))
    ]
    assert others == []


def test_no_comment_waives_unused_imports_for_a_whole_file_or_a_range_of_lines():
    """Outside the frozen fixture no comment waives F401 beyond its own line."""
    found = []
    for name in _tracked("*.py", "*.pyi"):
        if name.startswith("bench/corpus/"):
            continue
        for match in DIRECTIVE.finditer((ROOT / name).read_text(encoding="utf-8", errors="replace")):
            after = match.group(1).lstrip()
            if not after.startswith((":", "[")) or "F401" in after.upper():
                found.append(f"{name}: {match.group(0).strip()}")
    assert found == []
