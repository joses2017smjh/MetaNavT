"""The lint gate (`make lint`, CI's lint job): the ruff rule set in pyproject.toml and its one exemption.

Unused imports (F401) stayed out of the gate until M33b: the tree had many, two of them in the frozen
fixture (bench/corpus), which is never edited. The others were removed; the fixture keeps its two and
is the only place F401 is waived. Read with tomllib, so this runs where ruff is not installed (only the
lint job installs it).
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# every setting that takes files or rules out of the gate
EXEMPTIONS = ("exclude", "extend-exclude", "ignore", "extend-ignore", "per-file-ignores", "extend-per-file-ignores")


def _ruff() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]


def test_the_lint_gate_flags_unused_imports():
    assert "F401" in _ruff()["lint"]["select"]


def test_the_frozen_fixture_is_the_only_exemption_and_only_from_unused_imports():
    """Every file is linted: bench/corpus is waived from F401 alone, so the defect rules still run over it."""
    ruff = _ruff()
    found = {f"[tool.ruff] {k}": v for k, v in ruff.items() if k in EXEMPTIONS}
    found |= {f"[tool.ruff.lint] {k}": v for k, v in ruff["lint"].items() if k in EXEMPTIONS}
    assert found == {"[tool.ruff.lint] per-file-ignores": {"bench/corpus/**": ["F401"]}}
