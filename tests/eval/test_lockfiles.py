"""The pinned requirement files must cover pyproject.toml (scripts/check_lock.py).

CI and the image install requirements*.txt and then the package with --no-deps, so a
lock that lags pyproject.toml means a missing dependency at run time. Before this test
requirements.txt had no pin for two declared dependencies (mcp, snowballstemmer).
"""

import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_lock", ROOT / "scripts" / "check_lock.py")
check_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_lock)


def _copy(tmp_path: Path) -> Path:
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "requirements").mkdir()
    for rel in check_lock.PROFILES:
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def test_committed_lock_files_match_pyproject():
    assert check_lock.check(ROOT) == []


def test_every_profile_in_the_makefile_lock_target_is_checked():
    makefile = (ROOT / "Makefile").read_text()
    for rel in check_lock.PROFILES:
        assert f"-o {rel} " in makefile, f"{rel} is checked but `make lock` does not write it"
    assert makefile.count("$(PIP_COMPILE) ") == len(check_lock.PROFILES)


def test_a_dependency_missing_from_the_lock_is_reported(tmp_path):
    root = _copy(tmp_path)
    target = root / "requirements.txt"
    kept = [line for line in target.read_text().splitlines() if not line.startswith("snowballstemmer==")]
    target.write_text("\n".join(kept) + "\n")
    problems = check_lock.check(root)
    assert any("requirements.txt: no pin for 'snowballstemmer" in p for p in problems), problems


def test_a_pin_outside_the_declared_range_is_reported(tmp_path):
    root = _copy(tmp_path)
    target = root / "requirements" / "eval.txt"
    lines = ["pydantic==1.10.0" if line.startswith("pydantic==") else line for line in target.read_text().splitlines()]
    target.write_text("\n".join(lines) + "\n")
    problems = check_lock.check(root)
    assert any("pydantic==1.10.0 does not satisfy" in p for p in problems), problems
    assert any("requirements/eval.txt: pydantic==1.10.0 but requirements.txt pins" in p for p in problems), problems
