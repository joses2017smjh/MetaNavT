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


def test_the_ml_and_otel_extras_have_pinned_files():
    """The image installs each extra in EXTRAS from its pinned file. Before M33 ml and otel had none: the image
    resolved their ranges at build time, and torch from an index with no version."""
    assert check_lock.PROFILES["requirements/ml.txt"] == ("ml",)
    assert check_lock.PROFILES["requirements/otel.txt"] == ("otel",)


def test_the_ml_file_pins_the_cpu_torch_wheel_and_no_cuda_package():
    """The image installs requirements/ml.txt with --no-deps, so the file is exactly what the CPU image gets: torch's
    CPU build from PyTorch's CPU index (the index line is in the file, so `pip install -r` finds the wheel), and none
    of the CUDA packages that PyPI's torch for Linux would add."""
    path = ROOT / "requirements" / "ml.txt"
    pins = check_lock.read_pins(path)
    assert pins["torch"].endswith("+cpu"), pins["torch"]
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in path.read_text().splitlines()
    assert sorted(name for name in pins if name.startswith(("nvidia-", "cuda-")) or name == "triton") == []


def test_the_image_files_also_pin_setuptools():
    """The image installs app.txt, ml.txt and otel.txt with --no-deps and then runs pip check, so they must pin every
    package the image needs. pip-compile leaves setuptools out unless --allow-unsafe, and llama-index-core (in app.txt
    and ml.txt) needs a newer setuptools than python:3.11-slim ships (torch's setuptools>=77.0.3 is met by the image's
    79.0.1; otel.txt needs none); pip used to fetch the newest one, unpinned, while it resolved app.txt (found by
    running the install step in the pinned base image)."""
    for name in ("app", "ml", "otel"):
        lines = (ROOT / "requirements" / f"{name}.txt").read_text().splitlines()
        assert "# setuptools" not in lines and "# pip" not in lines, name  # how pip-compile lists one it left unpinned
    for name in ("app", "ml"):
        assert "setuptools" in check_lock.read_pins(ROOT / "requirements" / f"{name}.txt"), name


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
