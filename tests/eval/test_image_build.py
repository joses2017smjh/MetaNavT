"""The image's dependency step installs only the pinned files (no Docker needed).

There is no Docker on the development host, so the Dockerfile's install step is run here as the build runs it: the
RUN line under `sh -c`, with the build arguments as environment variables, from a directory holding the committed
requirements files, and a stand-in `pip` that records its arguments. CI's docker-smoke job builds the default image
(EXTRAS=app) and docker-ml-smoke the ml one; the CUDA override (docker-compose.gpu.yml) is built by hand only.
"""

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_lock", ROOT / "scripts" / "check_lock.py")
check_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_lock)

STAND_IN_PIP = """#!/bin/sh
echo "$*" >> "$PIP_LOG"
case "$*" in *"$PIP_FAIL_ON"*) [ -n "$PIP_FAIL_ON" ] && exit 1 ;; esac
exit 0
"""


def _dependency_step() -> str:
    """The RUN instruction that installs the requirements files, joined as Docker joins continued lines."""
    text = (ROOT / "Dockerfile").read_text().replace("\\\n", "")
    runs = [line.strip()[len("RUN "):] for line in text.splitlines() if line.strip().startswith("RUN ")]
    (step,) = [run for run in runs if "requirements/app.txt" in run]
    return step


def _build(tmp_path: Path, extras: str, torch_index_url: str = "", fail_on: str = ""):
    context = tmp_path / "context"
    (context / "requirements").mkdir(parents=True)
    for name in ("app", "ml", "otel"):
        shutil.copy(ROOT / "requirements" / f"{name}.txt", context / "requirements" / f"{name}.txt")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pip").write_text(STAND_IN_PIP)
    (bin_dir / "pip").chmod(0o755)
    log = tmp_path / "pip.log"
    log.touch()
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "EXTRAS": extras, "TORCH_INDEX_URL": torch_index_url,
           "TMPDIR": str(tmp_path), "PIP_LOG": str(log), "PIP_FAIL_ON": fail_on}
    done = subprocess.run(["sh", "-c", _dependency_step()], cwd=context, env=env, capture_output=True, text=True,
                          timeout=60)
    return done, log.read_text().splitlines()


def _torch_version() -> str:
    return check_lock.read_pins(ROOT / "requirements" / "ml.txt")["torch"].split("+")[0]


@pytest.mark.parametrize("extras, files", [
    ("app", ["app"]),
    ("app,ml", ["app", "ml"]),
    ("app,otel", ["app", "otel"]),
    ("app,ml,otel", ["app", "ml", "otel"]),
])
def test_each_extra_comes_from_its_pinned_file_with_no_resolution(tmp_path, extras, files):
    done, calls = _build(tmp_path, extras)
    assert done.returncode == 0, done.stderr
    assert calls == [f"install --no-deps -r requirements/{name}.txt" for name in files] + ["check"]


def test_an_extra_with_no_pinned_file_stops_the_build(tmp_path):
    done, calls = _build(tmp_path, "app,gpu")
    assert done.returncode != 0
    assert "gpu" in done.stderr
    assert "check" not in calls


def test_a_failed_install_stops_the_build(tmp_path):
    done, calls = _build(tmp_path, "app,ml,otel", fail_on="requirements/ml.txt")
    assert done.returncode != 0
    assert calls == ["install --no-deps -r requirements/app.txt", "install --no-deps -r requirements/ml.txt"]


def test_the_cuda_override_installs_the_locked_torch_version_from_its_index(tmp_path):
    """docker-compose.gpu.yml names a CUDA wheel index: every other ml pin still comes from the file, and torch is the
    same release from that index (its CUDA packages are resolved at build time; no lock file covers them)."""
    (cuda_index,) = re.findall(r"TORCH_INDEX_URL:\s*(\S+)", (ROOT / "docker-compose.gpu.yml").read_text())
    done, calls = _build(tmp_path, "app,ml", torch_index_url=cuda_index)
    assert done.returncode == 0, done.stderr
    without_torch = tmp_path / "ml-without-torch.txt"
    assert calls == ["install --no-deps -r requirements/app.txt", f"install --no-deps -r {without_torch}",
                     f"install torch=={_torch_version()} --index-url {cuda_index}", "check"]
    kept = (ROOT / "requirements" / "ml.txt").read_text().splitlines()
    assert without_torch.read_text().splitlines() == [line for line in kept if not line.startswith("torch==")]


PINNED = re.compile(r"[a-z0-9][a-z0-9._/-]*:[\w.-]+@sha256:[0-9a-f]{64}")


def _dockerfiles() -> list[Path]:
    """Every Dockerfile in the tree: the API image's and .frontend/'s (a standalone Next.js image no job builds)."""
    skip = {".git", "node_modules", "__pycache__", ".vendor", ".next", "android", "ios"}
    found = []
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in skip and not name.startswith((".venv", "venv"))]
        found += [Path(directory) / name for name in files if name == "Dockerfile" or name.startswith("Dockerfile.")]
    return sorted(found)


def _image_references() -> list[tuple[str, str]]:
    """(file, reference) for every image a build or a CI run pulls: each Dockerfile's syntax line and FROM lines (not
    the build stage a FROM selects by name), and every `image:` line of the Compose files and the workflows."""
    references = []
    for path in _dockerfiles():
        dockerfile, label = path.read_text().splitlines(), path.relative_to(ROOT).as_posix()
        stages = {m.group(1) for line in dockerfile if (m := re.match(r"FROM\s+\S+\s+AS\s+(\S+)", line.strip(), re.I))}
        for line in dockerfile:
            if m := re.match(r"#\s*syntax=(\S+)", line):
                references.append((label, m.group(1)))
            elif m := re.match(r"FROM\s+(\S+)", line.strip(), re.I):
                ref = m.group(1)
                if ref in stages or ("${" in ref and any(stage.startswith(ref.split("${")[0]) for stage in stages)):
                    continue
                references.append((label, ref))
    sources = [*sorted(ROOT.glob("docker-compose*.yml")), *sorted((ROOT / ".github" / "workflows").glob("*.yml"))]
    for path in sources:
        for line in path.read_text().splitlines():
            if m := re.match(r"\s*image:\s*(\S+)", line):
                references.append((path.name, m.group(1).strip("'\"")))
    return references


def test_every_base_image_is_pinned_by_digest():
    """A tag can point at a different image later, so the same commit could build differently. Each reference keeps
    the tag it was looked up from in front of the digest; doc/engineering-reference.md says how to refresh them."""
    references = _image_references()
    names = {ref.split("@")[0].rsplit(":", 1)[0] for _, ref in references}
    assert {"docker/dockerfile", "node", "alpine", "python", "pgvector/pgvector", "jaegertracing/all-in-one"} <= names
    assert {name for name, _ in references} >= {"Dockerfile", ".frontend/Dockerfile", "docker-compose.yml", "ci.yml"}
    assert [(name, ref) for name, ref in references if not PINNED.fullmatch(ref)] == []


def test_the_image_resolves_no_version_range():
    """Before M33 the ml and otel extras were installed from pyproject.toml's ranges (scripts/print_deps.py) under
    `-c requirements/app.txt`, and torch with no version."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "print_deps" not in dockerfile
    assert "pip install torch " not in dockerfile and " -c " not in _dependency_step()


def test_the_package_is_built_with_the_pinned_setuptools():
    """`pip install -e .` builds the package in an isolated environment by default, and pip fills that environment from
    the index with the newest setuptools pyproject.toml's [build-system] allows, at build time and unpinned (seen in the
    pinned python image: with --no-index that install fails on `setuptools>=68`). With --no-build-isolation the build
    uses the setuptools requirements/app.txt pins, which every image installs first."""
    text = (ROOT / "Dockerfile").read_text().replace("\\\n", "")
    installs = [line.strip() for line in text.splitlines() if line.strip().startswith("RUN ") and " -e ." in line]
    assert installs == ["RUN pip install --no-deps --no-build-isolation -e ."]
    assert "setuptools" in check_lock.read_pins(ROOT / "requirements" / "app.txt")
