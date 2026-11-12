"""Check that the pinned requirement files still match pyproject.toml. No network, no resolver.

`make lock` writes requirements.txt (eval + app) and one requirements/<profile>.txt per
install profile. CI and the image install those files and then the package with
--no-deps, so a dependency added to pyproject.toml without re-running `make lock`
would be silently missing. This script fails when:

  * a requirement of a profile's extras has no pin in that profile's file, or the pin
    does not satisfy the range in pyproject.toml;
  * a package is pinned to different versions in two files.

Usage: python scripts/check_lock.py [repo-root]     (exit 1 and one line per problem)
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# pinned file -> the extras it must cover (base dependencies are always included).
# Keep in step with the `lock` target in the Makefile.
PROFILES: dict[str, tuple[str, ...]] = {
    "requirements.txt": ("eval", "app"),
    "requirements/eval.txt": ("eval",),
    "requirements/app.txt": ("app",),
    "requirements/mobile.txt": ("eval", "mobile"),
    "requirements/agent.txt": ("eval", "mobile", "agent"),
    "requirements/dev.txt": ("dev",),
    "requirements/distill.txt": ("distill",),
    "requirements/ml.txt": ("ml",),  # the image's EXTRAS=...,ml layer; torch is the CPU wheel (+cpu)
    "requirements/otel.txt": ("otel",),  # the image's EXTRAS=...,otel layer
}
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)")


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = PIN.match(line.strip())
        if match:
            pins[canonicalize_name(match.group(1))] = match.group(2)
    return pins


def check(root: Path) -> list[str]:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    problems: list[str] = []
    seen: dict[str, tuple[str, str]] = {}  # package -> (version, file that pinned it first)
    for rel, extras in PROFILES.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: missing (run `make lock`)")
            continue
        pins = read_pins(path)
        wanted = list(project["dependencies"])
        for extra in extras:
            wanted.extend(project["optional-dependencies"][extra])
        for spec in wanted:
            req = Requirement(spec)
            name = canonicalize_name(req.name)
            if name not in pins:
                problems.append(f"{rel}: no pin for {spec!r} (extras {','.join(extras)}); run `make lock`")
            elif not req.specifier.contains(pins[name], prereleases=True):
                problems.append(f"{rel}: {name}=={pins[name]} does not satisfy {spec!r}; run `make lock`")
        for name, version in pins.items():
            first = seen.setdefault(name, (version, rel))
            if first[0] != version:
                problems.append(f"{rel}: {name}=={version} but {first[1]} pins {first[0]}")
    return problems


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    problems = check(root)
    for problem in problems:
        print(problem)
    if not problems:
        print(f"lock files match pyproject.toml ({len(PROFILES)} files)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
