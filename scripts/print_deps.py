"""Print the requirement ranges of pyproject.toml's base deps plus the given extras.

These are ranges, not pins: the pinned sets are requirements.txt and
requirements/*.txt (`make lock`). The image no longer uses this script: since
M33 it installs every extra in EXTRAS from its pinned file (requirements/ml.txt,
requirements/otel.txt) with --no-deps. Kept for a quick look at the declared
ranges, for example:
    python scripts/print_deps.py app,ml
"""

import sys
import tomllib

extras = [e for e in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if e]
with open("pyproject.toml", "rb") as fh:
    project = tomllib.load(fh)["project"]
deps = list(project["dependencies"])
for extra in extras:
    deps.extend(project["optional-dependencies"][extra])
print("\n".join(deps))
