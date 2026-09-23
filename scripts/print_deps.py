"""Print the pinned requirement strings for pyproject.toml's base deps plus the given extras.

Used by the Dockerfile so dependencies install in a cached layer before the
source tree is copied:  python scripts/print_deps.py app,ml > /tmp/requirements.txt
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
