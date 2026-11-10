"""Print the requirement ranges of pyproject.toml's base deps plus the given extras.

These are ranges, not pins: the pinned sets are requirements.txt and
requirements/*.txt (`make lock`). The Dockerfile uses this only for extras that
have no pinned file (ml, otel), under the pinned app file as a constraint:
    python scripts/print_deps.py app,ml > /tmp/extras.txt
    pip install -c requirements/app.txt -r /tmp/extras.txt
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
