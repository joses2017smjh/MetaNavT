"""Content-hash diff for incremental reindex: which files to embed again, which to drop.

Called by app/eval/ingest.plan_ingest, which the served indexer (app/engine/ingest.py) runs
on every API start with INDEX_ON_START.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def files_to_reindex(
    current: Mapping[str, str],
    previous: Mapping[str, str],
) -> list[str]:
    """Return paths that are new or whose hash changed. Deletes are files_to_remove."""
    out = []
    for path, digest in current.items():
        if previous.get(path) != digest:
            out.append(path)
    return sorted(out)


def files_to_remove(
    current: Mapping[str, str],
    previous: Mapping[str, str],
) -> list[str]:
    """Return paths that were indexed before and are gone now."""
    return sorted(path for path in previous if path not in current)
