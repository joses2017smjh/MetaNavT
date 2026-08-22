"""Content-hash incremental reindex: only files whose sha256 changed."""

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
    """Return paths that are new or whose hash changed. Deletes are ignored here."""
    out = []
    for path, digest in current.items():
        if previous.get(path) != digest:
            out.append(path)
    return sorted(out)
