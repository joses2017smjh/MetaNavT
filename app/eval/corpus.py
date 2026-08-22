"""Load and hash-verify the frozen corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.eval.hashing import content_hash


@dataclass
class CorpusSnapshot:
    root: Path
    aggregate_sha256: str
    files: list[dict]

    @property
    def n_files(self) -> int:
        return len(self.files)


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def verify_manifest(files_root: Path, manifest: dict) -> list[str]:
    """Return a list of mismatch descriptions (empty = ok)."""
    errors = []
    listed = {entry["path"]: entry for entry in manifest["files"]}
    on_disk = {
        str(p.relative_to(files_root)).replace("\\", "/"): p
        for p in files_root.rglob("*")
        if p.is_file()
    }
    extra = set(on_disk) - set(listed)
    missing = set(listed) - set(on_disk)
    for p in sorted(extra):
        errors.append(f"untracked file: {p}")
    for p in sorted(missing):
        errors.append(f"missing file: {p}")
    for rel, entry in listed.items():
        if rel not in on_disk:
            continue
        digest = content_hash(on_disk[rel].read_text(encoding="utf-8", errors="replace"))
        if digest != entry["sha256"]:
            errors.append(f"hash mismatch: {rel}")
    return errors
