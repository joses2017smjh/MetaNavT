"""Load and hash-verify the frozen corpus."""

from __future__ import annotations

import json
import os
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
    """Return a list of mismatch descriptions (empty = ok).

    A file's `sha256` is of the text read_text() returns, which turns CRLF into LF, so a
    copy whose line endings were rewritten (a checkout with core.autocrlf=true) still
    matches it, while the loader (app/eval/ingest.py) reads the bytes and indexes other
    text. The size on disk must therefore also equal the manifest's `bytes`.
    """
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
        size = on_disk[rel].stat().st_size
        if entry.get("bytes") is not None and size != entry["bytes"]:
            errors.append(f"size mismatch: {rel} ({size} bytes on disk, {entry['bytes']} in the manifest; line endings rewritten?)")
        digest = content_hash(on_disk[rel].read_text(encoding="utf-8", errors="replace"))
        if digest != entry["sha256"]:
            errors.append(f"hash mismatch: {rel}")
    return errors


def mtime_mismatches(files_root: Path, manifest: dict) -> list[str]:
    """Listed files whose mtime on disk (whole seconds) is not the one the manifest records."""
    out = []
    for entry in manifest["files"]:
        path = Path(files_root) / entry["path"]
        if not path.is_file() or int(path.stat().st_mtime) != int(entry["mtime"]):
            out.append(entry["path"])
    return out


def apply_manifest_mtimes(files_root: Path, manifest: dict) -> int:
    """Set every listed file's mtime to the manifest's, as bench/corpus/build.py left it. Contents are not
    touched, and nothing is set while the files disagree with the manifest (verify_manifest)."""
    problems = verify_manifest(Path(files_root), manifest)
    if problems:
        raise RuntimeError("corpus drift; mtimes not set:\n" + "\n".join(problems))
    for entry in manifest["files"]:
        os.utime(Path(files_root) / entry["path"], (entry["mtime"], entry["mtime"]))
    return len(manifest["files"])


def main(argv: list[str] | None = None) -> int:
    """make fixture-mtimes: the served indexer (app/engine/ingest.py) and its staleness filter read the disk's
    mtimes, and a checkout sets them to the moment each file was written. This puts back the fixture's own."""
    import argparse

    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--apply-mtimes", action="store_true", help="set bench/corpus/files mtimes to the MANIFEST's")
    args = p.parse_args(argv)
    files_root, manifest = root / "bench" / "corpus" / "files", load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    if args.apply_mtimes:
        print(f"set {apply_manifest_mtimes(files_root, manifest)} mtimes under bench/corpus/files from bench/corpus/MANIFEST.json")
    mismatched = mtime_mismatches(files_root, manifest)
    print(f"{len(manifest['files']) - len(mismatched)} of {len(manifest['files'])} files have the MANIFEST's mtime")
    return 1 if args.apply_mtimes and mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
