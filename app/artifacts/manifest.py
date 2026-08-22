"""ACM-style research artifact bundles over the personal file tree.

Badges follow ACM Artifact Review: Available (files exist + hashes),
Functional (declared tests execute), Reusable (readme / env / citations).
This is the reproducibility pack a lab actually ships: config + sbatch +
log + code + paper claim, not a Hugging Face Space.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from app.graph.file_graph import extract_run_id

ROLES = ("code", "config", "data", "log", "paper", "env", "checkpoint", "test")

ROLE_BY_SUFFIX = {
    ".py": "code",
    ".ipynb": "code",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
    ".sbatch": "env",
    ".sh": "env",
    ".out": "log",
    ".log": "log",
    ".md": "paper",
    ".txt": "paper",
    ".csv": "data",
    ".jsonl": "data",
    ".ckpt": "checkpoint",
    ".pt": "checkpoint",
    ".pth": "checkpoint",
}


@dataclass
class ArtifactFile:
    path: str
    role: str
    sha256: str = ""
    exists: bool = True
    bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "exists": self.exists,
            "bytes": self.bytes,
        }


@dataclass
class Badge:
    name: str  # available | functional | reusable
    earned: bool
    reason: str

    def as_dict(self) -> dict:
        return {"name": self.name, "earned": self.earned, "reason": self.reason}


@dataclass
class ArtifactBundle:
    entity: str
    files: list[ArtifactFile] = field(default_factory=list)
    badges: list[Badge] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "entity": self.entity,
            "files": [f.as_dict() for f in self.files],
            "badges": [b.as_dict() for b in self.badges],
            "claims": self.claims,
            "citations": self.citations,
        }

    def by_role(self, role: str) -> list[ArtifactFile]:
        return [f for f in self.files if f.role == role]


def _role_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    name = Path(path).name.lower()
    if name.startswith("test_") or name.endswith("_test.py") or "/tests/" in path.replace("\\", "/"):
        return "test"
    return ROLE_BY_SUFFIX.get(suffix, "data")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_run_artifact(root: str | Path, run_id: str | int) -> ArtifactBundle:
    """Pack every file whose path or body mentions this run id."""
    root = Path(root)
    rid = str(int(str(run_id).strip()))
    needle = re.compile(rf"run[_\-\s]?0*{re.escape(rid)}\b", re.I)
    files: list[ArtifactFile] = []
    claims: list[str] = []
    citations: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        text = ""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        if not (needle.search(rel) or needle.search(text) or extract_run_id(rel, text) == rid):
            continue
        role = _role_for(rel)
        files.append(
            ArtifactFile(
                path=rel,
                role=role,
                sha256=_sha256(raw),
                exists=True,
                bytes=len(raw),
            )
        )
        if role == "paper" and text:
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and len(line) > 20:
                    claims.append(line[:240])
                    citations.append(
                        {"path": rel, "start_byte": 0, "end_byte": min(len(raw), 400)}
                    )
                    break
    bundle = ArtifactBundle(entity=f"run:{rid}", files=files, claims=claims, citations=citations)
    bundle.badges = score_badges(bundle, tests_passed=None)
    return bundle


def score_badges(bundle: ArtifactBundle, *, tests_passed: bool | None) -> list[Badge]:
    present = [f for f in bundle.files if f.exists]
    missing = [f for f in bundle.files if not f.exists]
    has_code = any(f.role in {"code", "test"} for f in present)
    has_config = any(f.role == "config" for f in present)
    has_env = any(f.role == "env" for f in present)
    has_paper = any(f.role == "paper" for f in present)
    available = bool(present) and not missing
    functional = available and has_code and has_config
    if tests_passed is True:
        functional = functional and True
    elif tests_passed is False:
        functional = False
    reusable = functional and (has_env or has_paper) and bool(bundle.citations or bundle.claims)
    return [
        Badge("available", available, "all declared files exist with hashes" if available else "missing files"),
        Badge(
            "functional",
            functional,
            "code + config present" + ("" if tests_passed is None else f"; tests_passed={tests_passed}"),
        ),
        Badge(
            "reusable",
            reusable,
            "env or paper plus citations" if reusable else "need env/paper citations for reuse",
        ),
    ]


def files_from_paths(root: str | Path, paths: Iterable[str]) -> list[ArtifactFile]:
    root = Path(root)
    out = []
    for rel in paths:
        p = root / rel
        if p.is_file():
            raw = p.read_bytes()
            out.append(
                ArtifactFile(
                    path=rel.replace("\\", "/"),
                    role=_role_for(rel),
                    sha256=_sha256(raw),
                    exists=True,
                    bytes=len(raw),
                )
            )
        else:
            out.append(ArtifactFile(path=rel, role=_role_for(rel), exists=False))
    return out
