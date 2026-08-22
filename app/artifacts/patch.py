"""SEARCH/REPLACE patches (Aider / OpenHands / Cursor apply model).

propose never writes. apply requires approved=True, same as file moves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FilePatch:
    path: str
    old: str
    new: str

    def as_dict(self) -> dict:
        return {"path": self.path, "old": self.old, "new": self.new}


BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(?P<old>.*?)\n=======\n(?P<new>.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


def parse_search_replace(blob: str, default_path: str = "artifact.py") -> list[FilePatch]:
    path = default_path
    header = re.search(r"^path:\s*(\S+)", blob or "", re.M)
    if header:
        path = header.group(1)
    patches = []
    for m in BLOCK_RE.finditer(blob or ""):
        patches.append(FilePatch(path=path, old=m.group("old"), new=m.group("new")))
    return patches


def apply_search_replace(source: str, patch: FilePatch) -> str:
    if patch.old not in source:
        raise ValueError(f"search block not found in {patch.path}")
    return source.replace(patch.old, patch.new, 1)


def unified_hunk(path: str, old: str, new: str) -> str:
    old_lines = (old or "").splitlines()
    new_lines = (new or "").splitlines()
    parts = [f"--- a/{path}", f"+++ b/{path}", f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@"]
    for line in old_lines:
        parts.append("-" + line)
    for line in new_lines:
        parts.append("+" + line)
    return "\n".join(parts)
