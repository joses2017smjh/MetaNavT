"""Staleness Tier 1: version clusters from normalized keys + content hashes.

When two chunks share a key but differ in content, order by mtime and prefer
the newest unless the query is explicitly comparative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.eval.hashing import content_hash
from app.retrieval.types import Chunk, RetrievalHit

COMPARATIVE_RE = re.compile(
    r"\b(compare|versus|vs\.?|between|changed|difference|diff|older|newer)\b",
    re.IGNORECASE,
)

CONFIG_FIELD_RE = re.compile(
    r"^(learning_rate|lr|encoder|fusion|num_pairs|bark_type)\s*:\s*(.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def normalized_key(path: str, text: str = "") -> str:
    """Cluster key: same file role (suffix + de-versioned stem), not the whole run."""
    p = Path(path)
    stem = re.sub(r"(_v\d+|_old|_archive|[-._]bak)$", "", p.stem, flags=re.IGNORECASE)
    stem = re.sub(r"_v\d+", "", stem)
    suffix = p.suffix.lower() or "none"
    return f"{suffix}:{stem.lower()}"


@dataclass
class VersionMember:
    path: str
    chunk_id: str
    mtime: float
    content_hash: str
    current: bool


@dataclass
class VersionCluster:
    key: str
    members: list[VersionMember] = field(default_factory=list)

    @property
    def current_member(self) -> VersionMember | None:
        current = [m for m in self.members if m.current]
        if current:
            return max(current, key=lambda m: m.mtime)
        if not self.members:
            return None
        return max(self.members, key=lambda m: m.mtime)

    @property
    def superseded(self) -> list[VersionMember]:
        cur = self.current_member
        return [m for m in self.members if cur and m.path != cur.path]


def cluster_versions(chunks: Sequence[Chunk]) -> dict[str, VersionCluster]:
    groups: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        key = normalized_key(chunk.path, chunk.text)
        groups.setdefault(key, []).append(chunk)

    clusters: dict[str, VersionCluster] = {}
    for key, members in groups.items():
        hashes = {content_hash(m.text) for m in members}
        if len(members) < 2 and len(hashes) < 2:
            continue
        # only a cluster if hashes differ or paths differ
        paths = {m.path for m in members}
        if len(paths) < 2 and len(hashes) < 2:
            continue
        by_path: dict[str, Chunk] = {}
        for m in members:
            prev = by_path.get(m.path)
            if prev is None or m.mtime >= prev.mtime:
                by_path[m.path] = m
        vmembers = []
        max_mtime = max((c.mtime for c in by_path.values()), default=0.0)
        archive_hint = ("archive", "old", "superseded")
        for path, chunk in by_path.items():
            is_archive = any(h in path.replace("\\", "/").lower().split("/") for h in archive_hint)
            vmembers.append(
                VersionMember(
                    path=path,
                    chunk_id=chunk.chunk_id,
                    mtime=chunk.mtime,
                    content_hash=chunk.content_hash or content_hash(chunk.text),
                    current=(not is_archive) and chunk.mtime >= max_mtime - 1e-6,
                )
            )
        # If nothing marked current (all archive), newest wins
        if vmembers and not any(m.current for m in vmembers):
            newest = max(vmembers, key=lambda m: m.mtime)
            vmembers = [
                VersionMember(
                    path=m.path,
                    chunk_id=m.chunk_id,
                    mtime=m.mtime,
                    content_hash=m.content_hash,
                    current=(m.path == newest.path),
                )
                for m in vmembers
            ]
        if len({m.content_hash for m in vmembers}) >= 2 or len(vmembers) >= 2:
            clusters[key] = VersionCluster(key=key, members=vmembers)
    return clusters


def prefer_current(
    hits: Sequence[RetrievalHit],
    clusters: dict[str, VersionCluster],
    query: str,
) -> list[RetrievalHit]:
    """Drop superseded cluster members unless the query is comparative."""
    if COMPARATIVE_RE.search(query or ""):
        return list(hits)
    current_paths = set()
    superseded_paths = set()
    for cluster in clusters.values():
        cur = cluster.current_member
        if cur:
            current_paths.add(cur.path)
        for m in cluster.superseded:
            superseded_paths.add(m.path)
    kept: list[RetrievalHit] = []
    for hit in hits:
        path = hit.chunk.path
        if path in superseded_paths and path not in current_paths:
            # keep if no current member of same cluster is already present
            key = normalized_key(path, hit.chunk.text)
            cluster = clusters.get(key)
            cur = cluster.current_member if cluster else None
            if cur and any(h.chunk.path == cur.path for h in hits):
                continue
        kept.append(hit)
    # re-rank
    for i, hit in enumerate(kept, start=1):
        hit.rank = i
    return kept
