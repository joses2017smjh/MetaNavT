"""Deterministic file graph: containment, references, co-modification, provenance.

No LLM. Edges live alongside vectors conceptually; this module is in-memory
for the frozen corpus and can be persisted to Postgres later.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

RUN_ID_RE = re.compile(r"run[_\-]?0*(\d+)", re.IGNORECASE)
PATH_REF_RE = re.compile(
    r"(?:[\w.\-]+/)+[\w.\-]+\.(?:ya?ml|py|csv|md|out|sbatch|json|png|pdf)"
)
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)
INCLUDE_RE = re.compile(r"""(?:include|source|config|checkpoint)\s*[:=]\s*['\"]?([^\s'\"]+)""")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str
    weight: float = 1.0


@dataclass
class FileGraph:
    nodes: set[str] = field(default_factory=set)
    edges: list[Edge] = field(default_factory=list)

    def neighbors(self, node: str, hops: int = 1, kinds: Sequence[str] | None = None) -> set[str]:
        adj: dict[str, set[str]] = defaultdict(set)
        for e in self.edges:
            if kinds and e.kind not in kinds:
                continue
            adj[e.src].add(e.dst)
            adj[e.dst].add(e.src)
        frontier = {node}
        seen = {node}
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for nb in adj.get(n, ()):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
            frontier = nxt
        seen.discard(node)
        return seen

    def edges_of(self, kind: str) -> list[Edge]:
        return [e for e in self.edges if e.kind == kind]


def _norm(path: str) -> str:
    return str(path).replace("\\", "/")


def extract_run_id(path: str, text: str = "") -> str | None:
    for blob in (path, text):
        m = RUN_ID_RE.search(blob)
        if m:
            return str(int(m.group(1)))
    return None


def build_file_graph(
    files: Sequence[tuple[str, str, float]],
    *,
    co_mod_window_s: float = 3600.0,
) -> FileGraph:
    """files: (path, text, mtime)."""
    graph = FileGraph()
    by_dir: dict[str, list[str]] = defaultdict(list)
    by_run: dict[str, list[str]] = defaultdict(list)
    path_set = {_norm(p) for p, _, _ in files}

    for path, text, mtime in files:
        path = _norm(path)
        graph.nodes.add(path)
        parent = str(Path(path).parent).replace("\\", "/")
        if parent not in {".", ""}:
            graph.nodes.add(parent)
            graph.edges.append(Edge(parent, path, "contains"))
            by_dir[parent].append(path)
        run_id = extract_run_id(path, text)
        if run_id:
            by_run[run_id].append(path)
        for ref in PATH_REF_RE.findall(text or ""):
            ref_n = _norm(ref)
            if ref_n in path_set and ref_n != path:
                graph.edges.append(Edge(path, ref_n, "references"))
        if path.endswith(".py"):
            for imp in IMPORT_RE.findall(text or ""):
                stem = imp.replace(".", "/") + ".py"
                for candidate in path_set:
                    if candidate.endswith(stem) or Path(candidate).stem == imp.split(".")[-1]:
                        if candidate != path:
                            graph.edges.append(Edge(path, candidate, "imports"))
        for inc in INCLUDE_RE.findall(text or ""):
            inc_n = _norm(inc)
            for candidate in path_set:
                if candidate.endswith(inc_n) or Path(candidate).name == Path(inc_n).name:
                    if candidate != path:
                        graph.edges.append(Edge(path, candidate, "references"))

    for run_id, members in by_run.items():
        for a in members:
            for b in members:
                if a < b:
                    graph.edges.append(Edge(a, b, "same_run"))

    timed = [(_norm(p), mtime) for p, _, mtime in files]
    timed.sort(key=lambda x: x[1])
    for i, (pa, ta) in enumerate(timed):
        for pb, tb in timed[i + 1 :]:
            if tb - ta > co_mod_window_s:
                break
            if Path(pa).parent == Path(pb).parent:
                graph.edges.append(Edge(pa, pb, "co_modified"))

    return graph


def expand_with_graph(
    seed_paths: Iterable[str],
    graph: FileGraph,
    hops: int = 1,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for p in seed_paths:
        p = _norm(p)
        if p not in seen:
            ordered.append(p)
            seen.add(p)
        for nb in sorted(graph.neighbors(p, hops=hops)):
            if nb not in seen:
                ordered.append(nb)
                seen.add(nb)
    return ordered
