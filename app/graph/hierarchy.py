"""Hierarchical summaries: chunk → file → folder → corpus.

Broad questions hit folder summaries; specific ones hit chunks.
Folder summaries also become organizer topic labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from app.retrieval.types import Chunk


@dataclass
class NodeSummary:
    path: str
    level: str  # chunk | file | folder | corpus
    text: str
    children: list[str]


def _first_lines(text: str, n: int = 8) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return " ".join(lines[:n])[:400]


def extractive_summary(chunks: Sequence[Chunk], label: str) -> str:
    heads = []
    for chunk in chunks[:6]:
        heads.append(_first_lines(chunk.text, 3))
    joined = " | ".join(heads)
    return f"{label}: {joined}"[:800]


def build_hierarchy(
    chunks: Sequence[Chunk],
    summarizer: Callable[[Sequence[Chunk], str], str] | None = None,
) -> dict[str, NodeSummary]:
    summarizer = summarizer or extractive_summary
    nodes: dict[str, NodeSummary] = {}

    by_file: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_file[chunk.path].append(chunk)
        nodes[chunk.chunk_id] = NodeSummary(
            path=chunk.path,
            level="chunk",
            text=chunk.text[:400],
            children=[],
        )

    folder_children: dict[str, list[str]] = defaultdict(list)
    for path, file_chunks in by_file.items():
        nodes[path] = NodeSummary(
            path=path,
            level="file",
            text=summarizer(file_chunks, path),
            children=[c.chunk_id for c in file_chunks],
        )
        folder = str(Path(path).parent).replace("\\", "/")
        folder_children[folder].append(path)

    corpus_children = []
    for folder, files in folder_children.items():
        file_chunks = [c for p in files for c in by_file[p]]
        nodes[folder] = NodeSummary(
            path=folder,
            level="folder",
            text=summarizer(file_chunks, f"folder {folder}"),
            children=files,
        )
        corpus_children.append(folder)

    nodes["__corpus__"] = NodeSummary(
        path=".",
        level="corpus",
        text=summarizer(list(chunks)[:20], "corpus"),
        children=corpus_children,
    )
    return nodes


def route_level(query: str) -> str:
    q = (query or "").lower()
    if any(w in q for w in ("what's in", "what is in", "overview", "summarize the corpus", "what do i have")):
        return "corpus"
    if any(w in q for w in ("this folder", "directory", "under configs", "in logs")):
        return "folder"
    if any(w in q for w in ("whole file", "entire file", "this file")):
        return "file"
    return "chunk"
