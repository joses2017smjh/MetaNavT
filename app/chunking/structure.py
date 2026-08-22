"""Structure-aware chunking. Naive fixed-size is the baseline to beat."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Span:
    start: int
    end: int
    text: str
    kind: str = "fixed"


def _spans_from_slices(text: str, slices: list[tuple[int, int]], kind: str) -> list[Span]:
    out = []
    for start, end in slices:
        if start >= end:
            continue
        chunk = text[start:end]
        if chunk.strip():
            out.append(Span(start=start, end=end, text=chunk, kind=kind))
    return out


def fixed_size_chunks(text: str, size: int = 512, overlap: int = 50) -> list[Span]:
    """Character-level fixed window. Baseline."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap >= size:
        overlap = max(0, size // 5)
    spans = []
    i = 0
    n = len(text)
    if n == 0:
        return []
    while i < n:
        end = min(n, i + size)
        chunk = text[i:end]
        if chunk.strip():
            spans.append(Span(start=i, end=end, text=chunk, kind="fixed"))
        if end >= n:
            break
        i = end - overlap
    return spans


def markdown_heading_chunks(text: str, max_size: int = 2000) -> list[Span]:
    lines = text.splitlines(keepends=True)
    starts = [0]
    offset = 0
    for line in lines:
        if re.match(r"^#{1,6}\s+\S", line):
            if offset not in starts and offset > 0:
                starts.append(offset)
        offset += len(line)
    starts.append(len(text))
    slices = []
    for a, b in zip(starts, starts[1:]):
        if b - a > max_size:
            for span in fixed_size_chunks(text[a:b], size=max_size, overlap=80):
                slices.append((a + span.start, a + span.end))
        else:
            slices.append((a, b))
    return _spans_from_slices(text, slices, "markdown")


def csv_record_chunks(text: str) -> list[Span]:
    """One record per chunk, header prepended so each row is self-contained."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    header = lines[0]
    header_len = len(header)
    spans = [Span(start=0, end=header_len, text=header, kind="csv_header")]
    offset = header_len
    for line in lines[1:]:
        end = offset + len(line)
        blob = header + line if line.strip() else line
        if line.strip():
            spans.append(Span(start=offset, end=end, text=blob, kind="csv_row"))
        offset = end
    return spans


def jsonl_record_chunks(text: str) -> list[Span]:
    offset = 0
    spans = []
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        if line.strip():
            spans.append(Span(start=offset, end=end, text=line, kind="jsonl"))
        offset = end
    return spans


def python_ast_chunks(text: str) -> list[Span]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return fixed_size_chunks(text)
    slices: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    # module docstring / imports as a preamble
    first_body = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_body = node
            break
    if first_body is not None:
        preamble_end = _line_start(lines, first_body.lineno)
        if preamble_end > 0:
            slices.append((0, preamble_end))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = _line_start(lines, node.lineno)
            end = _line_start(lines, node.end_lineno + 1) if node.end_lineno else len(text)
            slices.append((start, min(end, len(text))))
    if not slices:
        return fixed_size_chunks(text)
    return _spans_from_slices(text, slices, "ast")


def _line_start(lines: list[str], lineno: int) -> int:
    if lineno <= 1:
        return 0
    return sum(len(l) for l in lines[: lineno - 1])


def yaml_chunks(text: str) -> list[Span]:
    """Keep small configs intact so keys like learning_rate stay next to run_id."""
    if len(text) < 2000:
        return [Span(0, len(text), text, "yaml")] if text.strip() else []
    lines = text.splitlines(keepends=True)
    starts = [0]
    offset = 0
    for i, line in enumerate(lines):
        if i > 0 and re.match(r"^[A-Za-z0-9_\-]+:", line) and not line.startswith(" "):
            starts.append(offset)
        offset += len(line)
    starts.append(len(text))
    slices = list(zip(starts, starts[1:]))
    spans = _spans_from_slices(text, slices, "yaml")
    # prepend path context is the caller's job; keep run_id in every split by
    # attaching the first key block as a header when present.
    if len(spans) <= 1:
        return spans
    header = spans[0].text
    out = [spans[0]]
    for span in spans[1:]:
        out.append(Span(span.start, span.end, header + span.text, "yaml"))
    return out


def chunk_text(text: str, path: str = "", strategy: str = "auto") -> list[Span]:
    ext = Path(path).suffix.lower()
    if strategy == "fixed":
        return fixed_size_chunks(text)
    if strategy == "auto":
        if ext in {".csv"}:
            return csv_record_chunks(text)
        if ext in {".jsonl"}:
            return jsonl_record_chunks(text)
        if ext == ".py":
            return python_ast_chunks(text)
        if ext in {".md", ".rst"}:
            return markdown_heading_chunks(text)
        if ext in {".yaml", ".yml"}:
            return yaml_chunks(text)
        if ext == ".json":
            try:
                json.loads(text)
                return [Span(0, len(text), text, "json")] if text.strip() else []
            except json.JSONDecodeError:
                return jsonl_record_chunks(text)
        return fixed_size_chunks(text)
    if strategy == "markdown":
        return markdown_heading_chunks(text)
    if strategy == "ast":
        return python_ast_chunks(text)
    raise ValueError(f"unknown chunk strategy {strategy}")
