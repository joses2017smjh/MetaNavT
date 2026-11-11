"""Structure-aware chunking. Naive fixed-size is the baseline to beat."""

from __future__ import annotations

import ast
import bisect
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


_PY_LINE_END = re.compile(r"\r\n|\r|\n")  # the line ends ast counts; str.splitlines also splits at \f, U+2028, ...
_NON_SPACE = re.compile(r"\S")


def python_ast_chunks(text: str) -> list[Span]:
    """One chunk per top-level function or class, and one per stretch of module-level code around them.

    A definition's chunk starts at its first decorator and ends with its last
    line. The code before the first definition (docstring, imports) is one
    chunk, and so is each non-blank stretch between or after definitions
    (constants, registration calls, a main guard), so every non-blank line of
    the module is in exactly one chunk. A module without definitions, or one
    that does not parse, takes the fixed window.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return fixed_size_chunks(text)
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not definitions:
        return fixed_size_chunks(text)
    line_starts = [0] + [m.end() for m in _PY_LINE_END.finditer(text)]

    def line_start(lineno: int) -> int:  # 1-based, as ast counts; past the last line is the end of the text
        return line_starts[lineno - 1] if lineno <= len(line_starts) else len(text)

    slices: list[tuple[int, int]] = []
    cursor = 0  # one forward cursor: slices never overlap
    for node in definitions:
        start = max(cursor, line_start(min([node.lineno] + [d.lineno for d in node.decorator_list])))
        if start > cursor:  # the preamble from 0; a later stretch from its first non-blank line
            slices.append((cursor if not slices else _first_line_with_code(text, line_starts, cursor, start), start))
        end = max(start, line_start(node.end_lineno + 1) if node.end_lineno else len(text))
        slices.append((start, end))
        cursor = end
    if cursor < len(text):
        slices.append((_first_line_with_code(text, line_starts, cursor, len(text)), len(text)))
    return _spans_from_slices(text, slices, "ast")


def _first_line_with_code(text: str, line_starts: list[int], start: int, end: int) -> int:
    """Start of the first line in text[start:end] that is not blank (end when there is none)."""
    match = _NON_SPACE.search(text, start, end)
    if match is None:
        return end
    return max(start, line_starts[bisect.bisect_right(line_starts, match.start()) - 1])


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
