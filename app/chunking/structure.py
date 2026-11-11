"""Structure-aware chunking. Naive fixed-size is the baseline to beat."""

from __future__ import annotations

import ast
import bisect
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


_LINE_END = re.compile(r"\r\n|\r|\n")  # the line ends ast, csv and JSON Lines readers count

# A structural chunk longer than this many characters is re-cut with the window (size MAX_CHUNK_CHARS, overlap
# RECUT_OVERLAP): a Markdown section, a JSON value, a Python definition or stretch of module-level code.
MAX_CHUNK_CHARS = 2000
RECUT_OVERLAP = 80


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


def _bounded(text: str, slices: list[tuple[int, int]], max_size: int = MAX_CHUNK_CHARS) -> list[tuple[int, int]]:
    """The slices, with each one longer than max_size re-cut into overlapping windows of max_size."""
    out = []
    for a, b in slices:
        if b - a > max_size:
            out.extend((a + span.start, a + span.end) for span in fixed_size_chunks(text[a:b], size=max_size, overlap=RECUT_OVERLAP))
        else:
            out.append((a, b))
    return out


_MD_HEADING = re.compile(r"#{1,6}\s+\S")
_MD_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")


def markdown_heading_chunks(text: str, max_size: int = MAX_CHUNK_CHARS) -> list[Span]:
    """A chunk starts at every # heading outside fenced code; a section over max_size is re-cut with the window.

    A fence opens with three or more backticks or tildes (up to three spaces
    in) and closes with a line of the same character, at least as long, and
    nothing after it; an unclosed fence runs to the end of the file. A line
    that only starts with backticks and has more of them later is inline code,
    not a fence.
    """
    starts = [0]
    fence = ""  # the open fence's marker, "" outside fenced code
    for start, end in _line_slices(text):
        line = text[start:end]
        marker = _MD_FENCE.match(line)
        if fence:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence) and not line[marker.end():].strip():
                fence = ""
        elif marker and not (marker.group(1)[0] == "`" and "`" in line[marker.end():]):
            fence = marker.group(1)
        elif start > 0 and _MD_HEADING.match(line):
            starts.append(start)
    starts.append(len(text))
    return _spans_from_slices(text, _bounded(text, list(zip(starts, starts[1:])), max_size), "markdown")


def _line_slices(text: str) -> list[tuple[int, int]]:
    """(start, end) of every line with its line end. Only \r\n, \r and \n end a line, as for the
    Python tokenizer and the csv and JSON Lines readers; str.splitlines also splits at U+2028 and others."""
    starts = [0] + [m.end() for m in _LINE_END.finditer(text)]
    if starts[-1] == len(text):
        starts.pop()  # the text ends with a line end: there is no empty last line
    return list(zip(starts, starts[1:] + [len(text)]))


def _csv_records(text: str, lines: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """(start, end) of every CSV record, or None when the text does not parse as strict CSV.

    A record may span lines (a quoted field holding a line break). Such a record is trusted only
    when it has as many fields as the first record: a stray quote that the parser closes at a
    later field boundary would otherwise merge the rows in between.
    """
    reader = csv.reader((text[a:b] for a, b in lines), strict=True)
    records, consumed = [], 0
    try:
        for row in reader:
            records.append((lines[consumed][0], lines[reader.line_num - 1][1], len(row), reader.line_num - consumed))
            consumed = reader.line_num
    except csv.Error:  # an unclosed quote, a quote inside an unquoted field, a field over the size limit
        return None
    width = next((n_fields for _, _, n_fields, _ in records if n_fields), 0)
    if any(n_lines > 1 and n_fields != width for _, _, n_fields, n_lines in records):
        return None
    return [(a, b) for a, b, _, _ in records]


def csv_record_chunks(text: str) -> list[Span]:
    """One record per chunk, header prepended so each row is self-contained.

    Records come from the csv module, so a quoted line break stays in its row.
    When the file does not parse (see _csv_records), every line is a record,
    so one bad quote cannot pull the rest of the file into one chunk. The
    header is the first non-blank record; blank records have no chunk.
    """
    lines = _line_slices(text)
    records = _csv_records(text, lines)
    spans: list[Span] = []
    header = None
    for start, end in records if records is not None else lines:
        raw = text[start:end]
        if not raw.strip():
            continue
        if header is None:
            header = raw
            spans.append(Span(start=start, end=end, text=raw, kind="csv_header"))
        else:
            spans.append(Span(start=start, end=end, text=header + raw, kind="csv_row"))
    return spans


def jsonl_record_chunks(text: str) -> list[Span]:
    """One chunk per non-blank line; a U+2028 or U+0085 inside a JSON string does not end one."""
    return _spans_from_slices(text, _line_slices(text), "jsonl")


_NON_SPACE = re.compile(r"\S")


def python_ast_chunks(text: str) -> list[Span]:
    """One chunk per top-level function or class, and one per stretch of module-level code around them.

    A definition's chunk starts at its first decorator and ends with its last
    line. The code before the first definition (docstring, imports) is one
    chunk, and so is each non-blank stretch between or after definitions
    (constants, registration calls, a main guard), so every non-blank line of
    the module is in exactly one chunk; a chunk longer than MAX_CHUNK_CHARS is
    re-cut into overlapping windows, whose overlaps repeat a few lines. A
    module without definitions, or one that does not parse, takes the fixed
    window.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):  # invalid, null bytes, or nested too deep to parse
        return fixed_size_chunks(text)
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not definitions:
        return fixed_size_chunks(text)
    line_starts = [0] + [m.end() for m in _LINE_END.finditer(text)]

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
    return _spans_from_slices(text, _bounded(text, slices), "ast")


def _first_line_with_code(text: str, line_starts: list[int], start: int, end: int) -> int:
    """Start of the first line in text[start:end] that is not blank (end when there is none)."""
    match = _NON_SPACE.search(text, start, end)
    if match is None:
        return end
    return max(start, line_starts[bisect.bisect_right(line_starts, match.start()) - 1])


_YAML_TOP_KEY = re.compile(r"([A-Za-z0-9_\-]+):")


def yaml_chunks(text: str) -> list[Span]:
    """Keep small configs intact so keys like learning_rate stay next to run_id.

    A config of MAX_CHUNK_CHARS or more is split before every top-level key.
    Every block's searched text then starts with the header block, so run_id
    stays in every split: the run_id block when there is one, else the first
    key's block (a leading comment is not a header). The header block itself
    is indexed as it is; path context is the caller's job.
    """
    if len(text) < MAX_CHUNK_CHARS:
        return [Span(0, len(text), text, "yaml")] if text.strip() else []
    starts, keys = [0], [None]
    for start, end in _line_slices(text):
        key = _YAML_TOP_KEY.match(text, start, end)
        if key is None:
            continue
        if start == 0:
            keys[0] = key.group(1)
        else:
            starts.append(start)
            keys.append(key.group(1))
    starts.append(len(text))
    blocks = [(a, b, key) for (a, b), key in zip(zip(starts, starts[1:]), keys) if text[a:b].strip()]
    if len(blocks) <= 1:
        return [Span(a, b, text[a:b], "yaml") for a, b, _ in blocks]
    keyed = [block for block in blocks if block[2] is not None]
    header = next((block for block in keyed if block[2] == "run_id"), keyed[0] if keyed else blocks[0])
    head = text[header[0]:header[1]]
    return [Span(a, b, text[a:b] if (a, b, key) == header else head + text[a:b], "yaml") for a, b, key in blocks]


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
            except json.JSONDecodeError:  # not one JSON value: one chunk per record line
                return jsonl_record_chunks(text)
            except RecursionError:  # nested deeper than the decoder recurses: cut like any long value
                pass
            return _spans_from_slices(text, _bounded(text, [(0, len(text))]), "json")
        return fixed_size_chunks(text)
    if strategy == "markdown":
        return markdown_heading_chunks(text)
    if strategy == "ast":
        return python_ast_chunks(text)
    raise ValueError(f"unknown chunk strategy {strategy}")
