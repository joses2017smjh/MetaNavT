"""M46, part (b): how a cached reply is read (PARSE_RULES), one parser per kind of call.

Every parser takes the reply text as the model returned it and whether the call was truncated (done_reason "length"),
and returns (the value the arm uses, a status). The statuses are counted per row. A value of None, or an empty list,
means the arm behaves as its parent for that query (HyDE, multi-query, decomposition), counts the unit as "yes" (the
grader), or has no rewrite; a heuristic is never substituted (app/eval/llm_query_arms.py). The truncation rule per kind
is llm_query_spec.TRUNCATION.
"""

from __future__ import annotations

import re

PARSE_RULES = {
    "hyde": "the reply stripped of surrounding whitespace; empty means no hypothesis",
    "lines": ("multi-query (keep 2) and decomposition (keep 4): split into lines; strip numbering and bullets "
              "(^[\\d.)\\-*\\s]+), then surrounding quotes and markdown (** __ ` # and surrounding *); drop empty lines, "
              "lines ending in ':' (preambles), lines equal (case-insensitive) to the query and lines equal "
              "(case-insensitive) to an earlier kept line; keep the first 2 or 4. A truncated reply loses its last (cut) line."),
    "grade": ("lowercase; strip the non-letters at the start (code fences); the first word: yes or no; anything else is "
              "unparseable"),
    "rewrite": ("the first non-empty line, surrounding quotes and a leading 'Rewritten query:' stripped; a truncated reply's "
                "first line is used only if a line break follows it"),
}
STATUSES = ("ok", "empty", "unparseable", "unchanged")
KEEP = {"multiquery": 2, "decompose": 4}

_BULLET = re.compile(r"^[\d.)\-*\s]+")
_QUOTES = "\"'“”‘’`"
_MARKDOWN = re.compile(r"^(?:#+\s*|\*\*|__)|(?:\*\*|__)$")
_LEADING_NONLETTERS = re.compile(r"^[^a-z]+")
_FIRST_WORD = re.compile(r"^[a-z]+")
_REWRITE_LABEL = re.compile(r"^rewritten query\s*:\s*", re.I)


def _strip_wrapping(text: str) -> str:
    """Surrounding quotes and markdown, repeated until nothing changes."""
    prev = None
    while prev != text:
        prev = text
        text = _MARKDOWN.sub("", text).strip()
        if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
            text = text[1:-1].strip()
        elif text and text[0] in _QUOTES and text.count(text[0]) == 1:
            text = text[1:].strip()
        elif text and text[-1] in _QUOTES and text.count(text[-1]) == 1:
            text = text[:-1].strip()
        if len(text) >= 2 and text[0] == "*" and text[-1] == "*":
            text = text[1:-1].strip()
    return text


def parse_hyde(reply: str | None, truncated: bool = False) -> tuple[str | None, str]:
    """The hypothesis (None when the reply is empty). A truncated reply is used as written: it is still a passage."""
    del truncated
    text = (reply or "").strip()
    return (text, "ok") if text else (None, "empty")


def parse_lines(reply: str | None, query: str, keep: int, truncated: bool = False) -> tuple[list[str], str]:
    """Alternative queries or sub-questions, at most `keep`. Status: ok (at least one line), empty (no reply),
    unchanged (every surviving candidate line equals the query), unparseable (a reply, but no line survives)."""
    raw = reply or ""
    if not raw.strip():
        return [], "empty"
    lines = raw.split("\n")
    if truncated:
        lines = lines[:-1]  # the cut line (empty when the reply ended with a line break)
    q = query.strip().lower()
    out: list[str] = []
    equal_to_query = 0
    for line in lines:
        text, prev = line.strip(), None
        while prev != text:  # "1. **query**" and "**1. query**" alike
            prev = text
            text = _strip_wrapping(_BULLET.sub("", text).strip())
        if not text or text.endswith(":"):
            continue
        if text.lower() == q:
            equal_to_query += 1
            continue
        if text.lower() in {x.lower() for x in out}:
            continue
        out.append(text)
        if len(out) >= keep:
            break
    if out:
        return out, "ok"
    return [], ("unchanged" if equal_to_query else "unparseable")


def parse_grade(reply: str | None, truncated: bool = False) -> tuple[str | None, str]:
    """"yes", "no", or None (unparseable or empty: the corrective rule counts it as yes)."""
    del truncated  # a grader reply is parsed from its first word as usual
    text = (reply or "").lower()
    if not text.strip():
        return None, "empty"
    word = _FIRST_WORD.match(_LEADING_NONLETTERS.sub("", text))
    if word and word.group(0) in ("yes", "no"):
        return word.group(0), "ok"
    return None, "unparseable"


def parse_rewrite(reply: str | None, truncated: bool = False) -> tuple[str | None, str]:
    """The rewritten query (None: no rewrite). Whether it differs from the query is the corrective rule's test."""
    raw = reply or ""
    if not raw.strip():
        return None, "empty"
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if truncated and i == len(lines) - 1:
            return None, "unparseable"  # the first line did not end before the cut
        text = _strip_wrapping(_REWRITE_LABEL.sub("", _strip_wrapping(line.strip())).strip())
        return (text, "ok") if text else (None, "unparseable")
    return None, "empty"


def parse(kind: str, reply: str | None, query: str, truncated: bool = False):
    """The parser of `kind` (llm_query_spec.KINDS)."""
    if kind == "hyde":
        return parse_hyde(reply, truncated)
    if kind in KEEP:
        return parse_lines(reply, query, KEEP[kind], truncated)
    if kind == "grade":
        return parse_grade(reply, truncated)
    if kind == "rewrite":
        return parse_rewrite(reply, truncated)
    raise KeyError(kind)
