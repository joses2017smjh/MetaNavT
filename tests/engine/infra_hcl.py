"""Reading the Terraform under infra/ as text, for the tests (no HCL parser is installed and none is added).

Enough of HCL for this repository's files: comments are dropped, strings (with ${...} templates) are skipped as
units, and a block or an attribute is found by its name at the right nesting depth. The Terraform files are written
so that each fact a test checks is one literal on one line (infra/modules/demo/*.tf).
"""

from __future__ import annotations

import re
from pathlib import Path


def _skip_string(text: str, i: int) -> int:
    """text[i] is the opening quote; the index after the closing one (escapes and ${...} templates skipped)."""
    i += 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text[i] == '"':
            return i + 1
        elif text.startswith("${", i) or text.startswith("%{", i):
            i = _skip_template(text, i + 2)
        else:
            i += 1
    raise ValueError("unterminated string")


def _skip_template(text: str, i: int) -> int:
    depth = 1
    while i < len(text):
        if text[i] == '"':
            i = _skip_string(text, i)
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unterminated template")


def strip_comments(text: str) -> str:
    """The text without # and // line comments and /* */ block comments (strings untouched)."""
    if "<<" in re.sub(r'"(?:[^"\\]|\\.)*"', "", text):
        raise ValueError("heredocs are not supported here; write the value as a string")
    out, i = [], 0
    while i < len(text):
        if text[i] == '"':
            j = _skip_string(text, i)
            out.append(text[i:j])
            i = j
        elif text[i] == "#" or text.startswith("//", i):
            j = text.find("\n", i)
            i = len(text) if j < 0 else j
        elif text.startswith("/*", i):
            i = text.index("*/", i) + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _close(text: str, i: int) -> int:
    """text[i] is {, [ or (; the index of its matching closer."""
    pairs = {"{": "}", "[": "]", "(": ")"}
    stack = [pairs[text[i]]]
    i += 1
    while i < len(text):
        c = text[i]
        if c == '"':
            i = _skip_string(text, i)
            continue
        if c in pairs:
            stack.append(pairs[c])
        elif c in ")]}":
            if c != stack.pop():
                raise ValueError(f"unbalanced {c!r} at {i}")
            if not stack:
                return i
        i += 1
    raise ValueError("unbalanced brackets")


def read_dir(directory: Path) -> str:
    """Every .tf file of one module, comments stripped, in name order."""
    return "\n".join(strip_comments(path.read_text()) for path in sorted(directory.glob("*.tf")))


def blocks(text: str, kind: str, *labels: str) -> list[tuple[tuple[str, ...], str]]:
    """((labels...), body) of every top-level `kind "label" ... {` block whose leading labels equal `labels`."""
    found, depth, i = [], 0, 0
    header = re.compile(r'([A-Za-z_][\w-]*)((?:\s+"[^"]*")*)\s*\{')
    while i < len(text):
        c = text[i]
        if c == '"':
            i = _skip_string(text, i)
            continue
        if depth == 0 and (i == 0 or text[i - 1] in "\n\t {}"):
            m = header.match(text, i)
            if m and m.group(1) == kind:
                names = tuple(re.findall(r'"([^"]*)"', m.group(2)))
                end = _close(text, m.end() - 1)
                if names[: len(labels)] == labels:
                    found.append((names, text[m.end(): end]))
                i = end + 1
                continue
        if c in "{[(":
            i = _close(text, i) + 1
            continue
        i += 1
    return found


def block(text: str, kind: str, *labels: str) -> str:
    """The body of the one top-level block `kind "labels"...`."""
    found = blocks(text, kind, *labels)
    if len(found) != 1:
        raise AssertionError(f"{len(found)} blocks {kind} {labels}")
    return found[0][1]


def nested(body: str, kind: str) -> list[str]:
    """Bodies of the blocks named `kind` directly inside `body` (e.g. every statement of a policy document)."""
    return [inner for _, inner in blocks(body, kind)]


def attr(body: str, name: str) -> str | None:
    """The expression of the attribute `name = ...` at the top level of `body`, as written (None if absent)."""
    i = 0
    pattern = re.compile(rf"\s*{re.escape(name)}\s*=(?!=)\s*")
    while i < len(body):
        c = body[i]
        if c == '"':
            i = _skip_string(body, i)
            continue
        if c in "{[(":
            i = _close(body, i) + 1
            continue
        if (i == 0 or body[i - 1] == "\n") and (m := pattern.match(body, i)):
            j = m.end()
            start = j
            while j < len(body) and body[j] != "\n":
                if body[j] == '"':
                    j = _skip_string(body, j)
                elif body[j] in "{[(":
                    j = _close(body, j) + 1
                else:
                    j += 1
            return body[start:j].strip()
        i += 1
    return None


def object_keys(expression: str) -> list[str]:
    """The keys of an HCL object literal `{ KEY = value ... }`, in order (top level only)."""
    expression = expression.strip()
    if not (expression.startswith("{") and expression.endswith("}")):
        raise ValueError(f"not an object literal: {expression[:60]!r}")
    inner, keys, i = expression[1:-1], [], 0
    while i < len(inner):
        c = inner[i]
        if c == '"':
            i = _skip_string(inner, i)
            continue
        if c in "{[(":
            i = _close(inner, i) + 1
            continue
        if (i == 0 or inner[i - 1] in "\n{,") and (m := re.compile(r'\s*"?([A-Za-z_][\w-]*)"?\s*=(?!=)').match(inner, i)):
            keys.append(m.group(1))
            i = m.end()
            continue
        i += 1
    return keys


def strings(expression: str) -> list[str]:
    """Every string literal in an expression, with its templates as written."""
    found, i = [], 0
    while i < len(expression):
        if expression[i] == '"':
            j = _skip_string(expression, i)
            found.append(expression[i + 1: j - 1])
            i = j
        else:
            i += 1
    return found
