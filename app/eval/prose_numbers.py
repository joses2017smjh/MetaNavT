"""Numbers in the prose of a Markdown document, checked against the files they cite (D36).

A number that comes from a results file (or from the code) carries a citation on the same line, after it: an HTML
comment `<!-- num: SPEC; SPEC; ... -->`, which GitHub does not render. The comment's k specs cite, in order, the k
numbers that precede it on that line (after any earlier citation on the line); a line must not start with one, since
on GitHub that ends the paragraph. A spec is one of

  bench/results/x.json#/a/0/b    a JSON pointer (RFC 6901) into a file under the repository root; a segment
                                 [field=value] selects the one list element whose field equals value
  #/a/1/b                        the same, in the file the previous spec on the line named
  bench/results/x.json#/a|len    the number of elements of the list or object there
  bench/results/x.json#/a|/1000  the value divided (|/N) or multiplied (|*N) by a constant, for a unit change
  app/x.py:"TEXT {} TEXT"        the number written at {} in the first line of that file the pattern matches

and the cited number must equal the value as the prose prints it: the value formatted with as many decimals as the
prose shows (f"{value:.{d}f}"), thousands separators and a leading + aside. A number written as inline code with
nothing else in the span (`0.938`, `0.0550`) is prose for this check; other inline code is code.

Every other number in the prose must be classified as not a result, by a rule of this module (a number glued to a
name, such as nDCG@10, fp16, qwen2.5:7b or k1=0.9; a list marker; an HTTP status code that is not part of a list of
numbers and not followed by a unit) or by a phrase in NOT_RESULTS (the reviewed list of the document's numbers that
are not results, each with its reason).
A phrase that no longer occurs in the document is an error, so the list cannot go stale. So a number typed into the
prose is either cited, and then checked, or classified on purpose.

Not checked: fenced code blocks (their numbers are code and commands) and generated blocks
(<!-- name:start ... --> ... <!-- name:end -->, which `make bench-table-check` checks). This module checks only the
documents in NOT_RESULTS (the engineering reference); README.md's numbers are generated blocks, and PLAN.md's status
column has its own test (tests/eval/test_plan_status.py).

    python -m app.eval.prose_numbers          # exit 1 and list the problems if a number is wrong or unclassified
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = "doc/engineering-reference.md"

CITATION = re.compile(r"<!--\s*num:(?P<body>.*?)-->")
# A number standing on its own: not glued on the left to a letter, digit or one of . @ # : + - = _ (so nDCG@10, fp16,
# qwen2.5:7b, k1=0.9, run_047 and v2 are names; of a range such as 1-50 only the first number counts), nor on the right
# to a letter or digit (H100, 80GB, 7b), except a multiplier x (6x is a number). A ~ or a / in front does not glue:
# ~1.3 and both numbers of 13/16 count.
NUMBER = re.compile(r"(?<![\w.@#:+\-=])[+\-]?\d+(?:,\d{3})*(?:\.\d+)?(?!(?!x\b)\w)")
SPEC = re.compile(r'\s*(?:(?P<file>[\w./\-]+)?(?P<ptr>#[^;|"\s]*)(?:\|(?P<op>len|[/*][0-9.]+))?'
                  r'|(?P<src>[\w./\-]+):"(?P<pat>(?:[^"\\]|\\.)*)")\s*(?:;|$)')
NUMERIC_CODE = re.compile(r"`[\d\s.,+\-→/%]+`")
HTTP_STATUS = {"200", "401", "403", "404", "409", "413", "415", "422", "429", "500", "503", "504"}
UNITS = {"ms", "s", "seconds", "second", "characters", "chars", "files", "documents", "queries", "questions", "chunks",
         "rows", "tokens", "requests", "attempts", "gb", "mb", "mib", "kb", "%", "x"}
GENERATED_START = re.compile(r"<!--\s*([\w-]+):start\b")

# The engineering reference's numbers that are not results, as phrases that contain them, with the reason.
NOT_RESULTS: dict[str, tuple[tuple[str, str], ...]] = {
    DOC: (
        ("independent 2026 extensions", "a year"),
        ("PostgreSQL 16/pgvector", "the PostgreSQL major version"),
        ("October 7", "the day of the October evaluation, which names its results directory"),
        ("family-bootstrap 95%", "the interval's confidence level, a method setting"),
        ("Python 3.11 or 3.12", "the Python versions a clean clone supports"),
        ("Python 3.11 and 3.12", "the Python versions CI runs"),
        ("run 47", "a run id of the fixture corpus"),
        ("run 40", "a run id of the fixture corpus"),
        ("Tier 1", "the name of the first staleness tier"),
        ("Tier 2", "the name of the second staleness tier"),
        ("every weight 1.0", "the list weight that defines the unweighted fusion"),
        ("mode 0600", "a file mode"),
        ("migration `0003`", "a migration's number"),
        ("migration `0004`", "a migration's number"),
        ("migration `0005`", "a migration's number"),
        ("2023–2026 RAG research", "years"),
        ("exits 0", "an exit status"),
        ("exits 1", "an exit status"),
        ("0 waits without a limit", "what the setting's value 0 means"),
        ("kappa is 0 by construction", "a property of Cohen's kappa on a one-class reference"),
        ("gold `8` inside `48`", "example strings of the exact-match rule"),
        # The injection section is being revised by another milestone (M32); its settings are classified here rather
        # than cited inline, so the two edits do not collide. They are the harness's own settings, not results.
        ("search snippets cut at 500 characters, results at 4,000", "the injection harness's output limits"),
        ("4,000 characters and flagged", "the injection harness's output limit"),
        ("k = 4 sampled trials (temperature 0.7", "the injection run's design: trials per task and temperature"),
        ("7-8B models", "the models' sizes, part of their names"),
        ("et al. 2023", "a citation's year"),
        ("et al. 2024", "a citation's year"),
        ("et al. 2025", "a citation's year"),
        ("ICLR 2026", "a citation's year"),
        ("the 2024 Next.js-13 demo list", "a year"),
        ("the constant 1", "what the feature equals when every mtime is the same"),
        ("below the `0.0550` baseline", "a value written in the fixture corpus's configs"),
        ("`0.055` reference threshold", "a value written in the fixture corpus"),
        ("(unset or 0: no limit; 1: no retry)", "what the setting's values 0 and 1 mean"),
        ("missing data table as 0 rows", "a described behaviour"),
        ("with 0 rows while", "a described behaviour"),
        ("psycopg2 2.9", "a library version"),
        ("pgvector 0.8 or later", "a library version"),
        ("Phase 10", "a section number of PLAN.md"),
    ),
}


@dataclass
class Token:
    line: int
    text: str
    start: int  # offset in the line as written (comments included)
    end: int


@dataclass
class Problem:
    line: int
    message: str

    def __str__(self) -> str:
        return f"{DOC}:{self.line}: {self.message}"


def _blank(text: str, pattern: str) -> str:
    """The text with every match of pattern replaced by spaces of the same length (offsets stay put)."""
    return re.sub(pattern, lambda m: " " * len(m.group(0)), text)


def prose_lines(text: str) -> list[tuple[int, str]]:
    """(line number, line) for every line outside fenced code blocks and generated blocks."""
    out, fence, generated = [], None, None
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if generated:
            if re.search(rf"<!--\s*{re.escape(generated)}:end\b", line):
                generated = None
            continue
        if fence:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            continue
        start = GENERATED_START.search(line)
        if start and not re.search(rf"<!--\s*{re.escape(start.group(1))}:end\b", line):
            generated = start.group(1)
            continue
        out.append((number, line))
    return out


def numbers_in(line: str, number: int) -> list[Token]:
    """The prose numbers of one line, with their offsets: outside inline code, link targets, HTML tags and comments."""
    masked = _blank(line, r"<!--.*?-->")
    masked = re.sub(r"`[^`]*`", lambda m: m.group(0) if NUMERIC_CODE.fullmatch(m.group(0)) else " " * len(m.group(0)), masked)
    masked = _blank(masked, r"\]\([^)]*\)")
    masked = _blank(masked, r"<[^>\n]+>")
    return [Token(number, m.group(0), m.start(), m.end()) for m in NUMBER.finditer(masked)]


def resolve_pointer(doc, pointer: str):
    """RFC 6901, plus [field=value] segments that select the one list element whose field equals value."""
    if pointer in ("", "#"):
        return doc
    if not pointer.startswith("#/"):
        raise ValueError(f"a JSON pointer starts with #/: {pointer!r}")
    cur = doc
    for raw in pointer[2:].split("/"):
        seg = raw.replace("~1", "/").replace("~0", "~")
        selector = re.fullmatch(r"\[([^=\]]+)=([^\]]*)\]", seg)
        if selector:
            field, want = selector.groups()
            if not isinstance(cur, list):
                raise ValueError(f"[{field}={want}] needs a list")
            hits = [x for x in cur if isinstance(x, dict) and field in x and str(x[field]) == want]
            if len(hits) != 1:
                raise ValueError(f"[{field}={want}] selects {len(hits)} elements, not one")
            cur = hits[0]
        elif isinstance(cur, list):
            if not seg.isdigit() or int(seg) >= len(cur):
                raise ValueError(f"no element {seg!r} in a list of {len(cur)}")
            cur = cur[int(seg)]
        elif isinstance(cur, dict):
            if seg not in cur:
                raise ValueError(f"no key {seg!r}")
            cur = cur[seg]
        else:
            raise ValueError(f"cannot descend into a {type(cur).__name__} with {seg!r}")
    return cur


def parse_specs(body: str) -> list[dict]:
    """The specs of one citation comment, in order; a spec that does not parse raises ValueError."""
    out, pos, body = [], 0, body.strip()
    while pos < len(body):
        m = SPEC.match(body, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot read a spec at {body[pos:pos + 40]!r}")
        out.append({k: v for k, v in m.groupdict().items() if v is not None})
        pos = m.end()
    return out


def cited_value(spec: dict, root: Path, last_file: str | None, cache: dict) -> tuple[float, str]:
    """(the number a spec cites, the file it read)."""
    if "src" in spec:
        rel = spec["src"]
        pattern = spec["pat"].replace('\\"', '"')
        if pattern.count("{}") != 1:
            raise ValueError(f'a source pattern holds exactly one {{}}: {pattern!r}')
        before, after = pattern.split("{}")
        regex = re.compile(re.escape(before) + r"([+\-]?\d[\d_,]*(?:\.\d+)?)" + re.escape(after))
        for line in _read(root, rel, cache, text=True).splitlines():
            m = regex.search(line)
            if m:
                return float(m.group(1).replace("_", "").replace(",", "")), rel
        raise ValueError(f"no line of {rel} matches {pattern!r}")
    rel = spec.get("file") or last_file
    if not rel:
        raise ValueError("a spec without a file must follow one with a file on the same line")
    value = resolve_pointer(_read(root, rel, cache), spec["ptr"])
    op = spec.get("op", "")
    if op == "len":
        if not isinstance(value, (list, dict)):
            raise ValueError("|len needs a list or an object")
        value = len(value)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"the pointer names a {type(value).__name__}, not a number")
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"the pointer names the text {value!r}, not a number") from None
    if op[:1] == "/":
        number /= float(op[1:])
    elif op[:1] == "*":
        number *= float(op[1:])
    return number, rel


def _read(root: Path, rel: str, cache: dict, text: bool = False):
    key = (rel, text)
    if key not in cache:
        path = (root / rel).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"{rel} is outside the repository")
        if not path.is_file():
            raise ValueError(f"no file {rel}")
        raw = path.read_text(encoding="utf-8")
        cache[key] = raw if text else json.loads(raw)
    return cache[key]


def matches(prose: str, value: float) -> bool:
    """Whether the prose number prints the value: same sign, same digits at the prose's precision."""
    shown = prose.replace(",", "")
    decimals = len(shown.split(".", 1)[1]) if "." in shown else 0
    if shown.startswith("+") and value < 0:
        return False
    formatted = f"{value:.{decimals}f}"
    if formatted.startswith("-") and set(formatted[1:]) <= {"0", "."}:
        formatted = formatted[1:]  # -0.000 prints as 0.000
    return formatted == shown.lstrip("+")


def _next_word(line: str, end: int) -> str:
    m = re.match(r"\s*([A-Za-z%]+)", line[end:])
    return m.group(1).lower() if m else ""


def _classified_by_rule(line: str, token: Token) -> str | None:
    """Why a number is not a result, by a rule of this module, or None."""
    if re.match(r"\s*(?:[-*]\s+)?$", line[: token.start]) and re.match(r"\.\s", line[token.end:]):
        return "a list marker"
    in_a_list = line[: token.start].rstrip().endswith("/") or line[token.end:].lstrip().startswith("/")
    if token.text in HTTP_STATUS and not in_a_list and _next_word(line, token.end) not in UNITS:
        return "an HTTP status code"
    return None


def check(text: str, root: Path = ROOT, not_results: tuple[tuple[str, str], ...] | None = None) -> list[Problem]:
    """Every problem with the numbers of one document's prose: a cited number that differs from its source, a
    citation that does not resolve, a number neither cited nor classified, a NOT_RESULTS phrase that no longer occurs."""
    phrases = NOT_RESULTS.get(DOC, ()) if not_results is None else not_results
    problems: list[Problem] = []
    cache: dict = {}
    used_phrases: set[str] = set()
    for number, line in prose_lines(text):
        tokens = numbers_in(line, number)
        cited: set[int] = set()
        previous_end = 0  # a citation claims only numbers after the previous citation on the line
        last_file = None  # the file a spec that starts with # reads: the one the previous spec on the line named
        for comment in CITATION.finditer(line):
            if not line[: comment.start()].strip():
                problems.append(Problem(number, "a citation must follow its number on the same line (a line that starts "
                                                "with an HTML comment ends the paragraph on GitHub)"))
            before = [i for i, t in enumerate(tokens) if previous_end <= t.start and t.end <= comment.start()]
            previous_end = comment.end()
            try:
                specs = parse_specs(comment.group("body"))
            except ValueError as exc:
                problems.append(Problem(number, f"citation {comment.group(0)!r}: {exc}"))
                continue
            if len(specs) > len(before):
                problems.append(Problem(number, f"citation names {len(specs)} numbers but only {len(before)} precede it "
                                                f"on its line: {comment.group(0)!r}"))
                continue
            for index, spec in zip(before[len(before) - len(specs):], specs):
                token = tokens[index]
                cited.add(index)
                try:
                    value, last_file = cited_value(spec, root, last_file, cache)
                except (ValueError, json.JSONDecodeError) as exc:
                    problems.append(Problem(number, f"{token.text}: citation {spec} does not resolve: {exc}"))
                    continue
                if not matches(token.text, value):
                    problems.append(Problem(number, f"{token.text} differs from its source {spec}: the value is {value!r}"))
        masked_line = _blank(line, r"<!--.*?-->")
        for index, token in enumerate(tokens):
            if index in cited or _classified_by_rule(line, token):
                continue
            covering = [p for p, _ in phrases if any(m.start() <= token.start and token.end <= m.end()
                                                     for m in re.finditer(re.escape(p), masked_line))]
            if covering:
                used_phrases.update(covering)
                continue
            problems.append(Problem(number, f"{token.text} is neither cited (<!-- num: ... -->) nor classified as not a "
                                            f"result (app/eval/prose_numbers.NOT_RESULTS): ...{line[max(0, token.start - 40):token.end + 30]}..."))
    for phrase, _ in phrases:
        if phrase not in used_phrases:
            problems.append(Problem(0, f"NOT_RESULTS phrase no longer classifies a number of the prose: {phrase!r}"))
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    problems = check((ROOT / DOC).read_text(encoding="utf-8"))
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} problem(s) with the numbers of {DOC}", file=sys.stderr)
        return 1
    print(f"every number in the prose of {DOC} is cited and matches its source, or is classified as not a result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
