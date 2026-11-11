import json
from pathlib import Path

from app.chunking.contextual import prepend_context, situate_chunk
from app.chunking.late import late_chunk_vectors, mean_pool_spans
from app.chunking.structure import chunk_text, csv_record_chunks, python_ast_chunks
from app.chunking.structure import Span, fixed_size_chunks, jsonl_record_chunks, markdown_heading_chunks, yaml_chunks
import numpy as np
import pytest


def _spans(spans: list[Span]) -> list[tuple[int, int, str]]:
    return [(s.start, s.end, s.kind) for s in spans]


def test_csv_one_record_per_chunk():
    text = "ablation,rmse\nfusion_off,0.06\nfusion_on,0.05\n"
    spans = csv_record_chunks(text)
    kinds = [s.kind for s in spans]
    assert "csv_header" in kinds
    assert kinds.count("csv_row") == 2
    assert "fusion_off" in spans[1].text and "ablation" in spans[1].text


def test_python_ast_splits_functions():
    src = "x = 1\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
    spans = python_ast_chunks(src)
    kinds = {s.kind for s in spans}
    assert "ast" in kinds
    texts = " ".join(s.text for s in spans)
    assert "def foo" in texts and "def bar" in texts


MODULE = '''"""Run helpers."""
import os


def first():
    return 1


RATE = 3e-4  # module-level code between two definitions
register(first)


@decorator
@other.decorator(arg=1)
def second():
    return 2


@dataclass
class Config:
    lr: float = RATE


if __name__ == "__main__":
    main()
'''


def _non_blank_lines(text: str) -> list[tuple[int, int, str]]:
    out, offset = [], 0
    for line in text.splitlines(keepends=True):
        if line.strip():
            out.append((offset, offset + len(line.rstrip("\r\n")), line))
        offset += len(line)
    return out


def test_python_chunks_cover_every_non_blank_line_exactly_once():
    spans = python_ast_chunks(MODULE)

    for start, end, line in _non_blank_lines(MODULE):
        touching = [s for s in spans if s.start < end and start < s.end]
        assert len(touching) == 1 and touching[0].start <= start and end <= touching[0].end, line
    assert [s.text for s in spans] == [
        '"""Run helpers."""\nimport os\n\n\n',
        "def first():\n    return 1\n",
        "RATE = 3e-4  # module-level code between two definitions\nregister(first)\n\n\n",
        "@decorator\n@other.decorator(arg=1)\ndef second():\n    return 2\n",  # a definition starts at its first decorator
        "@dataclass\nclass Config:\n    lr: float = RATE\n",
        'if __name__ == "__main__":\n    main()\n',
    ]
    assert all(s.kind == "ast" and s.text == MODULE[s.start:s.end] for s in spans)


def test_python_chunks_count_lines_the_way_ast_does():
    """A form feed or a U+2028 inside a string is not a line break for Python, though str.splitlines splits there."""
    src = "x = 1\n\x0c\ndef f():\n    return \"a\u2028b\"\n\n\ny = 2\n"
    assert [s.text for s in python_ast_chunks(src)] == ["x = 1\n\x0c\n", "def f():\n    return \"a\u2028b\"\n", "y = 2\n"]
    old_mac = "x = 1\rdef f():\r    pass\r"  # \r alone ends a line for Python as well
    assert [s.text for s in python_ast_chunks(old_mac)] == ["x = 1\r", "def f():\r    pass\r"]


def test_a_module_without_definitions_keeps_the_fixed_window():
    src = '"""Constants only."""\n\nTRELLIS_DEFAULT = True\n'
    assert [(s.start, s.end, s.kind) for s in python_ast_chunks(src)] == [(0, len(src), "fixed")]


def test_auto_strategy_by_extension():
    md = chunk_text("# A\nhello\n# B\nworld\n", path="paper/draft.md")
    assert any(s.kind == "markdown" for s in md)
    yml = chunk_text("run_id: 47\nlearning_rate: 3e-4\n", path="configs/run.yaml")
    assert yml and yml[0].kind == "yaml"


def test_fixed_window_spans_overlap_and_skip_blank_windows():
    text = "abcdefghijklmnopqrstuvwxyz"
    assert _spans(fixed_size_chunks(text, size=10, overlap=3)) == [(0, 10, "fixed"), (7, 17, "fixed"), (14, 24, "fixed"), (21, 26, "fixed")]
    assert _spans(fixed_size_chunks(text, size=10, overlap=10)) == [(0, 10, "fixed"), (8, 18, "fixed"), (16, 26, "fixed")]  # overlap >= size: size // 5
    assert _spans(fixed_size_chunks("ab" + " " * 10 + "cd", size=4, overlap=0)) == [(0, 4, "fixed"), (12, 14, "fixed")]  # blank windows dropped
    assert fixed_size_chunks("") == []
    with pytest.raises(ValueError):
        fixed_size_chunks(text, size=0)


def test_unknown_suffixes_take_the_default_fixed_window():
    text = "".join(f"line {i:04d}\n" for i in range(110))  # 1100 characters
    assert _spans(chunk_text(text, path="logs/run_047.out")) == [(0, 512, "fixed"), (462, 974, "fixed"), (924, 1100, "fixed")]
    assert _spans(chunk_text(text, path="configs/run.yaml", strategy="fixed")) == _spans(chunk_text(text, path="x.out"))
    with pytest.raises(ValueError):
        chunk_text(text, path="x.out", strategy="semantic")


def test_json_is_one_chunk_and_invalid_json_falls_back_to_records():
    text = '{"run_id": 47, "val_rmse": 0.0521}\n'
    assert chunk_text(text, path="checkpoints/run_047.ckpt.meta.json") == [Span(0, len(text), text, "json")]
    records = '{"a": 1}\n\n{"b": 2}\n'  # not one JSON value: one chunk per non-blank line
    assert _spans(chunk_text(records, path="x.json")) == [(0, 9, "jsonl"), (10, 19, "jsonl")]
    assert chunk_text("  \n", path="x.json") == []


def test_jsonl_is_one_chunk_per_record():
    text = '{"q": 1}\n\n{"q": 2}\r\n{"q": 3}'
    spans = jsonl_record_chunks(text)
    assert _spans(spans) == [(0, 9, "jsonl"), (10, 20, "jsonl"), (20, 28, "jsonl")]
    assert [s.text for s in spans] == [text[s.start:s.end] for s in spans]
    assert chunk_text(text, path="bench/gold/questions.jsonl") == spans


def test_markdown_splits_at_headings_and_recuts_a_long_section():
    text = "# A\nhello\n## B\nworld\n"
    assert _spans(markdown_heading_chunks(text)) == [(0, 10, "markdown"), (10, 21, "markdown")]
    long = "# A\n" + "x" * 2500 + "\n"
    assert _spans(markdown_heading_chunks(long)) == [(0, 2000, "markdown"), (1920, len(long), "markdown")]  # window 2000, overlap 80


def test_a_heading_inside_fenced_code_does_not_start_a_section():
    text = "# Setup\nRun:\n```bash\n# install the deps\npip install -e .\n```\n~~~\n## not a heading either\n~~~\n## Usage\ncall it\n"
    usage = text.index("## Usage")
    assert _spans(markdown_heading_chunks(text)) == [(0, usage, "markdown"), (usage, len(text), "markdown")]
    nested = "# A\n````\n```\n# still code: a shorter fence does not close a longer one\n````\n# B\n"
    b = nested.index("# B")
    assert _spans(markdown_heading_chunks(nested)) == [(0, b, "markdown"), (b, len(nested), "markdown")]
    unclosed = "# A\n```\n# code to the end of the file\n"
    assert _spans(markdown_heading_chunks(unclosed)) == [(0, len(unclosed), "markdown")]
    inline = "# A\n```inline``` code is not a fence\n# B\n"
    assert len(markdown_heading_chunks(inline)) == 2
    separator = "# A\nintro\u2028# not a heading: U+2028 does not end a Markdown line\n"
    assert _spans(markdown_heading_chunks(separator)) == [(0, len(separator), "markdown")]


def test_rst_takes_the_markdown_rule_without_underline_headings():
    """A known limit: an .rst title is underlined, not marked with #, so it does not start a chunk."""
    text = "Title\n=====\n\nIntro.\n\nSection\n-------\n\nBody.\n"
    assert _spans(chunk_text(text, path="docs/index.rst")) == [(0, len(text), "markdown")]


def test_small_yaml_is_one_chunk_and_a_long_one_splits_at_top_level_keys():
    small = "run_id: 47\nlearning_rate: 3e-4\n"
    assert yaml_chunks(small) == [Span(0, len(small), small, "yaml")]
    head = "run_id: 47\n"
    blocks = [head, "model:\n" + "".join(f"  layer_{i}: 1\n" for i in range(90)), "train:\n" + "".join(f"  step_{i}: 2\n" for i in range(90))]
    text = "".join(blocks)
    assert len(text) >= 2000
    spans = yaml_chunks(text)
    starts = [0, len(blocks[0]), len(blocks[0]) + len(blocks[1])]
    assert _spans(spans) == [(starts[0], starts[1], "yaml"), (starts[1], starts[2], "yaml"), (starts[2], len(text), "yaml")]
    assert [s.text for s in spans] == [head, head + blocks[1], head + blocks[2]]  # run_id travels with every block


def test_csv_rows_keep_crlf_and_skip_blank_lines():
    text = "h1,h2\r\na,1\r\n\r\nb,2\r\n"
    spans = csv_record_chunks(text)
    assert _spans(spans) == [(0, 7, "csv_header"), (7, 12, "csv_row"), (14, 19, "csv_row")]
    assert [s.text for s in spans] == ["h1,h2\r\n", "h1,h2\r\na,1\r\n", "h1,h2\r\nb,2\r\n"]


def test_a_json_value_longer_than_the_limit_is_recut_with_the_window():
    text = json.dumps({"rows": [{"i": i, "note": "x" * 20} for i in range(100)]}, indent=1)
    assert len(text) > 4000

    spans = chunk_text(text, path="results/sweep.json")

    assert _spans(spans) == [(s.start, s.end, "json") for s in fixed_size_chunks(text, size=2000, overlap=80)]
    assert len(spans) > 1 and all(s.end - s.start <= 2000 and s.text == text[s.start:s.end] for s in spans)


def test_a_python_definition_longer_than_the_limit_is_recut_with_the_window():
    body = "".join(f"    field_{i:03d}: int = {i}\n" for i in range(120))
    src = "import os\n\n\nclass Big:\n" + body + "\n\ndef small():\n    return 1\n"
    start, end, small = src.index("class Big"), src.index("\n\ndef small"), src.index("def small")
    windows = [(start + s.start, start + s.end, "ast") for s in fixed_size_chunks(src[start:end], size=2000, overlap=80)]

    spans = python_ast_chunks(src)

    assert len(windows) == 2 and _spans(spans) == [(0, start, "ast"), *windows, (small, len(src), "ast")]
    assert all(s.end - s.start <= 2000 for s in spans)


def test_a_parser_that_gives_up_does_not_abort_the_chunker():
    """ast and json raise RecursionError on deep nesting (3.11, 3.12); one such file must not stop an indexing run."""
    deep_py = "x = " + "1+" * 100000 + "1\n"
    assert _spans(python_ast_chunks(deep_py)) == _spans(fixed_size_chunks(deep_py))
    nul = "def f():\n    return 1\x00\n"  # null bytes: SyntaxError, or ValueError on older interpreters
    assert _spans(python_ast_chunks(nul)) == _spans(fixed_size_chunks(nul))
    deep_json = "[" * 100000 + "]" * 100000  # one JSON value too deep to decode: cut like any long value
    assert _spans(chunk_text(deep_json, path="results/deep.json")) == [
        (s.start, s.end, "json") for s in fixed_size_chunks(deep_json, size=2000, overlap=80)
    ]


def test_csv_records_keep_a_quoted_line_break_in_one_row():
    header, row, last = "run,note\r\n", '47,"fusion on\r\nsecond line"\r\n', "48,plain\r\n"
    text = header + row + last

    spans = csv_record_chunks(text)

    assert _spans(spans) == [(0, len(header), "csv_header"), (len(header), len(header + row), "csv_row"),
                             (len(header + row), len(text), "csv_row")]
    assert [s.text for s in spans[1:]] == [header + row, header + last]


def test_a_stray_quote_does_not_merge_the_rest_of_a_csv_into_one_row():
    unclosed = 'run,note\n47,"unclosed\n48,b\n49,c\n'  # never closed: strict parsing fails
    assert [s.text for s in csv_record_chunks(unclosed)] == ["run,note\n", 'run,note\n47,"unclosed\n', "run,note\n48,b\n", "run,note\n49,c\n"]
    closed_late = 'run,note\n47,"a\n48,b",x,y\n49,c\n'  # parses, but the two-line record has four fields, not two
    assert [s.end - s.start for s in csv_record_chunks(closed_late)] == [len(line) for line in closed_late.splitlines(keepends=True)]


def test_csv_header_is_the_first_non_blank_record():
    assert [s.text for s in csv_record_chunks("\nh1,h2\na,1\n")] == ["h1,h2\n", "h1,h2\na,1\n"]


def test_jsonl_records_split_only_at_line_ends():
    first, second = '{"note": "a\u2028b"}\n', '{"note": "c\x85d"}\n'  # both are legal inside a JSON string
    assert _spans(jsonl_record_chunks(first + second)) == [(0, len(first), "jsonl"), (len(first), len(first + second), "jsonl")]


def test_late_chunk_mean_pool():
    tokens = np.vstack([np.ones((4, 3)), np.zeros((4, 3))]).astype(np.float32)
    pooled = mean_pool_spans(tokens, [(0, 4), (4, 8)])
    assert pooled.shape == (2, 3)
    assert pooled[0].mean() > pooled[1].mean()


def test_late_chunk_from_char_spans():
    text = "alpha beta gamma delta"
    spans = [Span(0, 10, text[:10], "fixed"), Span(11, len(text), text[11:], "fixed")]
    tok = np.random.RandomState(0).randn(4, 8).astype(np.float32)
    vecs = late_chunk_vectors(tok, text, spans)
    assert vecs.shape == (2, 8)


def test_contextual_prepend(monkeypatch):
    span = Span(0, 5, "chunk", "fixed")
    out = situate_chunk(
        "configs/run.yaml",
        "full document text",
        span,
        llm=lambda prompt: "This chunk is the learning rate field of run 47.",
    )
    assert "learning rate" in out.context
    assert out.embedded_text.startswith(out.context)
    assert prepend_context("body", "ctx") == "ctx\n\nbody"
