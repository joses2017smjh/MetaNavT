from pathlib import Path

from app.chunking.contextual import prepend_context, situate_chunk
from app.chunking.late import late_chunk_vectors, mean_pool_spans
from app.chunking.structure import chunk_text, csv_record_chunks, python_ast_chunks
from app.chunking.structure import Span
import numpy as np


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
