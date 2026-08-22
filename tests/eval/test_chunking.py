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
