"""The reranker must work with sentence-transformers (predict) and FlagEmbedding (compute_score)."""

import pytest

from app.retrieval.rerank import CrossEncoderReranker, cross_encoder_scores
from app.retrieval.types import Chunk


class _STModel:  # sentence_transformers.CrossEncoder shape
    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return [len(p[1]) / 10.0 for p in pairs]


class _FlagModel:  # FlagEmbedding.FlagReranker shape
    def compute_score(self, pairs):
        return [len(p[1]) / 10.0 for p in pairs]


class _Neither:
    pass


def _chunk(i, text):
    return Chunk(chunk_id=f"c{i}", path=f"p{i}", text=text, start_byte=0, end_byte=len(text))


@pytest.mark.parametrize("model", [_STModel(), _FlagModel()])
def test_both_apis_score_and_rank(model):
    assert cross_encoder_scores(model, [("q", "ab"), ("q", "abcd")]) == [0.2, 0.4]
    pairs = [(_chunk(1, "short"), 0.0), (_chunk(2, "a much longer passage"), 0.0)]
    ranked = CrossEncoderReranker(model=model)(query="q", pairs=pairs)
    assert [c.chunk_id for c, _ in ranked] == ["c2", "c1"]


def test_scalar_and_empty_inputs():
    class Scalar:
        def predict(self, pairs, batch_size=32, show_progress_bar=False):
            return 0.7

    assert cross_encoder_scores(Scalar(), [("q", "x")]) == [0.7]
    assert cross_encoder_scores(_STModel(), []) == []


def test_unknown_model_type_is_a_type_error():
    with pytest.raises(TypeError):
        cross_encoder_scores(_Neither(), [("q", "x")])
