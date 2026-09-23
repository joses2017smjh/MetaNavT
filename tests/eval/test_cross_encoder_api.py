"""The reranker must work with sentence-transformers (predict) and FlagEmbedding (compute_score),
and return one documented scale: sigmoid probabilities in [0, 1]."""

import math

import pytest

from app.retrieval.rerank import SCORE_SCALE, CrossEncoderReranker, cross_encoder_scores
from app.retrieval.types import Chunk

LOGITS = {"ab": 0.0, "abcd": 2.0}


class _STModel:  # sentence_transformers.CrossEncoder: predict() with an activation override
    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        raw = [LOGITS.get(p[1], len(p[1]) / 10.0) for p in pairs]
        if activation_fn is None:  # library default: sigmoid
            return [1 / (1 + math.exp(-x)) for x in raw]
        return [activation_fn(x) for x in raw]


class _OldSTModel:  # sentence-transformers < 5 spelled it activation_fct
    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fct=None):
        raw = [LOGITS.get(p[1], len(p[1]) / 10.0) for p in pairs]
        return [activation_fct(x) if activation_fct else 1 / (1 + math.exp(-x)) for x in raw]


class _FlagModel:  # FlagEmbedding.FlagReranker: compute_score(normalize=...)
    def compute_score(self, pairs, normalize=False):
        raw = [LOGITS.get(p[1], len(p[1]) / 10.0) for p in pairs]
        return [1 / (1 + math.exp(-x)) for x in raw] if normalize else raw


class _FlagRawOnly:  # older FlagEmbedding: no normalize kwarg, raw logits
    def compute_score(self, pairs):
        return [LOGITS.get(p[1], len(p[1]) / 10.0) for p in pairs]


class _Neither:
    pass


def _chunk(i, text):
    return Chunk(chunk_id=f"c{i}", path=f"p{i}", text=text, start_byte=0, end_byte=len(text))


@pytest.mark.parametrize("model", [_STModel(), _OldSTModel(), _FlagModel(), _FlagRawOnly()])
def test_every_api_yields_sigmoid_probabilities(model):
    scores = cross_encoder_scores(model, [("q", "ab"), ("q", "abcd")])
    assert scores == pytest.approx([0.5, 1 / (1 + math.exp(-2.0))], abs=1e-9)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert "sigmoid" in SCORE_SCALE


@pytest.mark.parametrize("model", [_STModel(), _FlagModel()])
def test_ranking_is_by_score(model):
    pairs = [(_chunk(1, "ab"), 0.0), (_chunk(2, "abcd"), 0.0)]
    ranked = CrossEncoderReranker(model=model)(query="q", pairs=pairs)
    assert [c.chunk_id for c, _ in ranked] == ["c2", "c1"]


def test_scalar_and_empty_inputs():
    class Scalar:
        def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
            return 0.0

    assert cross_encoder_scores(Scalar(), [("q", "x")]) == [0.5]
    assert cross_encoder_scores(_STModel(), []) == []


def test_unknown_model_type_is_a_type_error():
    with pytest.raises(TypeError):
        cross_encoder_scores(_Neither(), [("q", "x")])
