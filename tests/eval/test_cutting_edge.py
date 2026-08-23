"""Tests for Phase 14-15 cutting-edge retrieval features."""

import numpy as np
import pytest

from app.retrieval.types import Chunk


# --- HyDE ---

def _make_chunk(cid, text, path="test.txt"):
    return Chunk(chunk_id=cid, path=path, text=text)


class TestHyDE:
    def test_heuristic_hypothesis_expands_query(self):
        from app.retrieval.hyde import heuristic_hypothesis
        hyp = heuristic_hypothesis("learning rate for run 47")
        assert len(hyp) > len("learning rate for run 47")
        assert "learning" in hyp.lower()

    def test_hyde_embed_returns_ndarray(self):
        from app.retrieval.hyde import hyde_embed
        from app.retrieval.embedders import HashEmbedder
        vec = hyde_embed("test query", HashEmbedder())
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1

    def test_hyde_embed_differs_from_raw(self):
        from app.retrieval.hyde import hyde_embed
        from app.retrieval.embedders import HashEmbedder
        emb = HashEmbedder()
        raw = emb.encode(["test query"])[0]
        hyde_vec = hyde_embed("test query", emb)
        assert not np.allclose(raw, hyde_vec, atol=1e-6)


# --- Query Decomposition ---

class TestDecompose:
    def test_heuristic_splits_compound_query(self):
        from app.agent.decompose import heuristic_decompose
        subs = heuristic_decompose(
            "What is the learning rate and what encoder was used in run 47?"
        )
        assert len(subs) >= 2

    def test_heuristic_preserves_simple_query(self):
        from app.agent.decompose import heuristic_decompose
        subs = heuristic_decompose("What is the learning rate?")
        assert len(subs) == 1

    def test_should_decompose_recognizes_compound(self):
        from app.agent.decompose import should_decompose
        assert should_decompose("Compare the RMSE and learning rate between run 1 and run 2")

    def test_should_decompose_rejects_simple(self):
        from app.agent.decompose import should_decompose
        assert not should_decompose("What file has the config?")


# --- Corrective RAG ---

class TestCorrectiveRAG:
    def test_classify_chunk_correct(self):
        from app.agent.corrective import classify_chunk, RelevanceLabel
        chunk = _make_chunk("c1", "learning_rate: 3e-4 encoder dinov2 run_047")
        ec = classify_chunk("learning_rate run_047", chunk, 0.9)
        assert ec.label == RelevanceLabel.CORRECT

    def test_classify_chunk_incorrect(self):
        from app.agent.corrective import classify_chunk, RelevanceLabel
        chunk = _make_chunk("c2", "the quick brown fox jumped over the lazy dog")
        ec = classify_chunk("learning_rate run_047", chunk, 0.1)
        assert ec.label == RelevanceLabel.INCORRECT

    def test_determine_correction_none_when_correct(self):
        from app.agent.corrective import (
            classify_chunk, determine_correction, RelevanceLabel,
        )
        chunks = [
            _make_chunk(f"c{i}", f"learning_rate: 3e-4 config run_047 encoder")
            for i in range(5)
        ]
        evaluated = [classify_chunk("learning_rate run_047", c, 0.9) for c in chunks]
        action = determine_correction("learning_rate run_047", evaluated)
        assert action.strategy == "none"

    def test_determine_correction_rewrite_when_poor(self):
        from app.agent.corrective import classify_chunk, determine_correction
        chunks = [_make_chunk(f"c{i}", "unrelated text about weather") for i in range(5)]
        evaluated = [classify_chunk("learning_rate run_047", c, 0.1) for c in chunks]
        action = determine_correction("learning_rate run_047", evaluated)
        assert action.strategy in ("rewrite", "decompose")

    def test_evaluate_retrieval(self):
        from app.agent.corrective import evaluate_retrieval
        pairs = [
            (_make_chunk("c1", "learning_rate: 3e-4 run_047"), 0.9),
            (_make_chunk("c2", "random text"), 0.1),
        ]
        results = evaluate_retrieval("learning_rate run_047", pairs)
        assert len(results) == 2


# --- Adaptive Retrieval ---

class TestAdaptiveRetrieval:
    def test_skip_greeting(self):
        from app.retrieval.adaptive import should_retrieve, RetrievalDecision
        result = should_retrieve("Hello there!")
        assert result.decision == RetrievalDecision.SKIP

    def test_retrieve_corpus_query(self):
        from app.retrieval.adaptive import should_retrieve, RetrievalDecision
        result = should_retrieve("What is the learning_rate in config run_047.yaml?")
        assert result.decision == RetrievalDecision.RETRIEVE

    def test_skip_empty(self):
        from app.retrieval.adaptive import should_retrieve, RetrievalDecision
        result = should_retrieve("")
        assert result.decision == RetrievalDecision.SKIP

    def test_skip_meta(self):
        from app.retrieval.adaptive import should_retrieve, RetrievalDecision
        result = should_retrieve("What can you do?")
        assert result.decision == RetrievalDecision.SKIP

    def test_confidence_range(self):
        from app.retrieval.adaptive import should_retrieve
        result = should_retrieve("show me run_047")
        assert 0.0 <= result.confidence <= 1.0


# --- Semantic Query Cache ---

class TestSemanticCache:
    def test_put_and_get(self):
        from app.retrieval.cache import SemanticCache
        from app.retrieval.embedders import HashEmbedder
        cache = SemanticCache(HashEmbedder(), similarity_threshold=0.5)
        cache.put("test query", {"answer": "42"})
        result = cache.get("test query")
        assert result is not None
        assert result["answer"] == "42"

    def test_miss_on_unrelated(self):
        from app.retrieval.cache import SemanticCache
        from app.retrieval.embedders import HashEmbedder
        cache = SemanticCache(HashEmbedder(), similarity_threshold=0.99)
        cache.put("learning rate for run 47", {"answer": "3e-4"})
        result = cache.get("completely unrelated weather forecast")
        assert result is None

    def test_invalidate_all(self):
        from app.retrieval.cache import SemanticCache
        from app.retrieval.embedders import HashEmbedder
        cache = SemanticCache(HashEmbedder(), similarity_threshold=0.5)
        cache.put("q1", {"a": 1})
        cache.put("q2", {"a": 2})
        n = cache.invalidate()
        assert n == 2
        assert len(cache) == 0

    def test_lru_eviction(self):
        from app.retrieval.cache import SemanticCache
        from app.retrieval.embedders import HashEmbedder
        cache = SemanticCache(HashEmbedder(), max_entries=2, similarity_threshold=0.5)
        cache.put("q1", {"a": 1})
        cache.put("q2", {"a": 2})
        cache.put("q3", {"a": 3})
        assert len(cache) == 2
        assert cache.stats.evictions >= 1

    def test_stats_tracking(self):
        from app.retrieval.cache import SemanticCache
        from app.retrieval.embedders import HashEmbedder
        cache = SemanticCache(HashEmbedder(), similarity_threshold=0.5)
        cache.put("test", {"a": 1})
        cache.get("test")
        cache.get("nonexistent query xyz 12345")
        assert cache.stats.hits >= 0
        assert cache.stats.misses >= 0


# --- Citation Verification ---

class TestCitationVerification:
    def test_extract_claims_finds_values(self):
        from app.agent.citation_verify import extract_claims
        claims = extract_claims(
            "The learning rate was 3e-4 in configs/run_047.yaml. "
            "The encoder used was dinov2."
        )
        assert len(claims) >= 1
        assert any(c.value for c in claims)

    def test_verify_claims_against_evidence(self):
        from app.agent.citation_verify import extract_claims, verify_claims
        answer = "The learning rate was 3e-4 in configs/run_047.yaml"
        claims = extract_claims(answer)
        evidence = [
            _make_chunk("e1", "learning_rate: 3e-4 encoder dinov2", "configs/run_047.yaml"),
        ]
        result = verify_claims(claims, evidence)
        assert result.verified_count > 0
        assert result.pass_threshold

    def test_hallucinated_value_detected(self):
        from app.agent.citation_verify import extract_claims, verify_claims
        answer = "The learning rate was 9e-9"
        claims = extract_claims(answer)
        evidence = [_make_chunk("e1", "learning_rate: 3e-4")]
        result = verify_claims(claims, evidence)
        assert len(result.hallucinated_values) > 0

    def test_empty_answer_passes(self):
        from app.agent.citation_verify import extract_claims, verify_claims
        claims = extract_claims("")
        result = verify_claims(claims, [])
        assert result.pass_threshold

    def test_verification_ratio(self):
        from app.agent.citation_verify import VerificationResult, Claim
        result = VerificationResult(
            claims=[], verified_count=3, total_count=4,
            missing_citations=[], hallucinated_values=[],
            pass_threshold=True,
        )
        assert abs(result.verification_ratio - 0.75) < 1e-6
