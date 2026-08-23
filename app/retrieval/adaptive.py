"""Adaptive retrieval: decide *whether* to retrieve before spending compute.

Not every query needs the full hybrid pipeline.  Greetings, meta-questions
about the system, and queries the model can answer from parametric knowledge
alone should skip retrieval entirely.

This is a latency optimisation, not an accuracy change — on the gold set
it should be neutral (every gold question *does* need retrieval).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class RetrievalDecision(Enum):
    RETRIEVE = "retrieve"
    SKIP = "skip"
    UNCERTAIN = "uncertain"


@dataclass
class AdaptiveResult:
    decision: RetrievalDecision
    reason: str
    confidence: float  # 0..1


_SKIP_PATTERNS = [
    (re.compile(r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b", re.I),
     "greeting"),
    (re.compile(r"^\s*(thanks?|thank\s+you|thx)\b", re.I),
     "thanks"),
    (re.compile(r"^\s*(?:who|what)\s+are\s+you\b", re.I),
     "meta_identity"),
    (re.compile(r"^\s*(?:what\s+can\s+you\s+do|help|how\s+do\s+(?:I|you)\s+use)\b", re.I),
     "meta_capability"),
    (re.compile(r"^\s*(?:define|what\s+(?:is|are)\s+(?:a|an|the)\s+)"
                r"(?:neural\s+network|gradient\s+descent|backprop|transformer|"
                r"attention|convolution|rnn|lstm|gan|vae|diffusion)\b", re.I),
     "textbook_concept"),
]

_RETRIEVE_SIGNALS = [
    re.compile(r"run[_\s]?\d+", re.I),
    re.compile(r"config", re.I),
    re.compile(r"\.yaml|\.py|\.csv|\.sbatch|\.out", re.I),
    re.compile(r"learning[_\s]?rate|lr\b|rmse|epoch|encoder|fusion", re.I),
    re.compile(r"file|document|corpus|data|log|script|draft|paper", re.I),
    re.compile(r"which|where|show|open|list|find|search|retrieve", re.I),
    re.compile(r"current|latest|archive|version", re.I),
]


def should_retrieve(query: str) -> AdaptiveResult:
    """Classify whether the query needs corpus retrieval.

    Returns SKIP for greetings/meta/textbook, RETRIEVE for corpus-specific
    queries, and UNCERTAIN when signals are mixed.
    """
    q = (query or "").strip()
    if not q:
        return AdaptiveResult(RetrievalDecision.SKIP, "empty_query", 1.0)

    for pattern, reason in _SKIP_PATTERNS:
        if pattern.search(q):
            return AdaptiveResult(RetrievalDecision.SKIP, reason, 0.9)

    retrieve_hits = sum(1 for p in _RETRIEVE_SIGNALS if p.search(q))
    if retrieve_hits >= 2:
        return AdaptiveResult(RetrievalDecision.RETRIEVE, "strong_corpus_signal", 0.95)
    if retrieve_hits == 1:
        return AdaptiveResult(RetrievalDecision.RETRIEVE, "corpus_signal", 0.7)

    word_count = len(q.split())
    if word_count <= 3:
        has_question = q.rstrip().endswith("?")
        if not has_question:
            return AdaptiveResult(RetrievalDecision.UNCERTAIN, "short_ambiguous", 0.4)

    return AdaptiveResult(RetrievalDecision.RETRIEVE, "default_retrieve", 0.6)
