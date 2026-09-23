"""Citation verification: post-generation gate that checks every claim is sourced.

After the agent produces an answer with citations, this module verifies:
1. Every cited path actually exists in the retrieved evidence
2. Every factual claim in the answer can be traced to a cited chunk
3. No hallucinated values appear (numbers, configs, names not in evidence)

Unverified claims are flagged. If the ratio of verified claims is below
the threshold, the answer is rejected (fail-loud).

Two strictness levels:
- verify_claims(..., cited_only=False): a value counts as verified when it appears
  anywhere in the retrieved evidence (the pre-M4 behaviour).
- verify_claims(..., cited_only=True): a value counts only when it appears inside
  a chunk the answer actually cites (path, and byte range when the citation
  carries one). This is what M4's deterministic check uses: "every number or
  config value in the answer appears in the cited bytes".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.retrieval.types import Chunk


@dataclass
class Claim:
    text: str
    value: str | None = None
    cited_path: str | None = None
    verified: bool = False
    evidence_snippet: str = ""


@dataclass
class VerificationResult:
    claims: list[Claim]
    verified_count: int
    total_count: int
    missing_citations: list[str]
    hallucinated_values: list[str]
    pass_threshold: bool

    @property
    def verification_ratio(self) -> float:
        return self.verified_count / self.total_count if self.total_count > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "verified": self.verified_count,
            "total": self.total_count,
            "ratio": round(self.verification_ratio, 4),
            "pass": self.pass_threshold,
            "missing_citations": self.missing_citations,
            "hallucinated_values": self.hallucinated_values,
            "claims": [
                {
                    "text": c.text,
                    "value": c.value,
                    "cited_path": c.cited_path,
                    "verified": c.verified,
                }
                for c in self.claims
            ],
        }


def extract_claims(answer: str) -> list[Claim]:
    """Extract factual claims from an answer text.

    A claim is a sentence or fragment that asserts a specific value,
    name, path, or configuration detail.
    """
    claims: list[Claim] = []
    # Sentence boundaries only: a period followed by whitespace or a newline. Splitting on
    # every period cut decimals like 0.055 in half, so those values were never checked.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer or "")

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue

        values = _extract_values(sent)
        paths = re.findall(r"[\w/]+\.(?:yaml|py|csv|sbatch|out|md|json|txt)", sent)

        if values or paths:
            for val in values:
                claims.append(Claim(
                    text=sent,
                    value=val,
                    cited_path=paths[0] if paths else None,
                ))
            if paths and not values:
                claims.append(Claim(
                    text=sent,
                    cited_path=paths[0],
                ))
        elif _has_factual_assertion(sent):
            claims.append(Claim(text=sent))

    return claims


def cited_chunks(
    evidence: Sequence[Chunk],
    citations: Sequence[dict | tuple | str] | None,
) -> list[Chunk]:
    """The evidence chunks an answer cites.

    A citation may be a path (str), a (path, start, end) tuple or a dict with
    path / start_byte / end_byte. With a byte range, the chunk must cover it;
    with a bare path, every chunk of that path counts.
    """
    if not citations:
        return []
    keep: list[Chunk] = []
    for cit in citations:
        if isinstance(cit, str):
            path, start, end = cit, None, None
        elif isinstance(cit, dict):
            path, start, end = cit.get("path"), cit.get("start_byte"), cit.get("end_byte")
        else:
            path, start, end = (list(cit) + [None, None])[:3]
        for chunk in evidence:
            if chunk.path != path:
                continue
            if start is not None and end is not None and not (chunk.start_byte <= int(start) and int(end) <= chunk.end_byte):
                continue
            if chunk not in keep:
                keep.append(chunk)
    return keep


def verify_claims(
    claims: list[Claim],
    evidence: Sequence[Chunk],
    cited_paths: Sequence[str] | None = None,
    threshold: float = 0.5,
    *,
    citations: Sequence[dict | tuple | str] | None = None,
    cited_only: bool = False,
) -> VerificationResult:
    """Verify each claim against the retrieved evidence chunks.

    cited_only=True restricts value checks to the chunks named in `citations`
    (falling back to `cited_paths`): a number that is in the evidence but not in
    the cited bytes is reported as hallucinated_values, because the citation
    does not support it.
    """
    evidence_paths = {c.path for c in evidence}
    cited_set = set(cited_paths or [])
    if cited_only:
        scope = cited_chunks(evidence, citations) or cited_chunks(evidence, list(cited_set))
    else:
        scope = list(evidence)
    evidence_text = " ".join(c.text for c in scope)

    missing_citations: list[str] = []
    hallucinated_values: list[str] = []
    verified_count = 0

    for claim in claims:
        if claim.cited_path:
            if claim.cited_path not in evidence_paths and claim.cited_path not in cited_set:
                missing_citations.append(claim.cited_path)
                claim.verified = False
                continue

        if claim.value:
            if claim.value in evidence_text:
                claim.verified = True
                for chunk in scope:
                    if claim.value in chunk.text:
                        claim.evidence_snippet = chunk.text[:200]
                        break
            else:
                claim.verified = False
                hallucinated_values.append(claim.value)
                continue
        elif claim.cited_path and claim.cited_path in evidence_paths:
            claim.verified = True
        elif _fuzzy_match_claim(claim.text, evidence_text):
            claim.verified = True
        else:
            claim.verified = False

        if claim.verified:
            verified_count += 1

    total = len(claims)
    return VerificationResult(
        claims=claims,
        verified_count=verified_count,
        total_count=total,
        missing_citations=missing_citations,
        hallucinated_values=hallucinated_values,
        pass_threshold=verified_count >= total * threshold if total > 0 else True,
    )


def _extract_values(text: str) -> list[str]:
    """Extract specific values (numbers, configs, identifiers) from text.

    Citation tags ([path:start-end]) are removed first so their byte offsets are
    not mistaken for claimed values.
    """
    text = re.sub(r"\[[^\[\]\s]+?:\d+-\d+\]", " ", text or "")
    values: list[str] = []
    values.extend(re.findall(r"\b\d+(?:\.\d+)?[eE][+-]?\d+\b", text))
    values.extend(re.findall(r"\b0\.\d+\b", text))
    values.extend(re.findall(r"\b\d+\.\d+\b", text))
    values.extend(m.group(1) for m in re.finditer(r"\brun[_\s]?(\d+)\b", text, re.I))  # the run number, not "Run 47" literally
    values.extend(re.findall(r"\b(?:dinov2|resnet\d+|clip|vit\w*)\b", text, re.I))
    for m in re.finditer(r"(\d+)\s*(?:epochs?|iterations?|steps?|pairs?)", text, re.I):
        values.append(m.group(1))
    return list(set(values))


def _has_factual_assertion(text: str) -> bool:
    """Check if a sentence makes a factual assertion (not just filler)."""
    assertion_patterns = [
        r"\b(?:is|are|was|were|has|had|uses?|set\s+to|equals?|=)\b",
        r"\b(?:achieved|produced|resulted|scored|measured)\b",
        r"\b\d+\b",
    ]
    return any(re.search(p, text, re.I) for p in assertion_patterns)


def _fuzzy_match_claim(claim_text: str, evidence_text: str) -> bool:
    """Check if the key tokens from a claim appear in the evidence."""
    claim_tokens = {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_]+", claim_text)
        if len(t) > 2 and t.lower() not in _FILLER
    }
    if not claim_tokens:
        return True
    evidence_lower = evidence_text.lower()
    matches = sum(1 for t in claim_tokens if t in evidence_lower)
    return matches / len(claim_tokens) >= 0.6 if claim_tokens else True


_FILLER = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "were", "has", "had", "been", "have", "not", "but", "also", "its",
    "which", "about", "into", "than", "can", "will", "would", "could",
    "should", "does", "did", "may", "might",
})
