"""Real byte slices and explicitly scoped structured fact checks."""
from dataclasses import replace
import json
import pytest

from app.agent.citation_verify import Claim, cited_chunks, verify_claims
from app.agent.evidence_checks import StructuredFact, compare_fact, verify_structured_claim
from app.eval.evidence_consistency import FAMILIES, evidence_case
from app.eval.index_loader import load_chunks
from app.retrieval.types import Chunk


@pytest.mark.parametrize("label,family", [(label, family) for label, families in FAMILIES.items() for family in families])
def test_balanced_structured_fixture_families(label, family):
    result = evidence_case(label, family, 3)
    assert result["passed"], result


def _chunk(text):
    return Chunk("test", "facts.txt", text, 0, len(text.encode()))


def test_subrange_does_not_support_values_elsewhere_in_chunk_or_fallback_to_path():
    text = "α before\nvalue: 0.055\nvalue: 0.999\n"
    chunk = _chunk(text)
    start = len("α before\n".encode())
    end = start + len("value: 0.055\n".encode())
    citation = {"path": chunk.path, "start_byte": start, "end_byte": end}
    assert cited_chunks([chunk], [citation])[0].text == "value: 0.055\n"
    result = verify_claims([Claim("metric value", value="0.999")], [chunk], cited_paths=[chunk.path], citations=[citation], cited_only=True)
    assert result.verified_count == 0
    bad = {**citation, "end_byte": 999}
    result = verify_claims([Claim("metric value", value="0.055")], [chunk], cited_paths=[chunk.path], citations=[bad], cited_only=True)
    assert result.verified_count == 0  # explicit invalid range cannot fall back to bare path


def test_literal_tokens_require_boundaries_and_are_not_entailment():
    chunk = _chunk("robot A latency is not 0.0559 ms; run 147")
    for value in ("0.055", "47"):
        assert verify_claims([Claim("metric", value=value)], [chunk]).verified_count == 0
    # Literal checks intentionally do not understand negation or units.
    assert verify_claims([Claim("robot latency is 0.0559 s", value="0.0559")], [chunk]).verified_count == 1
    fact = StructuredFact("robot A", "latency", "0.0559", "ms", negated=True)
    assert compare_fact(replace(fact, negated=False), fact).label == "conflict"
    assert compare_fact(replace(fact, unit="s", negated=False), replace(fact, negated=False)).label == "conflict"


def test_utf8_boundary_and_search_headers_fail_closed(tmp_path):
    text = "α calibration\nvalue: 0.055\n"
    chunk = _chunk(text)
    assert cited_chunks([chunk], [{"path": chunk.path, "start_byte": 1, "end_byte": 3}]) == []
    transformed = replace(chunk, text="HEADER: 0.999\n"+text, metadata={"text_is_verbatim": False, "evidence_raw": text})
    result = verify_claims([Claim("value", value="0.999")], [transformed], citations=[chunk.path], cited_only=True)
    assert result.verified_count == 0
    unbound = replace(transformed, metadata={"text_is_verbatim": False})
    assert cited_chunks([unbound], [chunk.path]) == []
    (tmp_path / "unicode.md").write_text(text)
    loaded = load_chunks(tmp_path)
    for source in loaded:
        raw = (tmp_path / source.path).read_bytes()[source.start_byte:source.end_byte].decode()
        assert source.metadata["evidence_raw"] == raw


def test_json_fact_schema_and_negative_values_are_strict():
    with pytest.raises(ValueError):
        StructuredFact.from_dict({"subject": "robot", "field": "latency", "value": [], "unit": "s"})
    with pytest.raises(ValueError):
        StructuredFact.from_dict({"subject": "robot", "field": "latency", "value": 1, "negated": "false"})
    base = StructuredFact("robot", "offset", "-5e-3", "m")
    assert compare_fact(replace(base, value="-5", unit="mm"), base).label == "supported"
    assert compare_fact(replace(base, value=True, unit=""), replace(base, value=1, unit="")).label == "conflict"


def test_conflicting_and_unknown_cited_sources_do_not_pass():
    fact = StructuredFact("robot", "offset", 1, "m", current=True)
    def chunk(f):
        raw = json.dumps({"facts": [{"subject": f.subject, "field": f.field, "value": f.value, "unit": f.unit, "current": f.current}]})
        return Chunk(str(f.current), f"{f.current}.json", raw, 0, len(raw.encode()))
    chunks = [chunk(fact), chunk(replace(fact, current=None))]
    result = verify_structured_claim(fact, chunks, [source.path for source in chunks])
    assert result.label == "unknown"


def test_unrelated_negative_facts_are_unknown_not_logical_contradictions():
    fact = StructuredFact("robot", "offset", 5, "mm", negated=True)
    assert compare_fact(replace(fact, value=6), fact).label == "unknown"
    assert compare_fact(replace(fact, value=6, negated=False), fact).label == "unknown"


def test_named_claim_path_cannot_borrow_a_value_from_another_cited_file():
    first = Chunk("first", "first.txt", "value: 0.111", 0, len("value: 0.111".encode()))
    second = Chunk("second", "second.txt", "value: 0.999", 0, len("value: 0.999".encode()))
    claim = Claim("first value is 0.999", value="0.999", cited_path="first.txt")
    result = verify_claims([claim], [first, second], citations=[first.path, second.path], cited_only=True)
    assert result.verified_count == 0 and claim.evidence_snippet == ""
    claim = Claim("second value is 0.999", value="0.999", cited_path="second.txt")
    result = verify_claims([claim], [first, second], citations=[first.path, second.path], cited_only=True)
    assert result.verified_count == 1 and claim.evidence_snippet == second.text
