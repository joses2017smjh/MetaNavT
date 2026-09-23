"""M4 offline: cited-answer parsing, cited-bytes verification, two-model jury, and the runner with a fake model."""

import json
import re

import pytest

from app.agent.citation_verify import cited_chunks, extract_claims, verify_claims
from app.agent.generate import NOT_IN_SOURCES, build_prompt, cited_answer, parse_citations, source_tag, strip_citations
from app.eval.judge import pairwise_ab_ba
from app.eval.jury import default_jury
from app.retrieval.types import Chunk, RetrievalHit


def _hit(path, text, start=0, rank=1):
    return RetrievalHit(chunk=Chunk(chunk_id=f"{path}::{start}", path=path, text=text, start_byte=start, end_byte=start + len(text.encode())), score=1.0, rank=rank)


HITS = [
    _hit("configs/run_047.yaml", "learning_rate: 3e-4\nencoder: dinov2\nrun_id: 47"),
    _hit("logs/run_047.out", "epoch 12 val_rmse 0.055", rank=2),
]


def test_prompt_shows_tags_and_citations_parse_back():
    prompt = build_prompt("learning rate for run 47?", HITS)
    assert source_tag(HITS[0]) in prompt and NOT_IN_SOURCES in prompt
    text = f"Run 47 uses a learning rate of 3e-4 {source_tag(HITS[0])}. Its RMSE was 0.055 {source_tag(HITS[1])}."
    cits = parse_citations(text)
    assert [(c.path, c.start_byte, c.end_byte) for c in cits] == [(h.chunk.path, 0, h.chunk.end_byte) for h in HITS]
    assert strip_citations(text) == "Run 47 uses a learning rate of 3e-4. Its RMSE was 0.055."


def test_cited_answer_flags_uncited_unknown_and_abstention():
    good = cited_answer(f"3e-4 {source_tag(HITS[0])}", HITS)
    assert good["citations"] and not good["uncited"] and good["cited_paths_in_retrieved"]
    bad = cited_answer("The learning rate is 3e-4.", HITS)
    assert bad["uncited"] and bad["citations"] == []
    unknown = cited_answer("3e-4 [configs/run_099.yaml:0-10]", HITS)
    assert unknown["uncited"] and unknown["unknown_citations"][0]["path"] == "configs/run_099.yaml" and not unknown["cited_paths_in_retrieved"]
    wrong_range = cited_answer("3e-4 [configs/run_047.yaml:5-9]", HITS)
    assert wrong_range["uncited"]  # a byte range that is not a retrieved chunk is not a valid citation
    assert cited_answer(NOT_IN_SOURCES, HITS)["abstained"] and not cited_answer(NOT_IN_SOURCES, HITS)["uncited"]


def test_values_must_be_in_the_cited_bytes():
    chunks = [h.chunk for h in HITS]
    # 0.055 is in the evidence (log) but the answer cites only the config
    claims = extract_claims("Run 47 uses learning rate 3e-4 and reached val_rmse 0.055.")
    loose = verify_claims(claims, chunks)
    assert loose.hallucinated_values == []
    strict = verify_claims(extract_claims("Run 47 uses learning rate 3e-4 and reached val_rmse 0.055."), chunks, citations=[{"path": "configs/run_047.yaml", "start_byte": 0, "end_byte": HITS[0].chunk.end_byte}], cited_only=True)
    assert strict.hallucinated_values == ["0.055"]
    assert cited_chunks(chunks, ["logs/run_047.out"])[0].path == "logs/run_047.out"
    assert cited_chunks(chunks, [("configs/run_047.yaml", 5, 9)])[0].path == "configs/run_047.yaml"
    assert cited_chunks(chunks, [("configs/run_047.yaml", 0, 999)]) == []  # range not covered by the chunk
    # citation tags are not mistaken for values
    assert "0" not in [c.value for c in extract_claims("lr 3e-4 [configs/run_047.yaml:0-40]")]


def test_two_model_jury_keeps_each_members_label():
    answers = {"qwen": "correct", "llama": "wrong"}
    jury = default_jury(models=["qwen", "llama"])
    for name, judge in jury.members[1:]:
        judge.complete = lambda prompt, _n=name: json.dumps({"score": 1.0 if answers[_n] == "correct" else 0.0, "label": answers[_n]})
    assert [n for n, _ in jury.members] == ["heuristic", "qwen", "llama"]
    v = jury.vote("q", "3e-4", ["learning_rate: 3e-4"], gold="3e-4")
    assert v.labels_by_member["qwen"] == "correct" and v.labels_by_member["llama"] == "wrong"


def test_pairwise_position_swap_detects_bias():
    always_first = lambda prompt: "A"  # noqa: E731 - a judge that always prefers the first position
    res = pairwise_ab_ba(always_first, "q", "answer one", "answer two")
    assert res["p_a_ab"] == 1.0 and res["p_a_ba"] == 0.0 and res["p_a"] == 0.5 and res["position_gap"] == 1.0
    consistent = lambda prompt: "B" if prompt.index("answer one") > prompt.index("answer two") else "A"  # noqa: E731
    res = pairwise_ab_ba(consistent, "q", "answer one", "answer two")
    assert res["p_a"] == 1.0 and res["position_gap"] == 0.0


def test_runner_offline_with_a_fake_model(tmp_path):
    from app.eval import llm_eval
    from app.eval.harness import BenchConfig

    def fake_complete(prompt: str) -> str:
        if "Rules:" in prompt:  # generation prompt: cite the first source tag, echo a value from it
            tag = re.search(r"Source 1 (\[[^\]]+\])\n(.*?)(?:\n\n|$)", prompt, re.S)
            body = tag.group(2)
            value = re.search(r"\b\d+(?:\.\d+)?(?:e-?\d+)?\b", body)
            return f"The value is {value.group(0) if value else 'unknown'} {tag.group(1)}."
        if "Which is better" in prompt:
            return "A"
        return json.dumps({"score": 0.9, "label": "correct"})

    cfg = BenchConfig(name="hybrid+router@hash", mode="hybrid", embedder="hash", enable_rerank=False, enable_router=True, staleness_tier1=True, log_triples=False, e2e=False)
    blob = llm_eval.run(config=cfg, generator_model="fake-gen", judge_models=["fake-a", "fake-b"], limit=6, complete_fn=fake_complete)

    assert blob["n_gold"] == 6 and len(blob["rows"]) == 6
    s = blob["summary"]
    assert s["checks"]["uncited_answers"]["count"] == 0
    assert set(s["kappa"]) == {"heuristic", "fake-a", "fake-b", "jury_majority"}
    assert s["kappa"]["heuristic"]["heuristic"] is True and s["kappa"]["fake-a"]["heuristic"] is False
    assert s["pairwise_llm_vs_extractive"]["fake-a"]["p_llm_better_mean"] == 0.5  # 'A' both ways = pure position bias
    assert s["pairwise_llm_vs_extractive"]["fake-a"]["position_gap_mean"] == 1.0
    assert "kappa_gate" in blob and blob["kappa_gate"] == 0.6
    assert "verdict" in s and "Verdict" in llm_eval.markdown_table(blob)
    row = blob["rows"][0]
    assert row["checks"]["cited_in_retrieved"] and row["checks"]["n_citations"] == 1
    assert "heuristic" in row["judges"] and row["generation"]["model"] == "fake-gen"


def test_exact_match_label_treats_equal_numbers_as_equal():
    from app.eval.judge import exact_match_label

    assert exact_match_label("learning_rate: 0.0001 [configs/run_040.yaml:0-249]", "1e-04") == "correct"
    assert exact_match_label("The rate is 3e-4.", "0.0003") == "correct"
    assert exact_match_label("The rate is 3e-4.", "0.0004") == "wrong"
    assert exact_match_label("run 40 used 2 stereo pairs", "2") == "correct"
    assert exact_match_label("bark_type: oak", "oak") == "correct"
    assert exact_match_label("encoder dinov2", "resnet50") == "wrong"
    assert exact_match_label("the encoder is a resnet", "resnet50 encoder") == "partial"
