"""B10 and D47: LLM-written HyDE hypotheses and contextual prefixes, generated once and stored, then scored with no model call."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import project_root

ROOT = project_root()
FILES = ROOT / "bench" / "corpus" / "files"


def _stub(prompt: str) -> str | None:
    """A stand-in for the model: fails on HyDE prompts about run 41 and on context prompts for results/ablation.csv."""
    is_context = "Document path: " in prompt
    if (not is_context and "run 41" in prompt) or "Document path: results/ablation.csv" in prompt:
        return None
    return "STUB " + prompt.splitlines()[-1][:40]


def _write_generations(tmp_path, gold):
    from app.eval.frontier_llm import OPTIONS, fixture_chunks, generate

    blob = {"git_sha": "test", "model": "stub:0b", "ollama": {"version": None, "digest": None, "details": None},
            "options": dict(OPTIONS), **generate(_stub, gold, fixture_chunks(FILES))}
    path = tmp_path / "frontier_generations.json"
    path.write_text(json.dumps(blob))
    return path, blob


def test_generation_stores_every_prompts_output_and_counts_failed_calls(tmp_path):
    from app.chunking.contextual import DEFAULT_PROMPT
    from app.retrieval.hyde import HYDE_PROMPT

    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    _, blob = _write_generations(tmp_path, gold)
    hyde, ctx = blob["hyde"], blob["context"]
    assert hyde["prompt"] == HYDE_PROMPT and ctx["prompt"] == DEFAULT_PROMPT and blob["options"]["temperature"] == 0.0
    assert hyde["n"] == len(gold) and [i["id"] for i in hyde["items"]] == [q.id for q in gold]
    failed = [i["id"] for i in hyde["items"] if not i["ok"]]
    assert failed == [q.id for q in gold if "run 41" in q.question] and hyde["n_failed"] == len(failed) > 0
    assert all(i["text"] == "" for i in hyde["items"] if not i["ok"])
    # One prefix per indexed chunk, keyed by chunk id and the hash of its bytes.
    from app.eval.index_loader import load_chunks

    chunks = load_chunks(FILES, mtimes="manifest")
    assert ctx["n"] == len(chunks) and {i["chunk_id"] for i in ctx["items"]} == {c.chunk_id for c in chunks}
    assert {i["chunk_id"]: i["span_sha256"] for i in ctx["items"]} == {c.chunk_id: c.metadata["source_span_sha256"] for c in chunks}
    assert ctx["n_failed"] == sum(1 for c in chunks if c.path == "results/ablation.csv") > 0


def test_a_hyde_prompt_with_no_stored_hypothesis_is_an_error_not_a_fallback(tmp_path):
    from app.eval.frontier_llm import hyde_completion
    from app.retrieval.hyde import HYDE_PROMPT

    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")[:3]
    _, blob = _write_generations(tmp_path, gold)
    complete = hyde_completion(blob)
    assert complete(HYDE_PROMPT.format(query=gold[0].question)) == blob["hyde"]["items"][0]["text"]
    with pytest.raises(KeyError):
        complete(HYDE_PROMPT.format(query="a question nobody generated for"))


def test_the_llm_hyde_arm_embeds_the_stored_hypothesis(tmp_path, monkeypatch):
    import app.eval.harness as harness
    from app.eval.harness import BenchConfig, run_config

    gold = [q for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl") if q.id in {"q001", "q006"}]  # q006 is about run 41
    path, blob = _write_generations(tmp_path, gold)
    seen: list[str] = []
    real = harness.hyde_embed

    def spy(query, embedder, complete=None, **kwargs):
        from app.retrieval.hyde import llm_hypothesis

        seen.append(llm_hypothesis(query, complete) if complete else "")
        return real(query, embedder, complete=complete, **kwargs)

    monkeypatch.setattr(harness, "hyde_embed", spy)
    cfg = BenchConfig(name="hybrid+hyde-llm", mode="hybrid", enable_rerank=False, enable_router=False, staleness_tier1=False,
                      e2e=False, log_triples=False, hyde=True, hyde_stage="first_stage", generations=str(path))
    row = run_config(cfg, gold, FILES)
    stored = {i["id"]: i for i in blob["hyde"]["items"]}
    assert seen[-2] == stored["q001"]["text"] and seen[-2].startswith("STUB")
    assert seen[-1].startswith("This document contains information about")  # q006's call failed: the keyword fallback
    assert row["llm"]["model"] == "stub:0b" and row["llm"]["hyde"] == {"n": 2, "n_failed": 1, "failed_ids": ["q006"]}
    assert row["settings"]["generations"] == str(path)


def test_llm_prefixes_are_prepended_by_chunk_id_and_must_match_the_chunk_bytes(tmp_path):
    from app.eval.frontier_llm import context_prefixes
    from app.eval.index_loader import load_chunks

    _, blob = _write_generations(tmp_path, load_gold(ROOT / "bench" / "gold" / "questions.jsonl")[:1])
    prefixes = context_prefixes(blob)
    plain = {c.chunk_id: c for c in load_chunks(FILES, mtimes="manifest")}
    situated = {c.chunk_id: c for c in load_chunks(FILES, contextual=True, prefixes=prefixes, mtimes="manifest")}
    assert plain.keys() == situated.keys()
    for cid, chunk in situated.items():
        entry = prefixes[cid]
        if entry["ok"]:
            assert chunk.text == entry["text"] + "\n\n" + plain[cid].text and chunk.metadata["text_is_verbatim"] is False
        else:  # a failed call: indexed without a prefix, as situate_chunk does with an empty completion
            assert chunk.text == plain[cid].text
        assert chunk.metadata["evidence_raw"] == plain[cid].metadata["evidence_raw"]
    stale = dict(prefixes)
    first = next(iter(stale))
    stale[first] = {**stale[first], "span_sha256": "0" * 64}
    with pytest.raises(ValueError, match="stored prefix"):
        load_chunks(FILES, contextual=True, prefixes=stale, mtimes="manifest")


def test_llm_arms_are_labelled_by_their_model_not_as_heuristics():
    from app.eval.harness import FRONTIER_NOT_EVALUATED, fallback_labels, frontier_configs

    llm = {"model": "qwen2.5:7b", "hyde": {"n": 136, "n_failed": 2, "failed_ids": ["q1", "q2"]}}
    labels = fallback_labels({"settings": {"hyde": True, "generations": "g.json", "embedder": "st:x"}, "llm": llm})
    assert not any("no LLM" in label for label in labels)
    assert "HyDE hypothesis by keyword heuristic on 2 of 136 questions (failed LLM calls)" in labels
    labels = fallback_labels({"settings": {"contextual": True, "generations": "g.json"}, "llm": {"context": {"n": 71, "n_failed": 0}}})
    assert labels == []
    neural = {c.name: c for c in frontier_configs(neural=True)}
    for name, parent in (("hybrid+hyde-llm@bge-small", "hybrid@bge-small"), ("hybrid+hyde-llm+bge-rerank@bge-small", "hybrid+bge-rerank@bge-small"),
                         ("hybrid+contextual-llm@bge-small", "hybrid@bge-small")):
        assert neural[name].parent == parent and neural[name].generations == "bench/results/frontier_generations.json"
    cpu = {c.name for c in frontier_configs()}
    assert {"hybrid+hyde-llm", "hybrid+hyde-llm+rerank", "hybrid+contextual-llm"} <= set(FRONTIER_NOT_EVALUATED) and not cpu & set(FRONTIER_NOT_EVALUATED)


def test_a_run_with_llm_arms_records_model_prompts_and_temperature_in_its_header(tmp_path):
    from app.eval.harness import BenchConfig, run_bench
    from app.eval.report import frontier_table

    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    path, _ = _write_generations(tmp_path, gold)
    base = dict(mode="hybrid", enable_router=False, staleness_tier1=False, e2e=False, log_triples=False, enable_rerank=False)
    configs = [BenchConfig(name="hybrid", **base),
               BenchConfig(name="hybrid+contextual", contextual=True, parent="hybrid", **base),
               BenchConfig(name="hybrid+contextual-llm", contextual=True, generations=str(path), parent="hybrid", **base)]
    out = tmp_path / "out.json"
    blob = run_bench(configs=configs, out=out, frontier=True)
    h = blob["generations"]
    assert h["model"] == "stub:0b" and h["options"]["temperature"] == 0.0 and h["prompts"]["context"] and h["prompts"]["hyde"]
    assert h["n_failed"]["context"] > 0 and h["file"] == str(path)
    comp = {(c["row"], c["other"]): c for c in blob["bootstrap"]["comparisons"]}
    assert ("hybrid+contextual-llm", "hybrid+contextual") in comp  # LLM-written against the heuristic prefix
    text = frontier_table(json.loads(out.read_text()))
    assert "stub:0b" in text and "temperature 0.0" in text
