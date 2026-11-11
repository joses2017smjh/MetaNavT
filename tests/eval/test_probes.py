"""The derived probe files (bench/probes/) are exactly what app/eval/probes.py writes, and stay outside fixture v1."""

from __future__ import annotations

from app.eval.gold import load_gold
from app.eval.harness import CACHE_PROBES, project_root
from app.eval.probes import PARAPHRASES, keywords, load_probes, paraphrases, render


def test_committed_probe_files_are_the_scripts_output():
    root = project_root()
    for rel, text in render(root).items():
        assert (root / rel).read_text() == text, f"run `make probes` ({rel})"
    assert CACHE_PROBES == PARAPHRASES


def test_probes_live_outside_the_frozen_fixture_and_inherit_their_sources_paths():
    root = project_root()
    assert not PARAPHRASES.startswith(("bench/gold/", "bench/corpus/"))
    gold = {q.id: q for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    probes = load_probes(root / PARAPHRASES)
    assert len(probes) == len({p.id for p in probes}) == len(gold_categories := {q.category for q in gold.values()}) * 3
    assert {p.category for p in probes} == gold_categories
    for p in probes:
        source = gold[p.source_id]
        assert p.relevant_paths == source.relevant_ids() and p.category == source.category
        assert p.question.strip().lower() != source.question.strip().lower()


def test_rules_only_drop_filler_words_or_wrap_the_question():
    assert keywords("what learning rate did run 40 use") == "learning rate run 40"
    gold = load_gold(project_root() / "bench" / "gold" / "questions.jsonl")
    for p in paraphrases(gold):
        source = next(q for q in gold if q.id == p.source_id).question.lower()
        if p.rule == "keywords":  # every kept word is a word of the source, in order
            words = iter(source.replace("how many", "number of").replace("?", "").split())
            assert all(any(w == s.strip("?,.:") for s in words) for w in p.question.split()), p
        else:
            assert source in p.question.lower(), p
