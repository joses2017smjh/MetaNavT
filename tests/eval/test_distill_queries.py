"""M43 (app/eval/distill_queries.py): the synthetic training queries. Documents only go to the model, every raw
completion is kept, and the one step that reads test queries only removes.

No torch and no Ollama: the completion is a stand-in and the embedder the hash embedder.
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from app.eval import distill_queries as dq
from app.eval.gold import load_gold
from app.retrieval.embedders import HashEmbedder

ROOT = Path(__file__).resolve().parents[2]
SMALL = {"fixture": {"description": "files", "documents": None, "per_form": 2},
         "scifact": {"description": "abstracts", "documents": 3, "per_form": 1}}


def _beir(tmp_path, name="scifact", n=8):
    """A BEIR directory with a corpus and nothing else: no queries.jsonl, no qrels."""
    root = tmp_path / name
    root.mkdir(parents=True)
    with (root / "corpus.jsonl").open("w") as fh:
        for i in range(n):
            words = " ".join(f"word{i}_{j}" for j in range(40 if i % 2 == 0 else 5))  # odd ids are too short to sample
            fh.write(json.dumps({"_id": str(i), "title": f"Title {i}", "text": words}) + "\n")
    return tmp_path


def test_sampling_is_seeded_and_reads_the_corpus_alone(tmp_path):
    data = _beir(tmp_path)
    docs = dq.sample_documents(ROOT, data, SMALL)  # works with no queries.jsonl and no qrels beside the corpus
    again = dq.sample_documents(ROOT, data, SMALL)
    assert docs == again
    assert len(docs["fixture"]) == 71  # every chunk of fixture v1
    ids = [did for did, _ in docs["scifact"]]
    assert len(ids) == 3 and all(int(i) % 2 == 0 for i in ids)  # only documents with at least MIN_WORDS words
    assert docs["scifact"][0][1].startswith("Title ")  # the text the BEIR runner indexes: title + text
    texts = {str(i): "w " * 40 for i in range(50)}
    assert dq.sample_ids(texts, texts, 5) == dq.sample_ids(reversed(list(texts)), texts, 5)  # the order of the ids given does not matter
    assert dq.sample_ids(texts, texts, 5, seed=1) != dq.sample_ids(texts, texts, 5)


def test_the_prompt_shows_the_document_and_no_example_query():
    a, b = dq.prompt_for("scifact", "FIRST DOCUMENT"), dq.prompt_for("scifact", "SECOND DOCUMENT")
    assert a.replace("FIRST", "X") == b.replace("SECOND", "X")  # nothing but the document varies
    assert "abstracts of scientific papers" in a and "- 1 natural-language questions" in a
    assert "- 4 natural-language questions" in dq.prompt_for("fixture", "doc")
    assert "?" not in dq.PROMPT  # no question is shown to the model
    template = dq.normalise(dq.PROMPT)
    for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl"):
        assert dq.normalise(q.question) not in template
    assert len(dq.prompt_for("fiqa", "x" * 10_000)) < len(dq.PROMPT) + dq.MAX_CHARS + 200  # a long document is cut


def test_a_completion_is_parsed_cleaned_and_capped():
    content = json.dumps({"questions": ["  What   is the seed of run 12? ", "what is the seed of run 12", 7, "ok"],
                          "keywords": ["run 12 seed value", "x"], "statements": ["Run 12 used seed 7."], "other": ["ignored form"]})
    forms, status = dq.parse_completion(content, 2)
    assert status == "ok"
    assert forms == {"questions": ["What is the seed of run 12?"], "keywords": ["run 12 seed value"], "statements": ["Run 12 used seed 7."]}
    assert dq.parse_completion(json.dumps({"questions": ["a b", "c d", "e f"]}), 2)[0]["questions"] == ["a b", "c d"]
    assert dq.parse_completion(None, 2)[1] == "no_completion"
    assert dq.parse_completion("Sure! Here are", 2)[1] == "not_json"
    assert dq.parse_completion("[1, 2]", 2)[1] == "not_json"
    assert dq.parse_completion(json.dumps({"questions": ["one"], "keywords": "not a list"}), 2)[1] == "no_queries"
    assert dq.parse_completion(json.dumps({"questions": [" ".join(["w"] * 61)]}), 2)[1] == "no_queries"  # too long to be a query


def test_generation_keeps_every_raw_completion_and_resumes(tmp_path):
    docs = {"fixture": [("a", "doc a"), ("b", "doc b")], "scifact": [("1", "doc one")]}
    calls = []

    def complete(prompt):
        calls.append(prompt)
        if "doc b" in prompt:
            return {"content": None, "eval_count": None, "done_reason": None, "error": "URLError"}
        return {"content": '{"questions": ["what is here"]}', "eval_count": 9, "done_reason": "stop", "error": None}

    out = tmp_path / "work" / "completions.jsonl"
    assert dq.generate(docs, complete, out, workers=2, corpora=SMALL) == 3
    rows = dq.read_jsonl(out)
    assert [(r["corpus"], r["doc_id"]) for r in rows] == [("fixture", "a"), ("fixture", "b"), ("scifact", "1")]
    assert rows[1]["content"] is None and rows[1]["error"] == "URLError"  # the failed call is a row, not a gap
    assert rows[0]["content"] == '{"questions": ["what is here"]}' and rows[0]["ms"] >= 0
    assert dq.generate(docs, complete, out, workers=2, corpora=SMALL) == 0 and len(calls) == 3  # nothing is asked twice
    docs["scifact"].append(("2", "doc two"))
    assert dq.generate(docs, complete, out, workers=1, corpora=SMALL, limit=5) == 1 and len(dq.read_jsonl(out)) == 4


def test_near_duplicates_of_test_queries_are_flagged_by_each_rule():
    tests = [{"set": "fixture", "id": "q001", "text": "what learning rate did run 40 use"},
             {"set": "scifact", "id": "7", "text": "Aspirin lowers the risk of stroke in adults."}]
    embed = HashEmbedder().encode
    queries = ["What learning rate did run 40 use?",          # the test query, differently cased and punctuated
               "what learning rate did run 40 use then",      # one more token: Jaccard 7/8
               "which optimiser schedule appears in the trellis notes"]  # unrelated
    near = dq.nearest_test_queries(queries, tests, embed)
    assert near[0]["equal"] and near[0]["jaccard"] == 1.0 and near[0]["jaccard_nearest"] == {"set": "fixture", "id": "q001"}
    assert "equal" in dq.flag_rules(near[0]) and "jaccard" in dq.flag_rules(near[0])
    assert not near[1]["equal"] and near[1]["jaccard"] == 0.875 and "jaccard" in dq.flag_rules(near[1])
    assert dq.flag_rules(near[2]) == [] and near[2]["jaccard"] < 0.3
    assert dq.flag_rules({"equal": False, "jaccard": 0.1, "cosine": 0.93}) == ["cosine"]  # meaning alone is enough
    assert dq.flag_rules({"equal": False, "jaccard": 0.69, "cosine": 0.8999}) == []
    assert dq.NEAR_DUPLICATE == {"jaccard": 0.7, "cosine": 0.90, "embedder": "BAAI/bge-small-en-v1.5"}
    assert dq.nearest_test_queries([], tests, embed) == []


def test_the_filter_drops_repeats_and_near_duplicates_and_splits_by_document():
    gold = [{"set": "fixture", "id": q.id, "text": q.question} for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")]
    completions = []
    for i in range(30):
        completions.append({"corpus": "fixture", "doc_id": f"file{i:02d}", "content": json.dumps({
            "questions": [f"which notebook cell plots figure {i}", gold[i]["text"].upper()],  # the second is a gold question
            "keywords": [f"figure {i} plotting notebook", "shared keyword query"]})})  # the last repeats across documents
    completions.append({"corpus": "scifact", "doc_id": "9", "content": "not json"})
    completions.append({"corpus": "scifact", "doc_id": "10", "content": None})
    kept, report = dq.filter_queries(completions, gold, HashEmbedder().encode, corpora=SMALL)
    fx = report["by_corpus"]["fixture"]
    assert fx["completions"] == {"ok": 30, "no_completion": 0, "not_json": 0, "no_queries": 0}
    assert report["by_corpus"]["scifact"]["completions"] == {"ok": 0, "no_completion": 1, "not_json": 1, "no_queries": 0}
    assert fx["repeats_dropped"] == 29  # "shared keyword query" survives once
    assert fx["near_duplicates_dropped"] == 30 and fx["dropped_by_rule"]["equal"] == 30
    assert fx["kept"] == len([q for q in kept if q["corpus"] == "fixture"]) == 61
    assert fx["checked"] == fx["kept"] + fx["near_duplicates_dropped"]
    assert {q["query"].lower() for q in kept}.isdisjoint({g["text"].lower() for g in gold})  # no gold question is a training query
    assert len(report["flagged"]) == 30 and all(f["rules"] and f["cosine_nearest"]["set"] == "fixture" for f in report["flagged"])
    assert report["test_queries"] == {"fixture": 136}
    assert fx["at_other_thresholds"]["cosine_at_least"]["0.80"] >= fx["at_other_thresholds"]["cosine_at_least"]["0.95"] >= 30
    split_of = {}
    for q in kept:
        assert split_of.setdefault(q["doc_id"], q["split"]) == q["split"]  # a document's queries stay together
    assert fx["validation_documents"] == 4 == sum(1 for s in split_of.values() if s == "validation")  # max(4, ceil(5% of 30))
    assert fx["kept_by_split"]["train"] + fx["kept_by_split"]["validation"] == fx["kept"]
    assert len(fx["nearest_kept"]) == 10 and fx["nearest_kept"][0]["cosine"] < dq.NEAR_DUPLICATE["cosine"]


def test_validation_documents_are_seeded_and_never_fewer_than_the_minimum():
    ids = [f"d{i}" for i in range(200)]
    val = dq.validation_documents(ids)
    assert len(val) == 10 and val == dq.validation_documents(reversed(ids))
    assert len(dq.validation_documents(ids[:20])) == 4 and len(dq.validation_documents(ids[:3])) == 3
    assert dq.validation_documents([]) == set()


def test_the_committed_files_have_reproducible_bytes(tmp_path):
    rows = [{"id": "a", "query": "é"}, {"id": "b", "query": "x"}]
    one = dq.write_jsonl_gz(tmp_path / "one.jsonl.gz", rows)
    two = dq.write_jsonl_gz(tmp_path / "sub" / "two.jsonl.gz", rows)
    assert one == two  # no timestamp and no file name in the gzip header
    assert dq.read_jsonl(tmp_path / "one.jsonl.gz") == rows
    assert gzip.open(tmp_path / "one.jsonl.gz", "rt", encoding="utf-8").read().count("\n") == 2


def test_the_filter_step_refuses_a_smoke_run(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "completions.jsonl").write_text(json.dumps({"corpus": "fixture", "doc_id": "a", "content": "{}"}) + "\n")
    (work / "generation.meta.json").write_text(json.dumps({"limit": 5, "sample": {"corpora": {"fixture": {"n_documents": 1}}}}))
    with pytest.raises(SystemExit, match="--limit"):
        dq.main(["filter", "--work", str(work), "--out-dir", str(tmp_path / "out")])
    (work / "generation.meta.json").write_text(json.dumps({"limit": None, "sample": {"corpora": {"fixture": {"n_documents": 2}}}}))
    with pytest.raises(SystemExit, match="not finished"):
        dq.main(["filter", "--work", str(work), "--out-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_generation_stops_when_the_server_went_away(tmp_path, monkeypatch):
    work = tmp_path / "work"
    monkeypatch.setattr(dq, "ollama_tags", lambda url: {"gemma4:12b": {"name": "gemma4:12b", "digest": "d" * 64, "details": {}}})
    monkeypatch.setattr(dq, "ollama_version", lambda url: "0.0.0")
    monkeypatch.setattr(dq, "sample_documents", lambda root, data_dir: {"fixture": [(f"d{i}", "text") for i in range(10)]})
    monkeypatch.setattr(dq, "ollama_chat", lambda prompt, model, url: {"content": None, "eval_count": None, "done_reason": None, "error": "URLError"})
    with pytest.raises(SystemExit, match="returned nothing"):
        dq.main(["generate", "--work", str(work), "--workers", "2"])
    assert len(dq.read_jsonl(work / "completions.jsonl")) == 10  # every failed call is still on record
    monkeypatch.setattr(dq, "ollama_tags", lambda url: {})
    with pytest.raises(SystemExit, match="is not served"):
        dq.main(["generate", "--work", str(tmp_path / "other")])


def test_the_declared_sample_and_model():
    assert {c: (s["documents"], s["per_form"]) for c, s in dq.CORPORA.items()} == {"fixture": (None, 4), "scifact": (1500, 1), "fiqa": (1800, 1)}
    assert dq.LLM_MODEL == "gemma4:12b" and dq.LLM_OPTIONS["temperature"] == 0.0 and dq.LLM_OPTIONS["seed"] == 0
    assert dq.VALIDATION == {"fraction": 0.05, "min_documents": 4, "seed": 1}
    assert np.isclose(dq.jaccard({"a", "b"}, {"b", "c"}), 1 / 3) and dq.jaccard(set(), {"a"}) == 0.0
