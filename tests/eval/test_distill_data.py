"""M43: the committed files of bench/distill/ are what their headers say, and hold no benchmark question.

The synthetic queries (queries.jsonl.gz), every raw completion they were read from (completions.jsonl.gz), the
teacher's scores (teacher_scores.jsonl.gz) and the students' manifest (students.json) are data a later CPU-only run
trains or re-scores from, so each is checked against its own header and against the declarations of the code: the
prompt that ran is the prompt in the repository, the lexical near-duplicate rules hold again here against the fixture's
gold questions (the cosine rule needs the embedder, and the BEIR test queries are not in the repository: for those the
header's counts stand), and a student's manifest names the files and the recipe it was trained on.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest

from app.eval import distill_queries as dq
from app.eval import distill_teacher as dt
from app.eval import small_rerankers as sr
from app.eval.distill_train import choose_epoch
from app.eval.gold import load_gold

ROOT = Path(__file__).resolve().parents[2]
DISTILL = ROOT / "bench" / "distill"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def meta():
    return json.loads((DISTILL / dq.META_FILE).read_text())


@pytest.fixture(scope="module")
def queries():
    return dq.read_jsonl(DISTILL / dq.QUERIES_FILE)


def test_the_queries_file_is_the_one_its_header_describes(meta, queries):
    assert _sha(DISTILL / dq.QUERIES_FILE) == meta["files"][dq.QUERIES_FILE]["sha256"]
    assert _sha(DISTILL / dq.COMPLETIONS_FILE) == meta["files"][dq.COMPLETIONS_FILE]["sha256"]
    assert len(queries) == meta["files"][dq.QUERIES_FILE]["n"] == sum(b["kept"] for b in meta["near_duplicates"]["by_corpus"].values())
    assert len({q["id"] for q in queries}) == len(queries)
    assert {q["corpus"] for q in queries} == set(dq.CORPORA) and {q["form"] for q in queries} <= set(dq.FORMS)
    assert meta["working_tree_dirty"] is False and meta["generation"]["working_tree_dirty"] is False


def test_the_prompt_and_the_model_that_ran_are_the_declared_ones(meta):
    gen = meta["generation"]
    assert gen["prompt"]["sha256"] == hashlib.sha256(dq.PROMPT.encode()).hexdigest() and gen["prompt"]["template"] == dq.PROMPT
    assert gen["prompt"]["contains_example_queries"] is False
    assert gen["llm"]["model"] == dq.LLM_MODEL and gen["llm"]["options"] == dq.LLM_OPTIONS and gen["llm"]["request"] == dq.LLM_REQUEST
    assert len(gen["llm"]["revision"]) == 64  # the digest of the weights Ollama served
    assert gen["limit"] is None  # the whole sample, not a smoke run
    for corpus, spec in dq.CORPORA.items():
        ran = gen["sample"]["corpora"][corpus]
        assert ran["per_form"] == spec["per_form"] and ran["documents"] == spec["documents"]
        assert ran["n_documents"] == (spec["documents"] or 71)
    assert "never queries.jsonl or qrels" in gen["sample"]["reads"]


def test_every_completion_is_kept_and_reads_as_recorded(meta):
    completions = dq.read_jsonl(DISTILL / dq.COMPLETIONS_FILE)
    assert len(completions) == sum(c["n_documents"] for c in meta["generation"]["sample"]["corpora"].values())
    for c in completions:
        assert dq.parse_completion(c.get("content"), dq.CORPORA[c["corpus"]]["per_form"])[1] == c["parse"]
    for corpus, block in meta["near_duplicates"]["by_corpus"].items():
        assert block["completions"] == {s: sum(1 for c in completions if c["corpus"] == corpus and c["parse"] == s) for s in dq.PARSE_STATUSES}
        assert block["completions"]["no_completion"] <= dq.MAX_FAILED_SHARE * sum(block["completions"].values())


def test_no_training_query_is_a_near_duplicate_of_a_gold_question_by_the_lexical_rules(meta, queries):
    rule = meta["near_duplicates"]["rule"]
    assert {k: rule[k] for k in ("jaccard", "cosine", "embedder")} == dq.NEAR_DUPLICATE
    assert meta["near_duplicates"]["test_queries"] == {"fiqa": 648, "fixture": 136, "scifact": 300}
    gold = [dq.normalise(q.question) for q in load_gold(ROOT / "bench" / "gold" / "questions.jsonl")]
    gold_sets = [set(g.split()) for g in gold]
    exact = set(gold)
    worst = 0.0
    for q in queries:
        norm = dq.normalise(q["query"])
        assert norm not in exact, q["id"]
        worst = max(worst, max(dq.jaccard(set(norm.split()), g) for g in gold_sets))
    assert worst < dq.NEAR_DUPLICATE["jaccard"]
    flagged = meta["near_duplicates"]["flagged"]
    assert len(flagged) == sum(b["near_duplicates_dropped"] for b in meta["near_duplicates"]["by_corpus"].values())
    assert all(f["rules"] and f["id"] not in {q["id"] for q in queries} for f in flagged)
    assert all(f["equal"] or f["jaccard"] >= dq.NEAR_DUPLICATE["jaccard"] or f["cosine"] >= dq.NEAR_DUPLICATE["cosine"] for f in flagged)
    for block in meta["near_duplicates"]["by_corpus"].values():  # what was kept is under the cosine rule too, by the header's own nearest list
        assert all(n["cosine"] < dq.NEAR_DUPLICATE["cosine"] for n in block["nearest_kept"])


def test_validation_is_split_by_source_document(meta, queries):
    for corpus in dq.CORPORA:
        mine = [q for q in queries if q["corpus"] == corpus]
        val = dq.validation_documents(q["doc_id"] for q in mine)
        assert all((q["split"] == "validation") == (q["doc_id"] in val) for q in mine)
        docs = {q["doc_id"] for q in mine}
        assert len(val) == min(len(docs), max(dq.VALIDATION["min_documents"], math.ceil(dq.VALIDATION["fraction"] * len(docs))))
        assert meta["near_duplicates"]["by_corpus"][corpus]["validation_documents"] == len(val)


@pytest.mark.skipif(not (DISTILL / dt.SCORES_FILE).exists(), reason="the teacher's scores are not committed yet")
def test_the_teachers_scores_cover_every_query_once(queries):
    meta = json.loads((DISTILL / dt.SCORES_META).read_text())
    rows = dq.read_jsonl(DISTILL / dt.SCORES_FILE)
    assert _sha(DISTILL / dt.SCORES_FILE) == meta["file"]["sha256"] and meta["queries"]["sha256"] == _sha(DISTILL / dq.QUERIES_FILE)
    assert [r["id"] for r in rows] == sorted(q["id"] for q in queries)
    by = {q["id"]: q for q in queries}
    for r in rows:
        n = len(r["candidates"])
        assert n == len(r["scores"]) == len(set(r["candidates"])) and dt.CANDIDATES_TOP <= n <= dt.CANDIDATES_TOP + 1 or r["corpus"] == "fixture"
        source = by[r["id"]]["doc_id"]
        assert source in r["candidates"]  # every list holds the document its query was written from
        assert (r["candidates"].index(source) + 1 == r["source_rank"]) if r["source_rank"] else r["candidates"][-1] == source
        assert all(math.isfinite(s) for s in r["scores"])
    t = meta["teacher"]
    assert (t["model"], t["precision"], t["max_length"], t["dtype"]) == (sr.TEACHER_MODEL, "fp32", 512, "torch.float32")
    assert t["load_and_score"]["passed"] and meta["working_tree_dirty"] is False and len({s["shard"][1] for s in meta["shards"]}) == 1
    assert meta["summary"] == dt.summarise(rows)
    assert "never a relevance judgment" in meta["labels"]


@pytest.mark.skipif(not (ROOT / sr.STUDENT_MANIFEST).exists(), reason="no student is recorded yet")
def test_each_student_was_trained_on_these_files_by_the_declared_recipe():
    manifest = json.loads((ROOT / sr.STUDENT_MANIFEST).read_text())
    assert set(manifest["students"]) == set(sr.STUDENTS)
    for tag, e in manifest["students"].items():
        assert e["recipe"] == sr.RECIPE and e["attempt"] == sr.ATTEMPT and e["smoke_run_max_steps"] is None
        assert e["init"]["tag"] == sr.STUDENTS[tag] and e["init"]["revision"] == sr.CANDIDATES[sr.STUDENTS[tag]]["revision"]
        assert e["data"] == {dq.QUERIES_FILE: _sha(DISTILL / dq.QUERIES_FILE), dt.SCORES_FILE: _sha(DISTILL / dt.SCORES_FILE)}
        assert e["working_tree_dirty"] is False and e["dtype"] == "torch.float32"
        assert len(e["epochs"]) == sr.RECIPE["epochs"] and e["chosen_epoch"] == choose_epoch([ep["validation_macro"] for ep in e["epochs"]])
        assert e["weights_sha256"] == e["files"][sr.WEIGHTS_FILE]["sha256"] and e["weights_bytes"] == e["files"][sr.WEIGHTS_FILE]["bytes"]
        assert e["reload_check"]["max_abs_difference"] <= e["reload_check"]["tolerance"] and len(e["reference_logits"]) == 3
        assert e["load_and_score"]["passed"] is True
        assert abs(e["first_step_loss"]["torch"] - e["first_step_loss"]["numpy"]) < 1e-3 * max(1.0, e["first_step_loss"]["numpy"])
