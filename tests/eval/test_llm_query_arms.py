"""M46 (app/eval/llm_query_*.py): what was pinned before any LLM call, the parsers, the arm wiring, the bar and its
decision sentences, and the pipeline end to end on fakes (ids, worklist, generation, merge, scoring, re-score).

No torch and no network: the embedder is the hash embedder, the cross-encoder a stand-in scoring by token overlap, the
LLM a fake /api/generate. What is tested is the wiring: every pinned value, that the parents rank exactly as the
committed runners do, that an LLM arm never reaches a heuristic, that the reranker reads the original query except in
per-sub, that nothing gold or qrels reaches a prompt, and that every interval and word re-scores from the sidecar.
"""

import ast
import copy
import gzip
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval import llm_query_arms as arms
from app.eval import llm_query_bar as bar
from app.eval import llm_query_gen as gen
from app.eval import llm_query_parse as parse
from app.eval import llm_query_spec as spec
from app.retrieval.bm25 import tokenize

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "bench" / "corpus" / "files"


# ---------------------------------------------------------------- what was pinned at D0


def test_the_models_are_pinned_by_digest():
    assert {t: m["name"] for t, m in spec.MODELS.items()} == {"qwen3.5-27b": "qwen3.5:27b", "gemma4-26b": "gemma4:26b", "qwen3.5-122b": "qwen3.5:122b"}
    assert spec.MODELS["qwen3.5-27b"]["digest"] == "64ae5b1d651fb2709d5928ccb8dcfb4bb2cc9c2c8001ac34497e36a6cf905b59"
    assert spec.MODELS["gemma4-26b"]["digest"] == "001e5dafc3c77684c2307ebc6ab8e336e10c9b18eca52acf547d72fc83c3ca8c"
    assert spec.MODELS["qwen3.5-122b"]["digest"] == "8b9d11d807c57feb1e2ecb0d6cbf40334c37dcc3523bed5540af4f927f112a37"
    assert spec.BAR_MODELS == ("qwen3.5-27b", "gemma4-26b") and spec.MAIN_MODEL == "qwen3.5-27b"
    committed = "".join(p.read_text() for p in sorted((ROOT / "bench" / "results").glob("jury_v3_*.json")))
    for tag in ("qwen3.5-27b", "gemma4-26b", "qwen3.5-122b"):  # the /api/tags digests M42's committed files recorded
        assert f'"digest": "{spec.MODELS[tag]["digest"]}"' in committed, tag
    assert spec.RETRIEVAL_MODELS["reranker"]["revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert spec.RETRIEVAL_MODELS["embedder"]["revision"] == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def test_the_request_is_pinned_and_derives_from_one_num_predict():
    assert spec.NUM_PREDICT == {"hyde": 256, "multiquery": 256, "decompose": 256, "rewrite": 256, "grade": 16}
    assert spec.options("grade") == {"temperature": 0, "seed": 0, "num_predict": 16}
    assert spec.options("hyde") == {"temperature": 0, "seed": 0, "num_predict": 256}
    assert spec.REQUEST["think"] is False and spec.REQUEST["stream"] is False and spec.REQUEST["keep_alive"] == "60m"
    assert spec.CONTEXT_LENGTH == 8192 and spec.REQUEST["server"]["OLLAMA_NUM_PARALLEL"] == 1
    assert spec.REQUEST["server"]["LLAMA_ARG_THREADS"] == spec.REQUEST["server"]["cpus"] == 4
    payload, error = gen.request_payload("qwen3.5:27b", "grade", "p")
    assert error is None and payload == {"model": "qwen3.5:27b", "prompt": "p", "stream": False, "think": False, "keep_alive": "60m",
                                         "options": {"temperature": 0, "seed": 0, "num_predict": 16}}
    assert "num_ctx" not in payload["options"]
    assert gen.request_payload("m", "nope", "p") == (None, "unknown kind 'nope'")
    # NUM_PREDICT is the one assignment allowed to change between D0 and D1: no other literal 256 or 16 in part (a)
    tree = ast.parse((ROOT / "app" / "eval" / "llm_query_spec.py").read_text())
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "NUM_PREDICT" for t in n.targets)]
    assert len(assigns) == 1
    inside = {id(n) for n in ast.walk(assigns[0])}
    stretch = [v for n in ast.walk(tree) if isinstance(n, ast.Dict) for k, v in zip(n.keys, n.values) if getattr(k, "value", None) == "stretch"]
    inside |= {id(n) for d in stretch for n in ast.walk(d)}  # the stretch server's 16 CPUs and 16 threads are not token caps
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value in (256, 16) and id(n) not in inside and type(n.value) is int]
    assert literals == []
    assert spec.overflowed(8192 - 256, "hyde") is False and spec.overflowed(8192 - 255, "hyde") is True and spec.overflowed(None, "grade") is False


def test_the_prompts_are_the_prototypes_own_text():
    from app.agent.decompose import DECOMPOSE_PROMPT
    from app.agent.deep_research import MULTI_QUERY_PROMPT, expand_queries
    from app.retrieval.hyde import HYDE_PROMPT

    q = "which run used dinov2"
    assert spec.build_prompt("hyde", q) == HYDE_PROMPT.format(query=q)
    assert spec.build_prompt("multiquery", q) == f"Write {3 - 1} short alternative search queries for: {q}\nOne per line, no numbering."
    assert spec.build_prompt("multiquery", q) == MULTI_QUERY_PROMPT.format(n_alt=2, query=q)
    assert spec.build_prompt("decompose", q) == DECOMPOSE_PROMPT.format(max_subs=4, query=q)
    assert "Return at most 4 sub-questions" in spec.build_prompt("decompose", q)
    seen = []
    expand_queries(q, n=3, llm=lambda p: (seen.append(p), "")[1])  # the prototype still sends the same text
    assert seen == [spec.build_prompt("multiquery", q)]
    assert spec.PROMPTS["grade"] == ("Does the passage below contain information that helps answer the search query? Reply with one word: yes or no."
                                     "\n\nSearch query: {query}\n\nPassage:\n{passage}\n\nAnswer:")
    assert spec.PROMPTS["rewrite"] == ("Rewrite the search query below so that it retrieves the documents that answer it better. Keep its meaning. "
                                       "Reply with the rewritten query only, on one line.\n\nQuery: {query}\n\nRewritten query:")
    assert spec.build_prompt("grade", q, "File: a.txt\nbody").endswith("Passage:\nFile: a.txt\nbody\n\nAnswer:")
    assert spec.fixture_passage("runs/a.yaml", "x" * 5000) == "File: runs/a.yaml\n" + "x" * 5000  # a fixture chunk is never cut
    assert spec.beir_passage("y" * 5000) == "y" * 2000 and spec.G == 5 and spec.GRADER_PASSAGE_CHARS == 2000
    with pytest.raises(ValueError):
        spec.build_prompt("grade", q)
    with pytest.raises(ValueError):
        spec.build_prompt("hyde", q, "passage")
    assert spec.grader_candidates(["a", "b", "c", "d", "e", "f"], ["c", "x", "a", "y", "z", "w"]) == ["a", "b", "c", "d", "e", "x", "y", "z"]


def test_the_arms_names_and_parents():
    names = [spec.arm_name(t, p, "qwen3.5-27b") for t, p, _, _ in spec.ARMS]
    assert names == [
        "hybrid+hyde[qwen3.5-27b]@bge-small",
        "hybrid+hyde[qwen3.5-27b]+bge-rerank@bge-small/fp32-512/top20",
        "hybrid+bge-rerank+hyde[qwen3.5-27b]@bge-small/fp32-512/top20",
        "hybrid+multiquery[qwen3.5-27b]@bge-small",
        "hybrid+multiquery[qwen3.5-27b]+bge-rerank@bge-small/fp32-512/top20",
        "hybrid+decompose[qwen3.5-27b]@bge-small",
        "hybrid+decompose[qwen3.5-27b]+bge-rerank@bge-small/fp32-512/top20",
        "hybrid+decompose[qwen3.5-27b]+bge-rerank-per-sub@bge-small/fp32-512/top20",
        "hybrid+corrective[qwen3.5-27b]@bge-small",
        "hybrid+bge-rerank+corrective[qwen3.5-27b]@bge-small/fp32-512/top20",
    ]
    assert spec.arm_name("multiquery", "none", "heuristic") == "hybrid+multiquery[heuristic]@bge-small"
    assert spec.P0 == "hybrid@bge-small" and spec.P1 == "hybrid+bge-rerank@bge-small/fp32-512/top20"
    assert [spec.parent_name(t, p) for t, p, _, _ in spec.ARMS] == [spec.P0, spec.P1, spec.P1, spec.P0, spec.P1, spec.P0, spec.P1, spec.P1, spec.P0, spec.P1]
    assert spec.K == {"fixture": 50, "beir": 100} and spec.RERANK == {"model": "BAAI/bge-reranker-v2-m3", "top": 20, "max_length": 512, "precision": "fp32"}
    assert set(spec.NOT_MEASURED) == {("multiquery", "post"), ("corrective", "pre")}
    # no name may equal a committed frontier row's (test_plan_status keys verdicts by row name)
    frontier = {r["config"] for r in json.loads((ROOT / "bench" / "results" / "frontier_neural.json").read_text())["results"]}
    every = [spec.arm_name(t, p, tag) for t, p, _, _ in spec.ARMS for tag in (*spec.MODELS, spec.HEURISTIC_TAG, spec.QWEN25_TAG)]
    assert not (set(every) & frontier) and len(set(every)) == len(every)


def test_the_bar_the_replication_table_and_the_decision_are_declared():
    assert spec.ATTEMPT == 1
    assert "lo_raw > 0" in spec.BAR["verdict"] and "exactly 0.0 covers zero" in spec.BAR["verdict"]
    assert "'above zero' on SciFact AND 'above zero' on FiQA AND not 'below zero' on the fixture's held-out half" in spec.BAR["passes"]
    assert "2000 resamples, seed 0, 95%" in spec.BAR["interval"] and "delta_raw" in spec.BAR["interval"]
    assert [r[2] for r in spec.REPLICATION] == ["the same word, open or withheld", "not run", "replicated", "not replicated", "contradicted",
                                                "gemma only", "contradicted", "neither passes"]
    assert spec.DECISION["banned_words"] == ("belongs", "earns", "helps only", "does not help")
    assert spec.DEV_QUERIES["n"] == 20 and spec.DEV_QUERIES["seed"] == 0 and spec.DEV_QUERIES["fixture"]["jaccard_drop_at"] == 0.5
    assert set(spec.TRUNCATION) == set(spec.KINDS) and "cut line is dropped" in spec.TRUNCATION["multiquery"]
    assert len(spec.DOES_NOT_SHOW) == 13


# ---------------------------------------------------------------- parsers and the truncation rule


def test_the_line_parser():
    reply = "Here are two alternatives:\n1. **alpha beta**\n2) \"gamma delta\"\n- which run used dinov2\n* `alpha beta`\n3. epsilon"
    assert parse.parse_lines(reply, "Which run used dinov2", 2) == (["alpha beta", "gamma delta"], "ok")
    assert parse.parse_lines(reply, "Which run used dinov2", 4) == (["alpha beta", "gamma delta", "epsilon"], "ok")  # repeats dropped
    assert parse.parse_lines("", "q", 2) == ([], "empty")
    assert parse.parse_lines("Sub-questions:\n", "q", 4) == ([], "unparseable")
    assert parse.parse_lines("Which run used dinov2\n", "which run used dinov2", 4) == ([], "unchanged")  # returned unchanged


def test_the_truncation_rule_per_kind():
    assert parse.parse_lines("one alt\ntwo alt is cu", "q", 2, truncated=True) == (["one alt"], "ok")  # the cut line dropped
    assert parse.parse_lines("one alt\ntwo alt\n", "q", 2, truncated=True) == (["one alt", "two alt"], "ok")
    assert parse.parse_lines("only a cut li", "q", 2, truncated=True) == ([], "unparseable")
    assert parse.parse_hyde("A passage that was cut mid", truncated=True) == ("A passage that was cut mid", "ok")  # used as written
    assert parse.parse_rewrite("a rewritten query that was cu", truncated=True) == (None, "unparseable")
    assert parse.parse_rewrite("a rewritten query\nand more that was cu", truncated=True) == ("a rewritten query", "ok")
    assert parse.parse_grade("yes, because the passa", truncated=True) == ("yes", "ok")


def test_the_grader_and_rewrite_parsers():
    for reply, want in (("yes", "yes"), ("No.", "no"), ("```\nYes\n```", "yes"), ("  **yes**", "yes"), ("YES - it does", "yes")):
        assert parse.parse_grade(reply) == (want, "ok"), reply
    assert parse.parse_grade("Maybe") == (None, "unparseable") and parse.parse_grade("yesterday") == (None, "unparseable")
    assert parse.parse_grade("") == (None, "empty")
    assert parse.parse_rewrite('Rewritten query: "learning rate of run 40"') == ("learning rate of run 40", "ok")
    assert parse.parse_rewrite("\n\n  lr of run_040 config\nexplanation") == ("lr of run_040 config", "ok")
    assert parse.parse_rewrite("   ") == (None, "empty")
    assert parse.parse_hyde("   ") == (None, "empty")


def test_a_cached_call_is_read_by_the_fallback_rule():
    ok = {"reply": "yes", "done_reason": "stop", "prompt_eval_count": 100, "error": None}
    assert arms.read_call(ok, "grade", "q") == ("yes", "ok", False)
    assert arms.read_call({**ok, "error": "HTTP 500"}, "grade", "q") == (None, "failed", False)
    assert arms.read_call({**ok, "prompt_eval_count": 8190}, "grade", "q") == (None, "overflowed", False)
    assert arms.read_call({**ok, "overflowed": True}, "grade", "q") == (None, "overflowed", False)  # the server log said so
    assert arms.read_call({**ok, "reply": "an alt\nanother cu", "done_reason": "length"}, "multiquery", "q") == (["an alt"], "ok", True)


# ---------------------------------------------------------------- the wiring on toy lists


class ToyBackend:
    """Units are letters; the first stage of a text is a fixed list per text; the reranker scores by a table and records
    the queries it is asked with."""

    k = 8

    def __init__(self, lists, scores=None, dense=None):
        self.lists, self.scores, self._dense = lists, scores or {}, dense or {}
        self.rerank_queries, self.searches = [], []

    def first_stage(self, query, query_vec=None):
        self.searches.append((query, query_vec))
        return list(self.lists[query if query_vec is None else ("vec", query_vec)])

    def dense(self, query_vec):
        return list(self._dense[query_vec])

    def hyde_vector(self, hypothesis, query):
        return f"vec:{hypothesis}"

    def rerank(self, query, units):
        self.rerank_queries.append(query)
        table = self.scores.get(query, {})
        ranked = [(u, table.get(u, 0.0)) for u in units]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def finish(self, qid, units):
        return list(units)


def _toy():
    q = "original question"
    lists = {q: list("abcdefgh"), "alt one": list("ijabklmn"), "sub one": list("opqabrst"), "sub two": list("uvwxyzab"),
             "rewrite": list("zyxwvuts"), ("vec", "vec:hyp"): list("hgfedcba")}
    scores = {q: {u: i for i, u in enumerate("hgfedcbaijklmnopqrstuvwxyz")}, "sub one": {"o": 100.0}, "sub two": {"u": 50.0}}
    return q, ToyBackend(lists, scores, {"vec:hyp": list("hgfedcbaxyz")})


def test_union_then_rerank_reads_only_the_original_query_and_per_sub_each_sub_question():
    q, backend = _toy()
    p0, p1 = arms.parents(backend, q)
    view = arms.View(subs=["sub one", "sub two"], grades={u: "yes" for u in "abcdefgh"})
    backend.rerank_queries.clear()
    out = arms.arm_lists(backend, q, view, p0, p1, placements=[("decompose", "union")])
    assert set(backend.rerank_queries) == {q}
    merged = arms.union([p0, list("opqabrst"), list("uvwxyzab")], 8)
    assert out[("decompose", "union")]["units"] == sorted(merged, key=lambda u: -backend.scores[q][u])  # the union's top 20 (of 8), reranked
    backend.rerank_queries.clear()
    out = arms.arm_lists(backend, q, view, p0, p1, placements=[("decompose", "per-sub")])
    assert backend.rerank_queries == [q, "sub one", "sub two"]
    per_sub = out[("decompose", "per-sub")]["units"]
    assert per_sub[:2] == ["o", "u"] and len(per_sub) == 8  # the highest score across lists first; truncated to k
    fired = out[("decompose", "per-sub")]["fired"]
    assert fired == {"lists": 3, "more_than_one_sub": True}


def test_an_empty_reply_gives_the_parents_list():
    q, backend = _toy()
    p0, p1 = arms.parents(backend, q)
    out = arms.arm_lists(backend, q, arms.View(grades={u: "yes" for u in "abcdefgh"}), p0, p1)
    for (t, p), a in out.items():
        assert a["units"] == (p0 if p == "none" else p1), (t, p)


def test_hyde_multiquery_and_post_rerank_wiring():
    q, backend = _toy()
    p0, p1 = arms.parents(backend, q)
    view = arms.View(hypothesis="hyp", alternatives=["alt one"], grades={u: "yes" for u in "abcdefgh"})
    out = arms.arm_lists(backend, q, view, p0, p1)
    assert out[("hyde", "none")]["units"] == list("hgfedcba")  # the HyDE vector's first stage
    assert ("original question", "vec:hyp") in backend.searches  # BM25 still reads the query; the vector replaces the dense leg's
    assert out[("hyde", "pre")]["units"] == [u for u, _ in backend.rerank(q, list("hgfedcba"))]
    assert set(out[("hyde", "post")]["units"]) == set(p1)  # a re-sort of P1's list: nothing added
    assert out[("multiquery", "none")]["units"] == arms.union([p0, list("ijabklmn")], 8)


def test_each_arm_counts_its_whole_pipeline_of_work():
    q, backend = _toy()
    p0, p1 = arms.parents(backend, q)
    view = arms.View(hypothesis="hyp", alternatives=["alt one"], subs=["sub one", "sub two"], rewrite="rewrite", grades={u: "yes" for u in "abcdefgh"})
    out = arms.arm_lists(backend, q, view, p0, p1)
    work = {key: (a["searches"], a["pairs"]) for key, a in out.items()}  # the toy lists have 8 units: P1 reranks 8
    assert work == {("hyde", "none"): (1, 0), ("hyde", "pre"): (1, 8), ("hyde", "post"): (2, 8), ("multiquery", "none"): (2, 0),
                    ("multiquery", "pre"): (2, 8), ("decompose", "none"): (3, 0), ("decompose", "union"): (3, 8),
                    ("decompose", "per-sub"): (3, 24), ("corrective", "none"): (1, 0), ("corrective", "post"): (1, 8)}
    no = arms.View(rewrite="rewrite", grades={u: "no" for u in "abcdefgh"})
    out = arms.arm_lists(backend, q, no, p0, p1, placements=[("corrective", "none"), ("corrective", "post")])
    assert {key: (a["searches"], a["pairs"]) for key, a in out.items()} == {("corrective", "none"): (2, 0), ("corrective", "post"): (2, 16)}
    assert out[("corrective", "post")]["input"] == list(dict.fromkeys(p0 + list("zyxwvuts")))  # P1's input, then the rewrite's top 20


def test_the_corrective_rule_on_toy_lists():
    q, backend = _toy()
    rec = arms.Recorder(backend)
    parent = list("abcdefgh")
    view = arms.View(grades={"a": "no", "b": "yes", "c": "no", "d": "yes", "e": "yes"}, rewrite="rewrite")
    out, fired = arms.corrective(rec, q, parent, view, lambda rw: backend.first_stage(rw), 8)
    assert out == list("bde") + list("fgh") + list("ac") and fired == {"demoted": 2, "rewritten": False}
    all_no = arms.View(grades={u: "no" for u in "abcde"}, rewrite="rewrite")
    out, fired = arms.corrective(rec, q, parent, all_no, lambda rw: backend.first_stage(rw), 8)
    assert out == list("zyxwvuts") and fired == {"demoted": 5, "rewritten": True}  # the new list, then the old top 5: cut at k
    out, _ = arms.corrective(rec, q, parent, all_no, lambda rw: list("zyabx"), 8)
    assert out == list("zyx") + list("abcde")  # the new list without the old top 5, then those 5 in order
    for rewrite in ("", None, "ORIGINAL question"):  # no rewrite: step 2's output stands
        out, fired = arms.corrective(rec, q, parent, replace(all_no, rewrite=rewrite), lambda rw: pytest.fail("no rerun"), 8)
        assert out == list("fgh") + list("abcde") and fired["rewritten"] is False
    with pytest.raises(KeyError):  # a top-G unit with no grade is an error, never a silent yes
        arms.corrective(rec, q, parent, arms.View(grades={}), lambda rw: [], 8)


def test_an_unusable_grade_counts_as_yes():
    q = "a question"
    src = arms.CacheSource("qwen3.5-27b", [], lambda u: f"passage {u}")
    ok = {"reply": "no", "done_reason": "stop", "prompt_eval_count": 50, "error": None}
    entries = [{"key": src.key(k, q), **ok, "reply": ""} for k in spec.QUERY_KINDS]
    entries += [{"key": src.key("grade", q, "a"), **ok, "reply": "", "error": "timeout"},
                {"key": src.key("grade", q, "b"), **ok, "reply": "perhaps"},
                {"key": src.key("grade", q, "c"), **ok, "prompt_eval_count": 8180},
                {"key": src.key("grade", q, "d"), **ok}]
    src = arms.CacheSource("qwen3.5-27b", entries, lambda u: f"passage {u}")
    view = src.view(q, ["a", "b", "c", "d"])
    assert view.grades == {"a": "yes", "b": "yes", "c": "yes", "d": "no"}  # failed, unparseable, overflowed: yes
    assert view.hypothesis is None and view.alternatives == [] and view.subs == [] and view.rewrite is None


def test_a_gemma_arm_cannot_read_a_qwen_entry():
    q = "a question"
    key = spec.cache_key(spec.MODELS["qwen3.5-27b"]["digest"], "hyde", spec.build_prompt("hyde", q))
    qwen = arms.CacheSource("qwen3.5-27b", [{"key": key, "reply": "x"}], lambda u: "p")
    gemma = arms.CacheSource("gemma4-26b", [{"key": key, "reply": "x"}], lambda u: "p")
    assert qwen.key("hyde", q) == key and gemma.key("hyde", q) != key
    assert len(gemma.missing(q, [])) == len(spec.QUERY_KINDS) and len(qwen.missing(q, [])) == len(spec.QUERY_KINDS) - 1


# ---------------------------------------------------------------- the verdict, passes, replication and the decision


def test_one_verdict_from_the_raw_bounds_and_its_display():
    assert bar.verdict_word(4e-05, 0.01) == "above zero" and bar.fmt_bound(4e-05, 0.0) == "+4.0e-05"
    assert bar.verdict_word(0.0, 0.01) == "covers zero"  # a bound exactly 0.0 covers zero
    assert bar.verdict_word(-0.01, -4e-05) == "below zero" and bar.fmt_bound(-4e-05, -0.0) == "-4.0e-05"
    assert bar.fmt_bound(0.0123) == "+0.0123" and bar.fmt_bound(0.0, 0.0) == "+0.0000"
    block = {"ndcg@10": {"delta": 0.001, "lo": 0.0, "hi": 0.002, "delta_raw": 0.001, "lo_raw": 4e-05, "hi_raw": 0.002}, "verdict": "above zero"}
    assert bar.fmt_block(block) == "+0.0010 [+4.0e-05, +0.0020]" and bar.stored_word(block) == "above zero"
    assert bar.mirrored("above zero") == "below zero" and bar.mirrored("covers zero") == "covers zero"


def test_the_paired_block_stores_rounded_and_raw_bounds_and_the_changed_count():
    from app.eval.stats import resample_indices

    a = {"ndcg@10": [0.5, 0.6, 0.7, 0.9], "recall@10": [1.0, 1.0, 1.0, 1.0]}
    b = {"ndcg@10": [0.5, 0.5, 0.7, 0.8], "recall@10": [1.0, 1.0, 1.0, 1.0]}
    block = bar.paired(a, b, resample_indices(4), ("ndcg@10", "recall@10"), bar.permutation_signs(4))
    assert block["n_changed"] == 2 and block["verdict"] in bar.spec.VERDICTS
    assert block["ndcg@10"]["delta"] == round(block["ndcg@10"]["delta_raw"], 4) and math.isclose(block["ndcg@10"]["delta_raw"], 0.05)
    assert 0 < block["permutation_p"] <= 1
    assert bar.bh_qvalues([0.01, 0.04, 0.03, 0.5]) == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert bar.minimum_detectable([0.1, -0.1, 0.1, -0.1]) == pytest.approx(2.8 * 0.11547005 / 2)


def _blobs(cells, gemma=None, direct=None, withheld=()):
    """Toy results files: cells = {(technique, placement): {dataset: word}} for qwen3.5:27b (gemma likewise)."""
    def block(row, other, word):
        lo = {"above zero": 0.01, "covers zero": -0.01, "below zero": -0.03}[word]
        hi = {"above zero": 0.03, "covers zero": 0.01, "below zero": -0.01}[word]
        return {"row": row, "other": other, "kind": "arm", "verdict": word, "n_changed": 3,
                "ndcg@10": {"delta": (lo + hi) / 2, "lo": lo, "hi": hi, "delta_raw": (lo + hi) / 2, "lo_raw": lo, "hi_raw": hi}}

    blobs = {}
    for d in spec.DATASETS:
        comps = []
        for tag, table in ((spec.MAIN_MODEL, cells), (spec.REPLICATION_MODEL, gemma or {})):
            for (t, p), words in table.items():
                if words.get(d):
                    comps.append(block(spec.arm_name(t, p, tag), spec.parent_name(t, p), words[d]))
                    if p == "none":
                        comps.append(block(spec.arm_name(t, p, tag), spec.P1, "below zero"))
        for (x, y), words in (direct or {}).items():
            comps.append(block(x, y, words[d]))
        blobs[d] = {"scopes": {bar.BAR_SCOPE[d]: {"comparisons": comps}}, "reproduction": {"status": "failed" if d in withheld else "reproduces"}}
    return blobs


ALL = {d: "above zero" for d in spec.DATASETS}
TIE = {d: "covers zero" for d in spec.DATASETS}
PASS_FIXTURE_TIE = {"fixture": "covers zero", "scifact": "above zero", "fiqa": "above zero"}


def _everything(word_table, technique):
    return {(t, p): word_table.get((t, p), TIE) for t, p, _, _ in spec.ARMS if t == technique}


def test_passes_all_three_and_open_or_withheld():
    assert bar.passes(PASS_FIXTURE_TIE) == "passes" and bar.all_three(PASS_FIXTURE_TIE) == "no" and bar.all_three(ALL) == "yes"
    assert bar.passes({**PASS_FIXTURE_TIE, "fixture": "below zero"}) == "does not pass"
    assert bar.passes({**PASS_FIXTURE_TIE, "fiqa": None}) == "open"
    assert bar.passes({**PASS_FIXTURE_TIE, "fiqa": "withheld", "scifact": None}) == "withheld"


def test_every_row_of_the_replication_table():
    below = {**PASS_FIXTURE_TIE, "fixture": "below zero"}
    assert bar.replication_label({**ALL, "fiqa": None}, ALL) == "open"
    assert bar.replication_label({**ALL, "fiqa": "withheld"}, ALL) == "withheld"
    assert bar.replication_label(ALL, {**ALL, "fiqa": None}) == "not run"
    assert bar.replication_label(ALL, PASS_FIXTURE_TIE) == "replicated"
    assert bar.replication_label(ALL, TIE) == "not replicated"
    assert bar.replication_label(ALL, below) == "contradicted"
    assert bar.replication_label(TIE, ALL) == "gemma only"
    assert bar.replication_label(below, ALL) == "contradicted"
    assert bar.replication_label(TIE, below) == "neither passes"


def _sentence(blobs, technique):
    d = bar.decide(blobs, technique)
    for word in spec.DECISION["banned_words"]:
        assert word not in d["sentence"], d["sentence"]
    return d


def test_decision_rule_0_open_and_withheld():
    cells = _everything({("hyde", "pre"): {**ALL, "fiqa": None}}, "hyde")
    assert _sentence(_blobs(cells), "hyde")["sentence"] == "HyDE: open (FiQA not measured)."
    assert _sentence(_blobs(_everything({}, "hyde"), withheld=("scifact",)), "hyde")["sentence"] == "HyDE: withheld (SciFact: the parents did not reproduce)."
    two = _everything({("hyde", "pre"): {"fixture": None, "scifact": None, "fiqa": "above zero"}}, "hyde")
    assert _sentence(_blobs(two), "hyde")["sentence"] == "HyDE: open (the fixture and SciFact not measured)."


def test_decision_rule_1_one_reranked_placement():
    cells = _everything({("corrective", "post"): PASS_FIXTURE_TIE}, "corrective")
    gem = _everything({("corrective", "post"): PASS_FIXTURE_TIE}, "corrective")
    d = _sentence(_blobs(cells, gem), "corrective")
    assert d["rule"] == 1 and d["sentence"] == ("Corrective grading passes the declared bar after the reranker. "
                                                "gemma4:26b replication: after the reranker: replicated.")
    cells[("corrective", "none")] = ALL
    d = _sentence(_blobs(cells, gem), "corrective")
    assert d["sentence"] == ("Corrective grading passes the declared bar after the reranker. Without a reranker it also passes against P0. "
                             "gemma4:26b replication: after the reranker: replicated; without a reranker: not replicated.")


def test_decision_rule_2_two_reranked_placements_and_their_direct_comparison():
    cells = _everything({("hyde", "pre"): ALL, ("hyde", "post"): ALL}, "hyde")
    pre, post = (spec.arm_name("hyde", p, spec.MAIN_MODEL) for p in ("pre", "post"))
    d = _sentence(_blobs(cells, direct={(pre, post): {"fixture": "covers zero", "scifact": "above zero", "fiqa": "above zero"}}), "hyde")
    assert d["rule"] == 2 and d["sentence"].startswith("HyDE passes the declared bar before the reranker and after the reranker; "
                                                       "in their direct comparison before the reranker is above after the reranker.")
    d = _sentence(_blobs(cells, direct={(pre, post): {"fixture": "covers zero", "scifact": "below zero", "fiqa": "below zero"}}), "hyde")
    assert "in their direct comparison after the reranker is above before the reranker." in d["sentence"]  # the mirrored word
    d = _sentence(_blobs(cells, direct={(pre, post): {"fixture": "covers zero", "scifact": "above zero", "fiqa": "covers zero"}}), "hyde")
    assert "their direct comparison does not separate them." in d["sentence"]


def test_decision_rules_3_and_4():
    cells = _everything({("multiquery", "none"): ALL}, "multiquery")
    d = _sentence(_blobs(cells), "multiquery")
    assert d["rule"] == 3 and d["sentence"].startswith("Multi-query expansion passes against P0 without a reranker; with the reranker it is "
                                                       "not shown to help.") and set(d["vs_P1"]) == set(spec.DATASETS)
    d = _sentence(_blobs(_everything({}, "multiquery"), _everything({("multiquery", "pre"): ALL}, "multiquery")), "multiquery")
    assert d["rule"] == 4 and d["sentence"] == ("No placement of multi-query expansion passes the declared bar on these datasets. "
                                                "gemma4:26b replication: before the reranker: gemma only.")
    d = _sentence(_blobs(_everything({}, "multiquery"), _everything({}, "multiquery")), "multiquery")
    assert d["sentence"].endswith("gemma4:26b replication: neither model passes any arm.")


def test_one_sentence_per_technique():
    blobs = _blobs({**_everything({}, "hyde"), **_everything({}, "multiquery"), **_everything({}, "decompose"), **_everything({}, "corrective")})
    assert [d["technique"] for d in bar.decisions(blobs)] == list(spec.TECHNIQUES)
    assert all(d["rule"] == 4 for d in bar.decisions(blobs))


# ---------------------------------------------------------------- the pipeline on fakes


class _Overlap:
    """A stand-in CrossEncoder: the logit is the share of query tokens in the passage, minus a bias."""

    default_prompt_name, prompts, max_seq_length = None, {}, 512

    def predict(self, pairs, batch_size=32, show_progress_bar=False, activation_fn=None):
        out = []
        for q, d in pairs:
            qt, dt = set(tokenize(q)), set(tokenize(d))
            x = len(qt & dt) / max(len(qt), 1) - 0.5
            out.append(activation_fn(x) if activation_fn else 1 / (1 + math.exp(-x)))
        return out

    def parameters(self):
        yield type("P", (), {"dtype": "torch.float32"})()


def _fake_reranker():
    from app.retrieval.rerank import cross_encoder_scores

    model = _Overlap()
    return (lambda q, texts: cross_encoder_scores(model, [(q, t) for t in texts])), spec.RETRIEVAL_MODELS["reranker"]["revision"], {"stand_in": True}


def fake_post(calls=None, *, empty=(), fail=()):
    """A fake /api/generate: a plausible reply per kind, recognised from the prompt; `empty` kinds reply nothing,
    `fail` kinds raise a transport error."""
    def post(base_url, payload):
        prompt = payload["prompt"]
        kind = ("grade" if prompt.startswith("Does the passage") else "rewrite" if prompt.startswith("Rewrite the search") else
                "multiquery" if prompt.startswith("Write 2 short") else "decompose" if prompt.startswith("Break the following") else "hyde")
        if calls is not None:
            calls.append((kind, payload))
        if kind in fail:
            return None, "URLError: refused"
        query = prompt.split("Question: ")[-1].split("\n")[0] if kind in ("hyde", "decompose") else prompt.split(": ", 1)[-1].split("\n")[0]
        reply = {"hyde": f"The configuration file states the learning rate and the dataset. {query}",
                 "multiquery": "learning rate config yaml\nexperiment results table",
                 "decompose": "What is the learning rate?\nWhich dataset was used?",
                 "rewrite": "learning rate configuration of the run",
                 "grade": "yes" if "learning" in prompt.split("Passage:")[-1].lower() else "no"}[kind]
        if kind in empty:
            reply = ""
        return {"response": reply, "thinking": "", "done_reason": "stop", "eval_count": 7, "prompt_eval_count": 120,
                "total_duration": 1_000_000, "load_duration": 0}, None
    return post


@pytest.fixture(scope="module")
def tmp_root(tmp_path_factory):
    """A repository root whose inputs are the real committed fixture and results files and whose bench/llm_query and
    M46 results are its own."""
    root = tmp_path_factory.mktemp("m46root")
    (root / "bench" / "results").mkdir(parents=True)
    for d in ("corpus", "gold", "distill"):
        os.symlink(ROOT / "bench" / d, root / "bench" / d)
    for name in ("neural.json", "rerank_depth.json", "frontier_neural.json", "frontier_neural.per_query.json", "frontier_generations.json",
                 "small_rerankers.json", "small_rerankers.per_query.json", "beir_scifact_small_rerankers.per_query.json",
                 "beir_fiqa_small_rerankers.per_query.json"):
        os.symlink(ROOT / "bench" / "results" / name, root / "bench" / "results" / name)
    (root / "bench" / "llm_query").mkdir()
    return root


def _fixture_backend(root, cache=None):
    from app.eval.index_loader import build_index

    index = build_index(root / "bench" / "corpus" / "files", embedder_name="hash", retrieve_k=50, enable_router=False, enable_rerank=False, mtimes="manifest")
    score, _, _ = _fake_reranker()
    return arms.FixtureBackend(index, lambda q, units: score(q, [index.by_id[u].text for u in units]), cache)


def test_the_fixture_parents_rank_as_the_m43_runner_does():
    from app.eval import small_rerankers as sr

    blob, sidecar = sr.run_fixture(ROOT, tags=["bge-rerank"], embedder="hash", load=lambda tag: (_Overlap(), {"model": tag}))
    backend = _fixture_backend(ROOT)
    from app.eval.gold import load_gold

    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    ranked = arms.rank_dataset(backend, [(q.id, q.question) for q in gold], {}, beir=False)
    scores = arms.fixture_scores(backend, gold, ranked, 61)
    for name in (spec.P0, spec.P1):
        for m in ("ndcg@10", "recall@10", "recall@50", "mrr@10"):
            assert scores[name][m] == sidecar["scores"][name][m], (name, m)


def test_the_hyde_path_ranks_as_the_harness_first_stage_with_a_stored_hypothesis():
    from app.eval.gold import load_gold
    from app.eval.harness import StageTimer, _retrieve, BenchConfig

    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")[:12]
    backend = _fixture_backend(ROOT)
    cfg = BenchConfig(name="h", mode="hybrid", embedder="hash", enable_rerank=False, enable_router=False, staleness_tier1=False, hyde=True,
                      hyde_stage="first_stage", e2e=False, log_triples=False)
    for q in gold:
        hyp = f"A document about {q.question} with the config values."
        paths, _ = _retrieve(backend.index, q.question, cfg, StageTimer(), {}, None, hyde_complete=lambda prompt: hyp)
        out = arms.arm_lists(backend, q.question, arms.View(hypothesis=hyp), *arms.parents(backend, q.question), placements=[("hyde", "none")])
        assert backend.paths(out[("hyde", "none")]["units"]) == paths


def _beir_dataset(root: Path, name="scifact", n_docs=60, n_queries=10):
    d = root / name
    (d / "qrels").mkdir(parents=True)
    with (d / "corpus.jsonl").open("w") as fh:
        for i in range(n_docs):
            fh.write(json.dumps({"_id": f"d{i}", "title": f"Topic {i % 7}", "text": f"protein number {i} folds in cells and tissue sample {i % 5} learning"}) + "\n")
    with (d / "queries.jsonl").open("w") as fh:
        for i in range(n_queries + 4):
            fh.write(json.dumps({"_id": f"q{i}", "text": f"protein number {i} in tissue sample {i % 5}", "metadata": {"evidence": {"d777SENTINEL": []}}}) + "\n")
    (d / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"q{i}\td{i}\t1\n" for i in range(n_queries)))
    (d / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"q{i}\tDOCSENTINEL\t9\n" for i in range(n_queries, n_queries + 4)))
    return d


def test_the_beir_parents_rank_as_the_beir_runner_does(tmp_path):
    from app.eval import beir

    droot = _beir_dataset(tmp_path)
    corpus, queries, qrels = beir.load_beir(droot, "test")
    engine = beir.Engine(beir.chunks_from_corpus(corpus), cache_dir=tmp_path / "cache", device="cpu")
    p0 = beir.BeirConfig(name=spec.P0, mode="hybrid", embedder="hash")
    p1 = beir.BeirConfig(name=spec.P1, mode="hybrid", embedder="hash", reranker="overlap", rerank_top=20)
    dense = beir.BeirConfig(name=spec.DENSE, mode="dense", embedder="hash")
    want = {c.name: beir.run_config(engine, c, queries, qrels)["ranking_sha256"] for c in (p0, p1, dense)}
    from app.retrieval.rerank import exact_token_hits, jaccard_overlap

    def overlap(q, units):
        return [exact_token_hits(q, corpus[u]) + jaccard_overlap(q, corpus[u]) + (0.25 if u.lower() in q.lower() else 0.0) for u in units]

    backend = arms.BeirBackend(engine, corpus, overlap, embedder="hash")
    ranked = arms.rank_dataset(backend, list(queries.items()), {}, beir=True)
    qids = list(queries)
    for name in (spec.P0, spec.P1, spec.DENSE):
        assert arms.trec_sha256(name, qids, ranked["rows"][name]) == want[name], name


def test_ids_read_only_the_qrels_id_column(tmp_path):
    d = _beir_dataset(tmp_path)
    ids = gen.qrels_query_ids(d / "qrels" / "train.tsv")
    assert ids == [f"q{i}" for i in range(10, 14)] and "DOCSENTINEL" not in json.dumps(ids) and "9" not in "".join(ids)
    assert gen.seeded_sample([f"x{i}" for i in range(50)], n=5) == gen.seeded_sample([f"x{i}" for i in reversed(range(50))], n=5)


def test_the_committed_query_ids():
    ids = json.loads((ROOT / "bench" / "llm_query" / "query_ids.json").read_text())
    assert len(ids["fixture"]["test"]) == 136 and len(ids["scifact"]["test"]) == 300 and len(ids["fiqa"]["test"]) == 648
    for d in spec.DATASETS:
        assert len(ids[d]["dev"]) == 20 and not set(ids[d]["dev"]) & set(ids[d]["test"])
    for d, rel in gen.M43_SIDECARS.items():
        assert ids[d]["test"] == json.loads((ROOT / rel).read_text())["ids"]
    pool, counts = gen.fixture_dev_pool(ROOT)
    assert ids["fixture"]["dev_pool"] == counts and set(ids["fixture"]["dev"]) <= set(pool) and counts["pool"] == 733
    assert all(i.startswith("fixture:") for i in ids["fixture"]["dev"])


def test_load_queries_reads_text_only_and_no_metadata_reaches_a_prompt(tmp_path):
    d = _beir_dataset(tmp_path)
    queries = gen.load_queries(d, ["q3", "q1"])
    assert list(queries) == ["q3", "q1"] and "SENTINEL" not in json.dumps(queries)
    for text in queries.values():
        for kind in spec.QUERY_KINDS:
            assert "SENTINEL" not in spec.build_prompt(kind, text)


def test_no_gold_field_reaches_a_prompt():
    from app.eval.gold import GoldQuestion

    q = GoldQuestion(id="q1", question="what learning rate did run 40 use", category="simple_factual", answer="ANSWERSENTINEL",
                     relevant_paths=["PATHSENTINEL/run_040.yaml"], notes="NOTESENTINEL")
    lines = gen.worklist_lines({q.id: q.question}, {q.id: {"p0_top": ["u1", "u2"], "p1_top": ["u2", "u3"]}}, lambda u: f"File: x/{u}.txt\nchunk")
    assert [x["kind"] for x in lines] == ["hyde", "multiquery", "decompose", "rewrite", "grade", "grade", "grade"]
    prompts = [spec.build_prompt(k, q.question) for k in spec.QUERY_KINDS] + [spec.build_prompt("grade", q.question, f"File: x/{u}.txt\nchunk") for u in ("u1", "u2", "u3")]
    assert [spec.prompt_sha256(p) for p in prompts] == [x["prompt_sha256"] for x in lines]
    assert not any("SENTINEL" in p for p in prompts)


def test_the_worklist_and_generation_never_open_qrels(tmp_path, tmp_root, monkeypatch):
    """A dataset directory with no qrels/: the worklist, generation and merge all run; scoring is the qrels reader."""
    from app.eval import beir

    data = tmp_path / "data"
    d = _beir_dataset(data)
    for f in (d / "qrels").iterdir():
        f.unlink()
    (d / "qrels").rmdir()
    ids_path = tmp_root / "bench" / "llm_query" / "query_ids.json"
    saved = ids_path.read_text() if ids_path.exists() else None
    ids_path.write_text(json.dumps({"scifact": {"test": [f"q{i}" for i in range(6)], "dev": []}}))
    try:
        corpus = beir.load_corpus(d)
        engine = beir.Engine(beir.chunks_from_corpus(corpus), cache_dir=data / "cache", device="cpu")
        score, _, _ = _fake_reranker()
        backend = arms.BeirBackend(engine, corpus, lambda q, units: score(q, [corpus[u] for u in units]), embedder="hash")
        wl = tmp_path / "wl"
        meta = gen.write_worklist(tmp_root, data, "scifact", "test", wl, rerank_cache=None, gpu="cpu", backend=backend)
        assert meta["n_queries"] == 6 and meta["committed_hash_check"]["checks"] == []
        summary = gen.generate(tmp_root, data, "scifact", "qwen3.5-27b", "1/2", worklist_dir=wl, cache_dir=tmp_path / "cache", base_url="x",
                               gpu="Fake GPU", post=fake_post(), determinism_n=3)
        assert summary["complete"] and summary["determinism"] == {"n": 3, "byte_identical": 3, "log": "determinism-1of2.jsonl"}
    finally:
        if saved is None:
            ids_path.unlink()
        else:
            ids_path.write_text(saved)


def test_generation_resumes_retries_and_is_final_after_two_jobs(tmp_path, tmp_root):
    wl, cache = tmp_path / "wl", tmp_path / "cache"
    backend = _fixture_backend(tmp_root)
    ids_path = tmp_root / "bench" / "llm_query" / "query_ids.json"
    ids_path.write_text(json.dumps({"fixture": {"test": ["q001", "q002"], "dev": []}}))
    try:
        gen.write_worklist(tmp_root, tmp_path, "fixture", "test", wl, rerank_cache=None, gpu="cpu", backend=backend)
        calls = []
        s1 = gen.generate(tmp_root, tmp_path, "fixture", "gemma4-26b", "1/1", worklist_dir=wl, cache_dir=cache, base_url="x", gpu="G", job="1",
                          post=fake_post(calls, fail=("rewrite",)), determinism_n=0)
        n_rewrite = sum(1 for k, _ in calls if k == "rewrite")
        assert n_rewrite == 2 * (gen.RETRIES + 1) and not s1["complete"]  # two queries, each tried 1 + RETRIES times
        calls.clear()
        s2 = gen.generate(tmp_root, tmp_path, "fixture", "gemma4-26b", "1/1", worklist_dir=wl, cache_dir=cache, base_url="x", gpu="G", job="1",
                          post=fake_post(calls), determinism_n=0)
        assert calls == [] and not s2["complete"]  # failed in this job already: the next job re-issues it
        s3 = gen.generate(tmp_root, tmp_path, "fixture", "gemma4-26b", "1/1", worklist_dir=wl, cache_dir=cache, base_url="x", gpu="G", job="2",
                          post=fake_post(calls, fail=("rewrite",)), determinism_n=0)
        assert s3["complete"] and s3["final_failures"] == 2 and all(k == "rewrite" for k, _ in calls)
        calls.clear()
        s4 = gen.generate(tmp_root, tmp_path, "fixture", "gemma4-26b", "1/1", worklist_dir=wl, cache_dir=cache, base_url="x", gpu="G", job="3",
                          post=fake_post(calls), determinism_n=0)
        assert calls == [] and s4["complete"]  # final failures are never re-issued; completed outputs never replaced
        merged = gen.merge_one(tmp_root, tmp_path, "fixture", "gemma4-26b", worklist_dir=wl, cache_dir=cache)
        assert merged["counts"]["rewrite"]["failed"] == 2 and merged["n_calls"] == len(gen.read_worklist(wl, "fixture")[0])
    finally:
        ids_path.unlink()
        for f in (tmp_root / "bench" / "llm_query").glob("fixture.*"):
            f.unlink()


@pytest.fixture(scope="module")
def fixture_pipeline(tmp_root, tmp_path_factory):
    """ids -> worklist -> generate (two fake models) -> merge -> score, on the real fixture with the hash embedder."""
    work = tmp_path_factory.mktemp("m46work")
    gold_ids = [f"q{i:03d}" for i in range(1, 137)]
    (tmp_root / "bench" / "llm_query" / "query_ids.json").write_text(json.dumps({"fixture": {"test": gold_ids, "dev": []}}))
    (tmp_root / "bench" / "llm_query" / "declaration.json").write_text(json.dumps({"D0": "d0", "D1": "d1"}))
    backend = _fixture_backend(tmp_root)
    gen.write_worklist(tmp_root, work, "fixture", "test", work / "wl", rerank_cache=None, gpu="cpu", backend=backend)
    for tag, empty in (("qwen3.5-27b", ()), ("gemma4-26b", ("multiquery",))):
        s = gen.generate(tmp_root, work, "fixture", tag, "1/1", worklist_dir=work / "wl", cache_dir=work / "cache", base_url="x", gpu="Fake GPU",
                         post=fake_post(empty=empty), determinism_n=2)
        assert s["complete"]
    manifest = gen.merge(tmp_root, work, worklist_dir=work / "wl", cache_dir=work / "cache", models=list(spec.BAR_MODELS), datasets=["fixture"])
    assert set(manifest["files"]) == {"fixture.qwen3.5-27b", "fixture.gemma4-26b"} and manifest["incomplete"] == {}
    blob, sidecar = arms.score_fixture(tmp_root, models=list(spec.BAR_MODELS), load_reranker=_fake_reranker, embedder="hash",
                                       rerank_cache=work / "rerank.jsonl", gpu="Fake GPU")
    out = tmp_root / bar.RESULTS["fixture"]
    out.write_text(json.dumps(blob, indent=2) + "\n")
    from app.eval import per_question

    per_question.write(per_question.sidecar_path(out), sidecar)
    return blob, sidecar, work


def test_the_fixture_rows_and_what_each_is_paired_with(fixture_pipeline):
    blob, sidecar, _ = fixture_pipeline
    names = [r["config"] for r in blob["rows"]]
    assert names[:2] == [spec.P0, spec.P1] and len(names) == 2 + 10 + 10 + 10 + 3
    assert set(blob["scopes"]) == {"held_out", "tune", "all_questions"} and blob["scopes"]["held_out"]["n"] == 68
    kinds = blob["n_intervals"]
    per_scope = {"arm": 20, "stand-in": 10, "qwen2.5-7b": 6, "reference": 1, "secondary:llm-minus-heuristic": 20,
                 "secondary:gemma-minus-qwen": 10, "secondary:pre-minus-post": 2, "secondary:per-sub-minus-union": 2,
                 "secondary:no-rerank-minus-P1": 8}
    assert kinds == {k: 3 * v for k, v in sorted(per_scope.items())}
    rows = {r["config"]: r for r in blob["rows"]}
    heur = rows[spec.arm_name("multiquery", "none", "heuristic")]
    assert "query expansion by keyword heuristic (no LLM)" in heur["fallbacks"] and spec.HEURISTIC_LABEL in heur["fallbacks"]
    gm = rows[spec.arm_name("multiquery", "none", "gemma4-26b")]
    assert gm["llm_counts"]["empty"] == 136 and gm["fired"]["lists_changed"] == 0  # every empty reply behaves as the parent
    assert any("parent behaviour on 136 of 136 queries (empty multi-query reply)" == x for x in gm["fallbacks"])
    q = rows[spec.arm_name("decompose", "per-sub", "qwen3.5-27b")]
    assert q["fired"]["more_than_one_sub"] == 136 and q["cost"]["llm_calls"] == 136 and q["cost"]["gpu"] == ["Fake GPU"]
    corr = rows[spec.arm_name("corrective", "none", "qwen3.5-27b")]
    assert corr["cost"]["llm_calls"] >= 136 * spec.G and "queries_with_demotions" in corr["fired"]
    assert all(r.get("ranking_sha256") for r in blob["rows"])
    assert blob["reproduction"]["status"] == "failed"  # the hash embedder is not the committed rows' bge-small
    assert "does_not_show" in blob and blob["llm_query"]["declaration"]["D0"] == "d0"


def test_scoring_needs_no_network_and_stops_on_a_missing_key(fixture_pipeline, tmp_root, monkeypatch):
    import urllib.request

    _, _, work = fixture_pipeline
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("scoring opened a URL"))
    for name in ("heuristic_hypothesis", "llm_hypothesis", "hyde_embed"):
        monkeypatch.setattr(f"app.retrieval.hyde.{name}", lambda *a, **k: pytest.fail("an LLM arm reached a HyDE helper"))
    monkeypatch.setattr("app.agent.decompose.heuristic_decompose", lambda *a, **k: pytest.fail("heuristic_decompose"))
    monkeypatch.setattr("app.agent.decompose.llm_decompose", lambda *a, **k: pytest.fail("llm_decompose"))
    monkeypatch.setattr("app.agent.deep_research.expand_queries", lambda *a, **k: pytest.fail("expand_queries"))
    monkeypatch.setattr("app.agent.deep_research.rrf_union_paths", lambda *a, **k: pytest.fail("rrf_union_paths"))
    blob, _ = arms.score_fixture(tmp_root, models=["qwen3.5-27b"], load_reranker=_fake_reranker, embedder="hash", stretch=True,
                                 rerank_cache=work / "rerank.jsonl", gpu="Fake GPU")  # no stand-ins: every LLM arm ran without them
    assert len([r for r in blob["rows"] if r.get("model") == "qwen3.5-27b"]) == 10
    path = tmp_root / arms.cache_file("fixture", "qwen3.5-27b")
    saved = path.read_bytes()
    rows = arms.load_cache(path)
    gen.write_gz(path, rows[1:])
    try:
        with pytest.raises(arms.MissingKeys):
            arms.score_fixture(tmp_root, models=["qwen3.5-27b"], load_reranker=_fake_reranker, embedder="hash", stretch=True, gpu="Fake GPU")
    finally:
        path.write_bytes(saved)


def test_every_interval_word_and_llm_block_rescores_from_the_sidecar_and_the_cache(fixture_pipeline, tmp_root):
    blob, sidecar, _ = fixture_pipeline
    assert bar.rescore_blob(blob, sidecar) == []
    checked, problems = bar.rescore(tmp_root, results={"fixture": bar.RESULTS["fixture"]})
    assert checked == 1 and problems == []
    moved = copy.deepcopy(blob)
    block = moved["scopes"]["held_out"]["comparisons"][3]
    block["ndcg@10"]["lo_raw"] += 1e-9
    block["verdict"] = "above zero" if block["verdict"] != "above zero" else "covers zero"
    problems = bar.rescore_blob(moved, sidecar)
    assert any("lo_raw" in p for p in problems) and any("verdict" in p for p in problems)
    moved = copy.deepcopy(blob)
    next(r for r in moved["rows"] if r.get("model") == "qwen3.5-27b")["cost"]["llm_calls"] += 1
    assert any("cost.llm_calls" in p for p in bar.rescore_llm_blocks(tmp_root, moved))


def test_the_bar_withholds_a_dataset_whose_parents_did_not_reproduce(fixture_pipeline, tmp_root):
    blobs = bar.load_blobs(tmp_root)
    assert set(blobs) == {"fixture"} and blobs["fixture"]["reproduction"]["status"] == "failed"  # hash embedder, not bge-small
    for row in bar.bar_rows(blobs, spec.MAIN_MODEL):
        assert row["passes"] == "withheld" and row["replication"] == "withheld" and row["words"] == {"fixture": "withheld", "scifact": None, "fiqa": None}
    assert [d["sentence"] for d in bar.decisions(blobs)][0] == "HyDE: withheld (the fixture: the parents did not reproduce)."
    ok = copy.deepcopy(blobs)
    ok["fixture"]["reproduction"]["status"] = "reproduces"
    for row in bar.bar_rows(ok, spec.MAIN_MODEL):  # the stored words, open without the BEIR files
        assert row["passes"] == "open" and row["words"]["fixture"] in spec.VERDICTS
    assert bar.decisions(ok)[0]["sentence"] == "HyDE: open (SciFact and FiQA not measured)."


def test_the_beir_pipeline_end_to_end(tmp_path, tmp_root):
    from app.eval import beir, per_question

    data = tmp_path / "data"
    d = _beir_dataset(data, n_queries=8)
    ids_path = tmp_root / "bench" / "llm_query" / "query_ids.json"
    saved = ids_path.read_text() if ids_path.exists() else "{}"
    ids_path.write_text(json.dumps({**json.loads(saved), "scifact": {"test": [f"q{i}" for i in range(8)], "dev": ["q9"]}}))
    try:
        corpus = beir.load_corpus(d)
        engine = beir.Engine(beir.chunks_from_corpus(corpus), cache_dir=data / "cache", device="cpu")
        score, _, _ = _fake_reranker()
        backend = arms.BeirBackend(engine, corpus, lambda q, units: score(q, [corpus[u] for u in units]), embedder="hash")
        wl, cache = tmp_path / "wl", tmp_path / "cache"
        gen.write_worklist(tmp_root, data, "scifact", "test", wl, rerank_cache=None, gpu="cpu", backend=backend)
        for tag in spec.BAR_MODELS:
            for shard in ("1/2", "2/2"):
                assert gen.generate(tmp_root, data, "scifact", tag, shard, worklist_dir=wl, cache_dir=cache, base_url="x", gpu="Fake GPU",
                                    post=fake_post(), determinism_n=0)["complete"]
        manifest = gen.merge(tmp_root, data, worklist_dir=wl, cache_dir=cache, models=list(spec.BAR_MODELS), datasets=["scifact"])
        assert "scifact.qwen3.5-27b" in manifest["files"] and len(manifest["files"]["scifact.qwen3.5-27b"]["shards"]) == 2
        with gzip.open(tmp_root / "bench" / "llm_query" / "parents.json.gz", "rt") as fh:
            assert "scifact" in json.load(fh)
        blob, sidecar = arms.score_beir(tmp_root, data, "scifact", models=list(spec.BAR_MODELS), load_reranker=_fake_reranker, gpu="Fake GPU",
                                        engine=engine, corpus=corpus, embedder="hash")
        names = [r["config"] for r in blob["rows"]]
        assert names[:3] == [spec.P0, spec.P1, spec.DENSE] and len(names) == 23 and set(blob["scopes"]) == {"test"}
        assert blob["n_intervals"]["secondary:no-rerank-minus-dense"] == 8 and blob["reproduction"]["status"] == "failed"
        out = tmp_root / bar.RESULTS["scifact"]
        out.write_text(json.dumps(blob) + "\n")
        per_question.write(per_question.sidecar_path(out), sidecar)
        checked, problems = bar.rescore(tmp_root, results={"scifact": bar.RESULTS["scifact"]})
        assert checked == 1 and problems == []
        cells = bar.cell(bar.load_blobs(tmp_root), "scifact", spec.arm_name("hyde", "none", spec.MAIN_MODEL), spec.P0)
        assert cells["word"] == "withheld"  # the parents did not reproduce (hash embedder): the cell is withheld
    finally:
        ids_path.write_text(saved)
        for p in (tmp_root / bar.RESULTS["scifact"], per_question.sidecar_path(tmp_root / bar.RESULTS["scifact"])):
            if p.exists():
                p.unlink()


def test_the_bar_cli_prints_from_the_files(fixture_pipeline, tmp_root, capsys, monkeypatch):
    assert gen.print_bar(tmp_root) == 0
    out = capsys.readouterr().out
    assert "fixture ceiling" in out and "qwen3.5:27b" in out and "rule 0: HyDE: withheld" in out and "uncorrected paired nDCG@10 intervals" in out


# ---------------------------------------------------------------- D0, D1 and llm-query-arms-check


def _git(repo, *args):
    import subprocess

    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
                                   cwd=repo, text=True).strip()


def test_the_check_allows_only_the_declared_d0_to_d1_changes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "app" / "eval").mkdir(parents=True)
    for rel in gen.PARTS.values():
        (repo / rel).write_text((ROOT / rel).read_text())
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "D0")
    d0 = _git(repo, "rev-parse", "HEAD")
    a = repo / gen.PARTS["a"]
    a.write_text(a.read_text().replace('"rewrite": 256, "grade": 16}', '"rewrite": 256, "grade": 24}'))
    b = repo / gen.PARTS["b"]
    b.write_text(b.read_text().replace('return (text, "ok") if text else (None, "empty")', 'return (text, "ok") if text.strip() else (None, "empty")'))
    _git(repo, "commit", "-qam", "D1: probe-driven changes")
    d1 = _git(repo, "rev-parse", "HEAD")
    decl = gen.declare(repo, d0, d1)
    assert sorted(decl["d0_to_d1"]["files"]) == sorted([gen.PARTS["a"], gen.PARTS["b"]]) and decl["d0_to_d1"]["changed_functions"] == {"b": ["parse_hyde"], "d": []}
    assert decl["d0_to_d1"]["num_predict"]["D1"]["grade"] == 24
    (repo / "bench" / "llm_query").mkdir(parents=True)
    (repo / gen.DECLARATION).write_text(json.dumps(decl))
    assert gen.arms_check(repo) == []
    manifest = {"files": {"fixture.qwen3.5-27b": {"shards": [{"shard": "1/1", "code_sha": [d0]}]}}}
    (repo / "bench" / "llm_query" / "manifest.json").write_text(json.dumps(manifest))
    assert any("does not descend from D1" in p for p in gen.arms_check(repo))  # a test shard generated at D0
    c = repo / gen.PARTS["c"]
    c.write_text(c.read_text() + "\n# a change\n")
    a.write_text(a.read_text().replace("G = 5", "G = 6"))
    _git(repo, "commit", "-qam", "D1b")
    (repo / gen.DECLARATION).write_text(json.dumps(gen.declare(repo, d0, _git(repo, "rev-parse", "HEAD"))))
    (repo / "bench" / "llm_query" / "manifest.json").unlink()
    problems = gen.arms_check(repo)
    assert any("part (c)" in p for p in problems) and any("part (a) differs" in p for p in problems)


# ---------------------------------------------------------------- reproduction, reproduce-only, the probe's counters


def test_the_beir_reproduction_reads_reproduces_on_the_committed_fiqa_runs():
    """The positive path: P0, P1 and dense@bge-small ranked exactly as the committed TREC runs give the committed hashes,
    the committed means and M43's per-query scores; the reading is 'reproduces'."""
    trec = ROOT / "bench" / "results" / "beir_fiqa_trec"
    qids = json.loads((trec / "query_ids.json").read_text())
    qrels: dict = {}
    for line in (trec / "qrels.txt").read_text().splitlines():
        qid, _, did, grade = line.split()
        qrels.setdefault(qid, {})[did] = int(grade)
    rows, inputs = {}, {}
    for name in (spec.P0, spec.P1, spec.DENSE):
        from app.eval.beir import _slug

        runs: dict = {}
        for line in (trec / f"{_slug(name)}.run").read_text().splitlines():
            qid, _, did, *_ = line.split()
            runs.setdefault(qid, []).append(did)
        rows[name] = [runs.get(q, []) for q in qids]
        inputs[name] = [x[:20] for x in rows[name]]
    ranked = {"rows": rows, "inputs": inputs}
    scores = arms.beir_scores(qids, qrels, ranked)
    rep = arms._beir_reproduction(ROOT, "fiqa", qids, ranked, scores)
    assert rep["status"] == "reproduces", json.dumps(rep, indent=1)[:2000]
    assert sum(1 for c in rep["checks"] if c.get("ranking_equal") is True) == 5  # 3 rows in beir_fiqa.json, 2 in the M43 file
    swapped = copy.deepcopy(ranked)
    swapped["rows"][spec.P1][0] = swapped["rows"][spec.P1][0][:2][::-1] + swapped["rows"][spec.P1][0][2:]
    rep = arms._beir_reproduction(ROOT, "fiqa", qids, swapped, arms.beir_scores(qids, qrels, swapped))
    assert rep["status"] in ("failed", "reproduces in every metric; near-tied units ordered differently in 1 queries")
    assert any(c.get("queries_ordered_differently") == 1 for c in rep["checks"])


def test_reproduce_only_ranks_the_parents_alone(tmp_root):
    blob, sidecar = arms.score_fixture(tmp_root, models=[], load_reranker=_fake_reranker, embedder="hash", reproduce_only=True)
    assert sidecar == {} and blob["reproduce_only"] and blob["reproduction"]["status"] == "failed"  # the hash embedder is not bge-small
    assert {c["row"] for c in blob["reproduction"]["checks"]} == {spec.P0, spec.P1}


def test_the_probe_counts_lines_that_lose_digits():
    assert gen.digits_lost("1. 401k rollover rules\n2) best 2023 tax rules\n- index funds") == 0
    assert gen.digits_lost("401k rollover rules\n2023 tax rules\n3. third") == 2
    assert parse.parse_lines("401k rollover rules", "q", 2) == (["k rollover rules"], "ok")  # the brief's rule as written
