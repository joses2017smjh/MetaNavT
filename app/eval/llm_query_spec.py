"""M46, part (a): every value pinned before any LLM call, and every prompt the models receive.

LLM query processing measured with real open LLMs: HyDE, multi-query expansion, decomposition and corrective grading,
each at the placements relative to the cross-encoder that make sense (ARMS), against its exact parent, on fixture v1's
held-out half, BEIR SciFact and BEIR FiQA (milestones2/briefs/M46.md is the design). Nothing here calls a model.

The code is in five parts, by what may change between the pre-probe commit D0 and the declaration commit D1:

  (a) this file: BAR, ARMS, PROMPTS, G, REQUEST, NUM_PREDICT, TRUNCATION, DECISION, REPLICATION, ATTEMPT,
      DEV_QUERIES, MODELS and all prompt construction (templates, builders, the grader passage and its cut, the
      grader candidates). Between D0 and D1 only the NUM_PREDICT assignment may change (compared by syntax tree);
      after D1 nothing.
  (b) app/eval/llm_query_parse.py: PARSE_RULES and the parsers (a fault the probe shows may be fixed before D1).
  (c) app/eval/llm_query_bar.py: the bootstrap wrapper, the verdict word, passes, the replication label, the
      decision rules and their sentences, --rescore (byte-identical from D0).
  (d) app/eval/llm_query_arms.py: the arm wiring and the runners (before D1, only a wiring bug the probe exposes,
      fixed with a failing test first).
  (e) app/eval/llm_query_gen.py and the rest: plumbing.

tests/eval/test_llm_query_arms.py pins every value here, so a later change has to be made on purpose; any change to
(a)-(d) after D1 is attempt 2 (ATTEMPT), whether or not a number was seen.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.agent.decompose import DECOMPOSE_PROMPT
from app.agent.deep_research import MULTI_QUERY_PROMPT
from app.retrieval.hyde import HYDE_PROMPT

ATTEMPT = 1  # a change to parts (a)-(d) after D1 makes the rows that follow attempt 2; attempt 1's rows stay

# ---------------------------------------------------------------- models (pinned by Ollama digest; a run whose /api/tags
# digest differs stops). Licences (M42 brief): Qwen3.5 and Gemma 4 are Apache-2.0.
MODELS: dict[str, dict[str, Any]] = {
    "qwen3.5-27b": {"name": "qwen3.5:27b", "digest": "64ae5b1d651fb2709d5928ccb8dcfb4bb2cc9c2c8001ac34497e36a6cf905b59",
                    "quantization": "Q4_K_M", "parameters": "26.9B", "family": "qwen35", "licence": "apache-2.0", "role": "main"},
    "gemma4-26b": {"name": "gemma4:26b", "digest": "001e5dafc3c77684c2307ebc6ab8e336e10c9b18eca52acf547d72fc83c3ca8c",
                   "quantization": "Q4_K_M", "parameters": "25.2B", "family": "gemma4", "licence": "apache-2.0", "role": "replication"},
    "qwen3.5-122b": {"name": "qwen3.5:122b", "digest": "8b9d11d807c57feb1e2ecb0d6cbf40334c37dcc3523bed5540af4f927f112a37",
                     "quantization": "Q4_K_M", "parameters": "125.1B", "family": "qwen35", "licence": "apache-2.0", "role": "stretch"},
}
MAIN_MODEL, REPLICATION_MODEL, STRETCH_MODEL = "qwen3.5-27b", "gemma4-26b", "qwen3.5-122b"
BAR_MODELS = (MAIN_MODEL, REPLICATION_MODEL)  # the bar applies to every LLM arm of these two; the stretch gets verdict words only
SERVER = {"ollama": "0.40.16", "store": "mnt_work/ollama/models-2026"}
RETRIEVAL_MODELS = {"embedder": {"model": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"},
                    "reranker": {"model": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"}}

# ---------------------------------------------------------------- the request (POST /api/generate, one call at a time)
# The generated-token cap per kind of call. The one assignment that may change between D0 and D1, and only if more than
# 5% of a kind is truncated on the dev queries (Held-out discipline); every other value derives from it.
NUM_PREDICT = {"hyde": 256, "multiquery": 256, "decompose": 256, "rewrite": 256, "grade": 16}
KINDS = ("hyde", "multiquery", "decompose", "rewrite", "grade")
QUERY_KINDS = ("hyde", "multiquery", "decompose", "rewrite")  # one call per query; grade is one call per (query, unit)
CONTEXT_LENGTH = 8192  # OLLAMA_CONTEXT_LENGTH at server launch; num_ctx is not sent
OPTIONS_BASE = {"temperature": 0, "seed": 0}
REQUEST: dict[str, Any] = {
    "endpoint": "/api/generate",
    "stream": False,
    "think": False,
    "keep_alive": "60m",
    "options": "OPTIONS_BASE plus num_predict = NUM_PREDICT[kind]",
    "num_ctx": "not sent (the server's OLLAMA_CONTEXT_LENGTH)",
    "server": {"OLLAMA_CONTEXT_LENGTH": CONTEXT_LENGTH, "OLLAMA_NUM_PARALLEL": 1, "OLLAMA_NOPRUNE": 1, "OLLAMA_NO_CLOUD": 1,
               "LLAMA_ARG_THREADS": 4, "LLAMA_ARG_THREADS_BATCH": 4, "cpus": 4,
               "stretch": {"LLAMA_ARG_THREADS": 16, "LLAMA_ARG_THREADS_BATCH": 16, "cpus": 16}},
    "overflow_guard": "a call is prompt-overflowed when prompt_eval_count + num_predict > OLLAMA_CONTEXT_LENGTH, or the shard's "
                      "server log reports a truncated prompt; it is counted and treated as failed",
    "retries": "a transport error is retried at most twice; there is no retry for content (temperature 0 repeats itself)",
    "load_check": "the job exits 5 unless /api/ps reports size == size_vram and the server log reports every layer offloaded",
}


def options(kind: str) -> dict[str, Any]:
    """The `options` of a call of `kind`."""
    return {**OPTIONS_BASE, "num_predict": NUM_PREDICT[kind]}


def overflowed(prompt_eval_count: int | None, kind: str) -> bool:
    """The prompt-overflow guard: the prompt plus the generated-token cap does not fit the server's context."""
    return prompt_eval_count is not None and prompt_eval_count + NUM_PREDICT[kind] > CONTEXT_LENGTH


# ---------------------------------------------------------------- prompts (only the query text, and for the grader one
# unit's text, reaches a prompt)
N_ALTERNATIVES = 2  # multi-query: expand_queries' prompt with n = 3
MAX_SUBS = 4  # decomposition: llm_decompose's prompt with max_subs = 4
GRADE_PROMPT = ("Does the passage below contain information that helps answer the search query? Reply with one word: yes or no."
                "\n\nSearch query: {query}\n\nPassage:\n{passage}\n\nAnswer:")
REWRITE_PROMPT = ("Rewrite the search query below so that it retrieves the documents that answer it better. Keep its meaning. "
                  "Reply with the rewritten query only, on one line.\n\nQuery: {query}\n\nRewritten query:")
PROMPTS: dict[str, str] = {
    "hyde": HYDE_PROMPT,  # verbatim: the qwen2.5:7b generations of frontier_generations.json used it
    "multiquery": MULTI_QUERY_PROMPT.format(n_alt=N_ALTERNATIVES, query="{query}"),
    "decompose": DECOMPOSE_PROMPT.format(max_subs=MAX_SUBS, query="{query}"),
    "rewrite": REWRITE_PROMPT,
    "grade": GRADE_PROMPT,
}
PROMPT_SOURCES = {"hyde": "app.retrieval.hyde.HYDE_PROMPT", "multiquery": "app.agent.deep_research.MULTI_QUERY_PROMPT, n_alt 2",
                  "decompose": "app.agent.decompose.DECOMPOSE_PROMPT, max_subs 4", "rewrite": "new (M46)", "grade": "new (M46)"}
G = 5  # the corrective grader reads each input list's top G units
GRADER_PASSAGE_CHARS = 2000  # a BEIR document is cut here; a fixture chunk is never cut


def template_sha256(kind: str) -> str:
    return hashlib.sha256(PROMPTS[kind].encode("utf-8")).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_prompt(kind: str, query: str, passage: str | None = None) -> str:
    """The prompt of a call: the template filled with the query text and, for the grader, the unit's passage. Nothing
    else (no answer, category, relevant path or qrel) is a parameter."""
    if kind == "grade":
        if passage is None:
            raise ValueError("a grader prompt needs a passage")
        return PROMPTS[kind].format(query=query, passage=passage)
    if passage is not None:
        raise ValueError(f"a {kind} prompt takes the query only")
    return PROMPTS[kind].format(query=query)


def fixture_passage(path: str, chunk_text: str) -> str:
    """The grader's passage for a fixture chunk: its file's path and its text, uncut (fixture files are small)."""
    return f"File: {path}\n{chunk_text}"


def beir_passage(document_text: str) -> str:
    """The grader's passage for a BEIR document: title + text as beir.load_corpus builds it, cut at GRADER_PASSAGE_CHARS."""
    return document_text[:GRADER_PASSAGE_CHARS]


def grader_candidates(p0: list[str], p1: list[str], g: int = G) -> list[str]:
    """The units the worklist grades for one query: P0's top g, then those of P1's top g not already listed."""
    out = list(dict.fromkeys(p0[:g]))
    out += [u for u in dict.fromkeys(p1[:g]) if u not in out]
    return out


def cache_key(digest: str, kind: str, prompt: str) -> str:
    """A cache entry's key: the model's digest, the options and the prompt (so a gemma entry cannot answer a qwen call)."""
    return hashlib.sha256(json.dumps([digest, options(kind), prompt], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


# Truncation (done_reason "length"), per kind; every truncated call is counted per row.
TRUNCATION = {
    "hyde": "used as written: it is still a passage",
    "multiquery": "the lines completed before the cut are used; the cut line is dropped",
    "decompose": "the lines completed before the cut are used; the cut line is dropped",
    "rewrite": "used only if its first line ended before the cut; otherwise there is no rewrite",
    "grade": "parsed from its first word as usual",
}

# ---------------------------------------------------------------- the rows
UNIT = {"fixture": "chunk", "beir": "document"}
K = {"fixture": 50, "beir": 100}  # every arm's output is truncated to k, the parent's length
RERANK = {"model": "BAAI/bge-reranker-v2-m3", "top": 20, "max_length": 512, "precision": "fp32"}
SUFFIX = "/fp32-512/top20"
RRF_K = 60
P0 = "hybrid@bge-small"  # BM25 + bge-small, RRF k 60; 50 chunks on the fixture, 100 documents on BEIR
P1 = "hybrid+bge-rerank@bge-small" + SUFFIX  # bge-reranker-v2-m3 over the fused top 20 against the ORIGINAL query, tail in fused order
DENSE = "dense@bge-small"  # BEIR only: the reference of a secondary (on FiQA P0 is below it in beir_fiqa.json)
TECHNIQUES = ("hyde", "multiquery", "decompose", "corrective")
TECHNIQUE_NAMES = {"hyde": "HyDE", "multiquery": "Multi-query expansion", "decompose": "Decomposition", "corrective": "Corrective grading"}
PLACEMENT_PHRASES = {"pre": "before the reranker", "union": "before the reranker (union, then rerank)",
                     "per-sub": "with each sub-question reranked against itself", "post": "after the reranker"}
# (technique, placement, parent, name template). "none" is the no-rerank arm (parent P0); the reranked placements have
# parent P1. The reranker reads the ORIGINAL query in every arm except per-sub.
ARMS: tuple[tuple[str, str, str, str], ...] = (
    ("hyde", "none", "P0", "hybrid+hyde[{m}]@bge-small"),
    ("hyde", "pre", "P1", "hybrid+hyde[{m}]+bge-rerank@bge-small" + SUFFIX),
    ("hyde", "post", "P1", "hybrid+bge-rerank+hyde[{m}]@bge-small" + SUFFIX),
    ("multiquery", "none", "P0", "hybrid+multiquery[{m}]@bge-small"),
    ("multiquery", "pre", "P1", "hybrid+multiquery[{m}]+bge-rerank@bge-small" + SUFFIX),
    ("decompose", "none", "P0", "hybrid+decompose[{m}]@bge-small"),
    ("decompose", "union", "P1", "hybrid+decompose[{m}]+bge-rerank@bge-small" + SUFFIX),
    ("decompose", "per-sub", "P1", "hybrid+decompose[{m}]+bge-rerank-per-sub@bge-small" + SUFFIX),
    ("corrective", "none", "P0", "hybrid+corrective[{m}]@bge-small"),
    ("corrective", "post", "P1", "hybrid+bge-rerank+corrective[{m}]@bge-small" + SUFFIX),
)
NOT_MEASURED = {
    ("multiquery", "post"): "an RRF re-sort after the reranker overrides it, which HyDE's post-rerank arm already shows",
    ("corrective", "pre"): "the reranker re-sorts the head and undoes a demotion",
}
HEURISTIC_TAG = "heuristic"  # fixture only: the same placements with the keyword stand-ins
QWEN25_TAG = "qwen2.5-7b"  # fixture only: the three HyDE placements with frontier_generations.json's hypotheses (no new call)
HEURISTIC_LABEL = ("keyword rules written with all 136 fixture questions in view, including the held-out half (the rules date "
                   "from 2026-08-22, the held-out split from 2026-11-09)")


def arm_name(technique: str, placement: str, tag: str) -> str:
    for t, p, _, name in ARMS:
        if (t, p) == (technique, placement):
            return name.format(m=tag)
    raise KeyError((technique, placement))


def parent_name(technique: str, placement: str) -> str:
    for t, p, parent, _ in ARMS:
        if (t, p) == (technique, placement):
            return P0 if parent == "P0" else P1
    raise KeyError((technique, placement))


def reranked(placement: str) -> bool:
    return placement != "none"


# ---------------------------------------------------------------- the bar (declared before any run)
DATASETS = ("fixture", "scifact", "fiqa")  # the fixture's cell is its held-out half
BAR: dict[str, Any] = {
    "name": "M46's declared bar",
    "metric": "nDCG@10 (BEIR: graded qrels, beir.graded_ndcg_at_k; fixture: file-level gold paths, aggregate_retrieval)",
    "comparison": "each arm minus its exact parent (same run, same queries, same first stage, same reranker shape; the LLM step is the only switch)",
    "parents": {"P0": P0, "P1": P1 + ": bge-reranker-v2-m3 over the fused top 20 against the ORIGINAL query, max_length 512, fp32, tail in fused order"},
    "interval": ("paired percentile bootstrap over queries: 2000 resamples, seed 0, 95%, one index matrix per dataset scope "
                 "(app/eval/stats.py); each interval computed once and stored rounded to 4 decimals (delta, lo, hi) and raw "
                 "(delta_raw, lo_raw, hi_raw)"),
    "verdict": ("from the RAW bounds: 'above zero' when lo_raw > 0, 'below zero' when hi_raw < 0, 'covers zero' otherwise "
                "(a bound exactly 0.0 covers zero); stored in the comparison block, and every reader uses the stored word"),
    "passes": ("'above zero' on SciFact AND 'above zero' on FiQA AND not 'below zero' on the fixture's held-out half; 'open' "
               "when a cell is missing; 'withheld' when a dataset's parents did not reproduce"),
    "why_the_fixture_needs_only_not_a_loss": ("the held-out half has 68 questions and P1 is near the fixture's ceiling there "
                                              "(--bar prints P1's held-out nDCG@10 and Recall@50 from small_rerankers.json)"),
    "also_reported": "M43's stricter reading, above zero on all three, as a column; not a decision",
    "applies_to": "every LLM arm of qwen3.5:27b and gemma4:26b; the placement decision uses qwen3.5:27b, gemma4:26b replicates",
    "verdict_words_only": "the qwen3.5:122b stretch rows, the heuristic stand-in rows, the qwen2.5:7b rows and every secondary comparison",
    "datasets": {"fixture": "fixture v1, held-out half: the test half of rerank_arms.split_questions (68); the tune half and all "
                            "136 reported beside it, no choice made on either",
                 "scifact": "BEIR SciFact test: 300 queries, corpus 5,183", "fiqa": "BEIR FiQA test: 648 queries, corpus 57,638",
                 "never_scored": "no BEIR train or dev query is scored for the record"},
    "multiplicity": "no correction, as elsewhere in the repository; each results file stores n_intervals by kind",
}
VERDICTS = ("above zero", "covers zero", "below zero")

# The replication label of an arm (qwen3.5:27b, gemma4:26b at the same arm): the first row that matches gives it.
REPLICATION: tuple[tuple[str, str, str], ...] = (
    ("open or withheld", "any", "the same word, open or withheld"),
    ("not open or withheld", "a dataset missing", "not run"),
    ("passes", "passes", "replicated"),
    ("passes", "does not pass; no below-zero cell", "not replicated"),
    ("passes", "below zero on any dataset", "contradicted"),
    ("does not pass; no below-zero cell", "passes", "gemma only"),
    ("below zero on any dataset", "passes", "contradicted"),
    ("does not pass", "does not pass", "neither passes"),
)

# The placement decision per technique (qwen3.5:27b); the rules apply in order and exactly one gives the sentence.
DECISION: dict[str, Any] = {
    "placements": "the no-rerank arm (parent P0) and the reranked placements (parent P1); a technique has at most two reranked placements",
    "never": "a no-rerank arm is never compared directly with a reranked arm (that delta is mostly the reranker)",
    "rules": (
        "0. any qwen3.5:27b arm of the technique is open or withheld: '<T>: open (<dataset> not measured)' or '<T>: withheld "
        "(<dataset>: the parents did not reproduce)'; nothing else is decided",
        "1. R (the reranked placements that pass) has one placement X: '<T> passes the declared bar <X>.'",
        "2. R has two placements X and Y: if X minus Y is above zero on SciFact AND FiQA and not below zero on the fixture's "
        "held-out half: '<T> passes the declared bar <X> and <Y>; in their direct comparison <X> is above <Y>.' (or Y, X); "
        "otherwise '<T> passes the declared bar <X> and <Y>; their direct comparison does not separate them.'",
        "3. R is empty and N (the no-rerank arm passes against P0) holds: '<T> passes against P0 without a reranker; with the "
        "reranker it is not shown to help.', with the no-rerank arm's delta vs P1 [95% CI] per dataset beside it",
        "4. R is empty and N does not hold: 'No placement of <T> passes the declared bar on these datasets.'",
    ),
    "clause_when_n_holds_in_1_and_2": "Without a reranker it also passes against P0.",
    "placement_phrases": PLACEMENT_PHRASES,
    "replication": "every sentence ends with the gemma4:26b replication label of each arm it names; rule 4 lists each arm whose "
                   "label is not 'neither passes', or says 'neither model passes any arm'",
    "cost": "each named arm's cost (LLM calls, tokens, LLM p50 s per query, GPU) is printed beside the sentence; never part of the bar",
    "banned_words": ("belongs", "earns", "helps only", "does not help"),
}

# ---------------------------------------------------------------- dev queries (the probe; never a test query)
DEV_QUERIES: dict[str, Any] = {
    "n": 20,
    "seed": 0,
    "fixture": {"pool": "bench/distill/queries.jsonl.gz, corpus fixture (M43's synthetic queries; never a gold question)",
                "drop": "every query whose nearest gold question (any of the 136) has token-set Jaccard >= 0.5 "
                        "(app.eval.distill_queries.normalise and jaccard)", "jaccard_drop_at": 0.5},
    "scifact": {"pool": "the query ids of qrels/train.tsv (id column only)"},
    "fiqa": {"pool": "the query ids of qrels/dev.tsv (id column only)"},
    "sample": "the pool's ids sorted, permuted with numpy RandomState(seed), the first n taken",
}
TEST_SPLIT = {"fixture": "bench/gold/questions.jsonl (136)", "scifact": "qrels/test.tsv (300)", "fiqa": "qrels/test.tsv (648)"}

DOES_NOT_SHOW = (
    "Retrieval ranking only: no answer quality (M47), no end-to-end latency on the demo host.",
    "The intervals cover query sampling only. There is one temperature-0 generation per prompt on one GPU type, so the "
    "variability of LLM outputs, prompts and hardware is not in any interval.",
    "One prompt per technique, untuned and shared by all datasets. The HyDE paper used per-task instructions, so these rows "
    "are a lower bound for prompt-engineered variants. SciFact's queries are claims, not questions; the prompts are used as "
    "written. Prompt sensitivity is not measured.",
    "Two Q4_K_M open models served by Ollama, temperature 0, reasoning off. Not API frontier models, not reasoning on.",
    "Latency on HPC GPUs, one request at a time, Q4 weights, prompt caching as Ollama does it. Five grader calls run in "
    "sequence where a server could batch them. Cost is in tokens and GPU seconds, not dollars.",
    "Pretraining exposure to BEIR or the repository cannot be excluded. The audit bounds verbatim copying and path mentions only.",
    "Fixture v1 is synthetic and small (61 files, 71 chunks, 68 held-out questions) and near its ceiling. A tie on the "
    "68-question held-out half is not evidence of no effect.",
    "bge-small as the first stage only. A stronger embedder (M41's Qwen3-Embedding-8B) may leave less room.",
    "bge-reranker-v2-m3 is the only reranker. M43's hybrid+mxbai-rerank-large-v2@bge-small/fp32-512/top20 row is above P1 on "
    "both BEIR sets in the committed files, so a stronger reranker may leave less room.",
    "On FiQA, P0 (hybrid@bge-small) is below dense@bge-small in beir_fiqa.json. A no-rerank win against P0 there is not a win "
    "against the best committed no-LLM first stage (see the secondary column).",
    "The HyDE post-rerank row is one wiring (an RRF re-sort that overrides the reranker), not post-rerank HyDE in general.",
    "The corrective grader is an ungated component: it has no calibration, and its quality is seen only through its effect on "
    "rankings and the descriptive agreement with qrels.",
    "One-round corrective, binary grader, no web search, G = 5 only. No combinations of techniques. Multi-query post-rerank and "
    "corrective pre-rerank are not measured. No multiplicity correction.",
)
