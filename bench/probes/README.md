# Probe files (not gold labels)

These files are derived evaluation inputs that live outside the frozen fixture. Fixture v1
(`bench/corpus/`, `bench/gold/`, the MANIFEST) is not touched by them, and nothing here is a
relevance judgment: no one checked a probe against the corpus.

They are written by a committed script and must equal its output:

    make probes                         # python -m app.eval.probes
    python -m app.eval.probes --check   # exit 1 on drift (tests/eval/test_probes.py runs the same check)

## paraphrases.jsonl

Rewrites of a sample of the gold questions, used by the semantic-cache arm of `make bench-frontier`
(`hybrid+rerank+cache`) to measure true cache hits: after the gold questions have been replayed into
the cache, each probe is looked up (and not added). A hit served from the entry of the question it
rewrites (`source_id`) is a true hit; a hit served from another question's entry is a false hit.

How they were made (`app/eval/probes.py`):

- Sample: in every gold category (categories in name order), the first, the middle and the last
  question in gold order. Eight categories, so 24 probes.
- One rewrite per sampled question. Three rules take turns; a rule that would leave the question
  unchanged passes to the next one:
  - `keywords`: drop question words, auxiliaries, articles and prepositions
    ("what learning rate did run 40 use" becomes "learning rate run 40"; "how many" becomes "number").
  - `ask`: wrap the question in a request ("can you tell me ...?", or "can you ...?" when the
    question starts with an imperative such as open, find or locate).
  - `prefix`: put "quick question: " in front.
- Each probe inherits `category` and `relevant_paths` from the question it rewrites. The rules only
  drop filler words or wrap the question, so the meaning, and with it the inherited paths, should
  stay the same; where a dropped word mattered, the inherited paths can be wrong for the probe.

Fields: `id`, `source_id` (the gold question), `question` (the rewrite), `rule`, `category`,
`relevant_paths`.

These rewrites are written by rules, not by people or a model, so they are closer to each other
and to their source than real users' paraphrases would be.
