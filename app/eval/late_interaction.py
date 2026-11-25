"""Late interaction (ColBERT MaxSim) on text, measured against its parent (M49): what was declared before any row ran.

PLAN.md row 4.2 said "built (MaxSim), not evaluated": app/multimodal/late_interaction.py scored toy and hash vectors
only, and no text model in the repository produced token vectors. This module names the multi-vector text models
measured, the rows they get, and the bar that decides whether row 4.2 moves, all as constants committed before any job
scored a test query; tests/eval/test_late_interaction.py pins every one of them except ATTEMPT. The constants are
plain literals on purpose: `make late-interaction-bar` reads BAR, SETTINGS and CANDIDATES back out of the declarations
commit with ast.literal_eval and compares them with each results header.

Two roles, both against `hybrid@bge-small` (BM25 + bge-small-en-v1.5, RRF k 60, equal weights), the default path:
  (a) the MaxSim leg as the dense first stage in place of bge-small: `hybrid@<tag>` (and `dense@<tag>` alone);
  (b) MaxSim as the reranker of the fused top 20: `hybrid+<tag>@bge-small/fp32-doc<L>/top20`, L the model's saved
      document length (the `doc` prefix keeps it apart from M43's cross-encoder `/fp32-512/top20` names).
Datasets: fixture v1's held-out half, BEIR SciFact test and BEIR FiQA test, each in two parts (`primary`: the parents,
the named rows and the three cross-encoder references; `extra`: the parents again, the 512-token sensitivity rows and
the stretch rows), each part its own results file.
"""

from __future__ import annotations

# The attempt these files belong to. ATTEMPT 2 exists only after a tripwire trip whose cause is a defect on the
# declared checklist, shown by a unit test that fails before the fix (TRIPWIRE["attempts"]); nothing else changes it.
ATTEMPT = 1

PERMISSIVE_LICENCES = ("apache-2.0", "mit")
CARD_URL = "https://huggingface.co/answerdotai/answerai-colbert-small-v1"

# The two named models (they decide row 4.2) and the two stretch models (they never do). Facts from the Hugging Face API
# on 2026-11-25 and from the files job J0 (m49-downloads 22150528) downloaded at each pinned revision: licence,
# parameters (safetensors header), the main commit, the weights' and the model card's sha256, and the saved settings
# read from artifact.metadata (Stanford-NLP format) or config_sentence_transformers.json (PyLate format). A run refuses
# another snapshot (check_revision) and asserts the settings it reads back from the loaded model against `saved`.
CANDIDATES = {
    "answerai-colbert-small": {
        "model": "answerdotai/answerai-colbert-small-v1", "licence": "apache-2.0", "parameters": 33396864,
        "revision": "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e", "format": "stanford",
        "weights_sha256": "35bb6ef696cc88b3d18b7520273dfd25315f6c47d265142ba13bba38678c92b6", "weights_bytes": 133610664,
        "readme_sha256": "5852bad161ce37c95108ffd1b1d862abdff536f2c9efb66bdc37db688571a7a1",
        "saved": {"dim": 96, "query_length": 32, "query_expansion": True, "expansion_attended": False, "document_length": 300,
                  "skiplist": "punctuation", "markers": ["[unused0]", "[unused1]"]},
        "why": "33M parameters, the size of the parent bge-small and of the MiniLM-L12 cross-encoder",
    },
    "colbertv2": {
        "model": "colbert-ir/colbertv2.0", "licence": "mit", "parameters": 109581056,
        "revision": "c1e84128e85ef755c096a95bdb06b47793b13acf", "format": "stanford",
        "weights_sha256": "3f58890b1dfdfec066ef12ba431fa9d56992da9e30a53489242c4156e37e9017", "weights_bytes": 438349816,
        "readme_sha256": "44e68292037d4fac7db91b9b24ae36535ae5807eef2b323adcc83a615b91751d",
        "saved": {"dim": 128, "query_length": 32, "query_expansion": True, "expansion_attended": False, "document_length": 180,
                  "skiplist": "punctuation", "markers": ["[unused0]", "[unused1]"]},
        "why": "the 2021 reference model of late interaction",
    },
}
STRETCH = {
    "gte-moderncolbert": {
        "model": "lightonai/GTE-ModernColBERT-v1", "licence": "apache-2.0", "parameters": 149015808,
        "revision": "25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639", "format": "pylate",
        "weights_sha256": "a08c46f4ef7c9ffd8486d6531844bf6c47188d42b260ed42db72e86d5304f0e9", "weights_bytes": 596076280,
        "readme_sha256": "8bac6a7e2d6c2de129f643aeb75e6f19e350d6c180e50362d86c0212a8c07b6c",
        "saved": {"dim": 128, "query_length": 48, "query_expansion": False, "expansion_attended": False, "document_length": 300,
                  "skiplist": "punctuation", "markers": ["[Q] ", "[D] "]},
        "why": "chosen as the strongest current open English ColBERT model (a selection on published BEIR scores)",
    },
    "mxbai-edge-colbert-17m": {
        "model": "mixedbread-ai/mxbai-edge-colbert-v0-17m", "licence": "apache-2.0", "parameters": 16797952,
        "revision": "592c6417c1c6687572043408ed1ae5196bce16b1", "format": "pylate",
        "weights_sha256": "63e9d246f0166ceafa8bd63ee52cb929335ca30110efdb28e7f877c7aa641ca1", "weights_bytes": 67195976,
        "readme_sha256": "0d177ea3615e2842ea6523d144d60d35c0cbc9d0e534012c319b4eb6bf7b2fb6",
        "saved": {"dim": 48, "query_length": 48, "query_expansion": False, "expansion_attended": False, "document_length": 512,
                  "skiplist": "punctuation", "markers": ["[Q] ", "[D] "]},
        "why": "the size that could serve on a CPU",
    },
}
SENSITIVITY_LENGTH = 512  # the token budget of the parent bge-small (max_seq_length 512) and of the served reranker
MAXSIM_BLOCK_BYTES = 1073741824  # 1 GiB: the largest float64 intermediate of the batched pass (query tokens x document tokens x 8 B)

SETTINGS = {
    "precision": "fp32",
    "batch_size": 32,
    "primary_document_length": "each model's saved setting: CANDIDATES[tag]['saved']['document_length'] (300 answerai, 180 colbertv2)",
    "sensitivity_document_length": 512,
    "doc512_override": "the backbone Transformer module's document_length, set after loading, read back and asserted; the run records the longest document's token count to show it took effect",
    "query_side": "the model's own markers and query expansion; no instruction or prompt beyond those markers",
    "document_text": {"fixture": "chunk.text, exactly what bge-small's dense leg reads (BM25 also reads search_context)",
                      "beir": "title + ' ' + text from app.eval.beir.load_corpus, as the parent"},
    "scoring": {"role_a": "maxsim_scores: one batched float64 pass per token store and query set, scores rounded to 10 decimals (DENSE_SCORE_DECIMALS), ranked with hybrid.rank_by_score (ties in corpus order)",
                "role_b": "app.multimodal.late_interaction.maxsim per (query, candidate), candidates encoded on the fly per query; stable sort, ties keep the fused order"},
    "maxsim_block_bytes": 1073741824,
    "maxsim_dtype": "float64",
    "fixture": {"retrieve_k": 50, "rerank_top": 20, "candidates": "the fused top 50 chunks, the first 20 reranked, the tail in fused order"},
    "beir": {"k": 100, "rerank_top": 20, "candidates": "the fused top 100 documents, the first 20 reranked", "split": "test"},
    "rrf_k": 60,
    "equivalence": {"pairs_per_dataset": 200, "maxsim_tolerance": 1e-05, "sentence_transformers_tolerance": 0.0001},
    "fingerprint_tolerance": 0.001,
    "runs": {"gzip": "mtime 0, compresslevel 9", "ids_only_if_beir_runs_exceed_bytes": 4000000},
    "device_order": "Turing first (-p gpu,eecs --constraint 'rtx8000|rtx2080'); an A40 only if neither started 12 h after submission",
}

# The published reference numbers (nDCG@10 x 100) on the answerai card, which quotes ColBERTv2's from its paper. They
# are a tripwire (TRIPWIRE), never a bar: other implementations produced them, ColBERTv2's with approximate (PLAID)
# search, and not every setting is stated. J1 checks that each card line below is in the README at the pinned revision,
# whose sha256 is `readme_sha256`, and that each value is in its line.
PUBLISHED = {
    "card": {"url": "https://huggingface.co/answerdotai/answerai-colbert-small-v1", "revision": "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e",
             "readme_sha256": "5852bad161ce37c95108ffd1b1d862abdff536f2c9efb66bdc37db688571a7a1", "table": "Against ColBERTv2.0"},
    "metric": "nDCG@10 x 100",
    "lines": {"scifact": "| **SciFact**     |    **74.77**     |    69.3     |",
              "fiqa": "| **FiQA2018**    |    **41.15**     |    35.6     |"},
    "values": {"answerai-colbert-small": {"scifact": "74.77", "fiqa": "41.15"},
               "colbertv2": {"scifact": "69.3", "fiqa": "35.6"}},
    "note": "other implementations; ColBERTv2's with approximate (PLAID) search; not every setting is stated: a tripwire, never a bar",
}

# Each card's own sentence-transformers example: the query, its four documents, the shapes and the scores the card
# prints. J1 encodes them with the loaded model and checks shapes exactly and scores within SETTINGS'
# fingerprint_tolerance, through sentence-transformers' similarity and through the repository's maxsim. colbertv2's card
# prints no scores, so it has no fingerprint.
FINGERPRINT = {
    "answerai-colbert-small": {
        "card": {"url": "https://huggingface.co/answerdotai/answerai-colbert-small-v1", "revision": "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e",
                 "readme_sha256": "5852bad161ce37c95108ffd1b1d862abdff536f2c9efb66bdc37db688571a7a1"},
        "query": "Which planet is known as the Red Planet?",
        "documents": ["Venus is often called Earth's twin because of its similar size and proximity.",
                      "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                      "Jupiter, the largest planet in our solar system, has a prominent red spot.",
                      "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."],
        "shapes_line": "# (32, 96) (17, 96)", "query_shape": [32, 96], "document_0_shape": [17, 96],
        "scores_line": "# tensor([[30.5692, 31.4895, 31.3029, 31.3072]])", "scores": ["30.5692", "31.4895", "31.3029", "31.3072"],
    },
    "gte-moderncolbert": {
        "card": {"url": "https://huggingface.co/lightonai/GTE-ModernColBERT-v1", "revision": "25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639",
                 "readme_sha256": "8bac6a7e2d6c2de129f643aeb75e6f19e350d6c180e50362d86c0212a8c07b6c"},
        "query": "Which planet is known as the Red Planet?",
        "documents": ["Venus is often called Earth's twin because of its similar size and proximity.",
                      "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                      "Jupiter, the largest planet in our solar system, has a prominent red spot.",
                      "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."],
        "shapes_line": "# (12, 128) (18, 128)", "query_shape": [12, 128], "document_0_shape": [18, 128],
        "scores_line": "# tensor([[11.2577, 11.5113, 11.3457, 11.4518]])", "scores": ["11.2577", "11.5113", "11.3457", "11.4518"],
    },
    "mxbai-edge-colbert-17m": {
        "card": {"url": "https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m", "revision": "592c6417c1c6687572043408ed1ae5196bce16b1",
                 "readme_sha256": "0d177ea3615e2842ea6523d144d60d35c0cbc9d0e534012c319b4eb6bf7b2fb6"},
        "query": "Which planet is known as the Red Planet?",
        "documents": ["Venus is often called Earth's twin because of its similar size and proximity.",
                      "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                      "Jupiter, the largest planet in our solar system, has a prominent red spot.",
                      "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."],
        "shapes_line": "# (12, 48) (18, 48)", "query_shape": [12, 48], "document_0_shape": [18, 48],
        "scores_line": "# tensor([[11.5693, 11.7584, 11.7099, 11.7229]])", "scores": ["11.5693", "11.7584", "11.7099", "11.7229"],
    },
}

# The declared bar. Verdict words are app.eval.report.verdict read off the stored lo and hi (four decimals).
BAR = {
    "name": "M49's all-three bar",
    "metric": "ndcg@10",
    "interval": "paired percentile bootstrap over the queries, 2000 resamples, seed 0, alpha 0.05 (app/eval/stats.py defaults, as M43)",
    "words": ("app.eval.report.verdict on the lo and hi stored in the results file (rounded to four decimals): 'above zero' if and "
              "only if lo > 0; 'below zero' if and only if hi < 0; 'covers zero' otherwise, including a stored lo or hi of exactly 0.0000"),
    "parent": "hybrid@bge-small",
    "datasets": {"fixture": "fixture v1, the held-out (test) half of app.eval.rerank_arms.split_questions (n 68); the tune half and all 136 are reported beside it, nothing is chosen on either",
                 "scifact": "BEIR SciFact test (300 queries, 5,183 documents)",
                 "fiqa": "BEIR FiQA test (648 queries, 57,638 documents)"},
    "only_test_qrels": True,
    "roles": {"a": "hybrid@<tag> vs hybrid@bge-small: BM25 plus the MaxSim leg in place of bge-small, RRF k 60, equal weights (fixture retrieve_k 50, BEIR k 100)",
              "b": "hybrid+<tag>@bge-small/fp32-doc<L>/top20 vs hybrid@bge-small: MaxSim reranks the fused top 20 (fixture: of the fused top 50, the tail in fused order; BEIR: of the fused top 100)"},
    "named_models": ["answerai-colbert-small", "colbertv2"],
    "pass": "PASS if the model's bar comparison in that role is above zero on all three datasets",
    "not_pass": "NOT PASS if it covers zero or is below zero on at least one dataset, with the triple written out; a loss on any dataset is written as a loss",
    "open": ("OPEN if at least one dataset is measured, none covers zero or is below zero, and at least one dataset has no row (its part did not "
             "finish, or could not before 2026-12-15); OPEN is never a pass. A model that loaded but has no row on any dataset is "
             "'not measured: no row finished before 2026-12-15'"),
    "secondary": "dense@<tag> vs dense@bge-small (fixture: dense_only@), the leg alone: reported with its verdict, outside the bar",
    "references": "each named role-(b) row vs hybrid+bge-rerank, hybrid+minilm-l12 and hybrid+gte-modernbert-base@bge-small/fp32-512/top20, recomputed in the same (primary) part: outside the bar",
    "does_not_count": ["the sensitivity rows (document length 512)", "the stretch models", "the tune half", "the all-136 scope", "recall", "MRR",
                       "latency", "storage", "the CARD_CLAIM reading", "the Bonferroni column"],
    "plan_row_4_2": {
        "pass": "measured: `<model>` passes in role (a|b) (claims) ...; against `gte-modernbert-base` (claims) ...; not on the default path: adopting it is an owner decision after the serving bar on the host",
        "no_pass": "measured, not adopted: no named model at its saved settings passes the all-three bar in either role (claims); the MaxSim code stays as a measured, tested option off the default path",
        "open": "appends: the bar is open for `<model>` in role (a|b): <dataset> not measured (<reason>)",
        "sensitivity_or_stretch_all_three": "appends: `<row>` (sensitivity or stretch, outside the declared bar) is above zero on all three datasets; the verdict is unchanged",
        "demo": "labelled a demo only if NO named model can be measured (both fail to load or the fidelity checks, or no GPU row can finish before 2026-12-15), naming the failed check's output, the job id and the NOTES entry",
        "one_measured": "if one named model is measured and the other is not, the row is 'measured' for the measured model, its nine plan-flagged claims only, the other named as 'not measured: <reason>'",
        "not_measured_means": "only a load failure, a failed fidelity check, or no row of that model finishing on any dataset before 2026-12-15; rows that loaded, passed the checks and scored are never withdrawn",
    },
}

MULTIPLICITY = {
    "per_pass": ("each PASS is an intersection-union test: three one-sided tests, each at 2.5% (the lower end of a 95% two-sided "
                 "interval above zero); its false-PASS rate is at most about 2.5% per (model, role)"),
    "row_4_2": ("row 4.2 moves on a union, 'at least one named model passes in at least one role': four named PASS claims give at most "
                "10% for any false PASS (union bound), or at most 20% if the four stretch PASS claims are also reported as findings"),
    "bar_words": ("the twelve per-dataset bar words are uncorrected 95% intervals: if every true effect were zero and the tests "
                  "independent, about 46% of runs (1 - 0.95^12) would show at least one 'above zero' or 'below zero' word by chance"),
    "others": ("every other interval is uncorrected too: the reference, secondary, sensitivity and stretch comparisons add 69 nDCG@10 "
               "intervals (23 per dataset: 11 in the primary part, 12 in the extra part) beside the 12 bar intervals, before the tune "
               "and all-136 scopes and before recall and MRR; each results file states its own count"),
    "bonferroni": {"alpha": 0.0125, "level": 0.9875, "for": "the twelve bar comparisons (0.05 / 4)", "draws": "the same bootstrap draws",
                   "deciding": False},
    "counts": {"bar_per_dataset": 4, "other_primary_per_dataset": 11, "other_extra_per_dataset": 12, "datasets": 3, "bar_total": 12,
               "other_total": 69},
}

# The answerai card says, of the model as a reranker: "it vastly outperforms cross-encoders its size!". Read here,
# outside the bar; it never moves row 4.2.
CARD_CLAIM = {
    "card_line": "If you're interested in using this model as a re-ranker (it vastly outperforms cross-encoders its size!), you can do so via the [rerankers](https://github.com/AnswerDotAI/rerankers) library:",
    "row": "hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20",
    "comparator": "hybrid+minilm-l12@bge-small/fp32-512/top20",
    "supported": "'supported here' only if the row is above zero against the comparator on all three datasets",
    "contradicted": "'contradicted on <datasets>' if it is below zero against the comparator on at least one dataset",
    "not_shown": "'not shown' otherwise, including a missing dataset",
    "excluded": "the doc512 rows are sensitivity rows and are not part of this reading",
    "caveats": ["the comparator was chosen by parameter count while its rows were already known to be weak: its held-out row against the parent in bench/results/small_rerankers.json is below zero",
                "unequal budgets: the MaxSim reranker reads 300 document tokens (answerai's saved length); the cross-encoder reads 512 tokens of query plus document"],
}

TRIPWIRE = {
    "evaluated": "after SciFact (the end of J2), for each named model's primary dense@<tag> SciFact row; `python -m app.eval.late_interaction --tripwire`",
    "low": {"below_bm25": "below the nDCG@10 of the committed bm25 row of bench/results/beir_scifact.json, read from that file at run time",
            "below_published_by": 0.05},
    "high": {"above_published_by": 0.05},
    "why": "exhaustive MaxSim should not lose to approximate search by that much at comparable settings, nor beat it by that much: either direction points to an adapter bug",
    "truncation_explained": ("a low trip of a model whose saved document length is below 512, whose dense@<tag>/doc512 SciFact row neither "
                             "low-trips nor high-trips, is 'truncation-explained' and does not stop FiQA; a high trip has no exception; "
                             "no exception if the doc512 row is missing"),
    "fixture_trip": ("if the run's counts show no fixture chunk cut at a model's primary document length, its primary and doc512 fixture "
                     "rows (dense_only, hybrid and role b) must have equal ranking sha256s, row by row; a difference trips, no exception"),
    "missing_published": ("a PUBLISHED figure absent from the pinned README (not written differently) drops that model's published-relative "
                          "thresholds for that dataset, recorded in NOTES before J2; the bm25 threshold and the fixture trip remain"),
    "checklist": ["settings read-back (dim, lengths, expansion, attention)", "the marker token ids ([unused0] / [unused1]) against artifact.metadata",
                  "the skiplist contents", "the query expansion and attention flags",
                  "rank agreement between maxsim_scores and sentence-transformers' similarity on 200 SciFact queries' top 100",
                  "transformers' mapping of the bert.* keys (no missing or unexpected keys; linear.weight loaded)"],
    "attempts": ("only a defect found on the checklist, shown by a unit test that fails before the fix and is not about a setting, licenses "
                 "ATTEMPT 2; at most one defect-driven attempt; a trip in ATTEMPT 2 is recorded, not investigated, and FiQA runs anyway; "
                 "the fix commit changes no pinned constant except ATTEMPT; ATTEMPT 2 reruns every declared row on every dataset and decides "
                 "the verdict, complete or not; ATTEMPT 1's files stay; no defect found: ATTEMPT 1's rows stand and FiQA runs at the same commit"),
    "mechanism": ("J2 writes all its files, then --tripwire writes a record and exits non-zero on a trip that is not truncation-explained; "
                  "J3a and J3b start only with J2's record saying 'clear' or 'truncation-explained', or INVESTIGATED=<NOTES heading>, or ATTEMPT=2"),
}

# Every row of every part, in run order (a model's rows next to each other, so it is released after its last one). The
# fixture names the dense leg alone `dense_only@` (as neural.json), BEIR `dense@` (as beir_*.json); the BEIR embedder of
# a role-(a) row is `mv:<tag>` or `mv:<tag>/doc512`. Each row is paired with its `parent` (the parents have none);
# `kind` says what its parent delta is: bar, secondary, reference, sensitivity or stretch. A named role-(b) row is also
# paired with each of its `references` in the same part.
ROWS = {
    "fixture": {
        "primary": [
            {"name": "dense_only@bge-small", "parent": None, "role": "parent", "kind": "parent"},
            {"name": "hybrid@bge-small", "parent": None, "role": "parent", "kind": "parent"},
            {"name": "dense_only@answerai-colbert-small", "parent": "dense_only@bge-small", "role": "a", "kind": "secondary", "model": "answerai-colbert-small", "document_length": 300},
            {"name": "hybrid@answerai-colbert-small", "parent": "hybrid@bge-small", "role": "a", "kind": "bar", "model": "answerai-colbert-small", "document_length": 300},
            {"name": "dense_only@colbertv2", "parent": "dense_only@bge-small", "role": "a", "kind": "secondary", "model": "colbertv2", "document_length": 180},
            {"name": "hybrid@colbertv2", "parent": "hybrid@bge-small", "role": "a", "kind": "bar", "model": "colbertv2", "document_length": 180},
            {"name": "hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "bar", "model": "answerai-colbert-small", "document_length": 300, "references": ["hybrid+bge-rerank@bge-small/fp32-512/top20", "hybrid+minilm-l12@bge-small/fp32-512/top20", "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"]},
            {"name": "hybrid+colbertv2@bge-small/fp32-doc180/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "bar", "model": "colbertv2", "document_length": 180, "references": ["hybrid+bge-rerank@bge-small/fp32-512/top20", "hybrid+minilm-l12@bge-small/fp32-512/top20", "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"]},
            {"name": "hybrid+bge-rerank@bge-small/fp32-512/top20", "parent": "hybrid@bge-small", "role": "reference", "kind": "reference", "reranker": "bge-rerank"},
            {"name": "hybrid+minilm-l12@bge-small/fp32-512/top20", "parent": "hybrid@bge-small", "role": "reference", "kind": "reference", "reranker": "minilm-l12"},
            {"name": "hybrid+gte-modernbert-base@bge-small/fp32-512/top20", "parent": "hybrid@bge-small", "role": "reference", "kind": "reference", "reranker": "gte-modernbert-base"},
        ],
        "extra": [
            {"name": "dense_only@bge-small", "parent": None, "role": "parent", "kind": "parent"},
            {"name": "hybrid@bge-small", "parent": None, "role": "parent", "kind": "parent"},
            {"name": "dense_only@answerai-colbert-small/doc512", "parent": "dense_only@bge-small", "role": "a", "kind": "sensitivity", "model": "answerai-colbert-small", "document_length": 512},
            {"name": "hybrid@answerai-colbert-small/doc512", "parent": "hybrid@bge-small", "role": "a", "kind": "sensitivity", "model": "answerai-colbert-small", "document_length": 512},
            {"name": "dense_only@colbertv2/doc512", "parent": "dense_only@bge-small", "role": "a", "kind": "sensitivity", "model": "colbertv2", "document_length": 512},
            {"name": "hybrid@colbertv2/doc512", "parent": "hybrid@bge-small", "role": "a", "kind": "sensitivity", "model": "colbertv2", "document_length": 512},
            {"name": "hybrid+answerai-colbert-small@bge-small/fp32-doc512/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "sensitivity", "model": "answerai-colbert-small", "document_length": 512},
            {"name": "hybrid+colbertv2@bge-small/fp32-doc512/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "sensitivity", "model": "colbertv2", "document_length": 512},
            {"name": "dense_only@gte-moderncolbert", "parent": "dense_only@bge-small", "role": "a", "kind": "stretch", "model": "gte-moderncolbert", "document_length": 300},
            {"name": "hybrid@gte-moderncolbert", "parent": "hybrid@bge-small", "role": "a", "kind": "stretch", "model": "gte-moderncolbert", "document_length": 300},
            {"name": "hybrid+gte-moderncolbert@bge-small/fp32-doc300/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "stretch", "model": "gte-moderncolbert", "document_length": 300},
            {"name": "dense_only@mxbai-edge-colbert-17m", "parent": "dense_only@bge-small", "role": "a", "kind": "stretch", "model": "mxbai-edge-colbert-17m", "document_length": 512},
            {"name": "hybrid@mxbai-edge-colbert-17m", "parent": "hybrid@bge-small", "role": "a", "kind": "stretch", "model": "mxbai-edge-colbert-17m", "document_length": 512},
            {"name": "hybrid+mxbai-edge-colbert-17m@bge-small/fp32-doc512/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "stretch", "model": "mxbai-edge-colbert-17m", "document_length": 512},
        ],
    },
    "beir": {
        "primary": [
            {"name": "dense@bge-small", "parent": None, "role": "parent", "kind": "parent"},
            {"name": "hybrid@bge-small", "parent": None, "role": "parent", "kind": "parent"},
            {"name": "dense@answerai-colbert-small", "parent": "dense@bge-small", "role": "a", "kind": "secondary", "model": "answerai-colbert-small", "document_length": 300},
            {"name": "hybrid@answerai-colbert-small", "parent": "hybrid@bge-small", "role": "a", "kind": "bar", "model": "answerai-colbert-small", "document_length": 300},
            {"name": "dense@colbertv2", "parent": "dense@bge-small", "role": "a", "kind": "secondary", "model": "colbertv2", "document_length": 180},
            {"name": "hybrid@colbertv2", "parent": "hybrid@bge-small", "role": "a", "kind": "bar", "model": "colbertv2", "document_length": 180},
            {"name": "hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "bar", "model": "answerai-colbert-small", "document_length": 300, "references": ["hybrid+bge-rerank@bge-small/fp32-512/top20", "hybrid+minilm-l12@bge-small/fp32-512/top20", "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"]},
            {"name": "hybrid+colbertv2@bge-small/fp32-doc180/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "bar", "model": "colbertv2", "document_length": 180, "references": ["hybrid+bge-rerank@bge-small/fp32-512/top20", "hybrid+minilm-l12@bge-small/fp32-512/top20", "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"]},
            {"name": "hybrid+bge-rerank@bge-small/fp32-512/top20", "parent": "hybrid@bge-small", "role": "reference", "kind": "reference", "reranker": "bge-rerank"},
            {"name": "hybrid+minilm-l12@bge-small/fp32-512/top20", "parent": "hybrid@bge-small", "role": "reference", "kind": "reference", "reranker": "minilm-l12"},
            {"name": "hybrid+gte-modernbert-base@bge-small/fp32-512/top20", "parent": "hybrid@bge-small", "role": "reference", "kind": "reference", "reranker": "gte-modernbert-base"},
        ],
        "extra": [
            {"name": "dense@bge-small", "parent": None, "role": "parent", "kind": "parent"},
            {"name": "hybrid@bge-small", "parent": None, "role": "parent", "kind": "parent"},
            {"name": "dense@answerai-colbert-small/doc512", "parent": "dense@bge-small", "role": "a", "kind": "sensitivity", "model": "answerai-colbert-small", "document_length": 512},
            {"name": "hybrid@answerai-colbert-small/doc512", "parent": "hybrid@bge-small", "role": "a", "kind": "sensitivity", "model": "answerai-colbert-small", "document_length": 512},
            {"name": "dense@colbertv2/doc512", "parent": "dense@bge-small", "role": "a", "kind": "sensitivity", "model": "colbertv2", "document_length": 512},
            {"name": "hybrid@colbertv2/doc512", "parent": "hybrid@bge-small", "role": "a", "kind": "sensitivity", "model": "colbertv2", "document_length": 512},
            {"name": "hybrid+answerai-colbert-small@bge-small/fp32-doc512/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "sensitivity", "model": "answerai-colbert-small", "document_length": 512},
            {"name": "hybrid+colbertv2@bge-small/fp32-doc512/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "sensitivity", "model": "colbertv2", "document_length": 512},
            {"name": "dense@gte-moderncolbert", "parent": "dense@bge-small", "role": "a", "kind": "stretch", "model": "gte-moderncolbert", "document_length": 300},
            {"name": "hybrid@gte-moderncolbert", "parent": "hybrid@bge-small", "role": "a", "kind": "stretch", "model": "gte-moderncolbert", "document_length": 300},
            {"name": "hybrid+gte-moderncolbert@bge-small/fp32-doc300/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "stretch", "model": "gte-moderncolbert", "document_length": 300},
            {"name": "dense@mxbai-edge-colbert-17m", "parent": "dense@bge-small", "role": "a", "kind": "stretch", "model": "mxbai-edge-colbert-17m", "document_length": 512},
            {"name": "hybrid@mxbai-edge-colbert-17m", "parent": "hybrid@bge-small", "role": "a", "kind": "stretch", "model": "mxbai-edge-colbert-17m", "document_length": 512},
            {"name": "hybrid+mxbai-edge-colbert-17m@bge-small/fp32-doc512/top20", "parent": "hybrid@bge-small", "role": "b", "kind": "stretch", "model": "mxbai-edge-colbert-17m", "document_length": 512},
        ],
    },
}
PLAN_FLAGS = [
    ["fixture:hybrid@answerai-colbert-small", "fixture:hybrid@bge-small"],
    ["scifact:hybrid@answerai-colbert-small", "scifact:hybrid@bge-small"],
    ["fiqa:hybrid@answerai-colbert-small", "fiqa:hybrid@bge-small"],
    ["fixture:hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20", "fixture:hybrid@bge-small"],
    ["scifact:hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20", "scifact:hybrid@bge-small"],
    ["fiqa:hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20", "fiqa:hybrid@bge-small"],
    ["fixture:hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20", "fixture:hybrid+gte-modernbert-base@bge-small/fp32-512/top20"],
    ["scifact:hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20", "scifact:hybrid+gte-modernbert-base@bge-small/fp32-512/top20"],
    ["fiqa:hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20", "fiqa:hybrid+gte-modernbert-base@bge-small/fp32-512/top20"],
    ["fixture:hybrid@colbertv2", "fixture:hybrid@bge-small"],
    ["scifact:hybrid@colbertv2", "scifact:hybrid@bge-small"],
    ["fiqa:hybrid@colbertv2", "fiqa:hybrid@bge-small"],
    ["fixture:hybrid+colbertv2@bge-small/fp32-doc180/top20", "fixture:hybrid@bge-small"],
    ["scifact:hybrid+colbertv2@bge-small/fp32-doc180/top20", "scifact:hybrid@bge-small"],
    ["fiqa:hybrid+colbertv2@bge-small/fp32-doc180/top20", "fiqa:hybrid@bge-small"],
    ["fixture:hybrid+colbertv2@bge-small/fp32-doc180/top20", "fixture:hybrid+gte-modernbert-base@bge-small/fp32-512/top20"],
    ["scifact:hybrid+colbertv2@bge-small/fp32-doc180/top20", "scifact:hybrid+gte-modernbert-base@bge-small/fp32-512/top20"],
    ["fiqa:hybrid+colbertv2@bge-small/fp32-doc180/top20", "fiqa:hybrid+gte-modernbert-base@bge-small/fp32-512/top20"],
]

# The names of the pinned constants (tests/eval/test_late_interaction.py pins each by value; ATTEMPT is checked to be 1
# or 2). Results headers record DECLARATIONS_COMMIT, the commit that added them.
PINNED = ("CANDIDATES", "STRETCH", "PUBLISHED", "FINGERPRINT", "SETTINGS", "SENSITIVITY_LENGTH", "MAXSIM_BLOCK_BYTES", "BAR",
          "MULTIPLICITY", "CARD_CLAIM", "ROWS", "PLAN_FLAGS", "TRIPWIRE", "PERMISSIVE_LICENCES", "CARD_URL")


def pinned() -> dict:
    """{name: value} of every pinned constant."""
    return {name: globals()[name] for name in PINNED}
