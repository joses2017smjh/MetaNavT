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


# ================================================================ not pinned: written after the declarations commit

# The commit that added the pinned constants above (and the test that pins them); every results header records it and
# `make late-interaction-bar` checks it is an ancestor of the file's git_sha, with BAR, SETTINGS and CANDIDATES equal.
DECLARATIONS_COMMIT = "26585e8429556e7424a7679d4bbf28523862dd3e"  # added in the next commit: a commit cannot name itself

# The stated training data of every model on both sides of a comparison (leakage check 1 of the brief), from the model
# cards at the revisions the rows load (and answerai's and colbertv2's artifact.metadata). `lists_scifact_or_fiqa`: True
# if the statement names SciFact or FiQA as training data, False if it names its training data and neither is in it,
# None when the card does not say (contamination cannot be ruled out from outside, either way).
TRAINING_DATA = {
    "answerai-colbert-small": {"stated": "MS MARCO triplets, by the JaColBERTv2.5 recipe (artifact.metadata: triples, collection and queries under msmarco/)",
                               "source": "artifact.metadata and the model card at the pinned revision", "lists_scifact_or_fiqa": False},
    "colbertv2": {"stated": "MS MARCO Passage Ranking (the card: 'trained on the MS MARCO Passage Ranking task'; artifact.metadata: MSMARCO triples and collection)",
                  "source": "the model card and artifact.metadata at the pinned revision", "lists_scifact_or_fiqa": False},
    "gte-moderncolbert": {"stated": "knowledge distillation on MS MARCO (lightonai/ms-marco-en-bge-gemma), the card; the card's NanoSciFact and NanoFiQA2018 rows are evaluations",
                          "source": "the model card at the pinned revision", "lists_scifact_or_fiqa": False},
    "mxbai-edge-colbert-17m": {"stated": "not stated on the card, which cites its tech report ('Fantastic (small) Retrievers and How to Train Them'), not read here",
                               "source": "the model card at the pinned revision", "lists_scifact_or_fiqa": None},
    "bge-small": {"stated": ("MTEB's training-data annotation of BAAI/bge-small-en-v1.5 (bge_training_data, from BAAI-MTP): NQ, NQHardNegatives, "
                             "AmazonReviewsClassification, MLQARetrieval, DuRetrieval, and pretraining pairs not in MTEB (wudao, cmrc2018, "
                             "simclue, csl, amazon_reviews_multi, wiki_atomic_edits, mlqa, xlsum, wikipedia, reddit, stackexchange, s2orc); "
                             "neither SciFact nor FiQA; the annotation itself marks the sentence-transformers data as not checked"),
                  "source": "mteb@551a5b54b1dfa07441deef71681a352b05c913b8 mteb/models/model_implementations/bge_models.py (sha256 4c7ae018...); the card at 5c38ec7c405e links BAAI-MTP",
                  "lists_scifact_or_fiqa": False,
                  "corpus_note": "s2orc (title, abstract) pairs: SciFact's corpus is S2ORC abstracts, so the parent may have read SciFact documents (not its claims or labels)"},
    "bge-rerank": {"stated": ("the card: 'a mixture of multilingual datasets': bge-m3-data, quora train data, fever train data; MTEB's annotation of "
                              "bge-m3's data (bge_m3_training_data): MIRACL, Mr.TyDi, MS MARCO, NQ, HotpotQA, T2, DuRetrieval, CodeSearchNet and "
                              "others, with s2orc among the pretraining data; neither SciFact nor FiQA"),
                   "source": "BAAI/bge-reranker-v2-m3's card at 953dc6f6f85a; mteb@551a5b54b1df bge_models.py", "lists_scifact_or_fiqa": False,
                   "corpus_note": "s2orc in bge-m3's pretraining: it may have read SciFact documents (not its claims or labels)"},
    "minilm-l12": {"stated": "MS MARCO Passage Ranking (the card; datasets: sentence-transformers/msmarco)",
                   "source": "cross-encoder/ms-marco-MiniLM-L12-v2's card at 7b0235231ca2", "lists_scifact_or_fiqa": False},
    "gte-modernbert-base": {"stated": "not listed on the card, which refers to the mGTE paper's training scheme, not read here",
                            "source": "Alibaba-NLP/gte-reranker-modernbert-base's card at f7481e605550", "lists_scifact_or_fiqa": None},
}

import argparse  # noqa: E402  (below the literals on purpose: ast reads the constants above)
import gzip  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import string  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = {**CANDIDATES, **STRETCH}
PARENT = "hybrid@bge-small"
MV_PREFIX = "mv:"
REFERENCE_TAGS = ("bge-rerank", "minilm-l12", "gte-modernbert-base")  # app.eval.small_rerankers tags of the three references
DATASETS = ("fixture", "scifact", "fiqa")
PARTS = ("primary", "extra")


class SettingsMismatch(RuntimeError):
    """A loaded model's settings, read back, are not the declared ones: the run stops."""


class FidelityCheckFailed(RuntimeError):
    """A loaded model failed a fidelity check (load-and-score, fingerprint, equivalence, card): the run stops."""


# ---------------------------------------------------------------- names


def results_file(dataset: str, part: str, attempt: int | None = None) -> str:
    """bench/results/<name>.json of a dataset's part; ATTEMPT 2 files carry `_attempt2` before `.json`."""
    attempt = ATTEMPT if attempt is None else attempt
    stem = {"fixture": "late_interaction", "scifact": "beir_scifact_late_interaction", "fiqa": "beir_fiqa_late_interaction"}[dataset]
    return f"bench/results/{stem}{'_extra' if part == 'extra' else ''}{'_attempt2' if attempt == 2 else ''}.json"


RUNS_DIR = "bench/results/late_interaction_runs"


def mv_name(tag: str, document_length: int) -> str:
    """The BEIR embedder / reranker name of a model at a document length: mv:<tag>, or mv:<tag>/doc512 off its saved length."""
    return MV_PREFIX + tag + ("" if document_length == MODELS[tag]["saved"]["document_length"] else f"/doc{document_length}")


def parse_mv(name: str) -> tuple[str, int]:
    """mv:<tag>[/doc<L>] -> (tag, document length)."""
    if not name.startswith(MV_PREFIX):
        raise ValueError(f"{name!r} is not a multi-vector name")
    body = name[len(MV_PREFIX):]
    tag, _, suffix = body.partition("/doc")
    if tag not in MODELS:
        raise KeyError(f"{tag!r} is not a declared multi-vector model: {sorted(MODELS)}")
    return tag, int(suffix) if suffix else MODELS[tag]["saved"]["document_length"]


def rows(dataset: str, part: str, skip: set[str] | None = None) -> list[dict]:
    """The declared rows of a dataset's part (ROWS), without those of the models in `skip` (a stretch model not loaded)."""
    return [r for r in ROWS["fixture" if dataset == "fixture" else "beir"][part] if r.get("model") not in (skip or set())]


def comparisons(names: list[str]) -> list[tuple[str, str]]:
    """Beside each row's delta against its parent: a named role-(b) row against each of its references."""
    present = set(names)
    out = []
    for part in PARTS:
        for r in ROWS["beir"][part] + ROWS["fixture"][part]:
            if r["name"] in present:
                out += [(r["name"], ref) for ref in r.get("references", []) if ref in present and (r["name"], ref) not in out]
    return out


def kind_of(name: str) -> dict:
    """The declared row (role, kind, model, document length) of a row name, in either naming."""
    for part in PARTS:
        for r in ROWS["beir"][part] + ROWS["fixture"][part]:
            if r["name"] == name:
                return r
    raise KeyError(name)


def check_revision(tag: str, revision: str | None) -> None:
    """A model's cached snapshot must be the pinned commit: a moved main branch is another model."""
    pinned = MODELS[tag]["revision"]
    if revision != pinned:
        raise RuntimeError(f"{tag}: the cached revision is {revision}, the pinned one is {pinned}")


# ---------------------------------------------------------------- the model, checked


def load_encoder(tag: str, document_length: int, device: str | None = None):
    """The pinned model at a document length, through the adapter (app.multimodal.late_interaction). Tests replace this
    function with a stub encoder; the run asserts its settings and passes its checks either way."""
    from app.eval.provenance import hf_revision
    from app.multimodal.late_interaction import MultiVectorTextEncoder

    c = MODELS[tag]
    check_revision(tag, hf_revision(c["model"]))
    return MultiVectorTextEncoder(c["model"], revision=c["revision"], device=device, batch_size=SETTINGS["batch_size"],
                                  document_length=document_length if document_length != c["saved"]["document_length"] else None)


def check_settings(encoder, tag: str, document_length: int) -> dict[str, Any]:
    """The settings the loaded model reads back, against the declared ones (`saved`, and the document length asked for):
    dim, query length, expansion on and not attended (named models), document length, a non-empty punctuation skiplist,
    the markers as single tokens, no processing override of the text length. Raises SettingsMismatch."""
    read = encoder.settings()
    declared = MODELS[tag]["saved"]
    want = {"dim": declared["dim"], "query_length": declared["query_length"], "query_expansion": declared["query_expansion"],
            "expansion_attended": declared["expansion_attended"], "document_length": document_length}
    wrong = {k: [v, read.get(k)] for k, v in want.items() if read.get(k) != v}
    if declared["skiplist"] == "punctuation" and (set(read.get("skiplist_words") or []) != set(string.punctuation) or not read.get("skiplist_ids")):
        wrong["skiplist"] = ["string.punctuation, resolved to token ids", read.get("skiplist_words")]
    for side, marker in zip(("query", "document"), declared["markers"]):
        # the loader prepends the marker as text with a trailing space ("[unused0] "; PyLate's "[Q] "): compare the token
        m = (read.get("markers") or {}).get(side) or {}
        if (m.get("prompt") or "").strip() != marker.strip() or not m.get("single_piece"):
            wrong[f"marker_{side}"] = [marker, m]
    if (read.get("text_max_length_override") or None) is not None:
        wrong["text_max_length_override"] = [None, read["text_max_length_override"]]
    if wrong:
        raise SettingsMismatch(f"{tag} at document length {document_length}: declared vs read back {wrong}")
    return {**read, "asserted": sorted(want) + ["skiplist", "markers"]}


def check_late_interaction(encoder, name: str = "") -> dict[str, Any]:
    """M41's load-and-score texts through MaxSim (app/eval/models_2026.py): the answer and the answer with text appended
    past the document cap (CHECK_LONG) must both outscore the other passage, and every score be finite. Raises
    app.eval.models_2026.ModelCheckFailed otherwise."""
    from app.eval.models_2026 import CHECK_ANSWER, CHECK_LONG, CHECK_OTHER, CHECK_QUERY, ModelCheckFailed
    from app.multimodal.late_interaction import maxsim

    q = encoder.encode_queries([CHECK_QUERY])[0]
    docs = encoder.encode_documents([CHECK_ANSWER, CHECK_OTHER, CHECK_LONG])
    scores = [maxsim(q, d) for d in docs]
    out = {"query": CHECK_QUERY, "score_answer": round(scores[0], 6), "score_other": round(scores[1], 6), "score_answer_long": round(scores[2], 6),
           "tokens": {"query": int(q.shape[0]), "answer": int(docs[0].shape[0]), "other": int(docs[1].shape[0]), "answer_long": int(docs[2].shape[0])}}
    if not all(math.isfinite(s) for s in scores):
        raise ModelCheckFailed(f"{name}: a MaxSim score is not finite ({out})")
    out["passed"] = bool(scores[0] > scores[1] and scores[2] > scores[1])
    if not out["passed"]:
        raise ModelCheckFailed(f"{name}: an answering passage does not outscore the other passage ({out})")
    return out


def check_fingerprint(encoder, tag: str) -> dict[str, Any] | None:
    """The card's own example (FINGERPRINT): shapes exactly, scores within the declared tolerance, through
    sentence-transformers' similarity and through the repository's maxsim. None for a model whose card prints none."""
    from app.multimodal.late_interaction import maxsim

    f = FINGERPRINT.get(tag)
    if f is None:
        return None
    q = encoder.encode_queries([f["query"]])[0]
    docs = encoder.encode_documents(f["documents"])
    st = encoder.similarity([q], docs)[0].tolist()
    mine = [maxsim(q, d) for d in docs]
    card = [float(s) for s in f["scores"]]
    tol = SETTINGS["fingerprint_tolerance"]
    out = {"query_shape": list(q.shape), "document_0_shape": list(docs[0].shape), "card_scores": f["scores"],
           "sentence_transformers": [round(s, 4) for s in st], "maxsim": [round(s, 4) for s in mine],
           "max_abs_difference": {"sentence_transformers": max(abs(a - b) for a, b in zip(st, card)),
                                  "maxsim": max(abs(a - b) for a, b in zip(mine, card))}, "tolerance": tol}
    out["passed"] = bool(out["query_shape"] == f["query_shape"] and out["document_0_shape"] == f["document_0_shape"]
                         and max(out["max_abs_difference"].values()) <= tol)
    if not out["passed"]:
        raise FidelityCheckFailed(f"{tag}: the card's fingerprint does not reproduce ({out})")
    return out


def check_equivalence(encoder, pairs: list[tuple[str, str]], device: str | None = None) -> dict[str, Any]:
    """maxsim, maxsim_scores (numpy, and torch on `device` when given) and sentence-transformers' similarity on the same
    (query, document) texts: their largest absolute differences, against the declared tolerances."""
    import numpy as np

    from app.multimodal.late_interaction import TokenStore, maxsim, maxsim_scores

    qs = encoder.encode_queries([q for q, _ in pairs])
    ds = encoder.encode_documents([d for _, d in pairs])
    mine = np.array([maxsim(q, d) for q, d in zip(qs, ds)])
    store = TokenStore.from_matrices(ds, encoder.dim)
    batched = maxsim_scores(qs, store, block_bytes=MAXSIM_BLOCK_BYTES).diagonal()
    st = np.array([encoder.similarity([q], [d])[0, 0] for q, d in zip(qs, ds)])
    out = {"pairs": len(pairs), "max_abs_difference": {"maxsim_scores_vs_maxsim": float(np.max(np.abs(batched - mine))),
                                                        "maxsim_scores_vs_sentence_transformers": float(np.max(np.abs(batched - st)))}}
    if device is not None:
        torch_batched = maxsim_scores(qs, store, block_bytes=MAXSIM_BLOCK_BYTES, device=device).diagonal()
        out["max_abs_difference"]["maxsim_scores_torch_vs_numpy"] = float(np.max(np.abs(torch_batched - batched)))
    eq = SETTINGS["equivalence"]
    out["passed"] = bool(out["max_abs_difference"]["maxsim_scores_vs_maxsim"] <= eq["maxsim_tolerance"]
                         and out["max_abs_difference"]["maxsim_scores_vs_sentence_transformers"] <= eq["sentence_transformers_tolerance"])
    return out


def card_checks(tag: str, readme: Path | None) -> dict[str, Any]:
    """Fidelity check 5: the README at the pinned revision has the declared sha256, and every PUBLISHED and
    FINGERPRINT line (and CARD_CLAIM's sentence, for answerai) is in it verbatim."""
    if readme is None or not Path(readme).exists():
        return {"passed": False, "error": "README.md is not in the cache"}
    data = Path(readme).read_bytes()
    lines = data.decode("utf-8").splitlines()
    sha = hashlib.sha256(data).hexdigest()
    found: dict[str, bool] = {}
    if tag == "answerai-colbert-small":
        for dataset, line in PUBLISHED["lines"].items():
            found[f"published_{dataset}"] = line in lines and all(v[dataset] in line for v in PUBLISHED["values"].values())
        found["card_claim"] = CARD_CLAIM["card_line"] in lines
    if tag in FINGERPRINT:
        f = FINGERPRINT[tag]
        found["fingerprint_shapes"] = f["shapes_line"] in lines
        found["fingerprint_scores"] = f["scores_line"] in lines
        found["fingerprint_texts"] = f'query = "{f["query"]}"' in lines and all(f'    "{d}",' in lines for d in f["documents"])
    out = {"readme_sha256": sha, "sha256_equal": sha == MODELS[tag]["readme_sha256"], "found": found}
    out["passed"] = out["sha256_equal"] and all(found.values())
    return out


# ---------------------------------------------------------------- one model at one document length, over one corpus

EQUIVALENCE_RANK_QUERIES = 200  # TRIPWIRE checklist 5: rank agreement with sentence-transformers on these queries' top 100


def store_key(corpus_sha256: str, tag: str, document_length: int, side: str = "document") -> str:
    """A token store's cache key: corpus, model, revision, side and document length (stores of two lengths never meet)."""
    c = MODELS[tag]
    return hashlib.sha256(json.dumps([corpus_sha256, c["model"], c["revision"], side, int(document_length)]).encode()).hexdigest()


class MvLeg:
    """The MaxSim leg of one model at one document length over one corpus, shared by the BEIR engine (app/eval/beir.py,
    every `mv:` branch) and the fixture's LateInteractionIndex. Loading asserts the declared settings and runs the
    load-and-score check and the card's fingerprint. build_store() encodes every document once into a TokenStore (memory
    only, never a shared cache); prepare() runs the one batched float64 pass for a query set, keeps each query's ranked
    top k (scores rounded to DENSE_SCORE_DECIMALS, hybrid.rank_by_score: ties in corpus order) and the score matrix (for
    role (b)'s stored-versus-on-the-fly cross-check), checks 200 of its scores against maxsim and sentence-transformers,
    and drops the store unless asked to keep it. rerank() is role (b): maxsim per (query, candidate), candidates encoded
    on the fly, a stable sort (ties keep the fused order)."""

    def __init__(self, name: str, chunks: list, *, corpus_sha256: str, device: str | None = None, check_models: bool = True,
                 keep_store: bool = False, encoder=None):
        self.name = name
        self.tag, self.document_length = parse_mv(name)
        self.chunks = list(chunks)
        self.ids = [c.chunk_id for c in self.chunks]
        self.index_of = {cid: i for i, cid in enumerate(self.ids)}
        self.keep_store = keep_store
        self.key = store_key(corpus_sha256, self.tag, self.document_length)
        t0 = time.perf_counter()
        self.encoder = encoder if encoder is not None else load_encoder(self.tag, self.document_length, device)
        self.load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        c = MODELS[self.tag]
        self.provenance: dict[str, Any] = {
            "mv_name": name, "model": c["model"], "tag": self.tag, "pinned_revision": c["revision"], "licence": c["licence"], "neural": True,
            "kind": "stretch" if self.tag in STRETCH else ("sensitivity" if self.document_length != c["saved"]["document_length"] else "named"),
            "document_length": self.document_length, "store_key": self.key,
            **(self.encoder.provenance() if hasattr(self.encoder, "provenance") else {}),
            "settings_read_back": check_settings(self.encoder, self.tag, self.document_length)}
        if self.provenance.get("weights_sha256") not in (None, c["weights_sha256"]):
            raise FidelityCheckFailed(f"{self.tag}: weights sha256 {self.provenance['weights_sha256']}, declared {c['weights_sha256']}")
        if self.provenance.get("all_parameters_dtype") not in (None, ["torch.float32"]):  # SETTINGS precision fp32, read back from the weights
            raise FidelityCheckFailed(f"{self.tag}: the weights loaded as {self.provenance['all_parameters_dtype']}, not fp32")
        if check_models:
            self.provenance["load_and_score"] = check_late_interaction(self.encoder, self.tag)
            fp = check_fingerprint(self.encoder, self.tag)
            if fp is not None:
                self.provenance["fingerprint"] = fp
        self.score_device = "cuda" if str(self.provenance.get("device", "")).startswith("cuda") else None
        self.store = None
        self.matrix = None
        self.rows: dict[str, int] = {}
        self.topk: dict[str, list[tuple[str, float]]] = {}
        self._rerank_checked: set[str] = set()
        self._rerank_differs: set[str] = set()

    # -- role (a)
    def build_store(self) -> None:
        """Encode every document's text once (chunk.text on the fixture, title + ' ' + text on BEIR), with the counts the
        run records: documents cut at the document length, documents with no scored token, vectors and bytes."""
        import numpy as np

        from app.multimodal.late_interaction import TokenStore

        if self.store is not None or self.matrix is not None:
            return
        texts = [c.text for c in self.chunks]
        t0 = time.perf_counter()
        mats = self.encoder.encode_documents(texts)
        encode_s = time.perf_counter() - t0
        tokenized = self.encoder.token_counts(texts, "document") if hasattr(self.encoder, "token_counts") else [int(m.shape[0]) for m in mats]
        self.store = TokenStore.from_matrices(mats, self.encoder.dim)
        del mats
        scored = self.store.lengths()
        n, dim, total = len(self.store), self.store.dim, int(scored.sum())
        self.provenance["store"] = {
            "documents": n, "dim": dim, "document_length": self.document_length,
            "cut_at_document_length": int(sum(1 for t in tokenized if t > self.document_length)),
            "longest_tokenized": int(max(tokenized) if tokenized else 0), "longest_scored": int(scored.max() if n else 0),
            "zero_scored_tokens": int((scored == 0).sum()), "scored_vectors": total, "mean_scored_per_document": round(total / max(n, 1), 3),
            "bytes": {"fp32": total * dim * 4, "fp16": total * dim * 2, "bge_small_fp32": n * 384 * 4},
            "ratio_to_bge_small": {"fp32": round(total * dim * 4 / max(n * 384 * 4, 1), 3), "fp16": round(total * dim * 2 / max(n * 384 * 4, 1), 3)},
            "encode_seconds": round(encode_s, 2), "where": "memory only (never the shared embedding cache)",
            "document_text": "chunk.text" if self.chunks and self.chunks[0].chunk_id != self.chunks[0].path else "title + ' ' + text",
        }
        if not np.isfinite(self.store.vectors).all():
            raise FidelityCheckFailed(f"{self.name}: a token vector is not finite")

    def prepare(self, queries, k: int) -> None:
        """The batched pass for every query text (a dict of id -> text, or a list), once: each query's ranked top k."""
        import numpy as np

        from app.multimodal.late_interaction import maxsim, maxsim_scores
        from app.retrieval.hybrid import DENSE_SCORE_DECIMALS, rank_by_score

        texts = list(dict.fromkeys(queries.values() if isinstance(queries, dict) else queries))
        if all(t in self.rows for t in texts):
            return
        if self.store is None:
            raise RuntimeError(f"{self.name}: prepare() after the store was dropped; build_store() first")
        q_mats = self.encoder.encode_queries(texts)
        q_tokens = self.encoder.token_counts(texts, "query") if hasattr(self.encoder, "token_counts") else [int(m.shape[0]) for m in q_mats]
        q_len = MODELS[self.tag]["saved"]["query_length"]
        t0 = time.perf_counter()
        matrix = maxsim_scores(q_mats, self.store, block_bytes=MAXSIM_BLOCK_BYTES, device=self.score_device)
        pass_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        rounded = np.round(matrix, DENSE_SCORE_DECIMALS)
        k = min(int(k), len(self.ids))
        for i, text in enumerate(texts):
            self.rows[text] = i
            self.topk[text] = [(self.ids[j], float(rounded[i, j])) for j in rank_by_score(rounded[i], k)]
        rank_s = time.perf_counter() - t1
        del rounded
        self.matrix = matrix
        self.provenance["queries"] = {"n": len(texts), "query_length": q_len, "over_query_length": int(sum(1 for t in q_tokens if t > q_len)),
                                      "longest_tokenized": int(max(q_tokens) if q_tokens else 0), "scored_rows": int(sum(int(m.shape[0]) for m in q_mats))}
        self.provenance["batched_pass"] = {
            "label": "batched, amortised: one float64 pass over every (query, document) pair, then a top-k per query; the per-query `dense` stage of this row's latency only reads that list",
            "device": self.score_device or "cpu (numpy)", "block_bytes": MAXSIM_BLOCK_BYTES, "wall_ms": round(pass_s * 1000.0, 2),
            "ms_per_query_amortised": round(pass_s * 1000.0 / max(len(texts), 1), 3), "ranking_ms": round(rank_s * 1000.0, 2),
            "queries": len(texts), "documents": len(self.ids), "k": k}
        self.provenance["equivalence"] = self._equivalence(texts, q_mats, maxsim)
        if not self.keep_store:
            self.store = None

    def _equivalence(self, texts: list[str], q_mats: list, maxsim) -> dict[str, Any]:
        """200 (query, document) scores read back from the matrix that ranked them, against maxsim (1e-5) and
        sentence-transformers' similarity (1e-4); and, on up to EQUIVALENCE_RANK_QUERIES queries, whether
        sentence-transformers orders each query's top 100 as the matrix does (TRIPWIRE checklist 5). Recorded, not asserted."""
        n_pairs = SETTINGS["equivalence"]["pairs_per_dataset"]
        d_max, st_max = 0.0, 0.0
        for p in range(n_pairs):
            i = p % len(texts)
            top = self.topk[texts[i]]
            j = self.index_of[top[(p * 7) % len(top)][0]]
            q, d = q_mats[i], self.store.doc(j)
            d_max = max(d_max, abs(float(self.matrix[i, j]) - maxsim(q, d)))
            st_max = max(st_max, abs(float(self.matrix[i, j]) - float(self.encoder.similarity([q], [d])[0, 0])))
        same, diff_max, checked = 0, 0.0, 0
        for i, text in enumerate(texts[:EQUIVALENCE_RANK_QUERIES]):
            cols = [self.index_of[cid] for cid, _ in self.topk[text][:100]]
            st = self.encoder.similarity([q_mats[i]], [self.store.doc(j) for j in cols])[0]
            mine = [self.matrix[i, j] for j in cols]
            diff_max = max(diff_max, max(abs(float(a) - float(b)) for a, b in zip(st, mine)))
            same += int(sorted(range(len(cols)), key=lambda x: -float(st[x])) == list(range(len(cols))))
            checked += 1
        eq = SETTINGS["equivalence"]
        return {"pairs": n_pairs, "max_abs_difference": {"matrix_vs_maxsim": d_max, "matrix_vs_sentence_transformers": st_max},
                "tolerance": {"maxsim": eq["maxsim_tolerance"], "sentence_transformers": eq["sentence_transformers_tolerance"]},
                "within_tolerance": bool(d_max <= eq["maxsim_tolerance"] and st_max <= eq["sentence_transformers_tolerance"]),
                "rank_agreement": {"queries": checked, "top": 100, "same_order_as_sentence_transformers": same,
                                   "max_abs_difference": diff_max,
                                   "note": "sentence-transformers scores in float32, so near-ties may swap; recorded for the tripwire checklist"}}

    def hits(self, text: str, k: int) -> list[tuple[str, float]]:
        if text not in self.topk:
            raise KeyError(f"{self.name}: the query was not in the batched pass (Engine.prepare)")
        return self.topk[text][:k]

    # -- role (b)
    def rerank(self, query: str, pairs: list) -> list:
        """maxsim per (query, candidate) with the candidates encoded now (a served reranker without a token index); a
        stable sort, so ties keep the fused order. The head must come back as a permutation of itself. When this
        model's role-(a) matrix exists for the query, the order its stored scores give is compared (recorded)."""
        from app.multimodal.late_interaction import maxsim

        q = self.encoder.encode_queries([query])[0]
        docs = self.encoder.encode_documents([c.text for c, _ in pairs])
        scores = [maxsim(q, d) for d in docs]
        order = sorted(range(len(pairs)), key=lambda i: scores[i], reverse=True)
        ranked = [(pairs[i][0], scores[i]) for i in order]
        if sorted(c.chunk_id for c, _ in ranked) != sorted(c.chunk_id for c, _ in pairs):
            raise RuntimeError(f"{self.name}: the reranked head is not a permutation of the candidates")
        if self.matrix is not None and query in self.rows:
            row = self.rows[query]
            stored = [self.matrix[row, self.index_of[c.chunk_id]] for c, _ in pairs]
            self._rerank_checked.add(query)
            if sorted(range(len(pairs)), key=lambda i: stored[i], reverse=True) != order:
                self._rerank_differs.add(query)
        return ranked

    def cross_check(self) -> dict[str, Any]:
        return {"queries_checked": len(self._rerank_checked), "queries_order_differs": len(self._rerank_differs),
                "what": "the same candidates ordered by the role-(a) store's scores (documents encoded in corpus batches) against on-the-fly encoding; batch padding can change floats"}

    def release(self) -> None:
        self.store = self.matrix = None
        self.rows, self.topk = {}, {}
        self.encoder = None


def _late_interaction_index_class():
    from app.retrieval.hybrid import InMemoryHybridIndex

    class LateInteractionIndex(InMemoryHybridIndex):
        """The fixture's role-(a) index: InMemoryHybridIndex with the dense leg replaced by a MaxSim leg (MvLeg) over the
        same chunks. Built with enable_router=False and enable_rerank=False; __init__ still fits the identical BM25 leg
        (search_context + text) and encodes the chunks with its HashEmbedder, which nothing here reads."""

        def __init__(self, chunks, leg: MvLeg, retrieve_k: int = 50):
            super().__init__(chunks, retrieve_k=retrieve_k, enable_router=False, enable_rerank=False)
            if [c.chunk_id for c in self.chunks] != leg.ids:
                raise ValueError("the MaxSim leg was built over other chunks")
            self.leg = leg

        def search_dense(self, query, k=None, query_vec=None):
            if query_vec is not None:
                raise ValueError("a MaxSim leg reads the query text, not a vector")
            return [(self.by_id[cid], score) for cid, score in self.leg.hits(query, k or self.retrieve_k)]

    return LateInteractionIndex


# ---------------------------------------------------------------- BEIR (the `late-interaction` ladder of app/eval/beir.py)

LATE_INTERACTION_REPRODUCE = {  # (committed file, rows), per dataset: the parents and the references, recomputed in each part
    "scifact": (("bench/results/beir_scifact.json", ("dense@bge-small", "hybrid@bge-small", "hybrid+bge-rerank@bge-small/fp32-512/top20")),
                ("bench/results/beir_scifact_models_2026.json", ("dense@bge-small", "hybrid@bge-small", "hybrid+bge-rerank@bge-small/fp32-512/top20")),
                ("bench/results/beir_scifact_small_rerankers.json", ("hybrid@bge-small", "hybrid+bge-rerank@bge-small/fp32-512/top20",
                                                                     "hybrid+minilm-l12@bge-small/fp32-512/top20", "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"))),
    "fiqa": (("bench/results/beir_fiqa.json", ("dense@bge-small", "hybrid@bge-small", "hybrid+bge-rerank@bge-small/fp32-512/top20")),
             ("bench/results/beir_fiqa_small_rerankers.json", ("hybrid@bge-small", "hybrid+bge-rerank@bge-small/fp32-512/top20",
                                                               "hybrid+minilm-l12@bge-small/fp32-512/top20", "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"))),
}
EXPECTED_REPRODUCTION = ("Expected, declared before any run (not defects): on a Turing GPU (RTX 8000 or 2080 Ti) FiQA's recomputed rankings "
                         "equal the committed Turing rows; on an A40, FiQA's dense@bge-small, hybrid@bge-small and bge rows give "
                         "ranking_equal false against beir_fiqa.json (adjacent near-tie swaps, equal metrics, M41); on SciFact the bge row's "
                         "ranking differs between files of different GPU architectures. Retrieval-block equality is the criterion everywhere.")

DOES_NOT_SHOW = [
    "a PLAID or compressed index: this is exhaustive, uncompressed MaxSim, the exact score PLAID-style search approximates; not ANN latency at scale; storage is for uncompressed vectors",
    "that it serves: no pgvector MaxSim, no served path; HPC timings are indicative; the serving bar on the host is not decided here; the image's sentence-transformers is 6.3, 6.1 here",
    "equivalence or 'no effect': a NOT PASS, and an interval that covers zero, is not evidence of equivalence; each dataset's paired half-width is the rough smallest gain that could have passed",
    "first-stage recall on the fixture: role (a) ranks 71 chunks with retrieve_k 50, so the fixture's Recall@50 sits beside its random baseline and is no recall claim",
    "results free of selection: the models were chosen knowing their published BEIR scores, which include SciFact and FiQA; only the fixture is free of that",
    "a level field: RRF k 60, retrieve_k, the reranking depth and the 512 budget were set around bge-small; the MaxSim legs are not re-tuned",
    "equal budgets in role (b): MaxSim reads 180 or 300 document tokens (512 in the sensitivity rows), the cross-encoders 512 of query plus document",
    "fine-tuned or domain-adapted ColBERT: zero-shot only",
    "that BEIR is free of training contamination: it cannot be checked from outside",
    "answer quality or the real corpus: retrieval metrics only; the fixture is small (68 held-out questions), templated and judged by file",
    "ColPali pages or image retrieval (PLAN rows 4.1, 4.3 and 4.4 unchanged)",
    "late interaction in general: two named models at their saved settings plus one 512 sensitivity length, and two stretch models",
]


def beir_configs(part: str, skip: set[str] | None = None) -> list:
    """The BEIR rows of a part (ROWS["beir"]) as BeirConfigs: the bge-small parents with no parent of their own, a
    role-(a) row on the `mv:` embedder, a role-(b) row with the `mv:` reranker over the fused top 20 of hybrid@bge-small
    (max_length = the document length), and the three cross-encoder references at M43's shape (fp32, 512, top 20)."""
    from app.eval import small_rerankers as sr
    from app.eval.beir import BGE_SMALL, BeirConfig

    top = SETTINGS["beir"]["rerank_top"]
    out = []
    for r in rows("beir", part, skip):
        mode = "dense" if r["name"].startswith("dense@") else "hybrid"
        if r["role"] == "parent":
            out.append(BeirConfig(name=r["name"], mode=mode, embedder=BGE_SMALL, parent=None))
        elif r["role"] == "a":
            out.append(BeirConfig(name=r["name"], mode=mode, embedder=mv_name(r["model"], r["document_length"]), parent=r["parent"]))
        elif r["role"] == "b":
            out.append(BeirConfig(name=r["name"], mode="hybrid", embedder=BGE_SMALL, reranker=mv_name(r["model"], r["document_length"]),
                                  rerank_top=top, precision=SETTINGS["precision"], max_length=r["document_length"], parent=r["parent"]))
        else:
            out.append(BeirConfig(name=r["name"], mode="hybrid", embedder=BGE_SMALL, reranker=sr.model_of(r["reranker"]), rerank_top=sr.SHAPE["rerank_top"],
                                  precision=sr.SHAPE["precision"], max_length=sr.SHAPE["max_length"], parent=r["parent"]))
    return out


def reference_resolver(root: Path | None = None):
    """The references' loader hook (M43's: a candidate's cached revision must be the pinned one)."""
    from app.eval import small_rerankers as sr

    return sr.resolver(None, root)


def header(dataset: str, part: str, names: list[str], stretch_status: dict[str, str] | None = None) -> dict[str, Any]:
    """The results header's `late_interaction` block as the run starts (finish() adds what the run measured)."""
    return {"milestone": "M49", "attempt": ATTEMPT, "declarations_commit": DECLARATIONS_COMMIT, "dataset": dataset, "part": part,
            "bar": BAR, "settings": SETTINGS, "candidates": CANDIDATES, "stretch": STRETCH, "multiplicity": dict(MULTIPLICITY),
            "card_claim": CARD_CLAIM, "published": PUBLISHED, "tripwire": {k: TRIPWIRE[k] for k in ("low", "high", "truncation_explained", "fixture_trip")},
            "training_data": TRAINING_DATA, "rows": list(names), "stretch_status": dict(stretch_status or {}),
            "expected_reproduction": EXPECTED_REPRODUCTION if dataset != "fixture" else None, "does_not_show": DOES_NOT_SHOW,
            "served_defaults_changed": False, "other_scoring_jobs": []}


def _bonferroni(a: list[float], b: list[float], idx) -> dict[str, Any]:
    from app.eval.stats import paired_delta_ci

    alpha = MULTIPLICITY["bonferroni"]["alpha"]
    d, lo, hi = paired_delta_ci(a, b, idx, alpha=alpha)
    return {"alpha": alpha, "level": MULTIPLICITY["bonferroni"]["level"], "delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "deciding": False}


def verdict_entry(dataset: str, config: str, reference: str, block: dict, *, scope: str, scores: dict[str, list[float]] | None, idx) -> dict[str, Any]:
    """One paired nDCG@10 comparison in the bar's scope, with dataset-qualified names, the stored interval and its word,
    what kind of comparison it is, whether PLAN.md must state it, and (a bar comparison) the Bonferroni interval."""
    from app.eval.report import verdict

    row = kind_of(config)
    kind = "reference" if row["role"] == "b" and reference != PARENT else row["kind"]
    entry = {"config": f"{dataset}:{config}", "reference": f"{dataset}:{reference}", "metric": "ndcg@10", "delta": block["delta"], "lo": block["lo"],
             "hi": block["hi"], "interval": verdict(block), "scope": scope, "role": row["role"], "kind": kind,
             "plan": [f"{dataset}:{config}", f"{dataset}:{reference}"] in PLAN_FLAGS}
    if kind == "bar":
        entry["half_width"] = round((block["hi"] - block["lo"]) / 2, 4)
        if scores is not None:
            entry["bonferroni"] = _bonferroni(scores[config], scores[reference], idx)
    return entry


def beir_verdicts(blob: dict, scores: dict[str, list[float]] | None, dataset: str) -> list[dict]:
    """Every paired nDCG@10 interval of a BEIR file: each row against its parent, each named role-(b) row against its
    references. `scores`: per-query nDCG@10 by row, for the Bonferroni column (from the run or the sidecar)."""
    from app.eval.stats import resample_indices

    boot = blob["bootstrap"]
    idx = resample_indices(boot["n_queries"], n_boot=boot["n_boot"], seed=boot["seed"])
    out = []
    for row in blob["results"]:
        dp = row.get("delta_vs_parent")
        if dp:
            out.append(verdict_entry(dataset, row["config"], dp["reference"], dp["ndcg@10"], scope="test", scores=scores, idx=idx))
    for c in boot.get("comparisons") or []:
        out.append(verdict_entry(dataset, c["row"], c["other"], c["ndcg@10"], scope="test", scores=scores, idx=idx))
    return out


def _gz(data: bytes) -> bytes:
    """Deterministic gzip: no file name, mtime 0, level 9, so the same content always has the same sha256."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as fh:
        fh.write(data)
    return buf.getvalue()


def write_runs(directory: Path, files: dict[str, list[dict]]) -> dict[str, dict]:
    """Each {name: one JSON object per query} as `directory`/<name> (late_interaction_runs/ beside the results file, which
    is RUNS_DIR in the repository), deterministic gzip of JSON lines; their sha256 and bytes, keyed RUNS_DIR/<name>."""
    out = {}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, lines in files.items():
        data = _gz("".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines).encode("utf-8"))
        (directory / name).write_bytes(data)
        out[f"{RUNS_DIR}/{name}"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "lines": len(lines)}
    return out


def read_run(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


RUN_BUDGET_SHARE = {"scifact": 300 / 948, "fiqa": 648 / 948}  # the 4 MB of SETTINGS["runs"], split by query count


def beir_run_files(dataset: str, captures: dict[str, dict], queries_order: list[str]) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """The committed BEIR runs of a part: the MaxSim leg's top 100 per query (PRE-filter: what retrieve returns and RRF
    fuses, before run_config drops did == qid) of the two named models' primary dense rows, and the reranked top-20
    order (pre-filter) of the four named role-(b) rows. Scores are dropped (ids only) when the leg runs would exceed this
    dataset's share of SETTINGS' 4 MB; RRF reads ranks, so the rederive works either way."""
    legs, heads = {}, {}
    for name, cap in captures.items():
        r = kind_of(name)
        if "leg" in cap and r["kind"] == "secondary":
            legs[f"{dataset}_{r['model']}.run.gz"] = [{"qid": q, "ids": [d for d, _ in cap["leg"][q]], "scores": [s for _, s in cap["leg"][q]]} for q in queries_order]
        if "head" in cap and r["role"] == "b" and r["model"] in CANDIDATES:
            suffix = "" if r["kind"] == "bar" else f"_doc{r['document_length']}"
            heads[f"{dataset}_rerank_{r['model']}{suffix}.json.gz"] = [{"qid": q, "head": cap["head"][q]} for q in queries_order]
    with_scores = sum(len(_gz("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in v).encode())) for v in legs.values())
    budget = int(SETTINGS["runs"]["ids_only_if_beir_runs_exceed_bytes"] * RUN_BUDGET_SHARE[dataset])
    ids_only = with_scores > budget
    if ids_only:
        legs = {k: [{"qid": x["qid"], "ids": x["ids"]} for x in v] for k, v in legs.items()}
    return {**legs, **heads}, {"leg_runs_with_scores_bytes": with_scores, "budget_bytes": budget, "ids_only": ids_only,
                               "pre_filter": "before run_config drops did == qid (BEIR's ignore-identical-ids); the rederive applies it after fusion"}


def published_vs_measured(blob: dict, dataset: str) -> dict[str, Any]:
    """The two named models' primary dense nDCG@10 against PUBLISHED, both directions (a tripwire figure, never a bar)."""
    out = {}
    by = {r["config"]: r for r in blob["results"]}
    for tag in CANDIDATES:
        row = by.get(f"dense@{tag}")
        text = PUBLISHED["values"][tag].get(dataset)
        if row and text:
            measured = row["retrieval"]["ndcg@10"]
            out[tag] = {"measured_ndcg@10": measured, "published_ndcg@10_x100": text, "difference": round(measured - float(text) / 100, 4)}
    return out


def legs_summary(blob: dict) -> dict[str, Any]:
    """Per `mv:` name, from the rows' model records: settings read back, checks, store counts, batched pass, equivalence."""
    out: dict[str, Any] = {}
    keys = ("model", "kind", "document_length", "revision", "weights_sha256", "dtype", "device", "versions", "settings_read_back", "load_and_score",
            "fingerprint", "store", "queries", "batched_pass", "equivalence")
    for row in blob["results"]:
        for role in ("embedder", "reranker"):
            m = (row.get("models") or {}).get(role) or {}
            name = m.get("mv_name")
            if name and name not in out:
                out[name] = {k: m.get(k) for k in keys if m.get(k) is not None}
    return out


def finish_beir(blob: dict, per_query: dict[str, list[dict]] | None, captures: dict[str, dict] | None, runs_dir: Path, dataset: str) -> dict[str, Any]:
    """After beir.run: the verdicts (with the Bonferroni column), the committed runs (written now, their sha256 recorded),
    the published-versus-measured differences, the legs' summary and this file's interval count."""
    scores = {name: [r["ndcg@10"] for r in rows_] for name, rows_ in (per_query or {}).items()}
    blob["verdicts"] = beir_verdicts(blob, scores or None, dataset)
    li = blob["late_interaction"]
    li["multiplicity"]["this_file"] = {"ndcg@10_intervals": len(blob["verdicts"]), "bar": sum(v["kind"] == "bar" for v in blob["verdicts"]),
                                       "uncorrected": "every interval of this file is an uncorrected 95% interval; the Bonferroni column of the bar rows is reported, never deciding"}
    queries_order = [r["id"] for r in next(iter(per_query.values()))] if per_query else []
    files, rule = beir_run_files(dataset, captures or {}, queries_order)
    li["runs"] = {**write_runs(runs_dir, files), "_rule": rule} if files else {}
    li["published_vs_measured"] = published_vs_measured(blob, dataset)
    li["legs"] = legs_summary(blob)
    return blob


def stretch_gate(part: str, skip_reason: str | None = None, device: str = "auto") -> tuple[set[str], dict[str, str]]:
    """Before an `extra` part: each stretch model loads and passes its checks (settings, load-and-score, the card's
    fingerprint), or its rows are skipped and recorded as "not loaded: <reason>"; with `skip_reason` (the timeout or the
    date rule) every stretch row is "not run: <reason>". A named model is never skipped: its failure stops the run."""
    if part != "extra":
        return set(), {}
    if skip_reason:
        return set(STRETCH), {tag: f"not run: {skip_reason}" for tag in STRETCH}
    from app.eval.models_2026 import free_gpu
    from app.eval.provenance import scrub

    skip, status = set(), {}
    for tag in STRETCH:
        try:
            leg = MvLeg(mv_name(tag, STRETCH[tag]["saved"]["document_length"]), [], corpus_sha256="", device=None if device == "auto" else device)
            status[tag] = "loaded; settings, load-and-score and the card's fingerprint passed"
            leg.release()
        except Exception as e:  # noqa: BLE001 - recorded, and the stretch rows are skipped
            skip.add(tag)
            status[tag] = f"not loaded: {type(e).__name__}: {scrub(str(e))[:400]}"
        free_gpu()
    return skip, status


# ---------------------------------------------------------------- fixture v1 (make bench-late-interaction PART=primary|extra)

FIXTURE_SIDECAR_SCHEMA = "late-interaction-per-question-v1"
SCOPES = ("held_out", "tune", "all_questions")
BGE_ROW = "hybrid+bge-rerank@bge-small/fp32-512/top20"
# The committed rows the fixture's recomputed parents and references stand for, on all 136 questions:
# (file, scope or None for the top-level results, rows)
FIXTURE_REPRODUCE = (("bench/results/neural.json", None, ("dense_only@bge-small", "hybrid@bge-small")),
                     ("bench/results/rerank_depth.json", None, (BGE_ROW,)),
                     ("bench/results/small_rerankers.json", "all_questions", ("hybrid+minilm-l12@bge-small/fp32-512/top20",
                                                                              "hybrid+gte-modernbert-base@bge-small/fp32-512/top20")))


def ranking_sha256(lists: list[list[str]]) -> str:
    """The sha256 of a row's per-question ranked chunk-id lists, in gold order (compact JSON)."""
    return hashlib.sha256(json.dumps(lists, separators=(",", ":")).encode("utf-8")).hexdigest()


def _paths_of(ids: list[str], by_id: dict) -> list[str]:
    out: list[str] = []
    for cid in ids:
        path = by_id[cid].path
        if path not in out:
            out.append(path)
    return out


def _ms(samples: list[float]) -> dict[str, float]:
    import numpy as np

    return {"p50_ms": round(float(np.percentile(samples, 50)), 3), "p95_ms": round(float(np.percentile(samples, 95)), 3), "n": len(samples)} if samples else {}


def fixture_verdicts(blob: dict, sidecar: dict) -> list[dict]:
    """Every paired nDCG@10 interval of the held-out scope (the bar's), with the Bonferroni column from the sidecar."""
    from app.eval.stats import resample_indices

    held = sidecar["split"]["held_out"]
    idx = resample_indices(len(held), n_boot=blob["held_out"]["bootstrap"]["n_boot"], seed=blob["held_out"]["bootstrap"]["seed"])
    scores = {name: [cols["ndcg@10"][i] for i in held] for name, cols in sidecar["scores"].items()}
    out = []
    for row in blob["held_out"]["results"]:
        for block in row.get("deltas") or []:
            out.append(verdict_entry("fixture", row["config"], block["reference"], block["ndcg@10"], scope="held_out", scores=scores, idx=idx))
    return out


def run_fixture(root: Path | None = None, *, part: str = "primary", device: str = "auto", skip_stretch: str | None = None, retrieve_k: int | None = None,
                embedder: str | None = None, gold=None, load_reranker: Callable | None = None, leg_factory: Callable | None = None) -> tuple[dict, dict, dict]:
    """(the results blob, its per-question sidecar, the committed runs {file: lines}) for one part on fixture v1.

    hybrid@bge-small's fused top 50 per question as M43 builds it; role (a) through a LateInteractionIndex per model (the
    identical BM25 leg, asserted per question, and the MaxSim leg over chunk.text, asserted equal to the parent's dense
    text); role (b) reranks the fused top 20 with the tail kept; the references (primary only) are M43's cross-encoders at
    fp32 / 512 / top 20. Scored over the held-out half (the bar), the tune half and all 136. Tests pass `gold`,
    `load_reranker` (tag -> (model, record)), `leg_factory` (name, chunks -> MvLeg) and `embedder`."""
    from datetime import datetime, timezone

    from app.eval import small_rerankers as sr
    from app.eval.corpus import load_manifest, verify_manifest
    from app.eval.gold import load_gold
    from app.eval.harness import BGE_SMALL, CI_METRICS
    from app.eval.index_loader import build_index, mtime_provenance
    from app.eval.metrics import aggregate_retrieval
    from app.eval.models_2026 import check_embedder, check_reranker, free_gpu, reproduces
    from app.eval.provenance import command_line, device_info, git_sha, git_state, model_device, model_provenance, run_provenance, sha256_file
    from app.eval.rerank_arms import split_questions
    from app.retrieval.rerank import CrossEncoderReranker, cross_encoder_provenance, get_cross_encoder

    root = Path(root or ROOT)
    started = git_state(root)
    files_root = root / "bench" / "corpus" / "files"
    gold_path = root / "bench" / "gold" / "questions.jsonl"
    injected = gold  # the dry run's pseudo questions: then the gold file is never opened, not even to hash it
    gold = gold if gold is not None else load_gold(gold_path)
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    n_files = manifest["n_files"]
    k = retrieve_k or SETTINGS["fixture"]["retrieve_k"]
    top = SETTINGS["fixture"]["rerank_top"]
    embedder = embedder or BGE_SMALL
    index = build_index(files_root, embedder_name=embedder, retrieve_k=k, enable_router=False, enable_rerank=False, mtimes="manifest")
    embedder_check = check_embedder(index.embedder) if embedder.startswith("st:") else None
    by_id = {c.chunk_id: c for c in index.chunks}
    corpus_sha = hashlib.sha256(json.dumps([(c.chunk_id, c.text) for c in index.chunks], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    fused = [[(h.chunk, h.score) for h in index.retrieve(q.question, k=k).hits] for q in gold]
    declared = rows("fixture", part)
    skip, stretch_status = stretch_gate(part, skip_stretch, device) if leg_factory is None else (set(), {})
    declared = [r for r in declared if r.get("model") not in skip]
    orders: dict[str, list[list[str]]] = {}
    orders["dense_only@bge-small"] = [[c.chunk_id for c, _ in index.search_dense(q.question, k=k)] for q in gold]
    orders["hybrid@bge-small"] = [[c.chunk_id for c, _ in pairs] for pairs in fused]
    legs: dict[str, dict] = {}
    runs: dict[str, list[dict]] = {}
    LateInteractionIndex = _late_interaction_index_class()
    leg_names = list(dict.fromkeys(mv_name(r["model"], r["document_length"]) for r in declared if r["role"] in ("a", "b")))
    for name in leg_names:
        if leg_factory is not None:
            leg = leg_factory(name, index.chunks)
        else:
            leg = MvLeg(name, index.chunks, corpus_sha256=corpus_sha, device=None if device == "auto" else device, keep_store=True)
        leg.build_store()
        leg.prepare({q.id: q.question for q in gold}, k)
        li_index = LateInteractionIndex(index.chunks, leg, retrieve_k=k)
        if [c.text for c in li_index.chunks] != [c.text for c in index.chunks]:
            raise RuntimeError(f"{name}: the MaxSim leg does not read the strings bge-small's dense leg reads")
        bm25_equal = all(li_index.search_bm25(q.question, k=k) == index.search_bm25(q.question, k=k) for q in gold)
        if not bm25_equal:
            raise RuntimeError(f"{name}: the BM25 leg of the LateInteractionIndex differs from the parent's")
        latency: list[float] = []
        for r in [r for r in declared if r["role"] in ("a", "b") and mv_name(r["model"], r["document_length"]) == name]:
            if r["role"] == "a" and r["name"].startswith("dense_only@"):
                orders[r["name"]] = [[c.chunk_id for c, _ in li_index.search_dense(q.question, k=k)] for q in gold]
                if r["kind"] == "secondary":
                    runs[f"fixture_{r['model']}.run.gz"] = [{"qid": q.id, "ids": [cid for cid, _ in leg.hits(q.question, k)],
                                                             "scores": [s for _, s in leg.hits(q.question, k)]} for q in gold]
            elif r["role"] == "a":
                orders[r["name"]] = [[h.chunk.chunk_id for h in li_index.retrieve(q.question, k=k).hits] for q in gold]
            else:
                lists = []
                for q, pairs in zip(gold, fused):
                    t0 = time.perf_counter()
                    ranked = leg.rerank(q.question, pairs[:top]) + pairs[top:]  # the tail keeps the fused order
                    latency.append((time.perf_counter() - t0) * 1000.0)
                    lists.append([c.chunk_id for c, _ in ranked])
                orders[r["name"]] = lists
                if r["model"] in CANDIDATES:
                    suffix = "" if r["kind"] == "bar" else f"_doc{r['document_length']}"
                    runs[f"fixture_rerank_{r['model']}{suffix}.json.gz"] = [{"qid": q.id, "head": lst[:top]} for q, lst in zip(gold, lists)]
        legs[name] = {**leg.provenance, "bm25_equal_to_parent": bm25_equal, "document_text": "chunk.text (asserted equal to bge-small's dense leg)",
                      "stored_vs_on_the_fly": leg.cross_check(), "rerank_latency": {**_ms(latency), "note": "per question on this run's device, on-the-fly encoding included"}}
        leg.release()
        free_gpu()
    rerankers: dict[str, dict] = {}
    for r in [r for r in declared if r["role"] == "reference"]:
        tag = r["reranker"]
        if load_reranker is not None:
            model, record = load_reranker(tag)
        else:
            source, record, after_load = sr.resolver(None, root)(tag)
            model = get_cross_encoder(source, precision=sr.SHAPE["precision"], max_length=sr.SHAPE["max_length"], device=None if device == "auto" else device,
                                      raise_errors=True)
            record = {**model_provenance(sr.model_of(tag), "reranker"), **record}
        info = {**record, "loaded": True, "device": model_device(model), "precision": sr.SHAPE["precision"], "max_length": sr.SHAPE["max_length"], "depth": top,
                **cross_encoder_provenance(model), "load_and_score": check_reranker(model, tag)}
        rerank = CrossEncoderReranker(model=model, model_name=tag, precision=sr.SHAPE["precision"], max_length=sr.SHAPE["max_length"])
        if gold:
            rerank(gold[0].question, fused[0][:top])  # warm-up
        lists, latency = [], []
        for q, pairs in zip(gold, fused):
            t0 = time.perf_counter()
            ranked = list(rerank(q.question, pairs[:top])) + pairs[top:]
            latency.append((time.perf_counter() - t0) * 1000.0)
            lists.append([c.chunk_id for c, _ in ranked])
        orders[r["name"]] = lists
        rerankers[tag] = {**info, "rerank_latency": {**_ms(latency), "note": "per question on this run's device, after one warm-up question"}}
        del model, rerank
        free_gpu()
    names = [r["name"] for r in declared]
    missing = [n for n in names if n not in orders]
    if missing:
        raise RuntimeError(f"rows not produced: {missing}")

    def rows_of(ids):
        return [{"id": gold[i].id, "category": gold[i].category, "relevant": gold[i].relevant_ids()} for i in ids]

    every = list(range(len(gold)))
    tune_ids, held_ids = split_questions(gold)
    path_orders = {name: [_paths_of(lst, by_id) for lst in orders[name]] for name in names}
    scored = {name: aggregate_retrieval([{**row, "retrieved": path_orders[name][i]} for i, row in zip(every, rows_of(every))], n_files=n_files) for name in names}
    scores = {name: {m: [float(q[m]) for q in s.per_query] for m in CI_METRICS} for name, s in scored.items()}
    pairs = [(r["name"], r["parent"]) for r in declared if r["parent"]] + comparisons(names)
    timestamp = datetime.now(timezone.utc).isoformat()
    blob: dict[str, Any] = {
        "git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": timestamp, "command": command_line(), "device": device_info(device),
        "corpus_sha256": manifest["aggregate_sha256"], "gold_sha256": sha256_file(gold_path) if injected is None else None, "n_gold": len(gold),
        "n_files": n_files, "mtimes": mtime_provenance(files_root, "manifest", base=root), "n_chunks": len(index.chunks),
        "late_interaction": header("fixture", part, names, stretch_status),
        "first_stage": {"config": PARENT, "retrieve_k": k, "reranked": f"the fused top {top} chunks; the rest keep the fused order",
                        "embedder": {**model_provenance(embedder, "embedder"), "device": model_device(getattr(index.embedder, "_model", None)),
                                     **(index.embedder.provenance() if hasattr(index.embedder, "provenance") else {}),
                                     **({"load_and_score": embedder_check} if embedder_check else {})}},
        "legs": legs, "rerankers": rerankers, "rankings": {name: ranking_sha256(orders[name]) for name in names},
        "split": {"rule": "alternate train / test inside each category, gold order (app.eval.rerank_arms.split_questions)", "tune": "train", "held_out": "test",
                  "tune_ids": [gold[i].id for i in tune_ids], "held_out_ids": [gold[i].id for i in held_ids],
                  "use": "the bar is read on the held-out half; nothing was chosen on the tune half"},
    }
    for scope, ids in zip(SCOPES, (held_ids, tune_ids, every)):
        scope_rows, boot = sr.scope_rows(scores, names, ids, pairs, CI_METRICS)
        for row in scope_rows:
            subset = aggregate_retrieval([{**rr, "retrieved": path_orders[row["config"]][i]} for i, rr in zip(ids, rows_of(ids))], n_files=n_files)
            row["retrieval"] = subset.as_dict()
        blob[scope] = {"bootstrap": boot, "results": scope_rows}
    blob["reproduces"] = [reproduces(root, rel, blob["all_questions"]["results"], [n for n in want if n in orders], scope=scope)
                          for rel, scope, want in FIXTURE_REPRODUCE if any(n in orders for n in want)]
    blob["reproduces_all"] = all(b["all"] for b in blob["reproduces"]) if blob["reproduces"] else None
    sidecar = {"schema": FIXTURE_SIDECAR_SCHEMA,
               "about": "per-question scores of the results file with the same name (.json), in gold order and unrounded: `make late-interaction-rescore` "
                        "recomputes every interval, paired delta and verdict of its three scopes from them",
               "git_sha": blob["git_sha"], "timestamp": timestamp, "metrics": list(CI_METRICS), "ids": [q.id for q in gold],
               "categories": [q.category for q in gold], "split": {"tune": tune_ids, "held_out": held_ids}, "scores": scores}
    blob["verdicts"] = fixture_verdicts(blob, sidecar)
    blob["late_interaction"]["multiplicity"]["this_file"] = {"ndcg@10_intervals": len(blob["verdicts"]), "bar": sum(v["kind"] == "bar" for v in blob["verdicts"]),
                                                              "scope": "held_out (the tune and all-136 scopes' intervals are not counted)", "uncorrected": True}
    return blob, sidecar, runs


# ---------------------------------------------------------------- every interval again, with no model (make late-interaction-rescore)


def deciding_attempt(root: Path | None = None) -> int:
    """2 once an ATTEMPT 2 primary file exists (it decides the verdict, complete or not), else 1."""
    root = Path(root or ROOT)
    return 2 if any((root / results_file(d, "primary", 2)).exists() for d in DATASETS) else 1


def all_files(root: Path | None = None) -> list[tuple[str, str, int, Path]]:
    """(dataset, part, attempt, path) of every late-interaction results file present."""
    root = Path(root or ROOT)
    return [(d, p, a, root / results_file(d, p, a)) for a in (1, 2) for d in DATASETS for p in PARTS if (root / results_file(d, p, a)).exists()]


def rescore_fixture(blob: dict, sidecar: dict) -> list[str]:
    """Every interval, paired delta, mean and verdict (its word, its Bonferroni column) of a fixture file that its sidecar does not reproduce."""
    from app.eval import small_rerankers as sr
    from app.eval.compare import _leaf_differences

    out = sr.rescore_fixture(blob, sidecar)
    if out and "not the same run" in out[0]:
        return out
    _leaf_differences("verdicts", blob.get("verdicts"), fixture_verdicts(blob, sidecar), out)
    return out


def rescore_beir(blob: dict, sidecar: dict) -> list[str]:
    """The same for a BEIR file (the BEIR runner's own attach_confidence over the sidecar's scores, then the verdicts)."""
    from app.eval import small_rerankers as sr
    from app.eval.compare import _leaf_differences

    out = sr.rescore_beir(blob, sidecar)
    if out and "not the same run" in out[0]:
        return out
    scores = {name: cols["ndcg@10"] for name, cols in sidecar["scores"].items()}
    _leaf_differences("verdicts", blob.get("verdicts"), beir_verdicts(blob, scores, blob["late_interaction"]["dataset"]), out)
    return out


def rescore(root: Path | None = None) -> tuple[int, list[str]]:
    """(files checked, differences) over every late-interaction results file present, from its sidecar."""
    from app.eval import per_question

    checked, problems = 0, []
    for dataset, _, _, path in all_files(root):
        rel = path.relative_to(Path(root or ROOT)).as_posix()
        side = per_question.sidecar_path(path)
        if not side.exists():
            problems.append(f"{rel}: no sidecar {side.name}")
            continue
        blob, sidecar = json.loads(path.read_text()), json.loads(side.read_text())
        diffs = rescore_fixture(blob, sidecar) if dataset == "fixture" else rescore_beir(blob, sidecar)
        problems += [f"{rel}: {d}" for d in diffs]
        checked += 1
    return checked, problems


# ---------------------------------------------------------------- the bar, from the three primary files (make late-interaction-bar)

ROLE_ROWS = {"a": "hybrid@{tag}", "b": "hybrid+{tag}@bge-small/fp32-doc{L}/top20"}
DATASET_LABELS = {"fixture": "fixture", "scifact": "SciFact", "fiqa": "FiQA"}
GTE_ROW = "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"
MINILM_ROW = "hybrid+minilm-l12@bge-small/fp32-512/top20"


def bar_row(tag: str, role: str) -> str:
    return ROLE_ROWS[role].format(tag=tag, L=MODELS[tag]["saved"]["document_length"])


def _load(root: Path, rel: str) -> dict | None:
    p = Path(root) / rel
    return json.loads(p.read_text()) if p.exists() else None


def _verdicts_by_pair(blob: dict | None) -> dict[tuple[str, str], dict]:
    return {(v["config"], v["reference"]): v for v in (blob or {}).get("verdicts") or []}


def bar_table(root: Path | None = None, attempt: int | None = None) -> dict[str, Any]:
    """The declared bar read off the deciding attempt's three `primary` files: per named model and role, each dataset's
    stored verdict against hybrid@bge-small (with its Bonferroni interval and half-width) and against the references,
    the PASS / NOT PASS / OPEN / not measured status, the CARD_CLAIM reading, the stretch and sensitivity rows above zero
    on all three datasets (from the `extra` files) and the PLAN.md row 4.2 outcome. A missing dataset never passes."""
    root = Path(root or ROOT)
    attempt = attempt or deciding_attempt(root)
    blobs = {d: _load(root, results_file(d, "primary", attempt)) for d in DATASETS}
    extras = {d: _load(root, results_file(d, "extra", attempt)) for d in DATASETS}
    table: dict[str, Any] = {"attempt": attempt, "datasets": {}, "models": {}, "card_claim": None, "all_three_outside_bar": []}
    for d in DATASETS:
        blob = blobs[d]
        if blob is None:
            continue
        scope = blob["held_out"] if d == "fixture" else blob
        parent = next((r for r in scope["results"] if r["config"] == PARENT), None)
        table["datasets"][d] = {"file": results_file(d, "primary", attempt), "git_sha": blob.get("git_sha"), "n": scope["bootstrap"]["n_queries"],
                                "device": (blob.get("device") or {}).get("device_name"), "parent_ndcg@10": (parent or {}).get("ci", {}).get("ndcg@10")}
    for tag in CANDIDATES:
        entry: dict[str, Any] = {}
        for role in ("a", "b"):
            name = bar_row(tag, role)
            cells: dict[str, Any] = {}
            for d in DATASETS:
                v = _verdicts_by_pair(blobs[d]).get((f"{d}:{name}", f"{d}:{PARENT}"))
                if v is None:
                    continue
                scope = blobs[d]["held_out"] if d == "fixture" else blobs[d]
                row = next(r for r in scope["results"] if r["config"] == name)
                cells[d] = {"word": v["interval"], "delta": v["delta"], "lo": v["lo"], "hi": v["hi"], "half_width": v.get("half_width"),
                            "bonferroni": v.get("bonferroni"), "ndcg@10": row["ci"]["ndcg@10"], "plan": v["plan"]}
                if role == "b":
                    cells[d]["references"] = {ref: _verdicts_by_pair(blobs[d]).get((f"{d}:{name}", f"{d}:{ref}")) for ref in (BGE_ROW, MINILM_ROW, GTE_ROW)}
            words = [cells[d]["word"] if d in cells else None for d in DATASETS]
            if not cells:
                status, passes = "not measured", None
            elif any(w in ("covers zero", "below zero") for w in words):
                status, passes = "NOT PASS", False
            elif all(w == "above zero" for w in words):
                status, passes = "PASS", True
            else:
                status, passes = "OPEN", None
            triple = ", ".join(f"{DATASET_LABELS[d]} {cells[d]['word'] if d in cells else 'not measured'}" for d in DATASETS)
            entry[role] = {"row": name, "datasets": cells, "status": status, "passes": passes, "triple": triple,
                           "missing": [d for d in DATASETS if d not in cells]}
        table["models"][tag] = entry
    # the card claim: answerai's primary role-(b) row against MiniLM-L12, outside the bar
    cells = {}
    for d in DATASETS:
        v = _verdicts_by_pair(blobs[d]).get((f"{d}:{CARD_CLAIM['row']}", f"{d}:{CARD_CLAIM['comparator']}"))
        if v:
            cells[d] = v
    below = [DATASET_LABELS[d] for d, v in cells.items() if v["interval"] == "below zero"]
    if below:
        reading = "contradicted on " + " and ".join(below)
    elif len(cells) == 3 and all(v["interval"] == "above zero" for v in cells.values()):
        reading = "supported here"
    else:
        reading = "not shown"
    table["card_claim"] = {"reading": reading, "datasets": {d: {k: v[k] for k in ("delta", "lo", "hi", "interval")} for d, v in cells.items()},
                           "caveats": CARD_CLAIM["caveats"]}
    # sensitivity and stretch rows above zero against their parent on all three datasets (they never change a verdict)
    names: dict[str, list[str]] = {}
    for d in DATASETS:
        for v in (extras[d] or {}).get("verdicts") or []:
            if v["reference"] == f"{d}:{PARENT}" and v["interval"] == "above zero" and v["kind"] in ("sensitivity", "stretch"):
                names.setdefault(v["config"].split(":", 1)[1], []).append(d)
    table["all_three_outside_bar"] = sorted(n for n, ds in names.items() if len(ds) == 3)
    table["outcome"] = plan_outcome(table)
    return table


def plan_outcome(table: dict) -> dict[str, Any]:
    """PLAN.md row 4.2's top-level outcome by the declared mapping (BAR["plan_row_4_2"]): "measured: ... passes",
    "measured, not adopted", or "labelled a demo", and the clauses it must carry (an OPEN bar, a row outside the bar
    above zero on all three, a named model not measured)."""
    measured = [t for t, e in table["models"].items() if any(e[r]["status"] != "not measured" for r in ("a", "b"))]
    passing = [(t, r) for t, e in table["models"].items() for r in ("a", "b") if e[r]["passes"] is True]
    if not measured:
        word = "labelled a demo"
    elif passing:
        word = "measured: passes"
    else:
        word = "measured, not adopted"
    open_clauses = [f"the bar is open for `{t}` in role ({r}): " + " and ".join(DATASET_LABELS[d] for d in e[r]["missing"]) + " not measured"
                    for t, e in table["models"].items() for r in ("a", "b") if e[r]["status"] == "OPEN"]
    return {"word": word, "passing": [f"`{t}` passes in role ({r})" for t, r in passing], "open": open_clauses,
            "not_measured": [t for t in CANDIDATES if t not in measured],
            "outside_bar_all_three": [f"`{n}`" for n in table["all_three_outside_bar"]]}


def declarations_check(root: Path | None = None) -> list[str]:
    """Each results file's declarations commit is an ancestor of its git_sha, and BAR, SETTINGS and CANDIDATES in that
    commit (read with ast.literal_eval) equal the file's header. Problems as strings; a commit this clone does not have
    (a shallow clone) is reported as such."""
    import ast
    import subprocess

    root = Path(root or ROOT)
    problems: list[str] = []
    cache: dict[str, dict] = {}
    for _, _, _, path in all_files(root):
        rel = path.relative_to(root).as_posix()
        blob = json.loads(path.read_text())
        li = blob.get("late_interaction") or {}
        decl, sha = li.get("declarations_commit"), blob.get("git_sha")
        if not decl:
            problems.append(f"{rel}: no declarations commit in its header")
            continue
        anc = subprocess.run(["git", "-c", "pack.threads=1", "merge-base", "--is-ancestor", decl, sha], cwd=root, capture_output=True, text=True)
        if anc.returncode != 0:
            problems.append(f"{rel}: the declarations commit {decl[:12]} is not an ancestor of {sha} ({anc.stderr.strip() or 'not an ancestor'})")
            continue
        if decl not in cache:
            src = subprocess.run(["git", "-c", "pack.threads=1", "show", f"{decl}:app/eval/late_interaction.py"], cwd=root, capture_output=True, text=True).stdout
            found = {}
            for node in ast.parse(src).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in ("BAR", "SETTINGS", "CANDIDATES"):
                    found[node.targets[0].id] = ast.literal_eval(node.value)
            cache[decl] = found
        for name, key in (("BAR", "bar"), ("SETTINGS", "settings"), ("CANDIDATES", "candidates")):
            if cache[decl].get(name) != li.get(key):
                problems.append(f"{rel}: {name} at the declarations commit differs from the header's")
    return problems


def cross_part_parents(root: Path | None = None, attempt: int | None = None) -> dict[str, Any]:
    """Per dataset: the `extra` part's recomputed parents against the `primary` part's (retrieval blocks; rankings too
    when both ran on the same GPU model). Read from the committed files: the two parts may run at the same time."""
    root = Path(root or ROOT)
    attempt = attempt or deciding_attempt(root)
    out = {}
    for d in DATASETS:
        p, e = _load(root, results_file(d, "primary", attempt)), _load(root, results_file(d, "extra", attempt))
        if not (p and e):
            continue
        same_device = (p.get("device") or {}).get("device_name") == (e.get("device") or {}).get("device_name")
        rows_ = {}
        for name in ([f"dense_only@bge-small", PARENT] if d == "fixture" else ["dense@bge-small", PARENT]):
            if d == "fixture":
                a = next(r for r in p["all_questions"]["results"] if r["config"] == name)
                b = next(r for r in e["all_questions"]["results"] if r["config"] == name)
                rank = (p.get("rankings") or {}).get(name) == (e.get("rankings") or {}).get(name)
            else:
                a = next(r for r in p["results"] if r["config"] == name)
                b = next(r for r in e["results"] if r["config"] == name)
                rank = a.get("ranking_sha256") == b.get("ranking_sha256")
            rows_[name] = {"retrieval_equal": a["retrieval"] == b["retrieval"], **({"ranking_equal": rank} if same_device else {})}
        out[d] = {"same_gpu_model": same_device, "rows": rows_}
    return out


# ---------------------------------------------------------------- the plausibility tripwire (after J2)


def tripwire(root: Path | None = None, attempt: int | None = None) -> dict[str, Any]:
    """TRIPWIRE over the fixture and SciFact files of an attempt: per named model, the primary dense SciFact row against
    the committed bm25 row of beir_scifact.json and against PUBLISHED (low, high), the truncation-explained exception
    (the doc512 row neither low- nor high-trips), and the fixture trip (no chunk cut at the primary length, yet a
    primary and a doc512 ranking differ). Status "clear", "truncation-explained" or "trip"."""
    root = Path(root or ROOT)
    attempt = attempt or ATTEMPT
    rels = {(d, p): results_file(d, p, attempt) for d in ("fixture", "scifact") for p in PARTS}
    blobs = {key: _load(root, rel) for key, rel in rels.items()}
    missing = [rel for key, rel in rels.items() if blobs[key] is None]
    committed = _load(root, "bench/results/beir_scifact.json") or {}
    bm25 = next((r["retrieval"]["ndcg@10"] for r in committed.get("results", []) if r["config"] == "bm25"), None)
    low_by, high_by = TRIPWIRE["low"]["below_published_by"], TRIPWIRE["high"]["above_published_by"]

    def judge(value: float | None, tag: str) -> dict[str, Any]:
        if value is None:
            return {"value": None}
        pub = float(PUBLISHED["values"][tag]["scifact"]) / 100
        low = (bm25 is not None and value < bm25) or value < round(pub - low_by, 4)
        return {"value": value, "bm25": bm25, "published": pub, "low": bool(low), "high": bool(value > round(pub + high_by, 4))}

    models: dict[str, Any] = {}
    for tag, c in CANDIDATES.items():
        sp, se = blobs[("scifact", "primary")], blobs[("scifact", "extra")]
        row = next((r for r in (sp or {}).get("results", []) if r["config"] == f"dense@{tag}"), None)
        row512 = next((r for r in (se or {}).get("results", []) if r["config"] == f"dense@{tag}/doc{SENSITIVITY_LENGTH}"), None)
        primary = judge(row["retrieval"]["ndcg@10"] if row else None, tag)
        doc512 = judge(row512["retrieval"]["ndcg@10"] if row512 else None, tag)
        explained = bool(primary.get("low") and not primary.get("high") and c["saved"]["document_length"] < SENSITIVITY_LENGTH
                         and doc512.get("value") is not None and not doc512.get("low") and not doc512.get("high"))
        # the fixture trip
        fp, fe = blobs[("fixture", "primary")], blobs[("fixture", "extra")]
        leg = ((fp or {}).get("legs") or {}).get(mv_name(tag, c["saved"]["document_length"])) or {}
        cut = (leg.get("store") or {}).get("cut_at_document_length")
        L, S = c["saved"]["document_length"], SENSITIVITY_LENGTH
        pairs = [(f"dense_only@{tag}", f"dense_only@{tag}/doc{S}"), (f"hybrid@{tag}", f"hybrid@{tag}/doc{S}"),
                 (f"hybrid+{tag}@bge-small/fp32-doc{L}/top20", f"hybrid+{tag}@bge-small/fp32-doc{S}/top20")]
        differ = [a for a, b in pairs if fp and fe and (fp.get("rankings") or {}).get(a) != (fe.get("rankings") or {}).get(b)]
        fixture_trip = bool(cut == 0 and differ)
        if primary.get("high") or fixture_trip or (primary.get("low") and not explained):
            status = "trip"
        elif primary.get("low"):
            status = "truncation-explained"
        elif primary.get("value") is None:
            status = "missing"
        else:
            status = "clear"
        models[tag] = {"scifact_dense": primary, "scifact_dense_doc512": doc512, "truncation_explained": explained,
                       "fixture": {"chunks_cut_at_primary_length": cut, "rankings_differing_from_doc512": differ, "trip": fixture_trip}, "status": status}
    statuses = [m["status"] for m in models.values()]
    overall = "trip" if "trip" in statuses or "missing" in statuses or missing else ("truncation-explained" if "truncation-explained" in statuses else "clear")
    return {"attempt": attempt, "status": overall, "models": models, "missing_files": missing,
            "files": {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in rels.values() if (root / rel).exists()},
            "bm25_source": "bench/results/beir_scifact.json, row bm25, retrieval.ndcg@10", "rule": TRIPWIRE}


# ---------------------------------------------------------------- re-derive the named rows on a CPU (make late-interaction-rederive)


def _verify_runs(root: Path, blob: dict, rel: str, problems: list[str]) -> None:
    for path, rec in ((blob.get("late_interaction") or {}).get("runs") or {}).items():
        if path.startswith("_"):
            continue
        p = root / path
        if not p.exists():
            problems.append(f"{rel}: committed run {path} is missing")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
            problems.append(f"{rel}: {path} does not have the sha256 its header records")


def _fuse(bm25_ids: list[str], leg_ids: list[str], k: int, rrf_k: int) -> list[str]:
    from app.retrieval.fuse import rrf_score_map

    fused = rrf_score_map([bm25_ids, leg_ids], k=rrf_k)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def rederive_fixture(root: Path, attempt: int) -> tuple[int, list[str]]:
    """The named models' fixture rows from the committed MaxSim lists and role-(b) heads, BM25 recomputed: rankings
    (sha256) and the all-136 retrieval blocks of dense_only@ and hybrid@; nDCG@10, MRR@10 and Recall@10 of role (b)."""
    from app.eval.gold import load_gold
    from app.eval.index_loader import load_chunks
    from app.eval.metrics import aggregate_retrieval
    from app.retrieval.hybrid import InMemoryHybridIndex

    rel = results_file("fixture", "primary", attempt)
    blob = _load(root, rel)
    if blob is None:
        return 0, []
    problems: list[str] = []
    _verify_runs(root, blob, rel, problems)
    chunks = load_chunks(root / "bench" / "corpus" / "files", mtimes="manifest")
    index = InMemoryHybridIndex(chunks, retrieve_k=SETTINGS["fixture"]["retrieve_k"], enable_router=False, enable_rerank=False)
    by_id = {c.chunk_id: c for c in chunks}
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    k, rows_all = SETTINGS["fixture"]["retrieve_k"], {r["config"]: r for r in blob["all_questions"]["results"]}

    def score(name: str, lists: list[list[str]], metrics: tuple[str, ...]) -> None:
        agg = aggregate_retrieval([{"id": q.id, "category": q.category, "relevant": q.relevant_ids(), "retrieved": _paths_of(lst, by_id)}
                                   for q, lst in zip(gold, lists)], n_files=blob["n_files"]).as_dict()
        for m in metrics:
            if agg.get(m) != rows_all[name]["retrieval"].get(m):
                problems.append(f"{rel}: {name} {m} re-derived {agg.get(m)}, results file {rows_all[name]['retrieval'].get(m)}")

    checked = 0
    for tag, c in CANDIDATES.items():
        run = Path(root) / RUNS_DIR / f"fixture_{tag}.run.gz"
        if run.exists():
            legs = {x["qid"]: x["ids"] for x in read_run(run)}
            dense = [legs[q.id][:k] for q in gold]
            hybrid = [_fuse([ch.chunk_id for ch, _ in index.search_bm25(q.question, k=k)], legs[q.id], k, SETTINGS["rrf_k"]) for q in gold]
            for name, lists in ((f"dense_only@{tag}", dense), (f"hybrid@{tag}", hybrid)):
                if ranking_sha256(lists) != blob["rankings"].get(name):
                    problems.append(f"{rel}: {name} ranking re-derived from the committed list differs from the file's")
                score(name, lists, tuple(m for m in rows_all[name]["retrieval"] if isinstance(rows_all[name]["retrieval"][m], float)
                                          and m in ("ndcg@10", "recall@10", "recall@50", "mrr@10", "recall@5")))
                checked += 1
        for length in (c["saved"]["document_length"], SENSITIVITY_LENGTH):
            name = f"hybrid+{tag}@bge-small/fp32-doc{length}/top20"
            suffix = "" if length == c["saved"]["document_length"] else f"_doc{length}"
            head_file = Path(root) / RUNS_DIR / f"fixture_rerank_{tag}{suffix}.json.gz"
            source = blob if suffix == "" else _load(root, results_file("fixture", "extra", attempt))
            if not head_file.exists() or source is None:
                continue
            heads = {x["qid"]: x["head"] for x in read_run(head_file)}
            rows_all_b = {r["config"]: r for r in source["all_questions"]["results"]}
            agg = aggregate_retrieval([{"id": q.id, "category": q.category, "relevant": q.relevant_ids(), "retrieved": _paths_of(heads[q.id], by_id)}
                                       for q in gold], n_files=blob["n_files"]).as_dict()
            for m in ("ndcg@10", "mrr@10", "recall@10"):
                if agg.get(m) != rows_all_b[name]["retrieval"].get(m):
                    problems.append(f"fixture: {name} {m} re-derived {agg.get(m)}, results file {rows_all_b[name]['retrieval'].get(m)}")
            checked += 1
    return checked, problems


def rederive_beir(root: Path, data_dir: Path, dataset: str, attempt: int) -> tuple[int, list[str]]:
    """The named models' BEIR rows from the committed MaxSim top 100 (pre-filter) and role-(b) heads, with BM25
    recomputed over the sha-pinned corpus and the test qrels: the dense@ and hybrid@ rankings (sha256 of their TREC run)
    and retrieval blocks; nDCG@10, MRR@10 and Recall@10 of role (b) (its Recall@100 equals the parent's by construction)."""
    from types import SimpleNamespace

    from app.eval.beir import Engine, chunks_from_corpus, download_dataset, graded_ndcg_at_k, load_beir, trec_run_text
    from app.eval.metrics import mrr_at_k, recall_at_k

    rel = results_file(dataset, "primary", attempt)
    blob = _load(root, rel)
    if blob is None:
        return 0, []
    problems: list[str] = []
    _verify_runs(root, blob, rel, problems)
    corpus, queries, qrels = load_beir(download_dataset(Path(data_dir), dataset))
    engine = Engine(chunks_from_corpus(corpus))
    k, top = SETTINGS["beir"]["k"], SETTINGS["beir"]["rerank_top"]
    bm25 = {qid: [d for d, _ in engine.bm25("beir", 0.9, 0.4).search(text, k=k)] for qid, text in queries.items()}
    rows_ = {r["config"]: r for r in blob["results"]}

    def metrics(rankings: dict[str, list[str]], names: tuple[str, ...]) -> dict[str, float]:
        per = []
        for qid, ids in rankings.items():
            rel_ids = list(qrels[qid])
            per.append({"ndcg@10": graded_ndcg_at_k(ids, qrels[qid], 10), "recall@10": recall_at_k(ids, rel_ids, 10),
                        "recall@100": recall_at_k(ids, rel_ids, 100), "mrr@10": mrr_at_k(ids, rel_ids, 10)})
        return {m: round(sum(p[m] for p in per) / len(per), 4) for m in names}

    checked = 0
    for tag, c in CANDIDATES.items():
        run = Path(root) / RUNS_DIR / f"{dataset}_{tag}.run.gz"
        if run.exists():
            legs = {x["qid"]: x["ids"] for x in read_run(run)}
            for name, make in ((f"dense@{tag}", lambda q: legs[q]), (f"hybrid@{tag}", lambda q: _fuse(bm25[q], legs[q], k, SETTINGS["rrf_k"]))):
                rankings = {qid: [d for d in make(qid) if d != qid] for qid in queries}
                sha = hashlib.sha256(trec_run_text(SimpleNamespace(name=name), rankings).encode("utf-8")).hexdigest()
                if sha != rows_[name].get("ranking_sha256"):
                    problems.append(f"{rel}: {name} ranking re-derived from the committed list differs from the file's")
                again = metrics(rankings, ("ndcg@10", "recall@10", "recall@100", "mrr@10"))
                for m, v in again.items():
                    if v != rows_[name]["retrieval"][m]:
                        problems.append(f"{rel}: {name} {m} re-derived {v}, results file {rows_[name]['retrieval'][m]}")
                checked += 1
        for length in (c["saved"]["document_length"], SENSITIVITY_LENGTH):
            name = f"hybrid+{tag}@bge-small/fp32-doc{length}/top20"
            suffix = "" if length == c["saved"]["document_length"] else f"_doc{length}"
            head_file = Path(root) / RUNS_DIR / f"{dataset}_rerank_{tag}{suffix}.json.gz"
            source = blob if suffix == "" else _load(root, results_file(dataset, "extra", attempt))
            if not head_file.exists() or source is None:
                continue
            heads = {x["qid"]: [d for d in x["head"] if d != x["qid"]] for x in read_run(head_file)}
            if any(len(h) < 10 for h in heads.values()):
                problems.append(f"{dataset}: {name} has a head shorter than ten documents after the filter")
            again = metrics(heads, ("ndcg@10", "recall@10", "mrr@10"))
            stored = {r["config"]: r for r in source["results"]}[name]["retrieval"]
            for m, v in again.items():
                if v != stored[m]:
                    problems.append(f"{dataset}: {name} {m} re-derived {v}, results file {stored[m]}")
            checked += 1
    return checked, problems


def rederive(root: Path | None = None, data_dir: Path | None = None, attempt: int | None = None) -> tuple[int, list[str]]:
    root = Path(root or ROOT)
    attempt = attempt or deciding_attempt(root)
    checked, problems = rederive_fixture(root, attempt)
    for dataset in ("scifact", "fiqa"):
        if (root / results_file(dataset, "primary", attempt)).exists():
            if data_dir is None:
                problems.append(f"{dataset}: no --data-dir (the sha-pinned BEIR zips app/eval/beir.py downloads)")
                continue
            n, p = rederive_beir(root, data_dir, dataset, attempt)
            checked, problems = checked + n, problems + p
    return checked, problems


# ---------------------------------------------------------------- J1: the fidelity checks, an orientation pass and a dry run (no test query)


def _readme(tag: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(MODELS[tag]["model"], "README.md", revision=MODELS[tag]["revision"])
    return Path(found) if isinstance(found, str) else None


def check_model(tag: str, device: str = "auto") -> dict[str, Any]:
    """Fidelity checks 1 to 5 of one model (no benchmark data): settings read back (and at 512 for a named model), the
    checkpoint's tensors against the loaded modules, M41's load-and-score texts, the card's fingerprint, maxsim /
    maxsim_scores / sentence-transformers on the same pairs, and the card's lines and sha256."""
    from app.eval.models_2026 import CHECK_ANSWER, CHECK_LONG, CHECK_OTHER, CHECK_QUERY

    import torch

    L = MODELS[tag]["saved"]["document_length"]
    enc = load_encoder(tag, L, None if device == "auto" else device)
    line: dict[str, Any] = {"tag": tag, "role": "named" if tag in CANDIDATES else "stretch"}
    line["settings"] = check_settings(enc, tag, L)
    line["provenance"] = {k: v for k, v in enc.provenance().items() if k != "settings"}
    line["weights"] = enc.weights_check()
    line["load_and_score"] = check_late_interaction(enc, tag)
    line["fingerprint"] = check_fingerprint(enc, tag)
    texts = [(CHECK_QUERY, CHECK_ANSWER), (CHECK_QUERY, CHECK_OTHER), (CHECK_QUERY, CHECK_LONG)]
    texts += [(FINGERPRINT.get(tag, FINGERPRINT["answerai-colbert-small"])["query"], d) for d in FINGERPRINT.get(tag, FINGERPRINT["answerai-colbert-small"])["documents"]]
    line["equivalence"] = check_equivalence(enc, texts, device="cuda" if torch.cuda.is_available() else "cpu")
    line["card"] = card_checks(tag, _readme(tag))
    if tag in CANDIDATES:
        enc.set_document_length(SENSITIVITY_LENGTH)
        line["doc512_settings"] = {k: v for k, v in check_settings(enc, tag, SENSITIVITY_LENGTH).items() if k in ("document_length", "saved_document_length")}
        long_tokens = enc.encode_documents([CHECK_LONG])[0].shape[0]
        line["doc512_effect"] = {"check_long_scored_tokens": int(long_tokens), "above_saved_length": int(long_tokens) > L}
    line["ok"] = bool(line["weights"].get("ok") and line["equivalence"]["passed"] and line["card"]["passed"] and line["load_and_score"]["passed"]
                      and (line["fingerprint"] is None or line["fingerprint"]["passed"]) and (tag not in CANDIDATES or line["doc512_effect"]["above_saved_length"]))
    del enc
    return line


def orient(data_dir: Path, device: str = "auto", tags: list[str] | None = None, n_queries: dict[str, int] | None = None) -> list[dict]:
    """J1's orientation (nothing committed; NOTES only): per model and document length, encode the SciFact corpus
    (corpus.jsonl only), count tokens, and time the batched float64 pass with random unit query matrices of the real
    shape; tokenise the FiQA corpus to project its encode and scoring time from SciFact's. No query file is opened."""
    import numpy as np

    from app.eval.beir import load_corpus
    from app.eval.models_2026 import free_gpu
    from app.multimodal.late_interaction import TokenStore, maxsim_scores

    n_queries = n_queries or {"scifact": 300, "fiqa": 648}
    out = []
    sci = list(load_corpus(Path(data_dir) / "scifact").values())
    fiqa = list(load_corpus(Path(data_dir) / "fiqa").values()) if (Path(data_dir) / "fiqa" / "corpus.jsonl").exists() else []
    for tag in tags or list(MODELS):
        lengths = [MODELS[tag]["saved"]["document_length"]] + ([SENSITIVITY_LENGTH] if tag in CANDIDATES else [])
        for L in lengths:
            enc = load_encoder(tag, L, None if device == "auto" else device)
            t0 = time.perf_counter()
            mats = enc.encode_documents(sci)
            encode_s = time.perf_counter() - t0
            store = TokenStore.from_matrices(mats, enc.dim)
            del mats
            q_rows = MODELS[tag]["saved"]["query_length"] if MODELS[tag]["saved"]["query_expansion"] else 16
            rng = np.random.RandomState(0)
            fake = [rng.standard_normal((q_rows, enc.dim)).astype(np.float32) for _ in range(n_queries["scifact"])]
            dev = "cuda" if str(enc.provenance().get("device", "")).startswith("cuda") else None
            t0 = time.perf_counter()
            maxsim_scores(fake, store, block_bytes=MAXSIM_BLOCK_BYTES, device=dev)
            pass_s = time.perf_counter() - t0
            sci_tok = enc.token_counts(sci, "document")
            fiqa_tok = enc.token_counts(fiqa, "document") if fiqa else []
            capped = lambda toks: sum(min(t, L) for t in toks)  # noqa: E731
            ratio = capped(fiqa_tok) / max(capped(sci_tok), 1) if fiqa_tok else None
            line = {"tag": tag, "document_length": L, "scifact": {"documents": len(sci), "scored_vectors": int(store.lengths().sum()),
                                                                     "store_bytes": store.nbytes, "cut_at_length": sum(t > L for t in sci_tok),
                                                                     "encode_s": round(encode_s, 1), "batched_pass_s_random_queries": round(pass_s, 1)},
                    "fiqa_projection": None if ratio is None else {"documents": len(fiqa), "cut_at_length": sum(t > L for t in fiqa_tok),
                                                                   "capped_token_ratio_to_scifact": round(ratio, 3),
                                                                   "store_bytes_estimate": int(store.nbytes * ratio),
                                                                   "encode_s": round(encode_s * ratio, 1),
                                                                   "batched_pass_s": round(pass_s * ratio * n_queries["fiqa"] / n_queries["scifact"], 1)},
                    "device": enc.provenance().get("device")}
            print(json.dumps(line), flush=True)
            out.append(line)
            del enc, store
            free_gpu()
    return out


def pseudo_dataset(data_dir: Path, work: Path, n: int = 48, seed: int = 0) -> Path:
    """A BEIR-format directory for the dry run: the real SciFact corpus with PSEUDO queries (the first twelve words of
    `n` seeded documents) and qrels pointing at those documents. No test query or qrel is read or written."""
    import random

    from app.eval.beir import load_corpus

    corpus_path = Path(data_dir) / "scifact" / "corpus.jsonl"
    target = Path(work) / "scifact"
    (target / "qrels").mkdir(parents=True, exist_ok=True)
    (target / "corpus.jsonl").write_bytes(corpus_path.read_bytes())
    corpus = load_corpus(target)
    picks = random.Random(seed).sample(sorted(corpus), n)
    (target / "queries.jsonl").write_text("".join(json.dumps({"_id": f"pseudo-{i}", "text": " ".join(corpus[d].split()[:12])}) + "\n" for i, d in enumerate(picks)))
    (target / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(f"pseudo-{i}\t{d}\t1\n" for i, d in enumerate(picks)))
    return Path(work)


def pseudo_gold(root: Path, n: int = 24) -> list:
    """The dry run's fixture questions: PSEUDO questions from chunk text (never the gold questions), each pointing at its file."""
    from app.eval.gold import GoldQuestion
    from app.eval.index_loader import load_chunks

    chunks = load_chunks(Path(root) / "bench" / "corpus" / "files", mtimes="manifest")
    step = max(1, len(chunks) // n)
    return [GoldQuestion(id=f"pseudo-{i}", question=" ".join(c.text.split()[:10]), category=("a", "b")[i % 2], answer="", relevant_paths=[c.path])
            for i, c in enumerate(chunks[::step][:n])]


def beir_run_kwargs(dataset: str, part: str, configs: list, status: dict[str, str], others: list | None = None, root: Path | None = None,
                    reproduce: bool = True) -> dict[str, Any]:
    """beir.run's keyword arguments for a late-interaction part (the main branch of app/eval/beir.py and the dry run)."""
    names = [c.name for c in configs]
    return dict(check_models=True, comparisons=comparisons(names), keep_per_query=True,
                reproduce=tuple((rel, tuple(n for n in want if n in names)) for rel, want in LATE_INTERACTION_REPRODUCE[dataset]) if reproduce else None,
                header={"late_interaction": {**header(dataset, part, names, status), "other_scoring_jobs": list(others or [])}},
                resolve_reranker=reference_resolver(root))


def dry_run(data_dir: Path, work: Path, device: str = "auto", root: Path | None = None) -> dict[str, Any]:
    """J1's dry run of every code path the scoring jobs take, on PSEUDO queries only: both parts of the fixture runner
    (pseudo questions from chunk text) and both parts of the BEIR ladder (the SciFact corpus with pseudo queries), written
    under `work` and never committed. A crash here costs nothing; in J2 it would."""
    from app.eval import beir, per_question

    root = Path(root or ROOT)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    gold = pseudo_gold(root)
    for part in PARTS:
        blob, sidecar, runs = run_fixture(root, part=part, device=device, gold=gold)
        blob["late_interaction"]["runs"] = write_runs(work / "late_interaction_runs", runs)
        (work / f"fixture_{part}.json").write_text(json.dumps(blob, indent=2) + "\n")
        per_question.write(work / f"fixture_{part}.per_query.json", sidecar)
        summary[f"fixture_{part}"] = {"rows": len(blob["held_out"]["results"]), "verdicts": len(blob["verdicts"]), "rescore": rescore_fixture(blob, sidecar)}
    pseudo = pseudo_dataset(data_dir, work / "pseudo")
    for part in PARTS:
        skip, status = stretch_gate(part, None, device)
        configs = beir_configs(part, skip)
        blob = beir.run(pseudo, configs, device=device, root=root, dataset="scifact", download=False,
                        **beir_run_kwargs("scifact", part, configs, status, root=root, reproduce=False))
        captures, per_query = blob.pop("_captures", None), blob.pop("_per_query", None)
        finish_beir(blob, per_query, captures, work / "late_interaction_runs", "scifact")
        (work / f"scifact_{part}.json").write_text(json.dumps(blob, indent=2) + "\n")
        sidecar = beir.per_query_sidecar(blob, per_query)
        summary[f"scifact_{part}"] = {"rows": len(blob["results"]), "verdicts": len(blob["verdicts"]), "rescore": rescore_beir(blob, sidecar),
                                      "stretch": status}
    return summary


# ---------------------------------------------------------------- command line


def _parse(argv: list[str] | None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="fidelity checks 1-5 of every model (J1; no benchmark data): one JSON line per model")
    p.add_argument("--out", type=Path, default=None, help="run fixture v1 (one part) and write this results file, its sidecar and its committed runs")
    p.add_argument("--part", choices=PARTS, default="primary")
    p.add_argument("--rescore", action="store_true", help="every interval, paired delta and verdict of the committed files again, from their sidecars (CPU, no model)")
    p.add_argument("--bar", action="store_true", help="the declared bar from the three primary files, the declarations-commit check and the cross-part parents")
    p.add_argument("--rederive", action="store_true", help="the named rows again from the committed MaxSim lists, BM25 and the qrels (CPU; needs --data-dir with the sha-pinned BEIR zips)")
    p.add_argument("--tripwire", action="store_true", help="the plausibility tripwire over the fixture and SciFact files (after J2); exit 1 on a trip that is not truncation-explained")
    p.add_argument("--record", type=Path, default=None, help="--tripwire: write the record here too")
    p.add_argument("--orient", action="store_true", help="J1: encode the SciFact corpus, time the batched pass, project FiQA (corpus files only)")
    p.add_argument("--dry-run", action="store_true", help="J1: every scoring path on pseudo queries, written under --work")
    p.add_argument("--work", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--models", default=None, help="--check / --orient: comma-separated tags")
    p.add_argument("--skip-stretch", default=None, help="--out --part extra: run without the stretch rows ('not run: <reason>')")
    p.add_argument("--other-jobs", type=Path, default=None, help="--out: a JSON list of the other scoring jobs at the same commit and attempt, recorded in the header")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args(argv)


def _fmt(block: dict | None) -> str:
    return f"{block['delta']:+.4f} [{block['lo']:+.4f}, {block['hi']:+.4f}]" if block else "-"


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    root = ROOT
    if args.rescore:
        checked, problems = rescore(root)
        for line in problems[:60]:
            print(f"  - {line}")
        print(f"late-interaction intervals: {'OK' if not problems else f'{len(problems)} differences'}, {checked} results files recomputed from their sidecars")
        return 1 if problems or not checked else 0
    if args.bar:
        table = bar_table(root)
        for tag, e in table["models"].items():
            for role in ("a", "b"):
                cells = "; ".join(f"{DATASET_LABELS[d]} {_fmt(c)} {c['word']} (Bonferroni {_fmt(c.get('bonferroni'))}, half-width {c.get('half_width')})"
                                  for d, c in e[role]["datasets"].items())
                print(f"{e[role]['status']:13s} {tag:24s} role ({role}) {e[role]['row']}: {cells}")
        print(f"card claim: {table['card_claim']['reading']}")
        print(f"outside the bar, above zero on all three: {table['all_three_outside_bar'] or 'none'}")
        print(f"PLAN.md row 4.2: {table['outcome']}")
        print(f"cross-part parents: {json.dumps(cross_part_parents(root))}")
        problems = declarations_check(root)
        for line in problems:
            print(f"  - {line}")
        print(f"declarations commit: {'OK' if not problems else f'{len(problems)} problems'} ({DECLARATIONS_COMMIT})")
        return 1 if problems else 0
    if args.rederive:
        checked, problems = rederive(root, args.data_dir)
        for line in problems[:60]:
            print(f"  - {line}")
        print(f"late-interaction rederive: {'OK' if not problems else f'{len(problems)} differences'}, {checked} rows re-derived from the committed lists")
        return 1 if problems or not checked else 0
    if args.tripwire:
        record = tripwire(root)
        text = json.dumps(record, indent=2) + "\n"
        if args.record:
            args.record.write_text(text)
        print(json.dumps({k: record[k] for k in ("attempt", "status", "missing_files")}))
        for tag, m in record["models"].items():
            print(f"{tag}: {m['status']} (SciFact dense {m['scifact_dense']}, doc512 {m['scifact_dense_doc512']}, fixture {m['fixture']})")
        return 0 if record["status"] in ("clear", "truncation-explained") else 1
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.check:
        from app.eval.models_2026 import free_gpu
        from app.eval.provenance import scrub

        failed = 0
        for tag in [t.strip() for t in (args.models.split(",") if args.models else MODELS) if t.strip()]:
            try:
                line = check_model(tag, args.device)
            except Exception as e:  # noqa: BLE001 - reported per model
                line = {"tag": tag, "ok": False, "error": scrub(f"{type(e).__name__}: {e}")}
            failed += (not line["ok"]) and tag in CANDIDATES
            print(json.dumps(line, default=str), flush=True)
            free_gpu()
        return 1 if failed else 0
    if args.orient:
        orient(args.data_dir, args.device, [t.strip() for t in args.models.split(",")] if args.models else None)
        return 0
    if args.dry_run:
        print(json.dumps(dry_run(args.data_dir, args.work, args.device), indent=2, default=str))
        return 0
    if args.out is not None:
        from app.eval import per_question

        blob, sidecar, runs = run_fixture(root, part=args.part, device=args.device, skip_stretch=args.skip_stretch)
        out = args.out if args.out.is_absolute() else root / args.out
        blob["late_interaction"]["runs"] = write_runs(out.parent / "late_interaction_runs", runs) if runs else {}
        blob["late_interaction"]["other_scoring_jobs"] = json.loads(args.other_jobs.read_text()) if args.other_jobs else []
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        per_question.write(per_question.sidecar_path(out), sidecar)
        for row in blob["held_out"]["results"]:
            print("held_out", row["config"], row["ci"]["ndcg@10"], [(d["reference"], d["ndcg@10"]) for d in row.get("deltas", [])])
        print("reproduces", json.dumps(blob["reproduces"]))
        print(f"wrote {out}")
        return 0
    print("nothing to do: pass --check, --out, --rescore, --bar, --rederive, --tripwire, --orient or --dry-run", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
