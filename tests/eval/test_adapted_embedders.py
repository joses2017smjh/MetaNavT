"""M48 (app/eval/adapted_embedders.py): what was declared before any adapted embedder was trained or measured.

The declarations are pinned value by value: a bar, a recipe or an arm list that could change quietly after a number
exists would not be a declaration. Tests that need torch skip where it is not installed (the login node's suites) and
run under the GPU environment (recorded in milestones2/m48/NOTES.md).
"""

import json
from pathlib import Path

from app.eval import adapted_embedders as ae

ROOT = Path(__file__).resolve().parents[2]


def test_the_bar_is_one_named_row_on_three_datasets():
    assert ae.BAR_ARM == "bge-small-marginmse"
    assert ae.BAR["row"] == "hybrid@bge-small-marginmse" and ae.BAR["parent"] == "hybrid@bge-small"
    assert ae.BAR["metric"] == "ndcg@10"
    assert ae.BAR["datasets"] == {"fixture": 68, "scifact": 300, "fiqa": 648}
    for needle in ("strictly above zero", "a stored 0.0 is not above zero", "held-out half (68 questions)", "SciFact test (300)",
                   "FiQA test (648)"):
        assert needle in ae.BAR["passes"]
    assert "seed 0 alone" in ae.BAR["one_seed"]
    assert "never read from the shared cache" in ae.BAR["parent_rule"]
    assert "RRF, k = 60, equal weights" in ae.BAR["hybrid"]
    assert "split_questions" in ae.BAR["held_out"] and "no choice of any kind is made on the tune half" in ae.BAR["held_out"]
    assert ae.BAR["read_from"] == ("bench/results/adapted_embedders.json (scope held_out)", "bench/results/beir_scifact_adapted_embedders.json",
                                   "bench/results/beir_fiqa_adapted_embedders.json")
    assert "first complete eval run per dataset at the C3" in ae.BAR["which_run"] and "never replaces it" in ae.BAR["which_run"]
    assert "2000 resamples, seed 0" in ae.BAR["interval"]
    assert tuple(ae.BAR["outcomes"]) == ae.OUTCOMES == ("passes", "does not pass", "open", "not run")
    assert "intersection-union" in ae.BAR["multiple_testing"] and "2.5%" in ae.BAR["multiple_testing"]
    assert "no metric uses an LLM judge" in ae.BAR["llm_judge"]
    assert ae.ATTEMPT == 1


def test_the_arms_are_declared_in_table_order():
    assert list(ae.ARMS) == ["bge-small-marginmse", "bge-small-marginmse-nofixture", "bge-small-marginmse-s1", "bge-small-marginmse-s2",
                             "bge-small-mnrl", "bge-base-marginmse"]
    shape = {t: (a["init"], a["loss"], a["training_set"], a["seed"], a["role"]) for t, a in ae.ARMS.items()}
    assert shape == {
        "bge-small-marginmse": ("bge-small", "marginmse", "all", 0, "bar"),
        "bge-small-marginmse-nofixture": ("bge-small", "marginmse", "beir", 0, "control"),
        "bge-small-marginmse-s1": ("bge-small", "marginmse", "all", 1, "seed"),
        "bge-small-marginmse-s2": ("bge-small", "marginmse", "all", 2, "seed"),
        "bge-small-mnrl": ("bge-small", "mnrl", "all", 0, "loss"),
        "bge-base-marginmse": ("bge-base", "marginmse", "all", 0, "base"),
    }
    assert ae.TRAINING_SETS == {"all": ("fixture", "scifact", "fiqa"), "beir": ("scifact", "fiqa")}
    assert ae.SCALE_KEYS == (("bge-small", "all"), ("bge-small", "beir"), ("bge-base", "all"))
    assert ae.SEED_ARMS == ("bge-small-marginmse", "bge-small-marginmse-s1", "bge-small-marginmse-s2")
    assert ae.CONTROL_ARM == "bge-small-marginmse-nofixture" and ae.MNRL_ARM == "bge-small-mnrl" and ae.BASE_ARM == "bge-base-marginmse"
    assert "confounded" in ae.ARMS[ae.CONTROL_ARM]["about"]


def test_the_inits_are_pinned_and_encode_as_every_committed_bge_row():
    assert ae.INITS["bge-small"] == {"model": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                                     "licence": "mit", "embedder": "st:BAAI/bge-small-en-v1.5"}
    assert ae.INITS["bge-base"]["revision"] == "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a" and ae.INITS["bge-base"]["optional"] is True
    assert ae.ENCODING == {"pooling": "cls", "normalize": True, "max_seq_length": 512, "do_lower_case": True, "dtype": "float32",
                           "query_prompt": None, "document_prompt": None}
    from app.retrieval.embedders import QUERY_PROMPT_NAMES

    assert not any(init["model"] in QUERY_PROMPT_NAMES for init in ae.INITS.values())  # bare queries, as every bge row


def test_the_recipe_is_m43s_with_a_fixed_scale():
    from app.eval import small_rerankers as sr

    numbers = {k: v for k, v in ae.RECIPE.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    assert numbers == {"lists_per_step": 8, "epochs": 4, "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_fraction": 0.1, "max_grad_norm": 1.0}
    for key in numbers:  # M43's optimisation recipe, unchanged
        assert sr.RECIPE[key] == numbers[key], key
    assert ae.RECIPE["corpus_mixing"] == sr.RECIPE["corpus_mixing"]
    assert ae.RECIPE["train_lists"] == {"fixture": 691, "scifact": 4267, "fiqa": 5129}
    assert ae.RECIPE["validation_lists"] == {"fixture": 42, "scifact": 223, "fiqa": 270}
    assert "std with ddof 0 in float64" in ae.RECIPE["scale"] and "every pair i < j" in ae.RECIPE["scale"]
    assert "mean over i < j of ((s_i - s_j) - (t_i - t_j))^2" in ae.RECIPE["loss"]
    assert "s * cos(E(q), E(d))" in ae.RECIPE["student_score"] and "preprocess()" in ae.RECIPE["student_score"]
    assert "allow_tf32 False" in ae.RECIPE["precision"] and "'highest'" in ae.RECIPE["precision"]
    assert "never a hyper-parameter" in ae.RECIPE["memory"]
    assert "never a gold question, a tune-half question or a BEIR query" in ae.RECIPE["validation"]
    assert "ties go to the earlier epoch" in ae.RECIPE["selection"] and "not used" in ae.RECIPE["selection"]
    assert ae.RECIPE["prompts"] is None


def test_the_mnrl_arm_rules_are_fixed():
    m = ae.RECIPE["mnrl"]
    assert (m["scale"], m["hard_negatives"], m["queries_per_batch"]) == (20.0, 7, 32)
    assert "10 lowest-scored candidates of a 20- or 21-candidate list" in m["hard_negative_rule"]
    assert "the first 7 of those in the list's order" in m["hard_negative_rule"]
    assert "teacher ranks the source document first" in m["lists"]
    assert "-inf, except the anchor's own positive column" in m["mask"]
    assert "never dropped" in m["batches"] and "square root" in m["batches"]
    assert "batches built once with seed 0" in m["selection"]
    assert "40 GB or more" in m["hardware"]


def test_the_twin_rule_and_the_files_are_named():
    assert ae.TWINS["thresholds"] == (0.80, 0.85) and ae.TWINS["fixture_reading_threshold"] == 0.85
    assert (ae.TWINS["assert_below"], ae.TWINS["assert_tolerance"], ae.TWINS["min_subgroup"]) == (0.90, 0.001, 10)
    assert ae.FIXTURE_READINGS == ("adaptation beyond templated twins", "not separable from templated twins")
    assert ae.CUTOFF == "2026-12-06" and ae.CPU_BAND == (0.90, 1.10)
    assert ae.RESULTS == {"fixture": "bench/results/adapted_embedders.json", "scifact": "bench/results/beir_scifact_adapted_embedders.json",
                          "fiqa": "bench/results/beir_fiqa_adapted_embedders.json"}
    assert all(p.endswith("adapted_embedders_base.json") for p in ae.BASE_RESULTS.values())
    assert (ae.SCALE_FILE, ae.TWINS_FILE, ae.MANIFEST_FILE) == ("bench/distill/embedder_scale.json", "bench/distill/embedder_twins.json",
                                                                "bench/distill/embedders.json")
    assert ae.SIDECAR_SCHEMA == "adapted-embedders-per-query-v1"
    assert ae.embedder_name(ae.BAR_ARM) == "ft:bge-small-marginmse"
    assert ae.row_names(ae.BAR_ARM, "fixture") == ("dense_only@bge-small-marginmse", "hybrid@bge-small-marginmse")
    assert ae.row_names("bge-small", "fiqa") == ("dense@bge-small", "hybrid@bge-small")


def test_the_declared_list_counts_are_the_committed_files():
    meta = json.loads((ROOT / "bench" / "distill" / "queries.meta.json").read_text())
    kept = {c: b["kept_by_split"] for c, b in meta["near_duplicates"]["by_corpus"].items()}
    assert {c: k["train"] for c, k in kept.items()} == ae.RECIPE["train_lists"]
    assert {c: k["validation"] for c, k in kept.items()} == ae.RECIPE["validation_lists"]
