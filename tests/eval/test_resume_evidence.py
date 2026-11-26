"""docs/RESUME_EVIDENCE.md must equal `make evidence` output; every claim row carries its provenance."""

from pathlib import Path

from app.eval.evidence import claims, render

ROOT = Path(__file__).resolve().parents[2]


def test_document_is_in_sync():
    assert (ROOT / "docs" / "RESUME_EVIDENCE.md").read_text() == render(), "run `make evidence`"


def test_every_claim_has_command_source_and_limits():
    rows = claims()
    assert len(rows) >= 10
    for r in rows:
        assert r["command"].startswith(("make ", "docker ")) and "bench/results/" in r["source"] and r["not_shown"]


def test_prose_sections_type_no_metric_digits():
    from app.eval.evidence import BULLETS, QA
    import re

    # a metric typed by hand looks like a decimal with two or more places, a signed delta, or scientific notation
    metric = re.compile(r"(?<![\w.])[-+]?\d*\.\d{2,}(?![\w.])|(?<![\w.])\d+(?:\.\d+)?[eE][-+]?\d+(?![\w.])")
    for text in (BULLETS, QA):
        found = metric.findall(text)
        assert not found, f"metric-like numbers typed by hand in prose: {found}"


def test_rrf_sweep_row_reads_the_held_out_delta_and_says_what_happens_to_the_default():
    """B12: one row for the RRF sweep, read from rrf_sweep.json: the tune-half winner minus k = 60 on the held-out half,
    the verdict word from its interval, and that the default stays unless that interval excludes zero in its favour."""
    import json

    from app.eval.evidence import _ci
    from app.eval.report import verdict

    blob = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())
    rows = [c for c in claims() if c["command"] == "make bench-rrf-sweep"]
    assert len(rows) == 1
    row, d = rows[0], blob["decision"]
    assert row["source"] == f"bench/results/rrf_sweep.json @ {blob['git_sha']}"
    if d["chosen_is_default"]:  # not the recorded run; this branch is also tested on a copy below
        assert row["ci"] == "—" and row["n"] == len(blob["split"]["tune_ids"])
        return
    block = d["held_out_ndcg@10"]
    assert (row["value"], row["ci"]) == _ci(block, "delta", digits=4) and row["n"] == len(blob["split"]["held_out_ids"])
    assert f"k = {d['chosen']['k']}, dense weight {d['chosen']['weights']['dense']:g}" in row["claim"]
    assert f"verdict from the interval: {verdict(block)}" in row["not_shown"]
    assert ("the default stays k = 60 with equal weights" in row["not_shown"]) == (not d["owner_decision_needed"])


def test_rrf_sweep_row_when_the_tune_half_chose_the_default(tmp_path):
    import json

    from app.eval.evidence import _rrf_sweep_claims

    blob = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())
    blob["decision"]["chosen"] = {"k": 60, "weights": {"bm25": 1.0, "dense": 1.0}}
    blob["decision"]["chosen_is_default"] = True
    blob["decision"]["held_out_ndcg@10"] = None
    blob["decision"]["owner_decision_needed"] = False
    (tmp_path / "bench" / "results").mkdir(parents=True)
    (tmp_path / "bench" / "results" / "rrf_sweep.json").write_text(json.dumps(blob))
    (row,) = _rrf_sweep_claims(tmp_path)
    assert row["ci"] == "—" and row["n"] == len(blob["split"]["tune_ids"]) and "chose the default" in row["claim"]
    assert "nothing to compare on the held-out half" in row["not_shown"]


def test_rrf_sweep_row_names_the_default_the_file_records(tmp_path):
    """The default setting is read from the file's decision block, not typed: a sweep recorded after the owner changed the
    default must not be reported against k = 60."""
    import json

    from app.eval.evidence import _rrf_sweep_claims

    blob = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())
    blob["decision"]["default"] = {"k": 100, "weights": {"bm25": 1.0, "dense": 1.0}}
    (tmp_path / "bench" / "results").mkdir(parents=True)
    (tmp_path / "bench" / "results" / "rrf_sweep.json").write_text(json.dumps(blob))
    (row,) = _rrf_sweep_claims(tmp_path)
    assert "minus k = 100 with equal weights" in row["claim"]
    assert "k = 60" not in row["claim"] + row["not_shown"]
    assert ("the default stays k = 100 with equal weights" in row["not_shown"]) == (not blob["decision"]["owner_decision_needed"])
    blob["decision"].update({"chosen": {"k": 100, "weights": {"bm25": 1.0, "dense": 1.0}}, "chosen_is_default": True,
                             "held_out_ndcg@10": None, "owner_decision_needed": False})
    (tmp_path / "bench" / "results" / "rrf_sweep.json").write_text(json.dumps(blob))
    (row,) = _rrf_sweep_claims(tmp_path)
    assert row["claim"].endswith("the tune half chose the default, k = 100 with equal weights")
    assert "no evidence that k = 100 is best" in row["not_shown"] and "k = 60" not in row["claim"] + row["not_shown"]


def test_hybrid_drops_row_reads_its_counts_and_ranks_from_the_file():
    """B17: the diagnosis row counts the gold files hybrid@bge-base drops and gains against dense_only@bge-base, from
    hybrid_drops_bge_base.json, with the rank ranges that place the drops just past the fused cut."""
    import json

    blob = json.loads((ROOT / "bench" / "results" / "hybrid_drops_bge_base.json").read_text())
    rows = [c for c in claims() if c["command"] == "make bench-hybrid-drops"]
    assert len(rows) == 1
    row, s = rows[0], blob["summary"]
    assert row["source"] == f"bench/results/hybrid_drops_bge_base.json @ {blob['git_sha']}" and row["n"] == s["n_questions"]
    assert row["value"] == f"{s['n_dropped']} dropped on {s['n_questions_with_drops']} questions, {s['n_gained']} gained"
    fused = [d["fused"]["chunk_rank"] for q in blob["questions"] for d in q["dropped"]]
    if fused:  # a recording with no drops has no rank range to state
        assert f"{min(fused)}-{max(fused)} in the fused list, which is cut at {blob['retrieve_k']}" in row["not_shown"]
    assert f"Recall@50 delta {s['recall@50']['delta']:+.4f}" in row["not_shown"] and "not a retrieval-quality claim" in row["not_shown"]


DOCS = ("README.md", "docs/RESUME_EVIDENCE.md", "doc/engineering-reference.md")


def test_the_docs_say_the_parity_gap_is_reported_not_gated():
    """Since #10 the parity job reports the API-minus-in-memory gap with its interval and fails only when the API is
    broken or the fixture on disk differs from its MANIFEST: no doc may call it a gate on that gap."""
    for rel in DOCS:
        text = (ROOT / rel).read_text().lower()
        for phrase in ("parity gate", "gated by the same parity job", "gated by the parity job"):
            assert phrase not in text, (rel, phrase)


def test_ts_rank_cd_is_labelled_the_default_lexical_backend_not_a_fallback():
    """M11 made the lexical backend an explicit setting (LEXICAL_BACKEND) whose default is ts_rank_cd; the served SQL
    statement never falls back to it (only the python retrieval mode's search_bm25 does, from ParadeDB, and labels the
    rows). Every doc still labels it as not BM25 wherever its numbers appear (owner rule 3), without calling the
    served default a fallback."""
    for rel in DOCS:
        for line in (ROOT / rel).read_text().splitlines():
            if "ts_rank_cd" in line:
                assert not ("ts_rank_cd (fallback)" in line or "a fallback, not BM25" in line
                            or "lexical fallback" in line or "default stack's fallbacks" in line), (rel, line[:160])
    served = [line for line in (ROOT / "README.md").read_text().splitlines() if line.startswith("| api:")]
    assert served and all("ts_rank_cd (default LEXICAL_BACKEND, not BM25)" in line for line in served)


def test_the_reranker_bug_bullet_says_what_the_history_shows():
    """OD02 (B28), owner-approved 2026-11-14: before 515e2c8 both callers used `compute_score` on a sentence-transformers
    CrossEncoder; the served retriever's broad except logged a warning and returned the fused list, and the benchmark
    labelled its no-model fallback and had no model-backed reranker row. The bullet says that, not that every earlier
    row was a silent fallback."""
    from app.eval.evidence import BULLETS

    bullet = next(line for line in BULLETS.splitlines() if line.startswith("- Measured hybrid BM25 + dense retrieval"))
    assert bullet.endswith(
        "found and fixed a reranker API bug: the code called a scoring method the loaded model did not have, the served "
        "retriever's broad except logged a warning and returned the fused list unreranked, and the benchmark had no "
        "model-backed reranker row until the fix.")
    assert "silent fallback" not in bullet


def test_the_reranker_bug_answer_opens_with_what_the_history_shows_and_keeps_its_lesson():
    """OD02, owner-approved 2026-11-14: the interview answer's first sentence is replaced (the served retriever's broad
    except returned the fused list unreranked; the benchmark labelled its no-model fallback in each row and had no
    model-backed reranker row until the fix); the rest of the answer is kept."""
    from app.eval.evidence import QA

    answer = next(p for p in QA.split("\n\n") if p.startswith("**What did the reranker bug teach you?**"))
    assert answer.startswith(
        "**What did the reranker bug teach you?** The code called a scoring method the loaded model did not have. In the "
        "served retriever a broad except caught the error, logged a warning and returned the fused list unreranked, so a "
        "response looked normal while reranking had been skipped; the benchmark labelled its no-model fallback in each "
        "row and had no model-backed reranker row until the fix. The fix was small; the lesson was structural:")
    assert "silent fallback" not in answer and "before M2" not in answer


# M36 T3: the owner's approved text, copied from the brief (milestones2/briefs/M36.md) without retyping
APPROVED_INJECTION_BULLET = (
    "Built an LLM-in-the-loop prompt-injection evaluation for the agent's file tools: local models pick "
    'MCP tool calls over corpus copies whose answering files carry planted instructions, and every '
    'episode is scored at the model level (attempted the planted action, stated the planted value) and at '
    'the system level (corpus changed, secret leaked, false value delivered), with τ-bench pass^k, '
    'any-of-k attack rates and paired bootstrap deltas. The current run plants each goal in seven '
    'phrasings (five of them AgentDojo templates) under no defense, datamarking, a least-privilege tool '
    'filter and a structured answer check between the model and the user; the answer check was the one '
    'defense that cut the planted value reaching the user for both models, with no measurable utility '
    'cost. An earlier run adds a capability control (the planted actions asked for by the user and signed '
    'by an operator key), and a later one an adaptive attacker that rewrote the planted text against each '
    'defense for up to seven rounds and did no measurably better than the fixed phrasings against the '
    'stronger model, a null result.'
)


def test_the_injection_bullet_is_the_approved_text_and_each_clause_holds_in_its_results_file():
    """Owner-approved 2026-11-14 (M36 T3): the prompt-injection bullet, word for word. Its clauses are read off the three
    results files it describes: the current run (v3), the earlier run with the capability control (v2) and the later
    adaptive run."""
    import json

    from app.eval.evidence import BULLETS

    bullet = next(line for line in BULLETS.splitlines() if line.startswith("- Built an LLM-in-the-loop prompt-injection"))
    assert bullet == "- " + APPROVED_INJECTION_BULLET
    v2, v3, adaptive = (json.loads((ROOT / "bench" / "results" / f"agent_injection_{run}.json").read_text())
                        for run in ("v2", "v3", "adaptive"))
    # seven phrasings, five of them AgentDojo templates, under the four configurations named
    assert len(v3["config"]["phrasings"]) == 7 and len(v3["config"]["agentdojo_templates"]) == 5
    assert v3["config"]["defenses"] == ["none", "spotlight", "least_privilege", "answer_check"]
    # the answer check was the one defense that cut the planted value reaching the user for both models, with no
    # measurable utility cost
    deltas = v3["paired_deltas"]
    assert all(deltas[m]["answer_check - none"]["planted_value_delivered"]["hi"] < 0 for m in deltas)
    for defense in ("spotlight", "least_privilege"):
        assert not all(deltas[m][f"{defense} - none"]["planted_value_delivered"]["hi"] < 0 for m in deltas)
    for key in ("utility_under_attack", "benign_success"):
        assert all(deltas[m]["answer_check - none"][key]["lo"] <= 0 <= deltas[m]["answer_check - none"][key]["hi"]
                   for m in deltas)
    # an earlier run adds the capability control; a later one the adaptive attacker, seven rounds, against the stronger
    # model, with no interval above zero
    assert v2["run_at_utc"] < v3["run_at_utc"] < adaptive["run_at_utc"]
    assert v2["config"]["capability_actions"] and not any(
        b.get("capability") for per in v3["summary"].values() for b in per.values())
    assert adaptive["config"]["rounds"] == 7
    stronger = max(v3["summary"], key=lambda m: v3["summary"][m]["none"]["benign_success"]["mean"])
    assert adaptive["config"]["models"] == [stronger]
    blocks = [lv["delta"] for d in adaptive["comparison"][stronger].values() for g in d.values() for lv in g.values()
              if isinstance(lv, dict) and "delta" in lv]
    assert blocks and not any(b["lo"] > 0 for b in blocks)


def test_the_scoring_answer_names_each_run_by_the_rescore_target_that_rebuilds_it():
    """M36 T5: agent_injection_v3 is the current run and agent_injection_v2 the second; the answer on checking the scoring
    names each run's rescore target, and each target is in the Makefile."""
    from app.eval.evidence import QA

    answer = next(p for p in QA.split("\n\n") if p.startswith("**How did you check your own scoring?**"))
    assert ("(the second run's, which also store every tool result, with `make bench-agent-injection-v2-rescore`, and the "
            "current run's with `make bench-agent-injection-v3-rescore`)") in answer
    makefile = (ROOT / "Makefile").read_text()
    assert all(f"\n{target}:" in makefile for target in ("bench-agent-injection-v2-rescore", "bench-agent-injection-v3-rescore"))


def test_the_injection_rows_lead_with_the_current_run_and_keep_the_others_labelled_by_run():
    """M36 T5: the owner's decision that v3 leads. The injection rows open with the current run's (agent_injection_v3)
    rates, any-of-the-phrasings with no defense and the answer check's effect on the planted value delivered; the second
    run's and the adaptive run's rows follow, each labelled by its run and citing its own results file. M50: the 2026
    open models' rows come right after the current run's, their cross-model rows ("<new> minus <committed>") with them,
    citing the comparison file."""
    rows = [c for c in claims() if c["claim"].startswith("Agent injection")]
    labels = [c["claim"].split(", ")[1].split(" (")[0] for c in rows]
    runs = ["2026 open models" if " minus " in label else label for label in labels]
    order = ["current run", "2026 open models", "second run", "adaptive run"]
    assert runs == sorted(runs, key=order.index) and {"current run", "second run", "adaptive run"} <= set(runs) <= set(order)
    for run, results in (("current run", ("agent_injection_v3.json",)),
                         ("2026 open models", ("agent_injection_v3_models_2026.json",
                                               "agent_injection_v3_models_2026_compare.json")),
                         ("second run", ("agent_injection_v2.json",)), ("adaptive run", ("agent_injection_adaptive.json",))):
        assert all(c["source"].startswith(tuple(f"bench/results/{f} @ " for f in results))
                   for c, r in zip(rows, runs) if r == run)
    assert all(c["source"].startswith("bench/results/agent_injection_v3_models_2026_compare.json @ ")
               for c, label in zip(rows, labels) if " minus " in label)
    assert "no defense: hijack questions in which at least one phrasing put the planted value" in rows[0]["claim"]
    assert "the structured answer check in the loop minus no defense, the planted value delivered" in rows[1]["claim"]


def test_the_search_policy_has_one_row_the_declared_bar_and_says_in_words_what_the_reward_did(tmp_path):
    """M44: the evidence table carries the bar declared before training (exact match against the untrained model) and
    nothing else from this run, whatever its verdict; the reward, secondary to the bar, is described in words read
    from the file, with no number the file does not hold."""
    import json

    from app.eval.evidence import _search_policy_claims

    def verdict(metric, reference, lo, hi, primary=False):
        word = "above zero" if lo > 0 else "below zero" if hi < 0 else "covers zero"
        return {"config": "policy-grpo", "reference": reference, "metric": metric, "delta": round((lo + hi) / 2, 4), "lo": lo, "hi": hi,
                "interval": word, "outcome": {"above zero": "win", "covers zero": "tie", "below zero": "loss"}[word], "primary": primary, "n_seeds": 3}

    def write(verdicts, uncited=(0.0882, 0.0245)):
        blob = {"git_sha": "abc1234", "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"}, "split": {"n_held_out": 68}, "verdicts": verdicts,
                "reward": {"formula": "R = 1.0 * em + 0.5 * nDCG@10(files shown) - 0.1 * n_search - 1.0 * uncited_or_empty"},
                "results": [{"config": "policy-untrained", "means": {"uncited_or_empty": uncited[0]}},
                            {"config": "policy-grpo", "means": {"uncited_or_empty": uncited[1]}}]}
        (tmp_path / "bench" / "results").mkdir(parents=True, exist_ok=True)
        (tmp_path / "bench" / "results" / "search_policy.json").write_text(json.dumps(blob))
        return _search_policy_claims(tmp_path)

    others = [verdict("reward", "policy-untrained", 0.04, 0.2), verdict("em_bounded", "policy-untrained", 0.01, 0.1),
              verdict("em", "single-shot-rag", -0.01, 0.09), verdict("reward", "single-shot-rag", 0.1, 0.26)]
    (row,) = write([verdict("em", "policy-untrained", -0.0049, 0.0931, primary=True), *others])
    assert row["value"] == "+0.0441" and row["ci"] == "[-0.0049, +0.0931]" and row["n"] == 68 and row["claim"].endswith("(the bar declared before training)")
    assert "each question's mean over 3 training seeds" in row["claim"] and "minus the same model untrained" in row["claim"]
    assert row["not_shown"].startswith("verdict from the interval: covers zero, a tie on the bar declared before training; "
                                       "against single-shot RAG with the same model, exact match is a tie too; "
                                       "secondary to the bar, the mean reward rose, with an interval above zero, about half of it from fewer uncited or empty answers;")
    # a win on the bar is still the one row; a reward whose interval covers zero is said to have not moved measurably
    (row,) = write([verdict("em", "policy-untrained", 0.01, 0.09, primary=True), verdict("reward", "policy-untrained", -0.02, 0.2)])
    assert "above zero, a win on the bar" in row["not_shown"] and "the mean reward did not move measurably (its interval covers zero)" in row["not_shown"]
    (row,) = write([verdict("em", "policy-untrained", -0.0049, 0.0931, primary=True), verdict("reward", "policy-untrained", 0.04, 0.2)], uncited=(0.12, 0.0))
    assert "nearly all of it from fewer uncited or empty answers" in row["not_shown"]
    (row,) = write([verdict("em", "policy-untrained", -0.0049, 0.0931, primary=True), verdict("reward", "policy-untrained", 0.04, 0.2)], uncited=(0.2, 0.0))
    assert "more than all of it from fewer uncited or empty answers" in row["not_shown"]
    assert _search_policy_claims(tmp_path / "nowhere") == []
