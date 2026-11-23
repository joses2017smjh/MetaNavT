"""The search policy's results file and its two sidecars (M44), and their recomputation on a CPU.

    python -m app.rl.results --arms DIR --run grpo/seed=0=RUN_DIR ... [--jobs FILE]   # assemble (make search-policy-eval)
    python -m app.rl.results --check                                                    # make search-policy-rescore

Three files under bench/results:

  search_policy.rollouts.json   every arm's episode for every held-out question: each piece the policy wrote, each
                                query, the files each search showed, the retrieved text, the answer, and the score;
  search_policy.training.json   every training run: its settings, one record per step (the training curve), and
                                every checkpoint's tune-half score, per question;
  search_policy.json            the header (what ran, on what), and everything computed from the two sidecars and
                                the gold file: per-arm means with bootstrap intervals, paired deltas, the verdicts
                                against the declared bar, the checkpoint each seed selected, the collapse rule, the
                                leakage report of the training questions.

--check recomputes that last part from the sidecars: each episode is scored again from its stored text and shown
files (no retrieval, no model, no torch), every interval is redrawn, the selected checkpoints and the collapse rule
are re-read from the curve, the leakage report is rebuilt from the corpus seeds. It exits 1 unless every value
equals the committed file.

The bar was declared before the first training run (BAR below; the milestone notes carry the same text with its
time): on fixture v1's held-out questions, trained minus untrained, the paired bootstrap delta in exact match is
the primary number, and with several seeds the trained score of a question is the mean over the seeds, so the
verdict cannot be the best seed's.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.eval.gold import GoldQuestion, load_gold
from app.eval.provenance import command_line, git_sha, git_state, run_provenance, scrub, scrub_names, sha256_file
from app.eval.report import verdict
from app.eval.rerank_arms import split_questions
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, mean_ci, paired_delta_ci, resample_indices, settings as bootstrap_settings
from app.rl.corpora import corpus_identities, training_set
from app.rl.episode import Episode, score_episode

ROOT = Path(__file__).resolve().parents[2]
RESULTS = "bench/results/search_policy.json"
ROLLOUTS = "bench/results/search_policy.rollouts.json"
TRAINING = "bench/results/search_policy.training.json"
UNTRAINED, SINGLE_SHOT = "policy-untrained", "single-shot-rag"
CI_METRICS = ("em", "reward", "em_bounded")  # mean with interval per arm; paired delta with interval per comparison
MEAN_METRICS = ("ndcg@10", "n_search", "uncited_or_empty", "answered", "cited", "answer_chars")
COLLAPSE_WINDOW, COLLAPSE_DROP, COLLAPSE_ANSWERED = 20, 0.1, 0.5
SHARED_DEVICE_KEYS = ("requested", "device", "device_name", "cuda_available", "torch", "python", "sentence_transformers")

BAR = {
    "declared": "before the first training run (milestone notes, 2026-11-22, section 1)",
    "questions": "fixture v1's held-out half: the `test` half of app.eval.rerank_arms.split_questions, the half rrf_sweep holds out",
    "comparison": "trained policy minus untrained policy, same model, prompt, environment and questions, greedy decoding",
    "primary": "the paired bootstrap delta in exact match (`em`)",
    "secondary": "the same delta in mean reward; and both again against single-shot RAG",
    "reading": "a 95% interval above zero is a win, one that covers zero is a tie, one below zero is a loss",
    "seeds": ("with several training seeds a question's trained score is the mean over the seeds (exact match in thirds), and the "
              "verdict is read from that row; every seed's own delta is in the table"),
    "checkpoint": ("per seed, the saved checkpoint with the highest mean reward on the TUNE half, greedy; a tie goes to the earliest "
                   "step; step 0 is not eligible"),
    "collapse": (f"a run has collapsed when its loss is not finite, or fewer than {COLLAPSE_ANSWERED:g} of the last {COLLAPSE_WINDOW} steps' "
                 f"episodes hold an answer, or the mean episode reward of its last {COLLAPSE_WINDOW} steps is more than {COLLAPSE_DROP:g} "
                 f"below that of its first {COLLAPSE_WINDOW}; a collapsed GRPO run is followed by one PPO-clip run"),
}
REWARD = {
    "formula": "R = 1.0 * em + 0.5 * nDCG@10(files shown, in order of first appearance; gold paths) - 0.1 * n_search - 1.0 * uncited_or_empty",
    "jury_term": "none: PLAN.md Phase 10's 0.3 * jury vote is not in this reward (no jury runs in the loop)",
    "overlap_teacher": "not used",
    "em": ("strict: the answer with its bracketed citations removed equals the gold answer after normalisation (lower case, whitespace "
           "collapsed, surrounding quotes and a final full stop dropped, standalone numbers compared as numbers)"),
    "em_bounded": ("guard column, never the reward: app.eval.judge.exact_match_label(bounded=True) is \"correct\" for the answer without "
                   "its citations (the gold is somewhere in it, not inside a longer number or word); it credits a verbose answer and a "
                   "list of candidates"),
    "cited": "a bracketed path in the answer is one of the files this episode's searches showed",
    "uncited_or_empty": "no <answer>, an answer with no value outside its citations, or not cited",
    "n_search": "searches the environment ran, the single-shot arm's own search included",
}
DOES_NOT_SHOW = [
    "Synthetic corpora: training and test corpora come from one builder (bench/corpus/build.py); nothing here is a real research directory.",
    ("Templated questions: every training question is built from a template that also builds test questions (same wording, other run "
     "ids, names and values), so a gain may be a gain on these templates only. The test questions whose template had no training "
     "question are listed under `leakage`."),
    "One small model (Qwen2.5-1.5B-Instruct with a LoRA adapter) and one prompt: no model sweep, no prompt sweep.",
    ("One test corpus of 61 files and 68 held-out questions, 40 of them the per-run lookups; the tune half of the same corpus chose the "
     "checkpoint."),
    ("Strict exact match can move on answer format alone; `em_bounded` and the answer length are the guard columns, and composite gold "
     "answers (a list, a sentence, two values joined by a semicolon) are counted apart from single values."),
    "No jury term in the reward, and no judge of answer quality beyond exact match.",
    ("Greedy decoding on one GPU type: the episodes are stored, and every number is recomputed from them, but regenerating them on other "
     "hardware may not reproduce every token."),
]


def gold_form(answer: str) -> str:
    """single_value: one token with no list separator (a number, a name, a path); composite: anything else."""
    text = answer.strip()
    return "single_value" if len(text.split()) == 1 and "," not in text and ";" not in text else "composite"


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _ci(values: Sequence[float], idx) -> dict:
    mean, lo, hi = mean_ci(values, idx)
    return {"mean": _round(mean), "lo": _round(lo), "hi": _round(hi)}


def _delta(a: Sequence[float], b: Sequence[float], idx) -> dict:
    delta, lo, hi = paired_delta_ci(a, b, idx)
    return {"delta": _round(delta), "lo": _round(lo), "hi": _round(hi)}


def rescore(rows: dict[str, dict], gold: dict[str, GoldQuestion]) -> dict[str, dict]:
    """{question id: score} from the stored episodes alone."""
    return {qid: score_episode(Episode.from_dict(row["episode"]), gold[qid].answer, gold[qid].relevant_paths) for qid, row in rows.items()}


def seed_mean(per_seed: Sequence[dict[str, dict]], ids: Sequence[str]) -> dict[str, dict]:
    """Per question, the mean over the seeds of every numeric score (exact match in thirds for three seeds)."""
    keys = (*CI_METRICS, *MEAN_METRICS)
    return {qid: {k: sum(float(scores[qid][k]) for scores in per_seed) / len(per_seed) for k in keys} for qid in ids}


def arm_row(config: str, scores: dict[str, dict], ids: Sequence[str], gold: dict[str, GoldQuestion], idx) -> dict:
    values = {k: [float(scores[q][k]) for q in ids] for k in (*CI_METRICS, *MEAN_METRICS)}
    row: dict[str, Any] = {"config": config, "n": len(ids), "ci": {k: _ci(values[k], idx) for k in CI_METRICS},
                           "means": {k: _round(sum(values[k]) / len(ids)) for k in MEAN_METRICS}}
    if all("stop" in scores[q] for q in ids):
        stops: dict[str, int] = {}
        for q in ids:
            stops[scores[q]["stop"]] = stops.get(scores[q]["stop"], 0) + 1
        row["stop"] = dict(sorted(stops.items()))
    groups: dict[str, dict[str, list[str]]] = {"by_category": {}, "by_gold_form": {}}
    for q in ids:
        groups["by_category"].setdefault(gold[q].category, []).append(q)
        groups["by_gold_form"].setdefault(gold_form(gold[q].answer), []).append(q)
    for name, parts in groups.items():
        row[name] = {part: {"n": len(qs), **{k: _round(sum(float(scores[q][k]) for q in qs) / len(qs)) for k in ("em", "em_bounded", "reward")}}
                     for part, qs in sorted(parts.items())}
    return row


def delta_block(scores: dict[str, dict], reference: str, ref_scores: dict[str, dict], ids: Sequence[str], idx) -> dict:
    block: dict[str, Any] = {"reference": reference}
    for k in CI_METRICS:
        block[k] = _delta([float(scores[q][k]) for q in ids], [float(ref_scores[q][k]) for q in ids], idx)
    block["n_questions_changed"] = {k: {"up": sum(1 for q in ids if float(scores[q][k]) > float(ref_scores[q][k])),
                                        "down": sum(1 for q in ids if float(scores[q][k]) < float(ref_scores[q][k]))} for k in ("em", "em_bounded")}
    return block


def selected_step(tune: Sequence[dict]) -> int | None:
    """BAR["checkpoint"]: the highest tune-half mean reward among steps after 0; a tie goes to the earliest step."""
    eligible = [e for e in tune if e["step"] > 0]
    return max(eligible, key=lambda e: (e["reward"], -e["step"]))["step"] if eligible else None


def collapse(curve: Sequence[dict]) -> dict:
    """BAR["collapse"] read off a training curve."""
    w = COLLAPSE_WINDOW
    finite = all(math.isfinite(float(r["loss"])) for r in curve)
    first, last = curve[:w], curve[-w:]

    def mean(rows: Sequence[dict], key: str) -> float | None:
        return _round(sum(float(r[key]) for r in rows) / len(rows)) if rows else None

    out = {"n_steps": len(curve), "loss_finite": finite, "reward_first": mean(first, "reward"), "reward_last": mean(last, "reward"),
           "answered_last": mean(last, "answered"), "em_first": mean(first, "em"), "em_last": mean(last, "em"), "window": w}
    reasons = []
    if not finite:
        reasons.append("the loss is not finite")
    if out["answered_last"] is not None and out["answered_last"] < COLLAPSE_ANSWERED:
        reasons.append(f"fewer than {COLLAPSE_ANSWERED:g} of the last {w} steps' episodes hold an answer")
    if out["reward_first"] is not None and out["reward_last"] < out["reward_first"] - COLLAPSE_DROP:
        reasons.append(f"the mean reward of the last {w} steps is more than {COLLAPSE_DROP:g} below the first {w}")
    return {**out, "collapsed": bool(reasons), "reasons": reasons}


def trained_family(name: str) -> str:
    """policy-grpo/seed=0 -> policy-grpo: the row a family's seeds are averaged into."""
    return name.split("/seed=")[0]


def compute(rollouts: dict, training: dict, gold_list: Sequence[GoldQuestion]) -> dict:
    """Everything in the results file that is a function of the two sidecars and the gold file."""
    gold = {q.id: q for q in gold_list}
    tune_idx, test_idx = split_questions(list(gold_list))
    ids = [gold_list[i].id for i in test_idx]
    idx = resample_indices(len(ids), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    arms = rollouts["arms"]
    scores = {name: rescore(arm["rows"], gold) for name, arm in arms.items()}
    for name, arm_scores in scores.items():
        if sorted(arm_scores) != sorted(ids):
            raise ValueError(f"{name} does not hold exactly the held-out questions")
    stored_differs = sorted(f"{name}/{qid}" for name, arm in arms.items() for qid, row in arm["rows"].items() if row["score"] != scores[name][qid])

    families: dict[str, list[str]] = {}
    for name, arm in arms.items():
        if arm["kind"] == "trained":
            families.setdefault(trained_family(name), []).append(name)
    means = {family: seed_mean([scores[m] for m in sorted(members)], ids) for family, members in families.items() if len(members) > 1}
    table: dict[str, dict[str, dict]] = {**{n: scores[n] for n in (SINGLE_SHOT, UNTRAINED) if n in scores}}
    for family in sorted(families):
        if family in means:
            table[family] = means[family]
        for member in sorted(families[family]):
            table[member] = scores[member]

    results = []
    for name, arm_scores in table.items():
        row = arm_row(name, arm_scores, ids, gold, idx)
        if name in means:
            row["kind"], row["seeds"] = "trained, mean of seeds", sorted(families[name])
        else:
            row["kind"] = arms[name]["kind"]
        references = [SINGLE_SHOT] if name == UNTRAINED else [UNTRAINED, SINGLE_SHOT] if name not in (SINGLE_SHOT, UNTRAINED) else []
        row["deltas"] = [delta_block(arm_scores, ref, table[ref], ids, idx) for ref in references if ref in table]
        results.append(row)
    by = {row["config"]: row for row in results}

    verdicts = []
    for family in sorted(families):
        headline = family if family in means else sorted(families[family])[0]
        for ref in (UNTRAINED, SINGLE_SHOT):
            block = next((d for d in by[headline]["deltas"] if d["reference"] == ref), None)
            if block is None:
                continue
            for metric in ("em", "reward", "em_bounded"):
                word = verdict(block[metric])
                verdicts.append({"config": headline, "reference": ref, "metric": metric, **block[metric], "interval": word,
                                 "outcome": {"above zero": "win", "covers zero": "tie", "below zero": "loss"}[word],
                                 "primary": ref == UNTRAINED and metric == "em", "n_seeds": len(families[family])})

    runs = []
    for name, run in sorted(training["runs"].items()):
        tune = run["tune"]
        step = selected_step(tune)
        untrained_tune = next((e for e in tune if e["step"] == 0), None)
        runs.append({"name": name, "selected_step": step,
                     "tune": [{"step": e["step"], "reward": e["reward"], "em": e["em"], "n_search": e["n_search"], "uncited_or_empty": e["uncited_or_empty"]}
                              for e in sorted(tune, key=lambda e: e["step"])],
                     "tune_reward_untrained": untrained_tune["reward"] if untrained_tune else None,
                     "tune_reward_selected": next((e["reward"] for e in tune if e["step"] == step), None),
                     "collapse": collapse(run["curve"]),
                     "seconds": {"rollouts": _round(sum(r["rollout_s"] for r in run["curve"]), 1), "updates": _round(sum(r["update_s"] for r in run["curve"]), 1),
                                 "tune_scoring": _round(sum(e.get("seconds", 0.0) for e in tune), 1)}})
    seeds = sorted({s for run in training["runs"].values() for s in run["config"]["corpus_seeds"]})
    _, leakage = training_set(seeds, gold_list) if seeds else ([], None)
    # The corpora are rebuilt here from their seeds; a run trained on them only if it recorded the same hashes.
    rebuilt = [{"seed": c["seed"], "corpus_sha256": c["corpus_sha256"], "gold_sha256": c["gold_sha256"], "n_files": c["n_files"], "n_gold": c["n_gold"]}
               for c in corpus_identities(seeds)] if seeds else []
    by_seed = {c["seed"]: (c["corpus_sha256"], c["gold_sha256"]) for c in rebuilt}
    corpora_differ = sorted(f"{name}/corpus {c['seed']}" for name, run in training["runs"].items() for c in run["config"].get("corpora") or []
                            if by_seed.get(c["seed"]) != (c["corpus_sha256"], c["gold_sha256"]))
    if leakage is not None:
        untrained_templates = set(leakage["fixture_questions_whose_template_is_not_trained"])
        leakage["held_out_questions_whose_template_is_not_trained"] = sorted(q for q in ids if q in untrained_templates)
        leakage["n_held_out_questions_whose_template_is_not_trained"] = len(leakage["held_out_questions_whose_template_is_not_trained"])
    forms: dict[str, int] = {}
    for q in ids:
        forms[gold_form(gold[q].answer)] = forms.get(gold_form(gold[q].answer), 0) + 1
    return {
        "bootstrap": bootstrap_settings(len(ids)),
        "split": {"rule": "alternate train / test inside each category, gold order (app.eval.rerank_arms.split_questions)",
                  "tune_ids": [gold_list[i].id for i in tune_idx], "held_out_ids": ids, "n_held_out": len(ids),
                  "held_out_gold_forms": {**dict(sorted(forms.items())),
                                          "rule": "single_value: one token with no comma or semicolon; composite: anything else"}},
        "results": results,
        "verdicts": verdicts,
        "training_runs": runs,
        "training_corpora": {"command": "python -m app.rl.corpora --seeds <seeds> --out <a directory outside the repository>", "seeds": seeds,
                             "rebuilt_from_seeds": rebuilt, "runs_whose_recorded_corpora_differ": corpora_differ},
        "leakage": leakage,
        "stored_scores_differ_from_rescored": stored_differs,
    }


# ---- assembling the files from a run's outputs

def _adapter_bytes(directory: Path) -> int | None:
    path = directory / "adapter_model.safetensors"
    return path.stat().st_size if path.exists() else None


def assemble(arms_dir: Path, runs: dict[str, Path], jobs: list[dict] | None, root: Path = ROOT) -> tuple[dict, dict, dict]:
    """(results, rollouts sidecar, training sidecar) from the arm files of app.rl.evaluate and the run directories of
    app.rl.train. A training run named grpo/seed=0 belongs to the arm policy-grpo/seed=0."""
    started = git_state(root)
    gold_list = load_gold(root / "bench" / "gold" / "questions.jsonl")
    arm_files = sorted(Path(arms_dir).glob("*.json"))
    if not arm_files:
        raise SystemExit(f"no arm files under {arms_dir}")
    arms: dict[str, dict] = {}
    header: dict[str, Any] = {}
    # Every arm must have run on the same model, index, questions, decoding and hardware; what may differ between
    # arms (the base-model arms are scored before training, the trained ones after) is recorded per arm. Of the
    # device, the GPU and the software must match; the node's CPU model and count and its kernel are the node's,
    # and arms scored on two nodes with the same GPU differ there.
    shared = ("model", "embedder", "libraries", "gpu", "precision", "corpus_sha256", "gold_sha256", "n_files", "n_chunks", "decoding",
              "max_new_tokens_per_piece", "information_chunks_per_search", "split", "question_ids")
    for path in arm_files:
        blob = json.loads(path.read_text())
        if blob["split"] != "held_out":
            raise SystemExit(f"{path.name} is a {blob['split']} file; the results file is the held-out half")
        for key, value in [*((k, blob[k]) for k in shared), ("device", {k: blob["device"].get(k) for k in SHARED_DEVICE_KEYS})]:
            if header.setdefault(key, value) != value:
                raise SystemExit(f"{path.name}: {key} differs from the other arms ({value!r} != {header[key]!r})")
        arms[blob["arm"]] = {"kind": blob["kind"], "adapter": blob["adapter"], "max_searches": blob["max_searches"],
                             "prompt_sha256": blob["prompt_sha256"], "seconds": blob["seconds"], "command": blob["command"],
                             "timestamp": blob["timestamp"], "git_sha": blob["git_sha"], "git_head": blob["git_head"],
                             "working_tree_dirty": blob["working_tree_dirty"], "device": blob["device"], "rows": blob["rows"]}
    training: dict[str, Any] = {"runs": {}}
    for name, directory in sorted(runs.items()):
        directory = Path(directory)
        config = json.loads((directory / "run.json").read_text())
        curve = [json.loads(line) for line in (directory / "curve.jsonl").read_text().splitlines() if line.strip()]
        tune = json.loads((directory / "tune.json").read_text()) if (directory / "tune.json").exists() else []
        best = directory / "best"
        training["runs"][f"policy-{name}"] = {
            "config": config, "curve": curve, "tune": tune,
            "adapter": {"sha256": {f: sha256_file(best / f) for f in ("adapter_config.json", "adapter_model.safetensors")} if best.exists() else None,
                        "bytes": _adapter_bytes(best)},
        }
    for name, run in training["runs"].items():
        arm = arms.get(name)
        if arm is not None and arm["adapter"]["sha256"] != run["adapter"]["sha256"]:
            raise SystemExit(f"{name}: the adapter that was scored is not the run's selected checkpoint")
    rollouts = {"what": "one greedy episode per arm and held-out question, with its score; search_policy.json is recomputed from this file",
                "arms": arms}
    # What is written is what --check will read: the episodes keep their text as the model and the corpus wrote it
    # (only a user or host name would be replaced), and everything is computed from the written form.
    rollouts = json.loads(json.dumps(scrub_names(rollouts)))
    training = json.loads(json.dumps(scrub(training, root=root)))
    computed = compute(rollouts, training, gold_list)
    first_run = next(iter(training["runs"].values()), None)
    results = {
        "git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "what": ("M44: the Search-R1 style search policy of app/rl, trained once with a LoRA adapter and scored on fixture v1's held-out "
                 "questions against the same model untrained and against single-shot RAG"),
        "episodes_generated_at": {name: {"git_sha": arm["git_sha"], "working_tree_dirty": arm["working_tree_dirty"], "timestamp": arm["timestamp"]}
                                  for name, arm in arms.items()},
        **{k: v for k, v in header.items() if k not in ("split", "question_ids")},
        "licence": {"model": "apache-2.0", "source": "Hugging Face API (/api/models/<id>: tag license:apache-2.0, card licence apache-2.0), read 2026-11-22",
                    "skipped": "Qwen/Qwen2.5-3B-Instruct, the size PLAN.md Phase 10 names: licence `other` (qwen-research, non-commercial)"},
        "environment": {"index": ("the in-memory hybrid index over the question's corpus: BM25 and bge-small lists fused with RRF, no router, "
                                  "no reranker (the benchmark's hybrid@bge-small construction)"),
                        "max_searches": {name: arm["max_searches"] for name, arm in arms.items()},
                        "prompt_sha256": {name: arm["prompt_sha256"] for name, arm in arms.items()}},
        "reward": REWARD,
        "bar": BAR,
        "algorithm": (None if first_run is None else {
            "name": first_run["config"]["algo"],
            "label": ("a policy gradient with GRPO's group-normalised advantage: per question, the group's mean reward is the baseline and "
                      "its standard deviation the scale; no critic"),
            "not": ("one optimiser step per batch of episodes, so the PPO ratio GRPO clips is identically 1 and nothing is clipped; no KL "
                    "term; not the veRL stack PLAN.md names, a small loop on transformers and peft"),
            "loss": "minus the advantage times the mean log-probability of the tokens the policy wrote; the prompt and every <information> block are context only",
        }),
        **computed,
        "does_not_show": DOES_NOT_SHOW,
        "slurm_jobs": jobs,
        "sidecars": {"rollouts": ROLLOUTS, "training": TRAINING},
    }
    return scrub(results, root=root), rollouts, training


COMPUTED_KEYS = ("bootstrap", "split", "results", "verdicts", "training_runs", "training_corpora", "leakage", "stored_scores_differ_from_rescored")


def check(results_path: Path, root: Path = ROOT) -> list[str]:
    """What differs between the committed results file and its recomputation from the sidecars (empty: nothing)."""
    blob = json.loads(Path(results_path).read_text())
    rollouts = json.loads((root / blob["sidecars"]["rollouts"]).read_text())
    training = json.loads((root / blob["sidecars"]["training"]).read_text())
    gold_path = root / "bench" / "gold" / "questions.jsonl"
    problems = []
    if sha256_file(gold_path) != blob["gold_sha256"]:
        problems.append("gold_sha256: the gold file is not the one the results were scored against")
    computed = json.loads(json.dumps(compute(rollouts, training, load_gold(gold_path))))
    for key in COMPUTED_KEYS:
        if computed[key] != blob.get(key):
            problems.append(f"{key}: differs from the recomputation")
    if computed["stored_scores_differ_from_rescored"]:
        problems.append(f"stored episode scores differ from their rescoring: {computed['stored_scores_differ_from_rescored'][:5]}")
    if computed["training_corpora"]["runs_whose_recorded_corpora_differ"]:
        problems.append(f"a run's training corpora are not the ones its seeds rebuild: {computed['training_corpora']['runs_whose_recorded_corpora_differ'][:5]}")
    for run in computed["training_runs"]:
        arm = rollouts["arms"].get(run["name"])
        chosen = ((arm or {}).get("adapter") or {}).get("selected") or {}
        if arm is not None and chosen.get("step") != run["selected_step"]:
            problems.append(f"{run['name']}: the scored adapter is step {chosen.get('step')}, the tune half selects step {run['selected_step']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="recompute the committed results file from its sidecars; exit 1 on any difference")
    p.add_argument("--results", type=Path, default=ROOT / RESULTS)
    p.add_argument("--arms", type=Path, help="the directory app.rl.evaluate wrote the held-out arm files to")
    p.add_argument("--run", action="append", default=[], metavar="NAME=DIR", help="a training run directory, e.g. grpo/seed=0=runs/seed0")
    p.add_argument("--jobs", type=Path, help="a JSON list of the Slurm jobs behind the run (id, what, GPU, elapsed, state)")
    args = p.parse_args(argv)
    if args.check:
        problems = check(args.results)
        for line in problems:
            print(f"differs: {line}")
        if not problems:
            blob = json.loads(args.results.read_text())
            print(f"exact: OK ({len(blob['results'])} rows, {len(blob['verdicts'])} verdicts and {len(blob['training_runs'])} training runs of "
                  f"{args.results.name} recomputed from its sidecars)")
        return 1 if problems else 0
    if not args.arms:
        p.error("--arms is required to assemble (or pass --check)")
    runs = {}
    for spec in args.run:
        name, _, directory = spec.rpartition("=")
        runs[name] = Path(directory)
    jobs = json.loads(args.jobs.read_text()) if args.jobs else None
    results, rollouts, training = assemble(args.arms, runs, jobs)
    out = args.results
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / Path(ROLLOUTS).name).write_text(json.dumps(rollouts, indent=1) + "\n")
    (out.parent / Path(TRAINING).name).write_text(json.dumps(training, indent=1) + "\n")
    out.write_text(json.dumps(results, indent=2) + "\n")
    for row in results["results"]:
        print(row["config"], row["n"], {k: row["ci"][k] for k in ("em", "reward")}, [{d["reference"]: d["em"]} for d in row["deltas"]])
    for v in results["verdicts"]:
        if v["primary"]:
            print("primary verdict:", json.dumps(v))
    print(f"wrote {out} and its sidecars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
