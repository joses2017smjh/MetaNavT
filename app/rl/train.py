"""Train the search policy once (M44, `make search-policy-train`): a LoRA adapter, GRPO's group-relative advantage.

Each step takes QUESTIONS training questions (variant corpora only, app/rl/corpora.py), samples GROUP episodes of
each at temperature 1 (app/rl/policy.py), scores every episode with the outcome reward (app/rl/episode.py), and
turns each question's rewards into advantages with app.rl.grpo.grpo_advantages: the group's mean is the baseline
and its standard deviation the scale, with no critic. The loss of an episode is minus its advantage times the mean
log-probability of the tokens the policy wrote; the prompt and every `<information>` block are context only.

What this is and is not. There is one optimiser step per batch of episodes, so the policy that is updated is the
policy that sampled: the PPO ratio GRPO clips is identically 1, nothing is clipped, and there is no KL term. It is
a policy gradient with GRPO's group baseline, and the results file says so. `--algo ppo` is the declared fallback
for a collapsed run: the same episodes, a learned value baseline in place of the group mean, and PPO_EPOCHS clipped
passes over each batch.

Checkpoints: every --checkpoint-every steps the adapter, the optimiser and the sampler's state go to OUT/last, so
a job that hits its time limit resumes with --resume. Every --eval-every steps the adapter is scored greedily on
fixture v1's TUNE half (never the held-out half), and OUT/best keeps the checkpoint with the highest tune-half
mean reward (a tie keeps the earlier step; step 0 is scored for reference and is not eligible).

The training curve is OUT/curve.jsonl (one line per step) and OUT/tune.json (one entry per scored checkpoint).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Sequence

from app.eval.corpus import load_manifest
from app.eval.gold import load_gold
from app.eval.provenance import command_line, device_info, git_sha, git_state, hf_revision, run_provenance, sha256_file
from app.eval.rerank_arms import split_questions
from app.rl.corpora import parse_seeds, training_set, variant_root, write_variants
from app.rl.episode import MAX_NEW_TOKENS, MAX_SEARCHES, corpus_env, greedy_episodes, run_episodes, score_episode
from app.rl.grpo import grpo_advantages

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDER = "BAAI/bge-small-en-v1.5"
QUESTIONS, GROUP = 8, 8
LEARNING_RATE = 5e-5
WARMUP_STEPS = 10
MAX_GRAD_NORM = 1.0
# Padded tokens per backward pass. The pilot's first run (fp32, more than twice this budget) ran out of memory on an
# RTX 8000 in its sixth step; bf16 holds about half as much per token.
TOKEN_BUDGET = {"bf16": 8000, "fp32": 4000}
MIN_TOKEN_BUDGET = 500
PPO_EPOCHS, PPO_CLIP, VALUE_LR = 2, 0.2, 1e-3


def micro_batches(order: Sequence[int], lengths: Sequence[int], budget: int) -> list[list[int]]:
    """`order` cut into runs whose padded size (count x longest) stays within `budget` tokens; never an empty run."""
    out: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in order:
        longest_next = max(longest, lengths[i])
        if current and longest_next * (len(current) + 1) > budget:
            out.append(current)
            current, longest_next = [], lengths[i]
        current.append(i)
        longest = longest_next
    if current:
        out.append(current)
    return out


def question_schedule(questions: Sequence, seed: int) -> tuple[list[int], list[int]]:
    """Two seeded orders over the training questions: the per-run lookups (templates simple.*), and every other
    template."""
    lookups = [i for i, q in enumerate(questions) if q.template.startswith("simple.")]
    others = [i for i, q in enumerate(questions) if not q.template.startswith("simple.")]
    rng = random.Random(f"search-policy-order-{seed}")
    rng.shuffle(lookups)
    rng.shuffle(others)
    return lookups, others


def step_questions(questions: Sequence, schedule: tuple[list[int], list[int]], step: int, per_step: int, mix: str = "balanced") -> list:
    """The questions of one step, a pure function of (seed, step), so a resumed run asks what an uninterrupted one
    would have. balanced: half per-run lookups and half from the other templates, each list walked in its own order
    and started again when it ends. In the pilot the untrained policy already answered nearly every per-run lookup
    among the training questions and almost none of the others, and the lookups are most of the training set: at
    their natural share most groups of episodes would carry no reward spread. natural: the training set's own
    shares."""
    lookups, others = schedule
    if mix == "natural":
        merged = sorted(lookups + others, key=lambda i: (hash_position(i, len(questions)), i))
        return [questions[merged[(step * per_step + k) % len(merged)]] for k in range(per_step)]
    half = per_step // 2
    picks = [lookups[(step * half + k) % len(lookups)] for k in range(half)]
    picks += [others[(step * (per_step - half) + k) % len(others)] for k in range(per_step - half)]
    return [questions[i] for i in picks]


def hash_position(i: int, n: int) -> int:
    """A fixed pseudo-random position for question i among n (the natural mix's order)."""
    return random.Random(f"search-policy-natural-{n}-{i}").randrange(10**9)


def group_advantages(rewards: Sequence[float], group: int) -> list[float]:
    """GRPO's advantage: each consecutive run of `group` rewards (one question's episodes) normalised on its own."""
    out: list[float] = []
    for start in range(0, len(rewards), group):
        out += grpo_advantages(rewards[start:start + group])
    return out


def accumulate(policy, optimizer, order: Sequence[int], lengths: Sequence[int], token_budget: int, backward) -> tuple[float, int]:
    """Run `backward(micro)` (which returns a loss already back-propagated) over `order` in micro-batches and return
    (the summed loss, the token budget that fitted). The gradient of a step is the sum over its micro-batches, so a
    pass that runs out of GPU memory is started again from zero gradients with half the budget, never patched up."""
    torch = policy.torch
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            return sum(backward(micro) for micro in micro_batches(order, lengths, token_budget)), token_budget
        except torch.OutOfMemoryError:
            if token_budget <= MIN_TOKEN_BUDGET:
                raise
            token_budget = max(token_budget // 2, MIN_TOKEN_BUDGET)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"out of GPU memory in the backward pass: retrying this step with {token_budget} tokens per pass", flush=True)


def policy_gradient_step(policy, optimizer, states, advantages: Sequence[float], *, token_budget: int | None = None,
                         max_grad_norm: float = MAX_GRAD_NORM) -> dict:
    """One optimiser step on sum_i(-advantage_i * mean log-prob of episode i's policy tokens) / n. Episodes with a
    zero advantage (a group whose rewards are all equal) have no gradient and are not run through the model."""
    torch = policy.torch
    n = len(states)
    live = sorted((i for i, a in enumerate(advantages) if a != 0.0), key=lambda i: len(states[i].ids))
    gaps: dict[int, float] = {}

    def backward(micro: list[int]) -> float:
        logps = policy.policy_logprobs([states[i] for i in micro])
        loss = sum(-advantages[i] * lp.mean() for i, lp in zip(micro, logps)) / n
        loss.backward()
        # the training pass against the sampler: the same tokens under the same weights must score the same
        for i, lp in zip(micro, logps):
            gaps[i] = abs(float(lp.detach().mean()) - sum(states[i].logprobs) / len(states[i].logprobs))
        return float(loss.detach())

    loss_value, used = accumulate(policy, optimizer, live, [len(s.ids) for s in states], token_budget or TOKEN_BUDGET[policy.precision], backward)
    grad_norm = 0.0
    if live:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), max_grad_norm))
        optimizer.step()
    return {"loss": loss_value, "grad_norm": grad_norm, "n_episodes_with_gradient": len(live), "token_budget": used,
            "logprob_gap_mean": sum(gaps.values()) / len(gaps) if gaps else 0.0, "logprob_gap_max": max(gaps.values(), default=0.0)}


class ValueHead:
    """The PPO fallback's critic: a linear read-out of the frozen base model's last hidden state at the end of the
    prompt (before the policy has written anything), trained on the episode's reward. It never touches the adapter."""

    def __init__(self, policy, lr: float = VALUE_LR):
        torch = policy.torch
        self.policy = policy
        self.linear = torch.nn.Linear(policy.head.in_features, 1).to(policy.device).float()
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.optimizer = torch.optim.Adam(self.linear.parameters(), lr=lr)

    def features(self, states):
        torch = self.policy.torch
        prompts = [s.segments[0][0] for s in states]
        width = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), width), self.policy.pad_id, dtype=torch.long)
        attn = torch.zeros((len(prompts), width), dtype=torch.long)
        for r, prompt in enumerate(prompts):
            ids[r, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            attn[r, width - len(prompt):] = 1
        with torch.no_grad(), self.policy._autocast():
            hidden = self.policy.body(input_ids=ids.to(self.policy.device), attention_mask=attn.to(self.policy.device), use_cache=False).last_hidden_state
        return hidden[:, -1].float()

    def values_and_fit(self, states, rewards: Sequence[float]) -> tuple[list[float], float]:
        """(the value of each episode's prompt BEFORE this update, the value loss): the baseline, then one step on it."""
        torch = self.policy.torch
        feats = torch.cat([self.features(states[i:i + 32]) for i in range(0, len(states), 32)])
        target = torch.tensor(list(rewards), dtype=torch.float32, device=self.policy.device)
        predicted = self.linear(feats).squeeze(1)
        loss = torch.nn.functional.mse_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return predicted.detach().tolist(), float(loss.detach())


def ppo_step(policy, optimizer, states, advantages: Sequence[float], *, epochs: int = PPO_EPOCHS, clip: float = PPO_CLIP,
             token_budget: int | None = None, max_grad_norm: float = MAX_GRAD_NORM) -> dict:
    """PPO-clip over one batch of episodes: `epochs` passes, each one optimiser step on the clipped surrogate
    -min(ratio * A, clip(ratio, 1 - clip, 1 + clip) * A), ratio per policy token against the log-probability the
    token had when it was sampled, averaged over an episode's policy tokens and then over episodes."""
    torch = policy.torch
    n = len(states)
    order = sorted(range(n), key=lambda i: len(states[i].ids))
    lengths = [len(s.ids) for s in states]
    budget = token_budget or TOKEN_BUDGET[policy.precision]
    loss_value, grad_norm = 0.0, 0.0
    outside: dict[int, tuple[int, int]] = {}  # episode -> (policy tokens whose ratio is outside the clip range, policy tokens), last pass

    def backward(micro: list[int]) -> float:
        logps = policy.policy_logprobs([states[i] for i in micro])
        loss = 0.0
        for i, lp in zip(micro, logps):
            old = torch.tensor(states[i].logprobs, dtype=torch.float32, device=lp.device)
            ratio = torch.exp(lp - old)
            a = advantages[i]
            loss = loss - torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            outside[i] = (int(((ratio < 1 - clip) | (ratio > 1 + clip)).sum()), int(ratio.numel()))
        loss = loss / n
        loss.backward()
        return float(loss.detach())

    for _ in range(epochs):
        loss_value, budget = accumulate(policy, optimizer, order, lengths, budget, backward)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), max_grad_norm))
        optimizer.step()
    return {"loss": loss_value, "grad_norm": grad_norm, "n_episodes_with_gradient": n, "token_budget": budget,
            "clip_fraction": sum(a for a, _ in outside.values()) / max(sum(b for _, b in outside.values()), 1)}


def batch_normalised(values: Sequence[float], eps: float = 1e-8) -> list[float]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return [(v - mean) / (std + eps) for v in values]


def summarise(scores: Sequence[dict]) -> dict:
    n = len(scores)

    def mean(key: str) -> float:
        return round(sum(float(s[key]) for s in scores) / n, 6)

    stops: dict[str, int] = {}
    for s in scores:
        stops[s["stop"]] = stops.get(s["stop"], 0) + 1
    return {"n": n, "reward": mean("reward"), "em": mean("em"), "em_bounded": mean("em_bounded"), "ndcg@10": mean("ndcg@10"),
            "n_search": mean("n_search"), "uncited_or_empty": mean("uncited_or_empty"), "answered": mean("answered"),
            "answer_chars": mean("answer_chars"), "stop": dict(sorted(stops.items()))}


def lr_at(step: int, base: float, warmup: int = WARMUP_STEPS) -> float:
    return base * min(1.0, (step + 1) / max(warmup, 1))


def scoring_step(done: int, eval_every: int, steps: int) -> bool:
    """Whether the checkpoint after `done` steps is scored on the tune half: every eval_every steps, and the last."""
    return done > 0 and (done % eval_every == 0 or done == steps)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="the run directory: curve.jsonl, tune.json, last/, best/")
    p.add_argument("--corpora", type=Path, required=True, help="where the variant corpora are (written there when missing); outside the repository")
    p.add_argument("--corpus-seeds", default="1-32")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--embedder", default=EMBEDDER)
    p.add_argument("--algo", default="grpo", choices=["grpo", "ppo"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--questions", type=int, default=QUESTIONS)
    p.add_argument("--group", type=int, default=GROUP)
    p.add_argument("--mix", default="balanced", choices=["balanced", "natural"], help="a step's questions: half per-run lookups and half other templates, or the training set's own shares")
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--max-searches", type=int, default=MAX_SEARCHES)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--token-budget", type=int, default=0, help="padded tokens per backward pass (0: TOKEN_BUDGET for the precision)")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=50, help="0: never score the tune half (the pilot)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--time-limit-min", type=float, default=0.0, help="stop after a checkpoint once this many minutes have passed (0: no limit)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"], help="auto: bf16 where the GPU has it natively, else fp32")
    p.add_argument("--probe", type=int, default=0, help="score this many TRAINING questions greedily before the first step and after the last (the pilot's check)")
    p.add_argument("--probe-show", type=int, default=6)
    args = p.parse_args(argv)

    import numpy as np
    import torch

    from app.retrieval.embedders import SentenceTransformerEmbedder
    from app.rl.policy import LORA, HFPolicy

    started = time.time()
    git_started = git_state(ROOT)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.corpus_seeds)
    missing = [s for s in seeds if not (variant_root(args.corpora, s) / "bench" / "corpus" / "MANIFEST.json").exists()]
    if missing:
        write_variants(missing, args.corpora)
    questions, leakage = training_set(seeds)
    if leakage["n_kept_with_a_fixture_question_text"] or leakage["variant_run_ids_also_in_fixture_v1"]:
        raise RuntimeError("a training question duplicates a fixture question; refusing to train")
    schedule = question_schedule(questions, args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    embedder = SentenceTransformerEmbedder(args.embedder)
    envs = {s: corpus_env(variant_root(args.corpora, s) / "bench" / "corpus" / "files", embedder) for s in seeds}
    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    tune_ids, _ = split_questions(gold)
    tune = [gold[i] for i in tune_ids]
    fixture_env = corpus_env(ROOT / "bench" / "corpus" / "files", embedder) if args.eval_every else None

    last, best = out / "last", out / "best"
    resume = args.resume and (last / "state.pt").exists()
    policy = HFPolicy(args.model, device=args.device, adapter=last if resume else None, lora=None if resume else LORA, trainable=True, seed=args.seed,
                      precision=args.precision)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    critic = ValueHead(policy) if args.algo == "ppo" else None
    step0, tune_log = 0, []
    curve_path, tune_path = out / "curve.jsonl", out / "tune.json"
    if resume:
        state = torch.load(last / "state.pt", map_location=policy.device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        if critic is not None and state.get("critic") is not None:
            critic.linear.load_state_dict(state["critic"])
            critic.optimizer.load_state_dict(state["critic_optimizer"])
        step0 = int(state["step"])
        kept = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()] if curve_path.exists() else []
        curve_path.write_text("".join(json.dumps(r) + "\n" for r in kept if r["step"] < step0))
        tune_log = [e for e in (json.loads(tune_path.read_text()) if tune_path.exists() else []) if e["step"] <= step0]
        print(f"resumed at step {step0}", flush=True)
    else:
        curve_path.write_text("")
    n_trainable = sum(p.numel() for p in policy.trainable_parameters())
    token_budget = args.token_budget or TOKEN_BUDGET[policy.precision]
    import peft
    import transformers

    # One entry per job that worked on this run: a run that hit its time limit and was resumed has several.
    run_path = out / "run.json"
    sessions = (json.loads(run_path.read_text()).get("sessions") or []) if resume and run_path.exists() else []
    sessions.append({"from_step": step0, "git_sha": git_sha(ROOT), **run_provenance(root=ROOT, started=git_started), "command": command_line(),
                     "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), "device": device_info(args.device),
                     "libraries": {"transformers": transformers.__version__, "peft": peft.__version__}})
    corpora = [{"seed": s, "corpus_sha256": load_manifest(variant_root(args.corpora, s) / "bench" / "corpus" / "MANIFEST.json")["aggregate_sha256"],
                "gold_sha256": sha256_file(variant_root(args.corpora, s) / "bench" / "gold" / "questions.jsonl")} for s in seeds]
    run_path.write_text(json.dumps({
        "sessions": sessions, "model_revision": hf_revision(args.model), "embedder_revision": hf_revision(args.embedder), "corpora": corpora,
        "algo": args.algo, "model": args.model, "embedder": args.embedder, "seed": args.seed, "steps": args.steps, "questions_per_step": args.questions,
        "group": args.group, "lr": args.lr, "warmup_steps": WARMUP_STEPS, "max_grad_norm": MAX_GRAD_NORM, "optimizer": "AdamW(betas=(0.9, 0.99), weight_decay=0)",
        "max_searches": args.max_searches, "max_new_tokens": args.max_new_tokens, "token_budget": token_budget, "lora": LORA,
        "n_trainable_parameters": n_trainable, "corpus_seeds": seeds, "n_training_questions": len(questions), "sampling": "temperature 1, no top-k, no top-p",
        "question_mix": args.mix, "n_lookup_questions": len(schedule[0]), "n_other_questions": len(schedule[1]),
        "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every, "precision": policy.precision,
        "gpu": torch.cuda.get_device_name(0) if policy.device.type == "cuda" else None,
        **({"ppo_epochs": PPO_EPOCHS, "ppo_clip": PPO_CLIP, "value_lr": VALUE_LR} if args.algo == "ppo" else {}),
    }, indent=1) + "\n")
    print(f"{args.algo} seed {args.seed}: {len(questions)} training questions from {len(seeds)} corpora, {n_trainable} trainable parameters, "
          f"{args.questions} x {args.group} episodes per step, precision {policy.precision}", flush=True)

    def encode(queries: list[str]):
        return embedder.encode_queries(queries)

    def score_tune(step: int) -> None:
        t0 = time.time()
        rows = greedy_episodes(policy, tune, fixture_env, max_searches=args.max_searches, encode_queries=encode)
        scores = [rows[q.id]["score"] for q in tune]
        entry = {"step": step, **summarise(scores), "seconds": round(time.time() - t0, 1),
                 "per_question": {q.id: {"reward": rows[q.id]["score"]["reward"], "em": rows[q.id]["score"]["em"]} for q in tune}}
        tune_log[:] = [e for e in tune_log if e["step"] != step] + [entry]
        eligible = [e for e in tune_log if e["step"] > 0]
        if step > 0 and max(eligible, key=lambda e: (e["reward"], -e["step"]))["step"] == step:
            policy.save_adapter(best)
            (best / "step.json").write_text(json.dumps({"step": step, "tune_reward": entry["reward"], "tune_em": entry["em"]}) + "\n")
        tune_path.write_text(json.dumps(tune_log, indent=1) + "\n")
        print(f"tune step {step}: reward {entry['reward']:.4f} em {entry['em']:.4f} searches {entry['n_search']:.2f} "
              f"uncited_or_empty {entry['uncited_or_empty']:.3f} ({entry['seconds']} s)", flush=True)

    def checkpoint(step: int) -> None:
        policy.save_adapter(last)
        torch.save({"step": step, "optimizer": optimizer.state_dict(),
                    "critic": critic.linear.state_dict() if critic is not None else None,
                    "critic_optimizer": critic.optimizer.state_dict() if critic is not None else None}, last / "state.pt")

    def probe(label: str) -> None:
        """Greedy episodes on TRAINING questions (never the fixture): how the policy follows the protocol, before
        the first step and after the last. The questions are the last --probe / 2 of each of the seed's two orders,
        which a run too short to reach the end of a list never trains on."""
        sample = [questions[i] for i in schedule[0][-(args.probe // 2):] + schedule[1][-(args.probe - args.probe // 2):]]
        eps, _ = run_episodes(policy, [q.question for q in sample], [envs[q.corpus] for q in sample], sample=False,
                              max_searches=args.max_searches, max_new_tokens=args.max_new_tokens, encode_queries=encode)
        probe_scores = [score_episode(ep, q.answer, q.relevant_paths) for ep, q in zip(eps, sample)]
        print(f"probe {label} (greedy, {len(sample)} training questions):", json.dumps(summarise(probe_scores)), flush=True)
        for ep, q, sc in list(zip(eps, sample, probe_scores))[:args.probe_show]:
            print(f"--- {q.key} [{q.template}] gold {q.answer!r} reward {sc['reward']} em {sc['em']} cited {sc['cited']} stop {sc['stop']}\n"
                  f"Q: {q.question}\n{ep.raw[:1500]}", flush=True)
        (out / f"probe_{label}.json").write_text(json.dumps({"summary": summarise(probe_scores), "episodes": [
            {"key": q.key, "template": q.template, "gold": q.answer, "score": sc, "episode": ep.to_dict()} for ep, q, sc in zip(eps, sample, probe_scores)]}, indent=1) + "\n")

    if args.probe and step0 == 0:
        probe("before")
    if args.eval_every and step0 == 0:
        score_tune(0)
    if args.eval_every and step0 > 0 and scoring_step(step0, args.eval_every, args.steps) and all(e["step"] != step0 for e in tune_log):
        score_tune(step0)  # the job before this one saved the checkpoint and stopped before its tune-half score was written
    for step in range(step0, args.steps):
        t0 = time.time()
        # The sampler is re-seeded from (seed, step), so a resumed run draws what an uninterrupted one would have.
        policy.generator.manual_seed(args.seed * 1_000_003 + step)
        batch = step_questions(questions, schedule, step, args.questions, args.mix)
        expanded = [q for q in batch for _ in range(args.group)]
        episodes, states = run_episodes(policy, [q.question for q in expanded], [envs[q.corpus] for q in expanded], sample=True,
                                        max_searches=args.max_searches, max_new_tokens=args.max_new_tokens, encode_queries=encode)
        scores = [score_episode(ep, q.answer, q.relevant_paths) for ep, q in zip(episodes, expanded)]
        rewards = [s["reward"] for s in scores]
        t1 = time.time()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args.lr)
        if args.algo == "grpo":
            advantages = group_advantages(rewards, args.group)
            stats = policy_gradient_step(policy, optimizer, states, advantages, token_budget=token_budget)
            token_budget = stats["token_budget"]  # a budget that ran out of memory once stays halved
        else:
            values, value_loss = critic.values_and_fit(states, rewards)
            advantages = batch_normalised([r - v for r, v in zip(rewards, values)])
            stats = {**ppo_step(policy, optimizer, states, advantages, token_budget=token_budget), "value_loss": value_loss}
        n_tokens = sum(s.n_policy_tokens for s in states)
        record = {"step": step, **summarise(scores),
                  "groups_with_a_reward_spread": sum(1 for g in range(0, len(rewards), args.group) if len(set(rewards[g:g + args.group])) > 1),
                  "nll_sampled": round(-sum(sum(s.logprobs) for s in states) / n_tokens, 6),
                  "policy_tokens": round(n_tokens / len(states), 2), "context_tokens": round(sum(len(s.ids) for s in states) / len(states), 1),
                  **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()},
                  "lr": lr_at(step, args.lr), "rollout_s": round(t1 - t0, 2), "update_s": round(time.time() - t1, 2)}
        if not math.isfinite(record["loss"]):
            print(f"step {step}: the loss is not finite; stopping (a collapse by the declared rule)", flush=True)
            with curve_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            return 3
        with curve_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"step {step}: reward {record['reward']:+.4f} em {record['em']:.3f} searches {record['n_search']:.2f} uncited_or_empty "
              f"{record['uncited_or_empty']:.3f} answered {record['answered']:.3f} nll {record['nll_sampled']:.4f} loss {record['loss']:+.5f} "
              f"grad {record['grad_norm']:.3f} gap {record.get('logprob_gap_max', 0.0):.4f} tokens {record['policy_tokens']:.0f}/{record['context_tokens']:.0f} "
              f"{record['rollout_s']:.1f}+{record['update_s']:.1f} s", flush=True)
        done = step + 1
        scored = bool(args.eval_every) and scoring_step(done, args.eval_every, args.steps)
        if done % args.checkpoint_every == 0 or done == args.steps or scored:
            checkpoint(done)
        if scored:
            score_tune(done)
        if args.time_limit_min and done < args.steps and done % args.checkpoint_every == 0 and (time.time() - started) / 60 > args.time_limit_min:
            print(f"time limit: stopped after step {done} of {args.steps}; continue with --resume", flush=True)
            return 0
    if args.probe:
        probe("after")
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"finished {args.steps} steps in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise SystemExit("training needs a GPU")
    raise SystemExit(main())
