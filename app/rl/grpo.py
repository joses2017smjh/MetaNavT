"""GRPO advantages + retrieved-token-masked loss. CPU dummy step for CI.

Shao et al. 2024 / DeepSeek-R1: group-relative advantages, no critic.
If GRPO collapses (Lazy Likelihood Displacement), fall back to PPO — not wired here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.rl.reward import search_r1_reward
from app.rl.search_r1 import SearchR1Env, Trajectory, token_train_mask


def grpo_advantages(rewards: Sequence[float], eps: float = 1e-8) -> list[float]:
    r = np.asarray(list(rewards), dtype=np.float64)
    if r.size == 0:
        return []
    if r.size == 1:
        return [0.0]
    std = float(r.std())
    return [float(x) for x in (r - r.mean()) / (std + eps)]


def masked_nll(logprobs: Sequence[float], train_mask: Sequence[bool]) -> float:
    """Mean negative log-prob over generated tokens only (retrieved spans excluded)."""
    vals = [float(lp) for lp, m in zip(logprobs, train_mask) if m]
    if not vals:
        return 0.0
    return float(-sum(vals) / len(vals))


def dummy_logprobs(n: int, rng: np.random.RandomState | None = None) -> list[float]:
    rng = rng or np.random.RandomState(0)
    return [float(x) for x in rng.normal(-1.2, 0.15, size=max(n, 0))]


def dummy_step(
    trajectories: Sequence[Trajectory],
    rewards: Sequence[float],
    *,
    lr: float = 0.01,
    rng: np.random.RandomState | None = None,
) -> dict:
    """One CPU GRPO step on a toy parameter vector. Not a trained policy."""
    rng = rng or np.random.RandomState(0)
    adv = grpo_advantages(rewards)
    theta = rng.normal(0, 0.01, size=8)
    losses = []
    for traj, a in zip(trajectories, adv):
        mask = traj.train_mask or token_train_mask(traj.raw)
        n = max(len(mask), 1)
        lp = dummy_logprobs(n, rng)
        nll = masked_nll(lp[: len(mask)], mask)
        losses.append(a * nll)
    grad = float(np.mean(losses)) if losses else 0.0
    theta = theta - lr * grad
    return {
        "loss": round(float(np.mean(losses)) if losses else 0.0, 6),
        "advantages": [round(a, 6) for a in adv],
        "theta_norm": round(float(np.linalg.norm(theta)), 6),
        "n": len(trajectories),
        "backend": "numpy-dummy",
    }


def dummy_train_on_index(index, question: str, gold: str, gold_paths: Sequence[str]) -> dict:
    env = SearchR1Env(index)
    group: list[Trajectory] = []
    rewards: list[float] = []
    for i in range(4):
        if i == 0:
            traj = env.rollout(question)
        elif i == 1:
            traj = env.rollout(question, policy=lambda q: f"<search>{q}</search>\n<answer></answer>")
        else:
            traj = env._heuristic_rollout(question)
        group.append(traj)
        rewards.append(search_r1_reward(traj, gold=gold, gold_paths=gold_paths)["reward"])
    step = dummy_step(group, rewards)
    step["rewards"] = rewards
    return step
