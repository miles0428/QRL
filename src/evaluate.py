"""Greedy evaluation + the single solve criterion + finite-shot hook.

The ONE solve definition used everywhere: mean reward >= 475 over 100 consecutive
episodes (see `SOLVE_REWARD` / `SOLVE_WINDOW`). `greedy_eval` runs the trained agent
with epsilon = 0 and reports the average episode reward (the Fundamental metric).

Model-agnostic: imports only the QFunction interface via duck typing (forward), never
qiskit. Works identically for the VQC and the MLP baseline.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

SOLVE_REWARD = 475.0
SOLVE_WINDOW = 100


def greedy_action(model, obs) -> int:
    with torch.no_grad():
        q = model(torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0))
    return int(torch.argmax(q, dim=1).item())


def greedy_eval(model, n_episodes: int = 100, seed: int = 0,
                env_id: str = "CartPole-v1") -> dict:
    """Run `n_episodes` greedy (epsilon=0) rollouts; return reward statistics.

    Reproducible: the env is seeded once with `seed`, subsequent resets use its
    seeded RNG. Returns mean/std/min/max, per-episode rewards, and whether the mean
    meets the solve threshold.
    """
    env = gym.make(env_id)
    was_training = model.training
    model.eval()
    rewards: list[float] = []
    obs, _ = env.reset(seed=seed)
    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(greedy_action(model, obs))
            total += float(r)
            done = terminated or truncated
        rewards.append(total)
    env.close()
    if was_training:
        model.train()

    arr = np.asarray(rewards, dtype=np.float64)
    return {
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "min_reward": float(arr.min()),
        "max_reward": float(arr.max()),
        "n_episodes": n_episodes,
        "solved": bool(arr.mean() >= SOLVE_REWARD),
        "rewards": rewards,
    }


def evaluate_finite_shot(model, shots: int):
    """Hook: evaluate `model` under finite sampling with a qiskit-aer estimator.

    Deliberately unimplemented (per spec) -- only the signature exists so the later
    finite-shot / noise robustness sweep plugs in without touching the trainer. Aer is
    used ONLY here, never during training. The VQC exposes its circuit/observables so a
    shot-based AerEstimator (or noise model) can be swapped in for the forward pass.
    """
    raise NotImplementedError(
        "Finite-shot evaluation is a stub hook (see spec). Implement in the shot-sweep phase "
        "using qiskit_aer.primitives.EstimatorV2 with the given `shots` (and optionally a noise model)."
    )
