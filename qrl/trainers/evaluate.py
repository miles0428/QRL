"""Greedy-policy rollout evaluation + solve criterion.

Model-agnostic, same as trainer.py -- only touches `model` via forward().
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

SOLVE_THRESHOLD = 475.0
SOLVE_WINDOW = 100


def greedy_action(model: nn.Module, state: np.ndarray) -> int:
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(model(state_t), dim=1).item())


def run_greedy_rollouts(
    model: nn.Module,
    n_episodes: int = SOLVE_WINDOW,
    env_id: str = "CartPole-v1",
    seed: int = 0,
    max_steps_per_episode: int = 500,
) -> list[float]:
    """Run n_episodes fully-greedy (epsilon=0) episodes, return per-episode rewards."""
    env = gym.make(env_id)
    rewards = []
    try:
        for episode in range(n_episodes):
            obs, _info = env.reset(seed=seed + episode)
            total = 0.0
            for _ in range(max_steps_per_episode):
                action = greedy_action(model, obs)
                obs, reward, terminated, truncated, _info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
    finally:
        env.close()
    return rewards


def evaluate(
    model: nn.Module,
    n_episodes: int = SOLVE_WINDOW,
    env_id: str = "CartPole-v1",
    seed: int = 0,
    solve_threshold: float = SOLVE_THRESHOLD,
) -> dict:
    """Final greedy evaluation: n_episodes rollouts, mean reward vs. solve_threshold."""
    rewards = run_greedy_rollouts(model, n_episodes=n_episodes, env_id=env_id, seed=seed)
    mean_reward = float(np.mean(rewards))
    return {
        "n_episodes": n_episodes,
        "mean_reward": mean_reward,
        "std_reward": float(np.std(rewards)),
        "solved": mean_reward >= solve_threshold,
        "solve_threshold": solve_threshold,
        "rewards": rewards,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from qrl.models.mlp import MLPQFunction

    model = MLPQFunction()
    result = evaluate(model, n_episodes=5, seed=0)
    print(f"mean_reward={result['mean_reward']:.1f} solved={result['solved']}")
    assert "rewards" in result and len(result["rewards"]) == 5
    print("evaluate.py smoke test OK")
