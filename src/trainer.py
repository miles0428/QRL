"""Model-agnostic DQN training loop.  [IMPLEMENTED AT CHECKPOINT 3]

HARD ARCHITECTURAL RULE: this file MUST NOT import anything from qiskit. It only ever
sees an object satisfying `models/base.QFunction`. If the trainer can tell whether the
model is quantum, the design is wrong.

Planned DQN (see README "Trainer spec"):
  * replay buffer, uniform sampling, batch 16
  * hard target-network update every `target_update_interval` gradient steps
  * loss = MSE(Q(s,a), r + gamma*(1-done)*max_a' Q_target(s',a'))
  * epsilon-greedy, linear 1.0 -> 0.01 over the first `decay_env_steps` env steps
  * `terminated` vs `truncated` handled correctly (bootstrap through time-limit truncation)
  * writes results/{name}_{seed}.csv per episode: idx, env_steps, reward, ma100,
    epsilon, mean_loss, wall_clock_s, and current w / lam values.
"""
from __future__ import annotations

from .models.base import QFunction  # noqa: F401  (interface only; NEVER import qiskit here)


def train(*args, **kwargs):
    raise NotImplementedError("train() is implemented at Checkpoint 3.")
