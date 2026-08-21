"""Abstract Q-function interface and shared observation normalization.

trainer.py depends only on QFunction (duck-typed via this ABC) and never
imports qiskit -- vqc.py and mlp.py are interchangeable behind this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

# CartPole-v1 termination bounds (Gymnasium docs): |x| > 2.4 or |theta| > 0.2095 rad ends
# the episode. x_dot and theta_dot are formally unbounded.
X_BOUND = 2.4
THETA_BOUND = 0.2095


def normalize_observation(states: torch.Tensor) -> torch.Tensor:
    """Map raw CartPole observations (x, x_dot, theta, theta_dot) to bounded inputs.

    Position and angle are divided by their termination bounds (roughly [-1, 1]
    in-distribution, unbounded outside it since the episode should have ended).
    Both velocities are formally unbounded, so they go through arctan instead of
    a fixed division, which would be an arbitrary and easily-violated bound.

    Called inside each model's forward() -- the replay buffer stores raw
    observations, not normalized ones, so normalization stays reproducible
    regardless of how the buffer is sampled.
    """
    x = states[..., 0] / X_BOUND
    x_dot = torch.arctan(states[..., 1])
    theta = states[..., 2] / THETA_BOUND
    theta_dot = torch.arctan(states[..., 3])
    return torch.stack([x, x_dot, theta, theta_dot], dim=-1)


class QFunction(nn.Module, ABC):
    """Q-function approximator: forward(states) -> Tensor[B, n_actions]."""

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        ...


if __name__ == "__main__":
    raw = torch.tensor([[0.1, -1.5, 0.05, 2.0], [3.0, 0.0, -0.3, -5.0]], dtype=torch.float32)
    normed = normalize_observation(raw)
    print("raw:\n", raw)
    print("normalized:\n", normed)
    assert normed.shape == raw.shape
    print("base.py smoke test OK")
