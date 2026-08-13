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


class PolicyFunction(nn.Module, ABC):
    """Policy network: forward(states) -> Tensor[B, n_actions] of LOGITS.

    Same tensor shape as QFunction, deliberately different meaning: these go
    through a softmax to give pi(a|s), they are not action values. Two
    consequences worth stating, because both are load-bearing elsewhere.

    Logits, not probabilities. The softmax lives in the loss
    (torch.distributions.Categorical(logits=...)), which is numerically stabler
    than taking log() of a softmax we computed ourselves, and it keeps the
    forward() contract identical to QFunction's.

    argmax is still the greedy policy. softmax is monotone, so
    argmax(logits) == argmax(pi). That is why src/evaluate.py works verbatim on
    a PolicyFunction -- greedy evaluation needs no policy-specific branch. It
    holds only while the inverse temperature is positive; see
    src/models/vqc_policy.py::VQCPolicy for how that is guaranteed.
    """

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        ...


class ValueFunction(nn.Module, ABC):
    """State-value critic: forward(states) -> Tensor[B], one scalar per state.

    Shape differs from QFunction/PolicyFunction on purpose: a critic scores the
    state, not the actions in it, so there is no action axis to index and
    returning [B, 1] would invite a silent broadcast against a [B] advantage
    vector. Implementations squeeze it themselves.

    The output has to reach the same magnitude a Q-function does -- V* for
    CartPole at gamma=0.99 is ~99, since V*(s) = Q*(s, a*) under the optimal
    policy. That is why src/models/vqc_value.py carries the same output-scaling
    weight `w` as VQCQFunction rather than the bare inverse temperature the
    policy head uses.
    """

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
