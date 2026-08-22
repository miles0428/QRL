"""QFunction contract and observation normalization, for the spin-control game.

This is the QDQN code from the qdqn-cartpole branch (src/models/vqc.py and
src/models/torch_statevector.py, both copied verbatim) rewired onto
QuantumSpinCartPole. Only two things differ from that branch:

  - normalize_observation, which was CartPole-specific and hardcoded four
    dimensions (x / X_BOUND, arctan(x_dot), theta / THETA_BOUND,
    arctan(theta_dot)). The spin observation has a different length and
    different bounds, so it needs its own map.
  - imports, which pointed at src.models.

The circuit, the backends, the trainable lam / w scalings and the three failure
modes documented in vqc.py are untouched.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def normalize_observation(states: torch.Tensor) -> torch.Tensor:
    """
    Map a spin observation to bounded RY encoding angles.

    Every component is already O(1) -- sx, sy in [-1, 1]; their deltas in
    [-2, 2]; one-hot previous-action flags in {0, 1} -- so unlike CartPole
    nothing here is formally unbounded. arctan is still used, uniformly:
    it is monotone, needs no per-mode bookkeeping as the observation layout
    changes between obs modes, and cannot push an angle outside (-pi/2, pi/2)
    if an unexpectedly large delta ever appears.

    Applied inside forward(), so the replay buffer keeps raw observations and
    normalization stays reproducible however the buffer is sampled -- same
    contract as the CartPole version.
    """
    return torch.arctan(states)


class QFunction(nn.Module, ABC):
    """Q-function approximator: forward(states) -> Tensor[B, n_actions]."""

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        ...


class PolicyFunction(nn.Module, ABC):
    """Policy approximator: forward(states) -> Tensor[B, n_actions] (logits).

    The loss (e.g. softmax-cross-entropy) is applied outside the model.
    Subclasses must implement forward(); see VQCPolicy and MLPPolicy.
    """

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        ...


class ValueFunction(nn.Module, ABC):
    """Value-function approximator: forward(states) -> Tensor[B] (not [B,1]).

    Returning shape [B] rather than [B,1] is a deliberate contract: the [B,1]
    shape silently broadcasts against advantage vectors in A2C/GAE and produces
    a [B,B] loss matrix. Every value model in this project returns [B].
    """

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        ...
