"""Classical MLP models for QRL: Q-function, policy, and value heads.

This module was deleted in commit 0ec94fb (QRL refactor) and is restored here
to keep the ``from qrl.models.mlp import`` imports in ``__init__.py`` working.
The three classes are simple classical counterparts to the VQC variants and use
the same ``QFunction`` / ``PolicyFunction`` / ``ValueFunction`` contracts from
``qrl.models.base``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qrl.models.base import PolicyFunction, QFunction, ValueFunction


class MLPQFunction(QFunction):
    """Classical MLP Q-function: states -> Q-values per action.

    Args:
        n_inputs:  Dimension of the observation space.
        n_actions: Number of discrete actions.
        hidden:    Width of the hidden layer.
    """

    def __init__(self, n_inputs: int, n_actions: int, hidden: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action: shape [B, n_actions]."""
        return self.net(states)


class MLPPolicy(PolicyFunction):
    """Classical MLP policy: states -> action logits (softmax applied by the loss).

    Args:
        n_inputs:  Dimension of the observation space.
        n_actions: Number of discrete actions.
        hidden:    Width of the hidden layer.
    """

    def __init__(self, n_inputs: int, n_actions: int, hidden: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return logits: shape [B, n_actions]."""
        return self.net(states)


class MLPValue(ValueFunction):
    """Classical MLP value function: states -> V(s).

    Args:
        n_inputs: Dimension of the observation space.
        hidden:   Width of the hidden layer.
    """

    def __init__(self, n_inputs: int, hidden: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return V(s): shape [B], not [B, 1] -- see ValueFunction contract."""
        return self.net(states).reshape(-1)
