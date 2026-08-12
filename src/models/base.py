"""Model-agnostic Q-function interface.

`trainer.py` only ever sees a `QFunction`. It never imports qiskit, never touches
`lam`/`w`/`vqc`, and cannot tell whether the approximator is a variational quantum
circuit or a classical MLP. Swapping ansatz == writing a new `QFunction` subclass +
one line in the factory (`src/models/__init__.py`). If the trainer had to change,
the abstraction leaked.

The optimizer parameter grouping (three learning rates for the VQC, one for the MLP)
lives here as `param_groups()` for the same reason: learning-rate structure is a
property of the model, not the training loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class QFunction(ABC, nn.Module):
    """Abstract Q-function approximator: states -> action-values.

    Subclasses must:
      * accept **raw** (un-normalized) observations in `forward` and do any
        normalization internally (the replay buffer stores raw obs);
      * return a tensor of shape ``[batch, n_actions]``;
      * expose `param_groups()` returning a list of dicts suitable for
        ``torch.optim.Adam`` (each with its own ``lr``).
    """

    #: number of discrete actions (CartPole -> 2). Set by subclasses.
    n_actions: int
    #: observation dimensionality (CartPole -> 4). Set by subclasses.
    obs_dim: int

    @abstractmethod
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Map a batch of raw observations ``[B, obs_dim]`` to Q-values ``[B, n_actions]``."""
        raise NotImplementedError

    @abstractmethod
    def param_groups(self) -> list[dict]:
        """Return Adam parameter groups, e.g.::

            [{"params": [...], "lr": 1e-3}, ...]

        The trainer builds the optimizer from exactly this, so per-group learning
        rates (the QDQN's critical 100x-larger output-scaling lr) are owned by the model.
        """
        raise NotImplementedError

    def num_trainable_params(self) -> int:
        """Total number of trainable scalar parameters (used for baseline matching)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def loggable_scalars(self) -> dict[str, float]:
        """Model-specific scalars to append as CSV columns each episode.

        Default: none. The VQC overrides this to expose ``w0, w1, lam0..lam3`` so the
        trainer can log the output/input scalings (the direct evidence for Failure
        Mode 1) WITHOUT the trainer knowing those parameters exist. The MLP leaves it
        empty, so its CSV simply has no such columns.
        """
        return {}
