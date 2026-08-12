"""Uniform experience replay buffer.

Stores **raw** observations (normalization happens inside the model's `forward`,
so the buffer is model-agnostic and the same buffer would work for any QFunction).
Sampling uses an injected numpy `Generator` so the sampled minibatch stream is a
deterministic function of the run seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    """A sampled minibatch of transitions (torch tensors, raw observations)."""
    states: torch.Tensor       # [B, obs_dim] float32
    actions: torch.Tensor      # [B] int64
    rewards: torch.Tensor      # [B] float32
    next_states: torch.Tensor  # [B, obs_dim] float32
    dones: torch.Tensor        # [B] float32 (1.0 if terminal)


class ReplayBuffer:
    """Fixed-capacity ring buffer with uniform sampling.

    Entries are ``(s, a, r, s', done)``. `done` is 1.0 only on true termination
    (pole fell / cart out of bounds), NOT on time-limit truncation -- the trainer is
    responsible for distinguishing terminated vs truncated when it calls `push`.
    """

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.rng = rng

        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    def push(self, s, a: int, r: float, s2, done: bool) -> None:
        i = self._pos
        self._states[i] = s
        self._actions[i] = a
        self._rewards[i] = r
        self._next_states[i] = s2
        self._dones[i] = 1.0 if done else 0.0
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, self._size, size=batch_size)
        return Batch(
            states=torch.from_numpy(self._states[idx]),
            actions=torch.from_numpy(self._actions[idx]),
            rewards=torch.from_numpy(self._rewards[idx]),
            next_states=torch.from_numpy(self._next_states[idx]),
            dones=torch.from_numpy(self._dones[idx]),
        )
