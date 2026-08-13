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


@dataclass
class RBatch:
    """A sampled minibatch for the Rainbow path (n-step returns + per-sample discount)."""
    states: torch.Tensor        # [B, obs_dim] float32
    actions: torch.Tensor       # [B] int64
    returns: torch.Tensor       # [B] float32  (accumulated n-step reward R_t^(n))
    boot_states: torch.Tensor   # [B, obs_dim] float32  (state s_{t+n} to bootstrap from)
    dones: torch.Tensor         # [B] float32  (1.0 if episode ended within the n-window)
    n_gammas: torch.Tensor      # [B] float32  (gamma**k for the bootstrap, k = actual steps)
    indices: np.ndarray         # buffer indices (for priority updates)
    weights: torch.Tensor       # [B] float32  (importance-sampling weights; 1.0 if PER off)


class RainbowReplayBuffer:
    """Replay buffer for the Rainbow value path: n-step transitions + Prioritized ER.

    Stores fully-formed n-step transitions ``(s_t, a_t, R_t^(n), s_{t+n}, done, gamma**k)``
    built by the trainer's n-step accumulator, so the bootstrap target is simply
    ``R + gamma**k * (1-done) * Q(s_{t+n})``. PER (Schaul et al. 2016) samples in
    proportion to ``priority**alpha`` (priority = |TD error| + eps) and corrects the bias
    with importance-sampling weights annealed by ``beta``. With ``per=False`` sampling is
    uniform and weights are 1.0 (so multi-step-only ablations reuse this same buffer).
    """

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator,
                 alpha: float = 0.5, eps: float = 1e-3):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.rng = rng
        self.alpha = float(alpha)
        self.eps = float(eps)

        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._returns = np.zeros((capacity,), dtype=np.float32)
        self._boot = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)
        self._n_gammas = np.zeros((capacity,), dtype=np.float32)
        self._priorities = np.zeros((capacity,), dtype=np.float64)

        self._size = 0
        self._pos = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    def push(self, s, a: int, R: float, s_boot, done: bool, n_gamma: float) -> None:
        i = self._pos
        self._states[i] = s
        self._actions[i] = a
        self._returns[i] = R
        self._boot[i] = s_boot
        self._dones[i] = 1.0 if done else 0.0
        self._n_gammas[i] = n_gamma
        self._priorities[i] = self._max_priority     # new transitions get max priority
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, per: bool = True, beta: float = 0.4) -> RBatch:
        n = self._size
        if per:
            pr = self._priorities[:n] ** self.alpha
            probs = pr / pr.sum()
            idx = self.rng.choice(n, size=batch_size, p=probs)
            w = (n * probs[idx]) ** (-beta)
            w = w / w.max()                          # normalize by max for stability
        else:
            idx = self.rng.integers(0, n, size=batch_size)
            w = np.ones(batch_size, dtype=np.float64)
        return RBatch(
            states=torch.from_numpy(self._states[idx]),
            actions=torch.from_numpy(self._actions[idx]),
            returns=torch.from_numpy(self._returns[idx]),
            boot_states=torch.from_numpy(self._boot[idx]),
            dones=torch.from_numpy(self._dones[idx]),
            n_gammas=torch.from_numpy(self._n_gammas[idx]),
            indices=idx,
            weights=torch.from_numpy(w.astype(np.float32)),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        p = np.abs(td_errors) + self.eps
        self._priorities[indices] = p
        self._max_priority = max(self._max_priority, float(p.max()))
