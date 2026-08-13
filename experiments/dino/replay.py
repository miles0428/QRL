"""Image replay buffer for the dino agent.

The backbone ``src.replay.ReplayBuffer`` stores flat ``[obs_dim]`` vectors; image
observations are ``[4,84,84]``, so this is the one piece that has to differ. It stores frames
as **uint8** (a 10k-capacity buffer is ~0.56 GB this way vs ~2.3 GB as float32) and returns
the SAME ``src.replay.Batch`` dataclass the reused ``dqn_update`` consumes — states as a uint8
tensor, which the model's ``forward`` divides by 255. Everything else (the DQN math, target
net, epsilon) is the backbone's, unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from src.replay import Batch, RBatch   # reuse the backbone's minibatch dataclasses


class ImageReplayBuffer:
    """Fixed-capacity uniform replay for ``[C,H,W]`` uint8 observations.

    Entries are ``(s, a, r, s', done)`` with ``done`` True ONLY on real termination (crash),
    never on time-limit truncation — same contract as the backbone buffer.
    """

    def __init__(self, capacity: int, obs_shape, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.rng = rng

        self._states = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self._next_states = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    def push(self, s, a: int, r: float, s2, done: bool) -> None:
        i = self._pos
        self._states[i] = s
        self._next_states[i] = s2
        self._actions[i] = a
        self._rewards[i] = r
        self._dones[i] = 1.0 if done else 0.0
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, self._size, size=batch_size)
        return Batch(
            states=torch.from_numpy(self._states[idx]),        # uint8 [B,4,84,84]; model /255 in forward
            actions=torch.from_numpy(self._actions[idx]),
            rewards=torch.from_numpy(self._rewards[idx]),
            next_states=torch.from_numpy(self._next_states[idx]),
            dones=torch.from_numpy(self._dones[idx]),
        )


class ImageRainbowReplayBuffer:
    """Image version of the backbone ``RainbowReplayBuffer``: n-step transitions + Prioritized ER.

    Stores fully-formed n-step transitions ``(s_t, a_t, R_t^(n), s_{t+n}, done, gamma**k)`` (built by
    the backbone ``NStepAccumulator``) with uint8 image states, and returns the SAME ``src.replay.RBatch``
    the reused ``rainbow_update`` consumes. Prioritized sampling / IS-weights are identical to the
    backbone's flat buffer — only the state storage differs. The brief calls for exactly this
    (multi-step + PER) for the harder credit-assignment case.
    """

    def __init__(self, capacity: int, obs_shape, rng: np.random.Generator,
                 alpha: float = 0.5, eps: float = 1e-3):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.rng = rng
        self.alpha = float(alpha)
        self.eps = float(eps)

        self._states = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self._boot = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._returns = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._n_gammas = np.zeros((self.capacity,), dtype=np.float32)
        self._priorities = np.zeros((self.capacity,), dtype=np.float64)

        self._size = 0
        self._pos = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    def push(self, s, a: int, R: float, s_boot, done: bool, n_gamma: float) -> None:
        i = self._pos
        self._states[i] = s
        self._boot[i] = s_boot
        self._actions[i] = a
        self._returns[i] = R
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
            w = w / w.max()
        else:
            idx = self.rng.integers(0, n, size=batch_size)
            w = np.ones(batch_size, dtype=np.float64)
        return RBatch(
            states=torch.from_numpy(self._states[idx]),        # uint8 [B,4,84,84]
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
