"""Experience replay buffer. Stores raw (unnormalized) observations --
normalization happens inside each model's forward()."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.push([0.0, 0.0, 0.0, 0.0], i % 2, 1.0, [0.0, 0.0, 0.0, 0.0], False)
    states, actions, rewards, next_states, dones = buf.sample(8)
    assert states.shape == (8, 4)
    assert len(buf) == 20
    print("replay.py smoke test OK")
