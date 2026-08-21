"""Reproducibility: seed python/numpy/torch RNGs.

Environment seeding is separate -- callers pass seed= to env.reset() directly
(see trainer.py), since Gymnasium seeds its own internal RNG per-reset rather
than through a global seed function.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    set_seed(0)
    a = [random.random(), float(np.random.rand()), torch.rand(1).item()]
    set_seed(0)
    b = [random.random(), float(np.random.rand()), torch.rand(1).item()]
    assert a == b, "seeding is not reproducible"
    print("seeds.py smoke test OK")
