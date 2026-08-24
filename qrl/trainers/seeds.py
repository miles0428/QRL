"""Reproducibility: seed python/numpy/torch RNGs.

Environment seeding is separate -- callers pass seed= to env.reset() directly
(see trainer.py), since Gymnasium seeds its own internal RNG per-reset rather
than through a global seed function.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_global_seeds(seed: int) -> None:
    """Alias for set_seed; used by backends/common.py for backward compatibility."""
    set_seed(seed)


def make_rng(seed: int) -> Any:
    """Return a numpy default_rng seeded consistently with set_seed.

    This is used by experiment scripts that need an independent RNG for
    action/reward sampling that is not affected by torch/numpy global states.
    """
    return np.random.default_rng(seed)


if __name__ == "__main__":
    set_seed(0)
    a = [random.random(), float(np.random.rand()), torch.rand(1).item()]
    set_seed(0)
    b = [random.random(), float(np.random.rand()), torch.rand(1).item()]
    assert a == b, "seeding is not reproducible"
    print("seeds.py smoke test OK")
