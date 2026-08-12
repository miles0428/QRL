"""Global seeding for byte-identical reproducibility.

The reproducibility contract (see README): same seed -> byte-identical results CSV.
That requires seeding python `random`, numpy, and torch, using a per-run numpy
`Generator` for replay sampling, AND passing `seed=` to `env.reset()` on the first
reset. This module handles the first three; the trainer owns the env seeding and the
sampling Generator (built via `make_rng`).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seeds(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python / numpy / torch global RNGs.

    Args:
        seed: the run seed.
        deterministic_torch: if True, request deterministic CPU algorithms. On a
            4-qubit CPU-only workload this has negligible cost and removes a source
            of run-to-run drift.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_torch:
        # CPU-only workload; these keep the torch side reproducible.
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def make_rng(seed: int) -> np.random.Generator:
    """Return a dedicated numpy Generator for replay sampling / epsilon draws.

    Kept separate from the global numpy state so that adding/removing an unrelated
    numpy call elsewhere does not shift the sampling stream.
    """
    return np.random.default_rng(seed)
