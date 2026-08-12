"""Classical MLP baseline, parameter-count matched to the VQC.  [IMPLEMENTED AT CHECKPOINT 3]

Fair comparison is the whole point: a 2x128 MLP (~17k params) vs a ~46-param VQC is
not evidence of quantum parameter-efficiency. This net uses a hidden width chosen so
its trainable param count is within +/-20% of the VQC's (~46). The exact count is
computed and asserted at construction and printed to the run log. Same normalization
pipeline as the VQC so the comparison isolates the approximator.
"""
from __future__ import annotations

from .base import QFunction


class MLPQFunction(QFunction):
    def __init__(self, model_cfg: dict, norm_cfg: dict):
        raise NotImplementedError("MLPQFunction is implemented at Checkpoint 3.")
