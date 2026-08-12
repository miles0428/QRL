"""VQC Q-function (quantum approximator).  [IMPLEMENTED AT CHECKPOINT 2]

Planned architecture (see README "Model spec"):

    s --arctan/normalize--> lam (o) s --> TorchConnector(EstimatorQNN) --> [<Z0>,<Z1>] --> (o) w --> Q(s,.)
                            ^ nn.Parameter                                                  ^ nn.Parameter

Key correctness constraints this module must satisfy (both are silent-failure traps):
  * BOTH scalings (`lam` input, `w` output) live HERE in torch, OUTSIDE the circuit,
    because lam_i * x_i is a product of two parameters and would break param-shift.
  * `w` (trainable output scaling) is mandatory: <Z> in [-1,1] cannot represent
    Q ~ 100 at gamma=0.99. Without it, training flatlines at ~10 reward.

Gradient: adjoint `ReverseEstimatorGradient` (config `gradient: reverse`) with a
shot-free statevector estimator for training; `ParamShiftEstimatorGradient`
(`gradient: paramshift`) kept available for the later finite-shot / hardware path.

Reference: Skolik, Jerbi, Dunjko (2022), Quantum 6, 720, arXiv:2103.15084.
"""
from __future__ import annotations

from .base import QFunction


class VQCQFunction(QFunction):
    def __init__(self, model_cfg: dict, norm_cfg: dict):
        raise NotImplementedError("VQCQFunction is implemented at Checkpoint 2.")
