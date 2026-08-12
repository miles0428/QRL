"""Classical MLP baseline, parameter-count matched to the VQC.

Fair comparison is the whole point (README design principle 2): a 2x128 MLP (~17k
params) vs a ~46-param VQC is not evidence of quantum parameter-efficiency. This net's
hidden width is chosen so its trainable param count lands within +/-20% of the VQC's.
Uses the SAME normalizer as the VQC (src/models/normalize.py) so the comparison
isolates the approximator, not the input pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import QFunction
from .normalize import CartPoleNormalizer

# Reference: the fundamental-run VQC (n_layers=5) has
#   5*4*2 = 40 variational + 4 input scalings (lam) + 2 output scalings (w) = 46 params.
# The MLP is asserted to be within +/-20% of this. If you change n_layers, update this.
_VQC_REF_PARAMS = 46
_MATCH_TOLERANCE = 0.20

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLPQFunction(QFunction):
    def __init__(self, model_cfg: dict, norm_cfg: dict):
        super().__init__()
        self.obs_dim = 4
        self.n_actions = 2
        self._lr = float(model_cfg["lr"])

        self.norm = CartPoleNormalizer(norm_cfg)

        hidden = list(model_cfg["hidden_dims"])
        act_name = str(model_cfg.get("activation", "relu")).lower()
        if act_name not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)}, got {act_name!r}")
        Act = _ACTIVATIONS[act_name]

        dims = [self.obs_dim] + hidden + [self.n_actions]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on the output layer
                layers.append(Act())
        self.net = nn.Sequential(*layers)

        # Report and (by default) enforce the parameter match. Set
        # `enforce_param_match: false` in the config for an intentionally unmatched net
        # (e.g. validating the trainer with a wider MLP); the real baseline keeps it on.
        n = self.num_trainable_params()
        lo, hi = _VQC_REF_PARAMS * (1 - _MATCH_TOLERANCE), _VQC_REF_PARAMS * (1 + _MATCH_TOLERANCE)
        enforce = bool(model_cfg.get("enforce_param_match", True))
        print(
            f"[mlp] hidden_dims={hidden} -> {n} trainable params "
            f"(VQC ref {_VQC_REF_PARAMS}; matched window [{lo:.0f}, {hi:.0f}]; "
            f"enforce_param_match={enforce})"
        )
        if enforce:
            assert lo <= n <= hi, (
                f"MLP has {n} params, outside +/-{_MATCH_TOLERANCE:.0%} of the VQC's "
                f"{_VQC_REF_PARAMS}. Adjust hidden_dims or set enforce_param_match: false."
            )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(torch.float32)
        return self.net(self.norm(states))

    def param_groups(self) -> list[dict]:
        # Classical net: one group, one learning rate (no input/output scalings).
        return [{"params": list(self.parameters()), "lr": self._lr}]
