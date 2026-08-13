"""Dueling VQC Q-function (Rainbow component) on the SHARED circuit.

Dueling DQN (Wang et al. 2016) splits the readout into a state-value stream ``V(s)`` and
a per-action advantage stream ``A(s,a)``, recombined as::

    Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )

Here both streams are Pauli-Z expectations of the ONE `build_circuit` ansatz — no new
gates. ``V`` reads one observable (``Z`` on ``value_qubit``) with its own output scaling
``w_v``; ``A`` reads ``n_actions`` observables (``Z`` on each ``advantage_qubits``) with
scaling ``w_a``. The mean-subtraction on ``A`` gives the identifiability the dueling
architecture needs. Runs on the same backends as `VQCQFunction` (train on torch_sv) and
carries the shot-noise NoisyNet exploration hook.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector

from .base import QFunction
from .normalize import CartPoleNormalizer
from .vqc import _maybe_shot_noise, build_circuit, build_estimator_qnn


class DuelingVQCQFunction(QFunction):
    def __init__(self, model_cfg: dict, norm_cfg: dict):
        super().__init__()
        self.n_qubits = int(model_cfg["n_qubits"])
        self.n_layers = int(model_cfg["n_layers"])
        self.reuploading = bool(model_cfg["reuploading"])
        self.gradient = str(model_cfg.get("gradient", "reverse")).lower()
        self.backend = str(model_cfg.get("backend", "torch_sv")).lower()
        if self.backend in ("reverse", "paramshift"):
            self.gradient, self.backend = self.backend, "qiskit_ml"
        self.value_qubit = int(model_cfg.get("value_qubit", 2))
        self.advantage_qubits = list(model_cfg.get("advantage_qubits", [0, 1]))
        self._lr_cfg = dict(model_cfg["lr"])

        self.obs_dim = self.n_qubits
        self.n_actions = len(self.advantage_qubits)
        assert self.obs_dim == 4, "normalization is specialized for CartPole's 4 observations"
        self.norm = CartPoleNormalizer(norm_cfg)

        self.circuit, input_params, weight_params = build_circuit(
            self.n_qubits, self.n_layers, self.reuploading)
        # readout order: [V observable, A observable per action]
        qubit_list = [self.value_qubit] + self.advantage_qubits
        self.observables = [
            SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=self.n_qubits)
            for q in qubit_list
        ]
        n_weights = len(weight_params)
        init_w = (torch.rand(n_weights) * 2.0 - 1.0) * np.pi
        if self.backend == "torch_sv":
            from .torch_sv import TorchStatevectorQNN
            self.vqc = TorchStatevectorQNN(self.circuit, self.observables, input_params,
                                           weight_params, initial_weights=init_w)
        elif self.backend == "qiskit_ml":
            qnn = build_estimator_qnn(self.circuit, self.observables, input_params,
                                      weight_params, self.gradient)
            self.vqc = TorchConnector(qnn, initial_weights=init_w)
        else:
            raise ValueError(f"Unknown backend {self.backend!r}")

        self.lam = nn.Parameter(torch.ones(self.obs_dim))      # input scaling
        self.w_v = nn.Parameter(torch.ones(1))                 # value output scaling
        self.w_a = nn.Parameter(torch.ones(self.n_actions))    # advantage output scaling
        self._explore_shots = None

    def set_exploration_shots(self, shots) -> None:
        self._explore_shots = shots

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(torch.float32)
        encoded = self.lam * self.norm(states)
        raw = self.vqc(encoded).to(self.w_v.dtype)             # (B, 1 + n_actions) in [-1,1]
        raw = _maybe_shot_noise(raw, self._explore_shots)
        v = raw[:, 0:1] * self.w_v                             # (B, 1)
        a = raw[:, 1:] * self.w_a                              # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))          # dueling recombination

    def param_groups(self) -> list[dict]:
        return [
            {"params": [self.lam], "lr": float(self._lr_cfg["input_scaling"])},
            {"params": list(self.vqc.parameters()), "lr": float(self._lr_cfg["variational"])},
            {"params": [self.w_v, self.w_a], "lr": float(self._lr_cfg["output_scaling"])},
        ]

    def loggable_scalars(self) -> dict[str, float]:
        out = {"w0": float(self.w_v.detach()[0])}              # keep a w0 column for the logger
        out.update({f"wa{i}": float(v) for i, v in enumerate(self.w_a.detach().cpu().numpy())})
        out["wv"] = float(self.w_v.detach()[0])
        return out
