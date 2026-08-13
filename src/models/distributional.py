"""Distributional (C51-style) VQC Q-function — the 6th Rainbow component, QUANTUM-NATIVE.

Categorical value distribution (Bellemare et al. 2017) read DIRECTLY off the quantum
circuit's measurement statistics — NO large classical head, so the model stays quantum
(~44-52 params, vs ~46 for the plain VQC):

  * the 4-qubit circuit's computational-basis distribution ``|amp|^2`` (16 probs) is the
    quantum measurement distribution;
  * the TOP qubit selects the action (2 actions -> 1 selector qubit), the remaining 3
    qubits index ``2^3 = 8`` value atoms on support ``[v_min, v_max]``;
  * ``p(s,a)`` = the conditional distribution ``|amp|^2`` over those 8 atoms (renormalised),
    so the quantum register's amplitudes ARE the atom probabilities — this is the
    quantum-native reading of C51.

Honest adaptation: the classical paper uses 51 atoms; a 4-qubit register only affords 8
atoms/action here. Finer atoms would need more qubits (the register size), not a classical
head — we keep it quantum on purpose (per "quantum version" requirement).

Dueling (optional, +``n_atoms`` params): a learnable value-atom bias added to every
action's logits — the distributional analogue of the dueling value stream. shot-noise
NoisyNet: Q estimated from S samples of the value distribution (finite measurement shots).
Requires backend="torch_sv" (needs the amplitude distribution).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from qiskit.quantum_info import SparsePauliOp

from .base import QFunction
from .normalize import CartPoleNormalizer
from .torch_sv import TorchStatevectorQNN
from .vqc import build_circuit


class DistributionalVQC(QFunction):
    def __init__(self, model_cfg: dict, norm_cfg: dict):
        super().__init__()
        self.n_qubits = int(model_cfg["n_qubits"])
        self.n_layers = int(model_cfg["n_layers"])
        self.reuploading = bool(model_cfg["reuploading"])
        self.backend = str(model_cfg.get("backend", "torch_sv")).lower()
        if self.backend != "torch_sv":
            raise ValueError("DistributionalVQC needs backend='torch_sv' (uses |amp|^2).")
        self.dueling = bool(model_cfg.get("dueling", True))
        self.n_actions = int(model_cfg.get("n_actions", 2))
        self.v_min = float(model_cfg.get("v_min", 0.0))
        self.v_max = float(model_cfg.get("v_max", 100.0))
        self._lr_cfg = dict(model_cfg["lr"])

        self.obs_dim = self.n_qubits
        assert self.obs_dim == 4
        # action-selector qubits (top bits) + atom qubits (remaining): quantum register split
        self.sel_bits = max(1, math.ceil(math.log2(self.n_actions)))
        self.n_atoms = 1 << (self.n_qubits - self.sel_bits)     # 4 qubits, 2 actions -> 8 atoms
        assert (1 << self.sel_bits) >= self.n_actions
        self.norm = CartPoleNormalizer(norm_cfg)

        self.circuit, input_params, weight_params = build_circuit(
            self.n_qubits, self.n_layers, self.reuploading)
        obs = [SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=self.n_qubits)]
        n_weights = len(weight_params)
        init_w = (torch.rand(n_weights) * 2.0 - 1.0) * np.pi
        self.qnn = TorchStatevectorQNN(self.circuit, obs, input_params, weight_params,
                                       initial_weights=init_w)
        self.dim = 1 << self.n_qubits

        self.lam = nn.Parameter(torch.ones(self.obs_dim))       # input scaling (outside circuit)
        # dueling value-atom bias (the ONLY head params; tiny, keeps the model quantum)
        self.v_bias = nn.Parameter(torch.zeros(self.n_atoms)) if self.dueling else None
        self.register_buffer("z", torch.linspace(self.v_min, self.v_max, self.n_atoms))
        self.dz = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self._explore_shots = None

    def set_exploration_shots(self, shots) -> None:
        self._explore_shots = shots

    def dist(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(torch.float32)
        encoded = self.lam * self.norm(states)
        prob = self.qnn.probabilities(encoded).to(torch.float32)        # (B, 16) |amp|^2
        B = prob.shape[0]
        # split: [action-selector bits | atom bits]  (big-endian: top bits = action)
        pa = prob.view(B, self.n_actions if self.sel_bits == 1 else (1 << self.sel_bits),
                       self.n_atoms)
        pa = pa[:, :self.n_actions, :]                                  # (B, A, atoms)
        # conditional distribution over atoms per action (quantum amplitudes ARE the probs)
        pa = pa / pa.sum(dim=2, keepdim=True).clamp_min(1e-8)
        if self.dueling:                                               # value-stream bias
            logits = torch.log(pa.clamp_min(1e-8)) + self.v_bias.view(1, 1, -1)
            pa = torch.softmax(logits, dim=2)
        return pa                                                     # (B, A, atoms)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        p = self.dist(states)
        if self._explore_shots is not None:                           # quantum-native NoisyNet
            S = int(self._explore_shots)
            with torch.no_grad():
                B, A, _ = p.shape
                idx = torch.multinomial(p.reshape(B * A, self.n_atoms), S, replacement=True)
                q = self.z[idx].mean(dim=1).reshape(B, A)             # S-shot Monte-Carlo Q
            return q
        return (p * self.z).sum(dim=2)                               # exact Q = Σ z·p

    def param_groups(self) -> list[dict]:
        groups = [
            {"params": [self.lam], "lr": float(self._lr_cfg["input_scaling"])},
            {"params": list(self.qnn.parameters()), "lr": float(self._lr_cfg["variational"])},
        ]
        if self.dueling:                        # tiny value-atom bias (only when dueling on)
            groups.append({"params": [self.v_bias], "lr": float(self._lr_cfg["output_scaling"])})
        return groups

    def loggable_scalars(self) -> dict[str, float]:
        return {"w0": float(self.v_max), "n_atoms": float(self.n_atoms)}
