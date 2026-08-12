"""VQC Q-function (quantum approximator).

Architecture (see README "Model spec"):

    s ─normalize─► λ ⊙ s_norm ─► TorchConnector(EstimatorQNN) ─► [⟨Z₀⟩,⟨Z₁⟩] ─► ⊙ w ─► Q(s,·)
                  ↑ nn.Parameter (input scaling)                                 ↑ nn.Parameter (output scaling)

Both scalings live HERE in torch, OUTSIDE the circuit, on purpose (Failure Mode 2):
``λ_i · x_i`` is a product of two parameters and would break parameter-shift if it
entered a gate. And ``w`` is mandatory (Failure Mode 1): ⟨Z⟩∈[-1,1] cannot represent
Q ~ 100 at γ=0.99, so training would flatline at ~10 reward without a trainable output
scale. ``w`` starts at 1 and is expected to climb toward the tens (Figure 3).

Gradient wiring is version-specific to qiskit-machine-learning 0.8.x and was validated
empirically (reverse vs param-shift agree to ~1e-16 on forward AND gradients):

  * ``gradient: reverse`` (default, exact adjoint, simulator-only) — V1 everywhere:
        estimator = qiskit.primitives.Estimator()                      # V1 reference, itself exact/shot-free
        gradient  = qiskit_algorithms.gradients.ReverseEstimatorGradient()
    Emits one benign "V1 Primitives are deprecated" warning (V1 goes away in qml 0.9 /
    qiskit 2.0, both pinned below here). This is the only path to the exact adjoint
    gradient in 0.8.x.
  * ``gradient: paramshift`` (kept for the later finite-shot / hardware path) — V2:
        estimator = qiskit.primitives.StatevectorEstimator()
        gradient  = qiskit_machine_learning.gradients.ParamShiftEstimatorGradient(estimator=est)
    with ``default_precision=0.0`` on the QNN, else the V2 estimator injects
    ~0.0156-std shot noise into the forward pass.

The naive ``EstimatorQNN(estimator=StatevectorEstimator(), gradient=ReverseEstimatorGradient())``
pairs a V2 forward primitive with a V1-only gradient and is NOT a supported combination.

Reference: Skolik, Jerbi, Dunjko (2022), *Quantum agents in the gym*, Quantum 6, 720,
arXiv:2103.15084.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from .base import QFunction


def build_ansatz(n_qubits: int, n_layers: int, reuploading: bool):
    """Construct the hardware-efficient ansatz and its parameter vectors.

    Each layer: (optional) ``RY(x[q])`` encoding on every qubit; trainable
    ``RY(θ), RZ(θ)`` on every qubit; CNOT ring ``q -> (q+1) % n``.
    ``reuploading=False`` encodes once before layer 0; ``True`` re-encodes each layer.

    Returns ``(circuit, input_params, weight_params)``.
    """
    x = ParameterVector("x", n_qubits)                    # input (encoding) params
    theta = ParameterVector("theta", 2 * n_qubits * n_layers)  # trainable RY,RZ per qubit per layer
    qc = QuantumCircuit(n_qubits, name="vqc")
    k = 0
    for layer in range(n_layers):
        if reuploading or layer == 0:                     # data (re-)uploading
            for q in range(n_qubits):
                qc.ry(x[q], q)
        for q in range(n_qubits):                         # trainable rotations
            qc.ry(theta[k], q); k += 1
            qc.rz(theta[k], q); k += 1
        for q in range(n_qubits):                         # CNOT ring (entanglement)
            qc.cx(q, (q + 1) % n_qubits)
    return qc, list(x), list(theta)


def build_estimator_qnn(qc, observables, input_params, weight_params, gradient: str) -> EstimatorQNN:
    """Wire an EstimatorQNN with the correct estimator+gradient pair for `gradient`.

    The circuit is built once (here) and reused for every forward/backward; there is
    no per-call transpilation. Statevector estimators need no ISA transpilation, so no
    pass_manager is used (also avoids qml issue #872).
    """
    if gradient == "reverse":
        from qiskit.primitives import Estimator                       # V1 reference: exact, shot-free
        from qiskit_algorithms.gradients import ReverseEstimatorGradient
        # Scope the (harmless) V1-deprecation warning so it doesn't spam the run log.
        warnings.filterwarnings("ignore", message=".*V1 Primitives are deprecated.*")
        return EstimatorQNN(
            circuit=qc, observables=observables,
            input_params=input_params, weight_params=weight_params,
            estimator=Estimator(),
            gradient=ReverseEstimatorGradient(),                       # exact adjoint, linear in n_params
            input_gradients=True,                                      # REQUIRED or `lam` never trains
        )
    if gradient == "paramshift":
        from qiskit.primitives import StatevectorEstimator             # V2, shot-free
        from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
        est = StatevectorEstimator()
        return EstimatorQNN(
            circuit=qc, observables=observables,
            input_params=input_params, weight_params=weight_params,
            estimator=est,
            gradient=ParamShiftEstimatorGradient(estimator=est),       # 2*n_params evals; hardware-ready
            input_gradients=True,
            default_precision=0.0,                                     # exact forward (no injected shot noise)
        )
    raise ValueError(f"Unknown gradient {gradient!r}. Expected 'reverse' or 'paramshift'.")


class VQCQFunction(QFunction):
    """Variational-quantum-circuit Q-function for CartPole (4 qubits, 2 actions)."""

    def __init__(self, model_cfg: dict, norm_cfg: dict):
        super().__init__()
        self.n_qubits = int(model_cfg["n_qubits"])
        self.n_layers = int(model_cfg["n_layers"])
        self.reuploading = bool(model_cfg["reuploading"])
        self.gradient = str(model_cfg["gradient"]).lower()
        self.observable_qubits = list(model_cfg["observable_qubits"])
        self._lr_cfg = dict(model_cfg["lr"])  # input_scaling / variational / output_scaling

        self.obs_dim = self.n_qubits
        self.n_actions = len(self.observable_qubits)

        # --- normalization config (CartPole-specific: dims 0=cart x, 1=cart v,
        #     2=pole angle, 3=pole angular v). Bounded dims divided by their
        #     termination bound; unbounded velocity dims squashed. ---
        assert self.obs_dim == 4, "normalization is specialized for CartPole's 4 observations"
        self.cart_pos_bound = float(norm_cfg["cart_position_bound"])
        self.pole_angle_bound = float(norm_cfg["pole_angle_bound"])
        vt = str(norm_cfg["velocity_transform"]).lower()
        if vt not in ("arctan", "tanh"):
            raise ValueError(f"velocity_transform must be 'arctan' or 'tanh', got {vt!r}")
        self._vtransform = torch.arctan if vt == "arctan" else torch.tanh

        # --- quantum circuit + QNN + TorchConnector (variational weights) ---
        self.circuit, input_params, weight_params = build_ansatz(
            self.n_qubits, self.n_layers, self.reuploading
        )
        self.observables = [
            SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=self.n_qubits)
            for q in self.observable_qubits
        ]
        qnn = build_estimator_qnn(
            self.circuit, self.observables, input_params, weight_params, self.gradient
        )
        n_weights = len(weight_params)
        # Reproducible initial variational angles from the (globally-seeded) torch RNG,
        # uniform in [-pi, pi].
        init_w = (torch.rand(n_weights) * 2.0 - 1.0) * np.pi
        self.vqc = TorchConnector(qnn, initial_weights=init_w)

        # --- the two torch-side scalings (outside the circuit) ---
        self.lam = nn.Parameter(torch.ones(self.obs_dim))     # input scaling  (Failure Mode 2)
        self.w = nn.Parameter(torch.ones(self.n_actions))     # output scaling (Failure Mode 1)

    # -- normalization -------------------------------------------------------
    def _normalize(self, s: torch.Tensor) -> torch.Tensor:
        """Raw CartPole obs [B,4] -> normalized [B,4] (bounded dims /bound, velocities squashed)."""
        cols = [
            s[:, 0] / self.cart_pos_bound,      # cart position
            self._vtransform(s[:, 1]),          # cart velocity (unbounded)
            s[:, 2] / self.pole_angle_bound,    # pole angle
            self._vtransform(s[:, 3]),          # pole angular velocity (unbounded)
        ]
        return torch.stack(cols, dim=1)

    # -- forward -------------------------------------------------------------
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(torch.float32)
        encoded = self.lam * self._normalize(states)        # trainable input scaling
        raw = self.vqc(encoded).to(self.w.dtype)            # [B, n_actions] in [-1, 1]
        return raw * self.w                                 # trainable output scaling -> Q-values

    # -- optimizer grouping (owned by the model) -----------------------------
    def param_groups(self) -> list[dict]:
        """Three groups, three learning rates. The output scaling `w` deliberately
        uses a ~100x larger lr so it can reach Q~100 while the circuit trains slowly."""
        return [
            {"params": [self.lam], "lr": float(self._lr_cfg["input_scaling"])},
            {"params": list(self.vqc.parameters()), "lr": float(self._lr_cfg["variational"])},
            {"params": [self.w], "lr": float(self._lr_cfg["output_scaling"])},
        ]

    # -- logging -------------------------------------------------------------
    def loggable_scalars(self) -> dict[str, float]:
        out = {f"w{i}": float(v) for i, v in enumerate(self.w.detach().cpu().numpy())}
        out.update({f"lam{i}": float(v) for i, v in enumerate(self.lam.detach().cpu().numpy())})
        return out
