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

# Allow `python src/models/vqc.py ...` (house-rule smoke command) despite the relative
# imports below: set the package context before they resolve.
if __name__ == "__main__" and __package__ in (None, ""):
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "src.models"

import warnings

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from .base import QFunction
from .normalize import CartPoleNormalizer


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


# `build_circuit` is the name the pipeline/house-rules refer to; the concrete builder is
# `build_ansatz`. New ansatz variations (CZ vs CNOT ring, reuploading on/off, extra
# observables) go THROUGH here so every backend executes the same circuit.
build_circuit = build_ansatz


def apply_entangler(circuit, entangler: str):
    """Return a copy of `circuit` with each CNOT-ring gate mapped to another entangler.

    ``build_circuit`` emits a CNOT ring; the ansatz ablation (STEP 5) needs CZ (or no)
    entanglement WITHOUT re-hardcoding the rest of the ansatz. ``cx`` returns the circuit
    unchanged. Only the two-qubit gates are rewritten; all rotations are preserved."""
    entangler = entangler.lower()
    if entangler == "cx":
        return circuit
    new = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    for instr in circuit.data:
        name = instr.operation.name.lower()
        qubits = [circuit.find_bit(q).index for q in instr.qubits]
        if name == "cx":
            if entangler == "none":
                continue
            if entangler == "cz":
                new.cz(qubits[0], qubits[1]); continue
            raise ValueError(f"unknown entangler {entangler!r} (expected cx|cz|none)")
        new.append(instr.operation, [new.qubits[i] for i in qubits])
    return new


def _maybe_shot_noise(o: torch.Tensor, shots) -> torch.Tensor:
    """Shot-noise NoisyNet exploration: perturb a Pauli-Z expectation <O> in [-1,1] by the
    sampling noise of a `shots`-shot estimate. For a +-1 observable the S-shot mean has
    variance (1-<O>^2)/S; we add zero-mean Gaussian noise of that std (detached, so it never
    enters a gradient). Returns `o` unchanged when `shots` is None. Reused by every value
    model that wants quantum-native exploration (anneal `shots` instead of epsilon)."""
    if shots is None:
        return o
    with torch.no_grad():
        var = ((1.0 - o.clamp(-1.0, 1.0) ** 2).clamp_min(0.0)) / float(shots)
        noise = torch.randn_like(o) * var.sqrt()
    return o + noise


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
        self.gradient = str(model_cfg.get("gradient", "reverse")).lower()
        # backend selects the circuit evaluator (all execute the SAME build_circuit):
        #   "torch_sv"  -> fast exact-autograd statevector (TRAIN ON THIS)
        #   "qiskit_ml" -> qiskit EstimatorQNN + TorchConnector, using `gradient`
        #   "reverse"/"paramshift" aliases pick qiskit_ml with that gradient.
        self.backend = str(model_cfg.get("backend", "qiskit_ml")).lower()
        if self.backend in ("reverse", "paramshift"):
            self.gradient, self.backend = self.backend, "qiskit_ml"
        # Entangler ablation (STEP 5): base build_circuit is a CNOT ring; "cz"/"none" swap it.
        self.entangler = str(model_cfg.get("entangler", "cx")).lower()
        # Readout: either single-Z per action (observable_qubits) OR arbitrary Z-strings
        # (observables = list of qubit-index tuples, e.g. [[0,1],[2,3]] for ZZ correlators).
        if model_cfg.get("observables") is not None:
            self._obs_spec = [tuple(o) for o in model_cfg["observables"]]
        else:
            self._obs_spec = [(q,) for q in model_cfg["observable_qubits"]]
        self.observable_qubits = [o[0] for o in self._obs_spec]  # kept for back-compat/logging
        self._lr_cfg = dict(model_cfg["lr"])  # input_scaling / variational / output_scaling

        self.obs_dim = self.n_qubits
        self.n_actions = len(self._obs_spec)

        # --- normalization config (CartPole-specific: dims 0=cart x, 1=cart v,
        #     2=pole angle, 3=pole angular v). Bounded dims divided by their
        #     termination bound; unbounded velocity dims squashed. ---
        assert self.obs_dim == 4, "normalization is specialized for CartPole's 4 observations"
        self.norm = CartPoleNormalizer(norm_cfg)

        # --- quantum circuit + QNN + TorchConnector (variational weights) ---
        self.circuit, input_params, weight_params = build_ansatz(
            self.n_qubits, self.n_layers, self.reuploading
        )
        self.circuit = apply_entangler(self.circuit, self.entangler)   # STEP 5 ablation hook
        self.observables = [
            SparsePauliOp.from_sparse_list([("Z" * len(qs), list(qs), 1.0)], num_qubits=self.n_qubits)
            for qs in self._obs_spec
        ]
        n_weights = len(weight_params)
        # Reproducible initial variational angles from the (globally-seeded) torch RNG,
        # uniform in [-pi, pi]. Drawn identically for EVERY backend so torch_sv and the
        # Qiskit path start byte-identical (the equivalence check depends on this).
        init_w = (torch.rand(n_weights) * 2.0 - 1.0) * np.pi
        if self.backend == "torch_sv":
            from .torch_sv import TorchStatevectorQNN
            self.vqc = TorchStatevectorQNN(
                self.circuit, self.observables, input_params, weight_params,
                initial_weights=init_w,
            )
        elif self.backend == "qiskit_ml":
            qnn = build_estimator_qnn(
                self.circuit, self.observables, input_params, weight_params, self.gradient
            )
            self.vqc = TorchConnector(qnn, initial_weights=init_w)
        else:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Expected 'torch_sv' or 'qiskit_ml' "
                "(or gradient aliases 'reverse'/'paramshift')."
            )

        # --- the two torch-side scalings (outside the circuit) ---
        self.lam = nn.Parameter(torch.ones(self.obs_dim))     # input scaling  (Failure Mode 2)
        self.w = nn.Parameter(torch.ones(self.n_actions))     # output scaling (Failure Mode 1)

        # Shot-noise "NoisyNet" exploration hook (Rainbow, quantum-native): when set to a
        # finite shot budget, `forward` perturbs <O> with the sampling noise of a shots-shot
        # estimate. Set only around action selection; None during updates/eval (exact).
        self._explore_shots = None

    # -- forward -------------------------------------------------------------
    def set_exploration_shots(self, shots) -> None:
        """Enable (int) / disable (None) shot-noise exploration on the next forward(s)."""
        self._explore_shots = shots

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(torch.float32)
        encoded = self.lam * self.norm(states)              # trainable input scaling
        raw = self.vqc(encoded).to(self.w.dtype)            # [B, n_actions] in [-1, 1]
        raw = _maybe_shot_noise(raw, self._explore_shots)   # quantum-native exploration
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


# ---------------------------------------------------------------------------
# Smoke test:  python src/models/vqc.py --backend {torch_sv|qiskit_ml|all}
# Builds the VQC on the requested backend(s), does one forward + backward on a random
# batch, and prints output shape + that lam/vqc/w gradients are alive. `--backend all`
# also prints the max forward disagreement between torch_sv and the Qiskit reverse path.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="VQC backend smoke test.")
    ap.add_argument("--backend", default="all", choices=["torch_sv", "qiskit_ml", "all"])
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--reuploading", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    norm_cfg = dict(cart_position_bound=2.4, pole_angle_bound=0.2095, velocity_transform="arctan")

    def _base_cfg():
        return dict(type="vqc", n_qubits=4, n_layers=args.n_layers,
                    reuploading=args.reuploading, gradient="reverse",
                    observable_qubits=[0, 1],
                    lr=dict(input_scaling=1e-3, variational=1e-3, output_scaling=1e-1))

    backends = ["torch_sv", "qiskit_ml"] if args.backend == "all" else [args.backend]
    outs = {}
    for backend in backends:
        torch.manual_seed(0)
        cfg = _base_cfg(); cfg["backend"] = backend
        m = VQCQFunction(cfg, norm_cfg)
        states = torch.randn(args.batch, 4)
        q = m(states)
        m.zero_grad(); q.pow(2).mean().backward()
        vqc_w = next(iter(m.vqc.parameters()))
        grads_ok = all(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in (m.lam, vqc_w, m.w))
        print(f"[{backend:9s}] out {tuple(q.shape)} | params {m.num_trainable_params()} | "
              f"lam/vqc/w grads alive: {grads_ok}")
        outs[backend] = q.detach().to(torch.float64)
    if len(outs) == 2:
        d = (outs["torch_sv"] - outs["qiskit_ml"]).abs().max().item()
        print(f"[all] max|torch_sv - qiskit_ml| forward = {d:.2e}")
