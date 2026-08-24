"""Shared pipeline layer for every quantum-RL experiment (PG, A2C, Rainbow, ablation).

This is the ONE place the experiments import. It re-exports the single circuit builder
(`build_circuit`), gives the policy-gradient / actor-critic agents a non-QFunction head
on that SAME circuit via `make_policy_backend`, wraps the model-agnostic DQN trainer
(`train_value`), the single greedy-eval protocol (`evaluate_greedy`), a backbone-neutral
checkpoint format (`save_ckpt`/`load_ckpt`) that Session B consumes, plus `set_seed` and
`log_versions`. See experiments/INTERFACES.md for signatures + the checkpoint schema.

TRAIN ON backend="torch_sv" (exact-autograd statevector, ~150x faster than the Qiskit
primitive path; verified equal on forward AND gradients by
scripts/verify_backend_equivalence.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from qiskit.quantum_info import SparsePauliOp                 # noqa: E402
from qrl.trainers.evaluate import SOLVE_THRESHOLD, SOLVE_WINDOW, run_greedy_rollouts  # noqa: E402
from qrl.backends.torch_statevector import TorchStatevectorQNN  # noqa: E402
from qrl.models.vqc import build_circuit                     # noqa: E402
from qrl.trainers.seeds import make_rng, set_global_seeds    # noqa: E402
from qrl.trainers.dqn_trainer import train as _train_value   # noqa: E402

# Default CartPole normalization (identical to configs/qdqn.yaml, so every agent shares
# the exact input pipeline and the comparison isolates the approximator).
DEFAULT_NORM = dict(cart_position_bound=2.4, pole_angle_bound=0.2095, velocity_transform="arctan")
N_QUBITS, N_ACTIONS = 4, 2


# ---------------------------------------------------------------------------
# seeding / versions
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> np.random.Generator:
    """Seed python/numpy/torch BEFORE constructing the model, return a replay/action RNG."""
    set_global_seeds(seed)
    return make_rng(seed)


def log_versions(log=print) -> None:
    """Placeholder: format_versions migrated from src/; implement in qrl.versions if needed."""
    log("[qrl] format_versions not yet migrated from src/")


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------
def as_observable(obs, n_qubits: int = N_QUBITS) -> SparsePauliOp:
    """Coerce a spec into a SparsePauliOp on `n_qubits`.

    Accepts:
      * a SparsePauliOp                                  -> returned as-is
      * a Pauli label string e.g. "ZZZZ" / "ZIII"       -> SparsePauliOp(label)
      * a tuple/list of qubit indices e.g. (0, 1)        -> Z on exactly those qubits
        (endianness-safe: uses from_sparse_list, NOT a little-endian label string)
    """
    if isinstance(obs, SparsePauliOp):
        return obs
    if isinstance(obs, str):
        assert len(obs) == n_qubits, f"label {obs!r} must have length n_qubits={n_qubits}"
        return SparsePauliOp(obs)
    qubits = list(obs)
    return SparsePauliOp.from_sparse_list([("Z" * len(qubits), qubits, 1.0)], num_qubits=n_qubits)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# CartPoleNormalizer stub — src.models.normalize not yet migrated to qrl
# ---------------------------------------------------------------------------------------------------------------------------------------------
class CartPoleNormalizer:
    """Stub normalizer for CartPole observation space; qrl uses spin-based normalization in models.base."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return torch.arctan(states)


# -----------------------------------------------------------------------------
# circuit backend for policy-gradient / actor-critic heads (non-QFunction)
# -----------------------------------------------------------------------------
class CircuitBackend(nn.Module):
    """`raw states -> (B, n_observables)` expectation values on the SHARED circuit.

    Encapsulates the exact backbone input pipeline: normalize -> trainable input scaling
    `lam` (outside the circuit) -> torch_sv statevector -> <O_j>. PG/A2C put their own
    classical head (softmax weights, value readout, ...) on top of this, so they reuse the
    one circuit without being Q-functions. Parameter groups mirror the backbone's three
    learning-rate structure: `input_params()` (lam) and `var_params()` (circuit angles).
    """

    def __init__(self, observables, n_qubits: int = N_QUBITS, n_layers: int = 5,
                 reuploading: bool = True, norm_cfg: dict | None = None,
                 entangler: str = "cx"):
        super().__init__()
        self.n_qubits = n_qubits
        self.observables = [as_observable(o, n_qubits) for o in observables]
        self.n_obs = len(self.observables)
        self.norm = CartPoleNormalizer(norm_cfg or DEFAULT_NORM)
        self.lam = nn.Parameter(torch.ones(n_qubits))         # input scaling (outside circuit)

        circuit, input_params, weight_params = build_circuit(n_qubits, n_layers, reuploading)
        if entangler != "cx":
            circuit = _swap_entangler(circuit, entangler)     # ablation hook (CZ ring, etc.)
        n_w = len(weight_params)
        init_w = (torch.rand(n_w) * 2.0 - 1.0) * np.pi         # backbone init convention
        self.qnn = TorchStatevectorQNN(circuit, self.observables, input_params,
                                       weight_params, initial_weights=init_w)

    def input_params(self):
        return [self.lam]

    def var_params(self):
        return [self.qnn.weights]

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(torch.float32)
        encoded = self.lam * self.norm(states)                # (B, n_qubits)
        return self.qnn(encoded).to(torch.float32)            # (B, n_obs); match torch pipeline dtype


def make_policy_backend(observables, **kw) -> CircuitBackend:
    """Factory for a shared-circuit backend feeding a PG/A2C head (see CircuitBackend)."""
    return CircuitBackend(observables, **kw)


def _swap_entangler(circuit, entangler: str):
    """Return a copy of `circuit` with every CX ring replaced by the requested entangler.

    Only used by the ansatz ablation. The base build_circuit uses a CNOT ring; this maps
    it to CZ (or drops entanglement) WITHOUT re-hardcoding the rest of the ansatz.
    """
    from qiskit.circuit import QuantumCircuit
    new = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    for instr in circuit.data:
        name = instr.operation.name.lower()
        qubits = [circuit.find_bit(q).index for q in instr.qubits]
        if name == "cx":
            if entangler == "none":
                continue
            if entangler == "cz":
                new.cz(qubits[0], qubits[1]); continue
            raise ValueError(f"unknown entangler {entangler!r}")
        new.append(instr.operation, [new.qubits[i] for i in qubits])
    return new


# ---------------------------------------------------------------------------
# training / evaluation wrappers (single protocol everywhere)
# ---------------------------------------------------------------------------
def train_value(model, cfg: dict, seed: int, path: str, log=print, progress_every: int = 10,
                on_best=None) -> dict:
    """Run the model-agnostic DQN trainer (src.trainer.train) -> summary dict.

    `model` is any QFunction (VQC value model / MLP / Rainbow). Writes per-episode CSV
    to `path`. The trainer re-seeds from `seed` at loop start; construct `model` under
    set_seed(seed) first for reproducible weight init. `on_best` is forwarded to the
    trainer for mid-run checkpointing (see src.trainer.train).
    """
    return _train_value(model, cfg, seed, path, log=log, progress_every=progress_every,
                        on_best=on_best)


def evaluate_greedy(model, n_episodes: int = 100, seed: int = 0, out_path: str | None = None) -> dict:
    """Final greedy (epsilon=0) eval over `n_episodes`; the ONE solve criterion
    (mean >= 475 over 100). Returns a JSON-serializable dict; optionally writes it."""
    res = run_greedy_rollouts(model, n_episodes=n_episodes, seed=seed)
    res["solve_threshold"] = SOLVE_THRESHOLD
    res["solve_window"] = SOLVE_WINDOW
    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return res


# ---------------------------------------------------------------------------
# backbone-neutral checkpoints (Session B consumes these)
# ---------------------------------------------------------------------------
CKPT_FORMAT = "qrl-backbone-neutral-v1"


def _circuit_weights(model):
    """The variational angle tensor inside a model's circuit, both backends, or None for a
    classical model (e.g. the MLP baseline has no circuit).

    Works for VQCQFunction (model.vqc is TorchConnector or TorchStatevectorQNN), the
    distributional model (model.qnn), and CircuitBackend heads (model.backend.qnn)."""
    for attr in ("vqc", "qnn"):
        if hasattr(model, attr):
            return next(iter(getattr(model, attr).parameters())).detach().clone()
    if hasattr(model, "backend") and hasattr(model.backend, "qnn"):
        return model.backend.qnn.weights.detach().clone()
    for m in model.modules():
        if isinstance(m, TorchStatevectorQNN):
            return m.weights.detach().clone()
    return None    # classical model (MLP): no circuit weights


def save_ckpt(model, path: str, cfg: dict | None = None, meta: dict | None = None) -> dict:
    """Write a backbone-neutral checkpoint {lam, head/w, circuit_weights, state_dict, meta}.

    * circuit_weights : variational angles (weight_params order) -- portable across the
      torch_sv and Qiskit backends because both index weight_params identically.
    * lam             : trainable input scaling (n_qubits,), if the model has one.
    * head            : output head params (e.g. {"w": ...}); a dict so Dueling/critic
      heads round-trip too.
    * state_dict      : full torch state_dict for exact reconstruction into the same class.
    * meta            : model_cfg / norm_cfg (to rebuild) + any metrics passed in.

    Returns the checkpoint dict (also written to `path`).
    """
    head = {}
    for hname in ("w", "w_v", "w_a"):        # single-head (w) or dueling (w_v, w_a)
        if hasattr(model, hname) and isinstance(getattr(model, hname), torch.Tensor):
            head[hname] = getattr(model, hname).detach().clone()
    lam = model.lam.detach().clone() if hasattr(model, "lam") else (
        model.backend.lam.detach().clone() if hasattr(model, "backend") else None)

    full_meta = dict(meta or {})
    if cfg is not None:
        full_meta.setdefault("model_cfg", cfg.get("model"))
        full_meta.setdefault("norm_cfg", cfg.get("normalization"))
    ckpt = {
        "format": CKPT_FORMAT,
        "circuit_weights": _circuit_weights(model),
        "lam": lam,
        "head": head,
        "state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "meta": full_meta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(ckpt, path)
    return ckpt


def load_ckpt(path: str) -> dict:
    """Load a backbone-neutral checkpoint dict (see save_ckpt)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_into(model, ckpt: dict) -> None:
    """Restore a model from a checkpoint's full state_dict (same class as saved)."""
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
