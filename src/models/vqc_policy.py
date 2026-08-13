"""Softmax-VQC policy: the QDQN circuit with a policy head instead of a Q head.

WHAT IS SHARED WITH THE Q-FUNCTION, AND WHY THAT IS THE POINT

Everything below the head is imported from src/models/vqc.py, not reimplemented:
`build_circuit` (same data re-uploading ansatz, same CIRCUIT_WEIGHT_ORDER),
`make_backend` (same torch-statevector evaluation -- `compile_circuit` +
`simulate`, autograd end to end, no qiskit primitive in the training path), the
same default observables, and the same trainable input scaling `lam`. A QDQN
agent and a QPG agent therefore differ in exactly two places: this file's last
three lines of forward(), and which trainer consumes it. That makes the
Q-learning vs policy-gradient comparison a comparison of *algorithms* rather
than of two independently-tuned circuits.

THE HEAD

VQCQFunction maps [-1, 1] -> [0, 1] and multiplies by a per-action weight w,
because CartPole's Q* at gamma=0.99 is ~99 and the head has to reach it. A
policy has no such target: only the *differences* between logits matter, since
softmax is shift-invariant. So the head here is a single inverse temperature,

    logits = beta * <O_a>,   pi(a|s) = softmax(logits)

which is the Softmax-VQC policy of Jerbi et al. 2021 (arXiv:2103.05577). beta
controls how sharp the policy is: at beta -> 0 it is uniform, at beta -> inf it
is deterministic-greedy. Making it trainable lets the agent anneal its own
exploration, which is why this needs no epsilon schedule at all.

BETA IS PARAMETRIZED THROUGH SOFTPLUS, NOT STORED RAW

`beta = softplus(beta_raw)` is strictly positive by construction. That is not
decoration. src/evaluate.py takes argmax over forward()'s output and is shared
verbatim with the DQN path (see PolicyFunction in base.py): argmax(logits)
equals argmax(pi) only while beta > 0. A raw parameter that gradient descent
walked through zero would silently inverse the greedy policy -- the evaluation
would report a *worst*-action agent without erroring anywhere. Softplus removes
that failure mode rather than relying on it not happening.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.base import PolicyFunction, normalize_observation
from src.models.vqc import (
    BACKENDS,
    CIRCUIT_WEIGHT_ORDER,
    DEFAULT_BACKEND,
    DEFAULT_GRADIENT_METHOD,
    DEFAULT_OBSERVABLES,
    build_circuit,
    make_backend,
)

DEFAULT_BETA = 1.0


def _inverse_softplus(y: float) -> float:
    """x such that softplus(x) == y. log(expm1(y)), guarded for tiny y."""
    if y <= 0:
        raise ValueError(f"beta must be positive, got {y}")
    return float(np.log(np.expm1(y)))


class VQCPolicy(PolicyFunction):
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 5,
        n_actions: int = 2,
        reuploading: bool = True,
        observables=DEFAULT_OBSERVABLES,
        backend: str = DEFAULT_BACKEND,
        gradient_method: str = DEFAULT_GRADIENT_METHOD,
        estimator=None,
        seed: int | None = None,
        beta_init: float = DEFAULT_BETA,
        trainable_beta: bool = True,
        per_layer_encoding: bool = False,
    ):
        super().__init__()
        observables = tuple(observables)
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose one of {BACKENDS}")
        if len(observables) != n_actions:
            raise ValueError(
                f"need exactly one observable per action: got {len(observables)} "
                f"observables {observables} for {n_actions} actions"
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.reuploading = reuploading
        self.observables = observables
        self.backend_name = backend
        self.gradient_method = gradient_method
        self.per_layer_encoding = per_layer_encoding

        circuit, input_params, weight_params = build_circuit(
            n_qubits, n_layers, reuploading, per_layer_encoding=per_layer_encoding
        )

        # Identical to VQCQFunction: one trainable scaling per encoding
        # parameter, applied outside the circuit in plain torch.
        self.lam = nn.Parameter(torch.ones(len(input_params)))
        self.beta_raw = nn.Parameter(
            torch.tensor([_inverse_softplus(beta_init)], dtype=torch.float32),
            requires_grad=trainable_beta,
        )
        self.circuit = circuit
        self._input_params = input_params
        self._weight_params = weight_params

        self.vqc = make_backend(
            backend,
            circuit,
            input_params,
            weight_params,
            observables,
            gradient_method=gradient_method,
            estimator=estimator,
            seed=seed,
        )

    @property
    def beta(self) -> torch.Tensor:
        return nn.functional.softplus(self.beta_raw)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = normalize_observation(states)  # [B, n_qubits]
        if self.per_layer_encoding and self.lam.numel() != normalized.shape[1]:
            # tile, not repeat_interleave -- the circuit parameter order is
            # (layer-major, qubit-minor). Same as VQCQFunction.forward.
            normalized = normalized.tile(1, self.lam.numel() // normalized.shape[1])
        scaled = self.lam * normalized
        expvals = self.vqc(scaled)  # [B, n_actions], each in [-1, 1]
        return self.beta * expvals  # logits; softmax is applied by the loss

    # --- backend-neutral weight transfer, mirroring VQCQFunction ---------------

    def circuit_weights(self) -> torch.Tensor:
        return self.vqc.weight.detach().clone()

    def load_circuit_weights(self, weights: torch.Tensor) -> None:
        target = self.vqc.weight
        flat = torch.as_tensor(weights, dtype=target.dtype).reshape(target.shape)
        with torch.no_grad():
            target.copy_(flat)

    def export_weights(self) -> dict:
        return {
            "lam": self.lam.detach().clone(),
            "beta_raw": self.beta_raw.detach().clone(),
            "circuit_weights": self.circuit_weights(),
            "meta": {
                "head": "softmax",
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
                "reuploading": self.reuploading,
                "observables": list(self.observables),
                "weight_order": CIRCUIT_WEIGHT_ORDER,
                "trained_backend": self.backend_name,
            },
        }

    def import_weights(self, blob: dict) -> None:
        meta = blob.get("meta", {})
        for key in ("n_qubits", "n_layers", "reuploading"):
            if key in meta and meta[key] != getattr(self, key):
                raise ValueError(
                    f"checkpoint {key}={meta[key]!r} does not match this model's "
                    f"{key}={getattr(self, key)!r}; the weight vector would not line up"
                )
        with torch.no_grad():
            self.lam.copy_(torch.as_tensor(blob["lam"], dtype=self.lam.dtype))
            self.beta_raw.copy_(torch.as_tensor(blob["beta_raw"], dtype=self.beta_raw.dtype))
        self.load_circuit_weights(blob["circuit_weights"])


if __name__ == "__main__":
    torch.manual_seed(0)
    model = VQCPolicy(backend="torch_sv", seed=0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VQCPolicy trainable params: {n_params}  (beta={model.beta.item():.4f})")

    states = torch.tensor(
        [[0.03, -0.5, 0.02, 0.4], [-0.1, 1.2, -0.05, -0.8], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    logits = model(states)
    assert logits.shape == (3, 2), logits.shape

    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(3)), probs.sum(-1)
    # The contract src/evaluate.py relies on (see PolicyFunction docstring).
    assert torch.equal(logits.argmax(-1), probs.argmax(-1))
    print("probs:\n", probs)

    # Every parameter must receive gradient -- lam and beta_raw included. This
    # is the dead-gradient check that matters: an encoding parameter whose
    # gradient never arrives trains silently at zero for the whole run.
    logits.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is not finite"
        print(f"  grad[{name}] norm = {p.grad.norm().item():.6f}")
    assert model.lam.grad.norm().item() > 0, "lam gradient is dead"
    assert model.beta_raw.grad.norm().item() > 0, "beta gradient is dead"

    blob = model.export_weights()
    clone = VQCPolicy(backend="torch_sv", seed=1)
    clone.import_weights(blob)
    assert torch.allclose(clone(states), logits, atol=1e-6)
    print("vqc_policy.py smoke test OK")
