"""Quantum critic: the QDQN circuit with a single observable and a scalar head.

Third head on the same body. `build_circuit` and `make_backend` are imported
from vqc.py exactly as vqc_policy.py imports them, so the actor and the critic
in an A2C run are the same ansatz, the same re-uploading, the same torch_sv
evaluation -- differing only in observable count and what the head does with the
expectation value.

WHY THE HEAD IS THE Q-FUNCTION'S, NOT THE POLICY'S

A policy needs only logit differences, so vqc_policy.py gets away with a bare
inverse temperature. A critic does not: it has to predict actual returns, and
V* for CartPole at gamma=0.99 is ~99 (V*(s) = Q*(s, a*) under the optimal
policy). So this reuses VQCQFunction's head verbatim -- map <O> from [-1, 1] to
[0, 1], then scale by a trainable `w`. Same reasoning as there: CartPole returns
are non-negative, so on [0, 1] a weight near 100 suffices, while multiplying
against [-1, 1] would need w ~ 500 to represent V ~ 99 from <O> ~ 0.2 and would
leave half the output range unused.

`w` IS INITIALIZED AT THE VALUE SCALE, NOT AT 1

On the DQN side `w` starts at 1 and climbs to ~tens over training. That works
there because a DQN run takes ~12,000 gradient steps. An A2C run takes about
SIXTY -- one per batch of episodes -- and Adam moves a parameter by roughly its
learning rate per step, so from w=1 at lr=0.1 the head reaches ~7 by the end of
a run and never comes near V* ~ 99. Measured, not predicted: a 10-gradient-step
probe left w at 2.00.

A critic stuck two orders of magnitude below the return scale predicts a nearly
constant value, which makes the GAE baseline a constant, which is REINFORCE's
batch-mean baseline again. A2C would degenerate into policy gradient while
appearing to run correctly -- `explained_variance` in the CSV is the column that
exposes it, and it sat at 0.056.

So `w_init` defaults to the largest return the discounting admits,
(1 - gamma^T) / (1 - gamma), which scripts/train_a2c.py computes from the
config's gamma and max_steps_per_episode rather than hardcoding CartPole's 99.3.
That puts the head at the right order of magnitude on step one and leaves the
circuit weights to shape V(s) ACROSS states, which is the part they can actually
learn in sixty steps.

ONE OBSERVABLE, NOT n_actions OF THEM

The arity check that requires one observable per action lives in
VQCQFunction.__init__ and VQCPolicy.__init__, not in build_circuit or
make_backend -- the circuit and the backend are indifferent. A critic is
therefore the same circuit read out through a single operator. "ZZZZ" is the
default: it reads the whole register, where a single-qubit Z would score the
state from one qubit's marginal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import ValueFunction, normalize_observation
from src.models.vqc import (
    BACKENDS,
    CIRCUIT_WEIGHT_ORDER,
    DEFAULT_BACKEND,
    DEFAULT_GRADIENT_METHOD,
    build_circuit,
    make_backend,
)

DEFAULT_VALUE_OBSERVABLE = "ZZZZ"
DEFAULT_W_INIT = 1.0  # callers that know the return scale should pass it; see the docstring


def value_scale(gamma: float, max_steps: int) -> float:
    """Largest return an episode of `max_steps` unit rewards can carry.

    (1 - gamma^T) / (1 - gamma), or T when gamma == 1. Used as the critic's
    output-weight initialization so the head starts at the right order of
    magnitude -- for CartPole at gamma=0.99, T=500 this is 99.34.
    """
    if gamma >= 1.0:
        return float(max_steps)
    return float((1.0 - gamma**max_steps) / (1.0 - gamma))


class VQCValue(ValueFunction):
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 5,
        reuploading: bool = True,
        observable: str = DEFAULT_VALUE_OBSERVABLE,
        backend: str = DEFAULT_BACKEND,
        gradient_method: str = DEFAULT_GRADIENT_METHOD,
        estimator=None,
        seed: int | None = None,
        output_rescaling: bool = True,
        per_layer_encoding: bool = False,
        w_init: float = DEFAULT_W_INIT,
    ):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose one of {BACKENDS}")
        if not isinstance(observable, str):
            raise TypeError(
                f"a critic takes exactly one observable, got {observable!r}. "
                f"Pass a Pauli string such as 'ZZZZ', not a list."
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reuploading = reuploading
        self.observable = observable
        self.backend_name = backend
        self.gradient_method = gradient_method
        self.output_rescaling = output_rescaling
        self.per_layer_encoding = per_layer_encoding

        circuit, input_params, weight_params = build_circuit(
            n_qubits, n_layers, reuploading, per_layer_encoding=per_layer_encoding
        )

        self.lam = nn.Parameter(torch.ones(len(input_params)))
        self.w = nn.Parameter(torch.full((1,), float(w_init)))
        self.circuit = circuit
        self._input_params = input_params
        self._weight_params = weight_params

        self.vqc = make_backend(
            backend,
            circuit,
            input_params,
            weight_params,
            (observable,),
            gradient_method=gradient_method,
            estimator=estimator,
            seed=seed,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = normalize_observation(states)  # [B, n_qubits]
        if self.per_layer_encoding and self.lam.numel() != normalized.shape[1]:
            normalized = normalized.tile(1, self.lam.numel() // normalized.shape[1])
        scaled = self.lam * normalized
        raw_out = self.vqc(scaled)  # [B, 1], in [-1, 1]
        if self.output_rescaling:
            raw_out = (raw_out + 1.0) / 2.0
        # [B], not [B, 1] -- see the ValueFunction docstring on why the shape matters.
        return (raw_out * self.w).reshape(-1)

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
            "w": self.w.detach().clone(),
            "circuit_weights": self.circuit_weights(),
            "meta": {
                "head": "value",
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers,
                "reuploading": self.reuploading,
                "observable": self.observable,
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
            self.w.copy_(torch.as_tensor(blob["w"], dtype=self.w.dtype))
        self.load_circuit_weights(blob["circuit_weights"])


if __name__ == "__main__":
    torch.manual_seed(0)
    critic = VQCValue(backend="torch_sv", seed=0)
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"VQCValue trainable params: {n_params}  (40 circuit + 4 lam + 1 w)")

    states = torch.tensor(
        [[0.03, -0.5, 0.02, 0.4], [-0.1, 1.2, -0.05, -0.8], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    values = critic(states)
    # [B], never [B, 1] -- a [B, 1] here would broadcast silently against the
    # [B] advantage vector in src/a2c_trainer.py and produce a [B, B] loss.
    assert values.shape == (3,), values.shape
    print("V(s):", values.detach().numpy())

    values.sum().backward()
    for name, p in critic.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is not finite"
        print(f"  grad[{name}] norm = {p.grad.norm().item():.6f}")
    assert critic.lam.grad.norm().item() > 0, "lam gradient is dead"
    assert critic.w.grad.norm().item() > 0, "w gradient is dead"

    # The return scale the head has to reach, derived rather than hardcoded.
    v_max = value_scale(gamma=0.99, max_steps=500)
    assert abs(v_max - 99.34) < 0.01, v_max
    assert value_scale(gamma=1.0, max_steps=500) == 500.0
    print(f"value_scale(0.99, 500) = {v_max:.2f}")

    # w_init must put the head at that order of magnitude on step one. This is
    # the fix for the degeneration described in the module docstring: at w=1 an
    # A2C run's ~60 gradient steps leave the critic near-constant, GAE's
    # baseline becomes a constant, and A2C silently becomes REINFORCE.
    scaled_critic = VQCValue(backend="torch_sv", seed=0, w_init=v_max)
    with torch.no_grad():
        v = scaled_critic(states)
    assert v.max() > 50.0, f"head is not at the return scale: {v}"
    assert v.max() <= v_max + 1e-4, v
    print(f"V(s) at w_init=value_scale: {v.numpy()}")

    blob = critic.export_weights()
    clone = VQCValue(backend="torch_sv", seed=1)
    clone.import_weights(blob)
    assert torch.allclose(clone(states), critic(states), atol=1e-6)
    print("vqc_value.py smoke test OK")
