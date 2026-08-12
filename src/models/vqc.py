"""Variational Quantum Circuit (VQC) Q-function for QDQN on CartPole-v1.

Circuit (per Chen et al. 2020, https://arxiv.org/abs/1907.00397; recipe follows
Skolik, Jerbi & Dunjko 2022, https://arxiv.org/abs/2103.15084):
  - 4 qubits, one per observation dimension.
  - n_layers variational layers (default 5). Each layer:
      * RY(x[q]) on every qubit -- the encoding. Placed only before layer 0
        unless `reuploading=True`, in which case it repeats at the start of
        every layer (data re-uploading, Perez-Salinas et al. 2020,
        https://doi.org/10.22331/q-2020-02-06-226).
      * RY(theta), RZ(theta) on every qubit -- trainable.
      * CNOT ring q -> (q+1) % n_qubits -- entanglement.
  - Observables Z@I@I@I and I@Z@I@I -> two expectation values in [-1, 1], one
    per CartPole action.

Two failure modes this module exists to avoid (see project README):

FAILURE MODE 1 (output scaling): raw <Z> in [-1, 1], but true CartPole Q-values
at gamma=0.99 reach ~100. Without a trainable output scaling `w`, the model
cannot represent the value function and training flatlines at ~reward 10.
`self.w` is a plain nn.Parameter multiplying the QNN's raw output.

FAILURE MODE 2 (do not scale inside the circuit): lam_i * x_i is a product of
an input value and a trainable weight. Parameter-shift requires parameters to
enter gates linearly, so computing that product *inside* the circuit silently
breaks gradients. Both `lam` (input scaling) and `w` (output scaling) live in
this PyTorch module, strictly outside TorchConnector -- `lam` multiplies the
already-normalized observation in plain torch before it is handed to the QNN
as plain (non-parameter) numeric input; the QNN never sees `lam` itself.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from src.models.base import QFunction, normalize_observation

OBSERVABLES = [SparsePauliOp("ZIII"), SparsePauliOp("IZII")]


def build_circuit(
    n_qubits: int = 4,
    n_layers: int = 5,
    reuploading: bool = False,
) -> tuple[QuantumCircuit, list, list]:
    """Build the RY-encoding / RY+RZ-variational / CNOT-ring circuit.

    Returns (circuit, input_params, weight_params). weight_params has
    n_layers * n_qubits * 2 entries (RY then RZ per qubit, per layer, in
    circuit order).
    """
    input_params = ParameterVector("x", n_qubits)
    circuit = QuantumCircuit(n_qubits)
    weight_params: list = []

    for layer in range(n_layers):
        if layer == 0 or reuploading:
            for q in range(n_qubits):
                circuit.ry(input_params[q], q)

        ry = ParameterVector(f"ry{layer}", n_qubits)
        rz = ParameterVector(f"rz{layer}", n_qubits)
        for q in range(n_qubits):
            circuit.ry(ry[q], q)
            circuit.rz(rz[q], q)
        weight_params.extend(list(ry))
        weight_params.extend(list(rz))

        for q in range(n_qubits):
            circuit.cx(q, (q + 1) % n_qubits)

    return circuit, list(input_params), weight_params


class VQCQFunction(QFunction):
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 5,
        n_actions: int = 2,
        reuploading: bool = False,
        estimator: BaseEstimatorV2 | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.reuploading = reuploading

        # Outside-the-circuit trainable scalings (Failure Modes 1 and 2 above).
        self.lam = nn.Parameter(torch.ones(n_qubits))
        self.w = nn.Parameter(torch.ones(n_actions))

        circuit, input_params, weight_params = build_circuit(n_qubits, n_layers, reuploading)
        self.circuit = circuit
        self._input_params = input_params
        self._weight_params = weight_params

        # Statevector path for training -- exact, differentiable, no shot noise.
        # qiskit-aer / finite shots are reserved for evaluate_finite_shot() below.
        if estimator is None:
            estimator = StatevectorEstimator()

        qnn = EstimatorQNN(
            circuit=circuit,
            observables=OBSERVABLES,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
            # Required for `lam` to receive a gradient: lam only appears
            # upstream of the QNN (lam * normalized_obs -> QNN input), so
            # autograd needs the QNN's own backward to return d(output)/d(input)
            # too, not just d(output)/d(weights). Left at its False default,
            # TorchConnector's backward silently returns None for the input
            # gradient and `lam.grad` stays dead -- this is the exact failure
            # the vqc.py smoke test below is designed to catch.
            input_gradients=True,
        )

        rng = np.random.default_rng(seed)
        initial_weights = rng.uniform(-0.1, 0.1, size=qnn.num_weights)
        self.vqc = TorchConnector(qnn, initial_weights=initial_weights)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = normalize_observation(states)  # [B, n_qubits]
        scaled = self.lam * normalized  # elementwise, in plain torch -- outside the circuit
        raw_out = self.vqc(scaled)  # [B, n_actions], batched TorchConnector call
        return raw_out * self.w


def evaluate_finite_shot(model: VQCQFunction, shots: int) -> VQCQFunction:
    """Hook: rebuild `model`'s QNN backed by a shot-based Aer estimator.

    Returns a new VQCQFunction sharing model's trained lam/w/circuit-weight
    values but evaluated with `shots`-shot sampling noise instead of an exact
    statevector, for later finite-shot robustness study (Skolik et al. 2023,
    https://doi.org/10.1140/epjqt/s40507-023-00166-1). Only the swap is
    implemented here -- the sweep across shot counts / evaluation episodes is
    intentionally not implemented yet, per the project's build order.
    """
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

    precision = 1.0 / np.sqrt(shots)  # AerEstimatorV2 exposes default_precision, not shots directly
    aer_estimator = AerEstimatorV2(options={"default_precision": precision})

    shot_model = VQCQFunction(
        n_qubits=model.n_qubits,
        n_layers=model.n_layers,
        n_actions=model.n_actions,
        reuploading=model.reuploading,
        estimator=aer_estimator,
    )
    shot_model.load_state_dict(model.state_dict())
    return shot_model


if __name__ == "__main__":
    torch.manual_seed(0)
    model = VQCQFunction(seed=0)
    n_circuit_params = len(model._weight_params)
    n_total_params = sum(p.numel() for p in model.parameters())
    print(f"circuit weight params: {n_circuit_params}")
    print(f"total trainable params (lam + circuit + w): {n_total_params}")

    states = torch.rand(8, 4, dtype=torch.float32) * 0.2 - 0.1
    q_values = model(states)
    print("output shape:", q_values.shape)
    assert q_values.shape == (8, 2), f"expected [8, 2], got {q_values.shape}"

    loss = q_values.sum()
    loss.backward()

    assert model.lam.grad is not None and model.lam.grad.abs().sum().item() > 0, "lam gradient is dead"
    assert model.w.grad is not None and model.w.grad.abs().sum().item() > 0, "w gradient is dead"
    assert model.vqc.weight.grad is not None and model.vqc.weight.grad.abs().sum().item() > 0, "vqc gradient is dead"

    print("lam grad norm:", model.lam.grad.norm().item())
    print("w grad norm:", model.w.grad.norm().item())
    print("vqc grad norm:", model.vqc.weight.grad.norm().item())
    print("vqc.py smoke test OK -- all three parameter groups have live gradients")
