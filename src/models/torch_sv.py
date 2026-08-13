"""``torch_sv`` — a fast, exact, autograd-native statevector backend for the ONE circuit.

This module does NOT define a circuit. It *interprets* a Qiskit ``QuantumCircuit``
(the one built by ``vqc.build_ansatz`` / ``build_circuit`` — the single source of truth)
gate-by-gate in batched PyTorch, so training runs at statevector speed with exact
autograd gradients while the gates themselves are never re-hardcoded here. Whatever
``build_circuit`` emits (CNOT vs CZ ring, reuploading on/off, extra layers) is executed
verbatim; ``scripts/verify_backend_equivalence.py`` checks this backend agrees with the
Qiskit ``EstimatorQNN`` reverse-adjoint path on BOTH forward values and gradients.

Why it is exact and fast on THIS problem (4 qubits, dim 16):
  * every observable is a product/sum of Pauli-Z's -> DIAGONAL in the computational
    basis, so <O> = sum_k |amp_k|^2 * eig_k -- no operator matmul;
  * the whole minibatch is one batched complex tensor torch autograd differentiates
    directly (no parameter-shift, no adjoint bookkeeping).

Endianness note: the *flat* amplitude ordering here is big-endian (qubit 0 is the MSB
axis), which differs from Qiskit's little-endian statevector. That is irrelevant to the
equivalence check because we compare EXPECTATION VALUES, not raw amplitudes: gate on
qubit ``q`` and ``<Z_q>`` both use the same logical axis ``1+q``, and observable
eigenvalues are read straight off each Pauli's per-qubit ``z`` mask (index = qubit),
so the mapping is endianness-safe by construction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# ---------------------------------------------------------------------------
# Batched single-qubit rotation / gate matrices (Qiskit conventions).
# Each returns a (B, 2, 2) complex128 tensor from a (B,) angle.
# ---------------------------------------------------------------------------
_CDTYPE = torch.complex128


def _rx(theta: torch.Tensor) -> torch.Tensor:
    h = theta / 2
    c = torch.cos(h).to(_CDTYPE)
    s = torch.sin(h).to(_CDTYPE)
    i = torch.tensor(1j, dtype=_CDTYPE)
    return torch.stack([torch.stack([c, -i * s], -1),
                        torch.stack([-i * s, c], -1)], -2)


def _ry(theta: torch.Tensor) -> torch.Tensor:
    h = theta / 2
    c = torch.cos(h).to(_CDTYPE)
    s = torch.sin(h).to(_CDTYPE)
    return torch.stack([torch.stack([c, -s], -1),
                        torch.stack([s, c], -1)], -2)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    h = theta / 2
    z = torch.zeros_like(h, dtype=_CDTYPE)
    e_m = torch.exp(-1j * h.to(_CDTYPE))
    e_p = torch.exp(1j * h.to(_CDTYPE))
    return torch.stack([torch.stack([e_m, z], -1),
                        torch.stack([z, e_p], -1)], -2)


_ROT = {"rx": _rx, "ry": _ry, "rz": _rz}
# Static single-qubit gates (no parameter), as fixed (2, 2) complex128 matrices.
_SQRT2 = 2.0 ** 0.5
_STATIC = {
    "h": torch.tensor([[1, 1], [1, -1]], dtype=_CDTYPE) / _SQRT2,
    "x": torch.tensor([[0, 1], [1, 0]], dtype=_CDTYPE),
    "z": torch.tensor([[1, 0], [0, -1]], dtype=_CDTYPE),
    "y": torch.tensor([[0, -1j], [1j, 0]], dtype=_CDTYPE),
}
_SUPPORTED = set(_ROT) | set(_STATIC) | {"cx", "cz", "barrier", "id"}


def _diag_eigs(observable: SparsePauliOp, n_qubits: int, dim: int) -> torch.Tensor:
    """Diagonal (n=dim,) real eigenvalues of a Z-only ``SparsePauliOp``.

    Read straight off each Pauli's per-qubit ``z`` mask (``z[q]`` == Z on logical
    qubit ``q``), so it is endianness-safe. Raises if any X/Y (non-diagonal) appears.
    """
    idx = torch.arange(dim)
    # bit_q(k): the value of logical qubit q in basis state k, big-endian flat index.
    bits = [((idx >> (n_qubits - 1 - q)) & 1).to(torch.float64) for q in range(n_qubits)]
    total = torch.zeros(dim, dtype=torch.float64)
    sp = observable if isinstance(observable, SparsePauliOp) else SparsePauliOp(observable)
    for pauli, coeff in zip(sp.paulis, sp.coeffs):
        if bool(np.any(pauli.x)):
            raise NotImplementedError(
                "torch_sv supports only Z-diagonal observables (I/Z Paulis); "
                f"got a term with X/Y: {pauli}"
            )
        d = torch.ones(dim, dtype=torch.float64)
        for q in range(n_qubits):
            if bool(pauli.z[q]):
                d = d * (1.0 - 2.0 * bits[q])   # +1 if qubit q is |0>, -1 if |1>
        if abs(float(np.imag(coeff))) > 1e-12:
            raise NotImplementedError(f"complex observable coefficient {coeff}")
        total = total + float(np.real(coeff)) * d
    return total


class TorchStatevectorQNN(nn.Module):
    """Differentiable statevector evaluator for a fixed Qiskit circuit.

    Mirrors the public surface used by ``VQCQFunction``: it is an ``nn.Module`` holding
    the variational ``weights`` as an ``nn.Parameter`` and, given a batch of *encoded*
    inputs (``lam * norm(states)``, exactly what the ``EstimatorQNN`` receives), returns
    ``(B, n_observables)`` expectation values in the observable's range.

    Args:
        circuit: the ``QuantumCircuit`` from ``build_ansatz`` (source of truth).
        observables: list of ``SparsePauliOp`` (Z-diagonal), one per readout.
        input_params / weight_params: the circuit's input and weight ``Parameter`` lists
            (as returned by ``build_ansatz``), defining index order.
        initial_weights: (n_weights,) tensor for reproducible init shared with the
            Qiskit path so both backends start byte-identical.
    """

    def __init__(self, circuit: QuantumCircuit, observables, input_params, weight_params,
                 initial_weights: torch.Tensor):
        super().__init__()
        self.n_qubits = circuit.num_qubits
        self.dim = 1 << self.n_qubits
        self.n_obs = len(observables)

        in_index = {p: i for i, p in enumerate(input_params)}
        w_index = {p: i for i, p in enumerate(weight_params)}
        self.n_inputs = len(input_params)
        self.n_weights = len(weight_params)

        # --- parse the circuit ONCE into (name, qubits, param_ref) ops ---
        self._ops = self._parse(circuit, in_index, w_index)

        # variational weights (shared init with the Qiskit backend)
        w0 = torch.as_tensor(initial_weights, dtype=torch.float64).reshape(-1)
        assert w0.numel() == self.n_weights, (
            f"initial_weights has {w0.numel()} entries, circuit needs {self.n_weights}"
        )
        self.weights = nn.Parameter(w0)

        eigs = torch.stack([_diag_eigs(o, self.n_qubits, self.dim) for o in observables], 0)
        self.register_buffer("eigs", eigs)                      # (n_obs, dim) float64

    # -- parsing -------------------------------------------------------------
    def _parse(self, circuit: QuantumCircuit, in_index: dict, w_index: dict) -> list:
        ops = []
        for instr in circuit.data:
            name = instr.operation.name.lower()
            if name in ("barrier", "id"):
                continue
            if name not in _SUPPORTED:
                raise NotImplementedError(
                    f"torch_sv cannot interpret gate {name!r}. Supported: {sorted(_SUPPORTED)}. "
                    "Add it here (it must go THROUGH build_circuit)."
                )
            qubits = [circuit.find_bit(q).index for q in instr.qubits]
            ref = None
            if name in _ROT:
                params = instr.operation.params
                assert len(params) == 1, f"{name} expects 1 param, got {params}"
                ref = self._param_ref(params[0], in_index, w_index)
            ops.append((name, qubits, ref))
        return ops

    @staticmethod
    def _param_ref(p, in_index: dict, w_index: dict):
        """Resolve a gate parameter to ('input', i) / ('weight', i) / ('const', value).

        ``build_circuit`` binds each rotation to a single plain Parameter (input scaling
        `lam` lives OUTSIDE the circuit), so anything else is rejected loudly."""
        if isinstance(p, ParameterExpression) and len(p.parameters) == 0:
            return ("const", float(p))
        if p in in_index:
            return ("input", in_index[p])
        if p in w_index:
            return ("weight", w_index[p])
        raise NotImplementedError(
            f"gate parameter {p!r} is not a plain input/weight Parameter. "
            "Compound expressions must be handled torch-side (like `lam`), not in a gate."
        )

    # -- statevector primitives ----------------------------------------------
    def _apply_1q(self, state: torch.Tensor, U: torch.Tensor, q: int) -> torch.Tensor:
        st = state.movedim(1 + q, 1)
        st = torch.einsum("bij,bj...->bi...", U, st)
        return st.movedim(1, 1 + q)

    def _apply_cx(self, P: torch.Tensor, c: int, t: int) -> torch.Tensor:
        P = P.clone()
        idx = [slice(None)] * (1 + self.n_qubits)
        idx[1 + c] = 1                                # control qubit == |1>
        tt = t if t < c else t - 1                    # target axis after control axis fixed
        P[tuple(idx)] = P[tuple(idx)].flip(dims=[1 + tt])
        return P

    def _apply_cz(self, P: torch.Tensor, a: int, b: int) -> torch.Tensor:
        P = P.clone()
        idx = [slice(None)] * (1 + self.n_qubits)
        idx[1 + a] = 1
        idx[1 + b] = 1                                # both qubits |1> -> phase -1
        P[tuple(idx)] = -P[tuple(idx)]
        return P

    def probabilities(self, encoded: torch.Tensor) -> torch.Tensor:
        """Run the circuit and return the computational-basis measurement distribution
        ``|amp|^2`` of shape (B, dim). This IS a quantum measurement distribution — used
        by the distributional (C51) head to parameterize the value-atom distribution."""
        encoded = encoded.to(torch.float64)
        B = encoded.shape[0]
        w = self.weights
        # |0...0>
        P = torch.zeros(B, self.dim, dtype=_CDTYPE, device=encoded.device)
        P[:, 0] = 1.0
        P = P.reshape(B, *([2] * self.n_qubits))

        for name, qubits, ref in self._ops:
            if name in _ROT:
                kind, i = ref
                if kind == "input":
                    angle = encoded[:, i]
                elif kind == "weight":
                    angle = w[i].expand(B)
                else:                                 # const
                    angle = torch.full((B,), ref[1], dtype=torch.float64, device=encoded.device)
                P = self._apply_1q(P, _ROT[name](angle), qubits[0])
            elif name in _STATIC:
                U = _STATIC[name].to(encoded.device).expand(B, 2, 2)
                P = self._apply_1q(P, U, qubits[0])
            elif name == "cx":
                P = self._apply_cx(P, qubits[0], qubits[1])
            elif name == "cz":
                P = self._apply_cz(P, qubits[0], qubits[1])

        psi = P.reshape(B, self.dim)
        return (psi.conj() * psi).real                # |amp|^2  (B, dim) float64

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        prob = self.probabilities(encoded)
        return torch.einsum("bk,jk->bj", prob, self.eigs)   # (B, n_obs)
