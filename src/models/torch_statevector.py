"""Batched statevector simulator for a Qiskit circuit, written in torch.

WHY THIS EXISTS
---------------
Profiling the pre-v3 training path showed 91% of wall-clock inside the
framework's gradient machinery, not inside anything intrinsic to the physics:
2.16 s per gradient step for a *four qubit* circuit, whose entire state is 16
complex amplitudes. Every Qiskit-side option we measured keeps that overhead:

  qiskit-ML + SPSA          2.16  s/grad step  (the pre-v3 baseline)
  qiskit-ML + param-shift    slower still -- 2 circuit evaluations per parameter
  qiskit-torch-module        1.42  s/grad step forced sequential

qiskit-torch-module is the right idea (adjoint gradients, batch parallel) but it
parallelizes with `multiprocessing.Process`. On Windows that is `spawn`, so each
forward and backward launches fresh interpreters that re-import torch and
qiskit; a single CartPole episode did not finish in ten minutes. Forced to
`num_threads=1` it runs but only reaches 1.42 s/step. That is a platform
property, not a defect -- on Linux (fork) its published fan-out works.

The cost being avoided is per-circuit-evaluation Python and dispatch overhead,
which is enormous relative to a 16-amplitude state. So this module evaluates
the circuit directly as torch tensor algebra and lets autograd differentiate
it. Gradients are exact (not SPSA's stochastic estimate) and cost one backward
pass, not the 2*n_params circuit evaluations parameter-shift needs.

QISKIT REMAINS THE SOURCE OF TRUTH
----------------------------------
This is a faster evaluator for a Qiskit circuit, not a reimplementation of one.

  - The circuit is a `qiskit.QuantumCircuit`, built by `vqc.build_circuit`, and
    is still what gets drawn for the circuit figure and transpiled for hardware.
    `compile_circuit` *reads* `circuit.data` -- the gate list, qubit indices and
    parameter bindings all come from the Qiskit object, so a change to
    `build_circuit` propagates here automatically and cannot silently desync.
  - Observables stay `qiskit.quantum_info.SparsePauliOp`, and the operator
    matrix used for the expectation value is produced by Qiskit's own
    `SparsePauliOp.to_matrix()`. Qiskit's little-endian Pauli-string convention
    ("ZZII" is Z on qubits 3 and 2) is therefore inherited rather than
    reimplemented -- the class of bug where a hand-rolled observable silently
    reads the wrong qubits cannot occur here.
  - Amplitude ordering matches `qiskit.quantum_info.Statevector`: index i is the
    basis state whose qubit q holds bit (i >> q) & 1.
  - Finite-shot and noisy evaluation still runs on qiskit-aer, unchanged. This
    path is statevector-only and exact, so it cannot model shot noise.

`scripts/verify_backend_equivalence.py` asserts agreement with Qiskit's own
`Statevector.expectation_value` to ~1e-6 over random inputs; that check is what
licenses using this for training.

SCOPE. Supports exactly the gate set `vqc.build_circuit` emits (parameterized
rx/ry/rz, cx) plus a few common constants. Anything else raises at compile
time rather than being silently skipped -- an unsupported gate that was quietly
ignored would produce a plausible-looking wrong answer.
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# Constant (non-parameterized) single-qubit gates, in Qiskit's convention.
_SQRT2 = float(np.sqrt(2.0))
_CONST_1Q: dict[str, np.ndarray] = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / _SQRT2,
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "id": np.eye(2, dtype=complex),
}
_ROTATIONS = ("rx", "ry", "rz")


class CompiledCircuit:
    """A Qiskit circuit reduced to a flat op list this simulator can execute.

    Each op is one of:
      ("rot", axis, qubit, source, index)  source: "enc" | "var" | "const"
      ("const1q", qubit, matrix)
      ("cx", control, target)

    `source="enc"` means the rotation angle is element `index` of the per-sample
    encoding input (so it varies across the batch); `"var"` means element
    `index` of the shared weight vector; `"const"` means the angle was already
    bound to a number in the circuit.
    """

    def __init__(self, n_qubits: int, ops: list, n_encoding: int, n_weights: int):
        self.n_qubits = n_qubits
        self.ops = ops
        self.n_encoding = n_encoding
        self.n_weights = n_weights
        self.dim = 2**n_qubits


def compile_circuit(
    circuit: QuantumCircuit,
    encoding_params: list,
    variational_params: list,
) -> CompiledCircuit:
    """Read a Qiskit circuit into an executable op list.

    `encoding_params` and `variational_params` are explicit ordered lists, never
    `circuit.parameters` -- that property is sorted alphabetically, which would
    map weight vector index i onto whichever gate happens to sort into position
    i. Index i here means exactly what `vqc.CIRCUIT_WEIGHT_ORDER` says it means.
    """
    enc_index = {p: i for i, p in enumerate(encoding_params)}
    var_index = {p: i for i, p in enumerate(variational_params)}
    if len(enc_index) != len(encoding_params) or len(var_index) != len(variational_params):
        raise ValueError("duplicate entries in encoding_params/variational_params")
    overlap = set(enc_index) & set(var_index)
    if overlap:
        raise ValueError(f"parameters declared as both encoding and variational: {overlap}")

    ops: list = []
    for instruction in circuit.data:
        op = instruction.operation
        name = op.name
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]

        if name in _ROTATIONS:
            (angle,) = op.params
            if isinstance(angle, Parameter):
                if angle in enc_index:
                    ops.append(("rot", name, qubits[0], "enc", enc_index[angle]))
                elif angle in var_index:
                    ops.append(("rot", name, qubits[0], "var", var_index[angle]))
                else:
                    raise ValueError(
                        f"{name} on qubit {qubits[0]} uses parameter {angle} which is in "
                        f"neither encoding_params nor variational_params"
                    )
            else:
                try:
                    ops.append(("rot", name, qubits[0], "const", float(angle)))
                except TypeError as exc:
                    raise ValueError(
                        f"{name} angle {angle!r} is a parameter *expression*. Only bare "
                        f"parameters are supported: an expression such as lam*x would in "
                        f"any case break adjoint/parameter-shift differentiation, which is "
                        f"why input scaling lives outside the circuit (see vqc.py)."
                    ) from exc
        elif name == "cx":
            ops.append(("cx", qubits[0], qubits[1]))
        elif name in _CONST_1Q:
            ops.append(("const1q", qubits[0], _CONST_1Q[name]))
        elif name in ("barrier", "delay"):
            continue
        else:
            raise ValueError(
                f"gate {name!r} is not supported by the torch statevector backend. "
                f"Supported: {_ROTATIONS}, cx, {sorted(_CONST_1Q)}. Add it here rather "
                f"than letting it be skipped silently."
            )

    return CompiledCircuit(circuit.num_qubits, ops, len(encoding_params), len(variational_params))


def observable_matrices(observables, n_qubits: int) -> torch.Tensor:
    """Dense matrices for the observables, straight from Qiskit.

    Returned as a single [n_obs, D, D] complex tensor. Using Qiskit's own
    `to_matrix()` is the point: the little-endian Pauli-string convention comes
    from Qiskit rather than being re-derived here.
    """
    mats = []
    for obs in observables:
        spo = obs if isinstance(obs, SparsePauliOp) else SparsePauliOp(obs)
        if spo.num_qubits != n_qubits:
            raise ValueError(
                f"observable {obs} acts on {spo.num_qubits} qubits, circuit has {n_qubits}"
            )
        mats.append(np.asarray(spo.to_matrix()))
    return torch.as_tensor(np.stack(mats), dtype=torch.complex64)


def _rotation_matrix(axis: str, theta: torch.Tensor) -> torch.Tensor:
    """Batched single-qubit rotation, Qiskit's convention. theta: [B] -> [B,2,2]."""
    half = theta / 2
    cos = torch.cos(half)
    sin = torch.sin(half)
    zero = torch.zeros_like(cos)

    if axis == "ry":
        real = torch.stack(
            [torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2
        )
        return torch.complex(real, torch.zeros_like(real))
    if axis == "rz":
        # diag(exp(-i theta/2), exp(+i theta/2))
        top = torch.complex(cos, -sin)
        bot = torch.complex(cos, sin)
        zc = torch.complex(zero, zero)
        return torch.stack([torch.stack([top, zc], -1), torch.stack([zc, bot], -1)], -2)
    if axis == "rx":
        c = torch.complex(cos, zero)
        s = torch.complex(zero, -sin)
        return torch.stack([torch.stack([c, s], -1), torch.stack([s, c], -1)], -2)
    raise ValueError(f"unknown rotation axis {axis!r}")


def _apply_1q(state: torch.Tensor, mat: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Apply a [B,2,2] (or [2,2]) matrix to `qubit`.

    `state` is [B, 2, 2, ..., 2]; axis 1 is the highest qubit index, so qubit q
    lives on axis 1 + (n_qubits - 1 - q). That is exactly the bit layout of
    Statevector's amplitude index (bit q of i is qubit q).
    """
    axis = 1 + (n_qubits - 1 - qubit)
    moved = state.movedim(axis, -1)
    shape = moved.shape
    flat = moved.reshape(shape[0], -1, 2)
    if mat.dim() == 2:
        # out[b,k,i] = sum_j mat[i,j] * flat[b,k,j]
        out = flat @ mat.transpose(0, 1)
    else:
        out = torch.einsum("bij,bkj->bki", mat, flat)
    return out.reshape(shape).movedim(-1, axis)


def _apply_cx(state: torch.Tensor, control: int, target: int, n_qubits: int) -> torch.Tensor:
    """CX: flip `target` on the half of the state where `control` is 1."""
    ac = 1 + (n_qubits - 1 - control)
    at = 1 + (n_qubits - 1 - target)
    moved = state.movedim((ac, at), (-2, -1))
    kept = moved[..., 0, :]
    flipped = moved[..., 1, :].flip(-1)
    return torch.stack((kept, flipped), dim=-2).movedim((-2, -1), (ac, at))


def simulate(
    compiled: CompiledCircuit,
    encoding: torch.Tensor,
    weights: torch.Tensor,
    obs_matrices: torch.Tensor,
) -> torch.Tensor:
    """Run the circuit on a batch and return expectation values.

    encoding: [B, n_encoding] real, one angle set per batch element.
    weights:  [n_weights] real, shared across the batch.
    returns:  [B, n_obs] real.

    Everything is differentiable torch, so `.backward()` gives exact gradients
    w.r.t. both `weights` and `encoding` in a single reverse pass. The gradient
    w.r.t. `encoding` is what makes the input-scaling parameter `lam` trainable
    -- see FAILURE MODE 3 in vqc.py.
    """
    batch = encoding.shape[0]
    n = compiled.n_qubits

    if encoding.shape[1] != compiled.n_encoding:
        raise ValueError(
            f"encoding has {encoding.shape[1]} columns, circuit expects {compiled.n_encoding}"
        )
    if weights.numel() != compiled.n_weights:
        raise ValueError(
            f"weight vector has {weights.numel()} entries, circuit expects {compiled.n_weights}"
        )

    # |0...0>
    state = torch.zeros(batch, compiled.dim, dtype=torch.complex64, device=encoding.device)
    state[:, 0] = 1.0
    state = state.reshape(batch, *([2] * n))

    for op in compiled.ops:
        if op[0] == "rot":
            _, axis, qubit, source, ref = op
            if source == "enc":
                theta = encoding[:, ref]
            elif source == "var":
                theta = weights[ref].expand(batch)
            else:  # angle was already bound to a number in the circuit
                theta = torch.full((batch,), ref, dtype=encoding.dtype, device=encoding.device)
            state = _apply_1q(state, _rotation_matrix(axis, theta), qubit, n)
        elif op[0] == "cx":
            state = _apply_cx(state, op[1], op[2], n)
        else:  # const1q
            _, qubit, matrix = op
            mat = torch.as_tensor(matrix, dtype=torch.complex64, device=encoding.device)
            state = _apply_1q(state, mat, qubit, n)

    psi = state.reshape(batch, compiled.dim)
    # <psi|O|psi> for each observable. Imaginary part is zero for Hermitian O
    # up to float error; .real is the value, not a truncation.
    expvals = torch.einsum("bi,oij,bj->bo", psi.conj(), obs_matrices, psi)
    return expvals.real
