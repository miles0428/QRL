"""Verify the TFQ/Cirq model against an independent Qiskit Statevector.

WHY THIS EXISTS
---------------
The Qiskit arm (src/) is checked against a reference that shares no code with
it: qiskit's own Statevector for the forward pass, and central finite
differences on that Statevector for the gradients (see
scripts/verify_backend_equivalence.py). The TFQ arm arrived with only a smoke
test -- right output shape, non-zero gradients -- which proves the plumbing
works but proves nothing about *what circuit* is being evaluated.

That gap matters here more than usual, because the single most dangerous bug in
this codebase's history is silent and identical in both frameworks: a symbol
ordering mismatch. `ControlledPQC` expects parameter values in the circuit's
sorted-symbol order, and ReUploadingPQC permutes them with `self.indices` to
match. If that permutation is wrong, every angle lands on the wrong gate --
forward runs, gradients flow, loss decreases, and the model trains a circuit
nobody designed. Exactly the failure qiskit's alphabetical parameter ordering
causes on the other arm.

So this rebuilds the SAME circuit independently in Qiskit, binds the SAME
parameter values, and compares expectation values. Qiskit is used here purely as
an offline oracle -- it is not in the TFQ training path and this script is not
imported by it.

Checks:
  forward   TFQ model output vs Qiskit Statevector.expectation_value
  gradient  TFQ autodiff vs central finite differences on that Statevector

Run in WSL:
    cd /mnt/c/Users/tseng/quantum\\ hackthon\\ 2026/qdqn-cartpole
    ~/tfq/.venv/bin/python tfq/verify_against_qiskit.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from tfq.model import W_IN, W_OUT, W_VAR, build_q_model


def qiskit_expectations(states, theta, lmbd, n_qubits, n_layers):
    """Ground truth: the tutorial's circuit rebuilt in Qiskit, evolved exactly.

    Rebuilt from the tutorial's *description* rather than translated from the
    Cirq object, so a mistake in the Cirq construction cannot propagate into its
    own reference. The layer order is: variational RX,RY,RZ -> CZ ring ->
    encoding RX, repeated n_layers times, then one final variational layer.

    Qubit convention: cirq.GridQubit.rect(1, n) gives qubits left to right, and
    observables are Z(q0)*Z(q1) and Z(q2)*Z(q3). Qiskit Pauli strings are
    little-endian (leftmost character = highest qubit index), so Z0*Z1 is
    "IIZZ" and Z2*Z3 is "ZZII".
    """
    theta = np.asarray(theta).reshape((n_layers + 1, n_qubits, 3))
    lmbd = np.asarray(lmbd).reshape((n_layers, n_qubits))
    ops = [SparsePauliOp("IIZZ"), SparsePauliOp("ZZII")]  # Z0*Z1, Z2*Z3

    out = np.zeros((len(states), len(ops)))
    for b, state in enumerate(states):
        qc = QuantumCircuit(n_qubits)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qc.rx(theta[layer, q, 0], q)
                qc.ry(theta[layer, q, 1], q)
                qc.rz(theta[layer, q, 2], q)
            for q in range(n_qubits - 1):
                qc.cz(q, q + 1)
            if n_qubits != 2:
                qc.cz(0, n_qubits - 1)
            for q in range(n_qubits):
                # tanh(lambda * x) -- the squashing lives outside the circuit in
                # both implementations; the circuit only ever sees a number.
                qc.rx(np.tanh(lmbd[layer, q] * state[q]), q)
        for q in range(n_qubits):
            qc.rx(theta[n_layers, q, 0], q)
            qc.ry(theta[n_layers, q, 1], q)
            qc.rz(theta[n_layers, q, 2], q)

        sv = Statevector.from_instruction(qc)
        for j, op in enumerate(ops):
            out[b, j] = np.real(sv.expectation_value(op))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--grad-tol", type=float, default=2e-3)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--skip-grad", action="store_true")
    args = p.parse_args()

    tf.random.set_seed(0)
    np.random.seed(0)

    model = build_q_model(args.n_qubits, args.n_layers, 2)
    tv = model.trainable_variables

    # Non-trivial values everywhere, so a mis-scaling cannot cancel out, and
    # states spread across the space -- an ordering bug can hide near zero.
    theta = np.random.uniform(0, 2 * np.pi, size=tv[W_VAR].shape).astype(np.float32)
    lmbd = np.linspace(0.4, 1.6, int(tv[W_IN].shape[0])).astype(np.float32)
    w = np.array([[2.0, -3.0]], dtype=np.float32)
    tv[W_VAR].assign(theta)
    tv[W_IN].assign(lmbd)
    tv[W_OUT].assign(w)

    states = np.random.uniform(-2.0, 2.0, size=(args.batch, args.n_qubits)).astype(np.float32)

    got = model([tf.convert_to_tensor(states)]).numpy()
    raw_ref = qiskit_expectations(states, theta[0], lmbd, args.n_qubits, args.n_layers)
    ref = ((raw_ref + 1.0) / 2.0) * w  # the Rescaling layer, applied outside the circuit

    max_abs = float(np.max(np.abs(got - ref)))
    print(f"circuit: {args.n_qubits} qubits, {args.n_layers} layers "
          f"({int(np.prod(tv[W_VAR].shape))} variational + {int(tv[W_IN].shape[0])} lambda + 2 w)")
    print(f"batch:   {args.batch}")
    print(f"max |TFQ - qiskit Statevector|: {max_abs:.3e}   (tol {args.tol:g})")
    print("\nfirst 3 rows")
    print("  TFQ:      ", np.array2string(got[:3], precision=6, suppress_small=True))
    print("  reference:", np.array2string(ref[:3], precision=6, suppress_small=True))

    if max_abs > args.tol:
        print(
            "\nMISMATCH: the TFQ model does not evaluate the reference circuit.\n"
            "Most likely, in order: (1) ReUploadingPQC.indices permutes symbols wrongly, so\n"
            "angles land on the wrong gates; (2) the encoding gates sit in the wrong position\n"
            "within a layer; (3) the observables map to the wrong qubits."
        )
        return 1
    print(f"\nOK -- TFQ forward matches the qiskit statevector reference to {args.tol:g}")

    if args.skip_grad:
        return 0

    # Gradients. Forward agreement proves the circuit is right; it says nothing
    # about differentiation, and every wrong-gradient bug in this project has
    # been invisible in the forward pass.
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(model([tf.convert_to_tensor(states)]))
    grads = tape.gradient(loss, tv)

    def total(theta_v, lmbd_v, w_v):
        raw = qiskit_expectations(states, theta_v[0], lmbd_v, args.n_qubits, args.n_layers)
        return float(np.sum(((raw + 1.0) / 2.0) * w_v))

    print("\n=== gradient check vs central finite differences on the qiskit reference ===")
    ok = True
    for label, idx, base in (("theta", W_VAR, theta), ("lambda", W_IN, lmbd), ("w", W_OUT, w)):
        flat = base.reshape(-1).astype(np.float64)
        fd = np.zeros(flat.size)
        for i in range(flat.size):
            hi, lo = flat.copy(), flat.copy()
            hi[i] += args.eps
            lo[i] -= args.eps
            hi_r, lo_r = hi.reshape(base.shape), lo.reshape(base.shape)
            if label == "theta":
                fd[i] = (total(hi_r, lmbd, w) - total(lo_r, lmbd, w)) / (2 * args.eps)
            elif label == "lambda":
                fd[i] = (total(theta, hi_r, w) - total(theta, lo_r, w)) / (2 * args.eps)
            else:
                fd[i] = (total(theta, lmbd, hi_r) - total(theta, lmbd, lo_r)) / (2 * args.eps)

        analytic = grads[idx].numpy().reshape(-1)
        scale = max(1.0, float(np.max(np.abs(fd))))
        err = float(np.max(np.abs(analytic - fd))) / scale
        status = "OK  " if err <= args.grad_tol else "FAIL"
        print(f"  {status} {label:7s} n={analytic.size:3d}  max rel err {err:.2e}  "
              f"|analytic|={np.linalg.norm(analytic):.6f}  |reference|={np.linalg.norm(fd):.6f}")
        ok = ok and err <= args.grad_tol

    if not ok:
        print("\nGRADIENT MISMATCH: right circuit, wrong differentiation.")
        return 1
    print("\nOK -- TFQ gradients match finite differences on the qiskit reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
