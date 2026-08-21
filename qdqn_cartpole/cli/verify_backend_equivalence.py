"""Verify a backend evaluates the circuit we think it does.

A backend migration's most dangerous failure is silent: the flat weight vector
gets permuted relative to the gates (qiskit orders parameters alphabetically by
default, so `ry10` sorts between `ry1` and `ry2`), or the Pauli-string
endianness flips, and everything still runs -- it just optimizes a different
circuit. Neither the shape check nor the gradient check in vqc.py's smoke test
would notice.

This script pins that down against a reference that depends on no ML framework
at all: bind the parameters into the qiskit circuit by name, evolve a
Statevector, and take the expectation value directly. Any backend that
disagrees with that has an ordering or convention bug.

Two checks, both against that same qiskit-only reference:

  forward   backend expectation values vs Statevector.expectation_value.
  gradient  (--check-grad) backend autograd/adjoint gradients vs CENTRAL
            FINITE DIFFERENCES taken on the Statevector reference. This matters
            more than it looks for the v3 torch_sv backend: matching forward
            values only proves the circuit is right, and every wrong-gradient
            failure this project has hit (dead lam, permuted weight vector) is
            invisible in the forward pass. Finite differences share no code
            with the backend's differentiation, so agreement is real evidence.

Usage:
    python scripts/verify_backend_equivalence.py --backend torch_sv --check-grad
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..models.base import normalize_observation
from ..models.vqc import VQCQFunction, build_circuit


def reference_expectations(
    states: torch.Tensor,
    lam: torch.Tensor,
    circuit_weights: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    reuploading: bool,
    observables,
    per_layer_encoding: bool = False,
) -> np.ndarray:
    """Ground truth: plain qiskit Statevector, no EstimatorQNN, no torch."""
    circuit, input_params, weight_params = build_circuit(
        n_qubits, n_layers, reuploading, per_layer_encoding=per_layer_encoding
    )
    ops = [SparsePauliOp(o) for o in observables]

    normalized = normalize_observation(states)
    if len(input_params) != normalized.shape[1]:
        normalized = normalized.tile(1, len(input_params) // normalized.shape[1])
    scaled = (lam * normalized).detach().numpy()
    weights = circuit_weights.detach().numpy()

    out = np.zeros((states.shape[0], len(ops)))
    for b in range(states.shape[0]):
        binding = {p: float(scaled[b, i]) for i, p in enumerate(input_params)}
        binding.update({p: float(weights[i]) for i, p in enumerate(weight_params)})
        bound = circuit.assign_parameters(binding)
        sv = Statevector.from_instruction(bound)
        for j, op in enumerate(ops):
            out[b, j] = np.real(sv.expectation_value(op))
    return out


def _apply_output_head(raw: np.ndarray, w, model) -> np.ndarray:
    """Everything VQCQFunction.forward does AFTER the circuit, mirrored here.

    The circuit returns expectation values in [-1, 1]; the model then optionally
    maps them to [0, 1] and scales by w. This has to track forward() exactly --
    when output_rescaling was added, this reference did not follow and the check
    reported a 2.576 mismatch against a simulator that was in fact correct.
    A reference that silently drifts from the thing it certifies is worse than
    no reference, so the post-circuit head lives in one function used by both
    the forward and the finite-difference checks.
    """
    if getattr(model, "output_rescaling", False):
        raw = (raw + 1.0) / 2.0
    return raw * w.detach().numpy()


def finite_difference_gradients(
    model, states: torch.Tensor, eps: float = 1e-3
) -> dict[str, np.ndarray]:
    """d/dp sum(model(states)) for every parameter group, by central differences
    on the qiskit Statevector reference.

    Deliberately does not touch the backend or torch autograd -- the whole point
    is that this number is produced by a different mechanism than the one under
    test. O(n_params) statevector evaluations, so keep the batch small.
    """
    lam0 = model.lam.detach().clone()
    w0 = model.w.detach().clone()
    weights0 = model.circuit_weights()

    def total(lam, weights, w):
        raw = reference_expectations(
            states, lam, weights, model.n_qubits, model.n_layers,
            model.reuploading, model.observables,
            getattr(model, "per_layer_encoding", False),
        )
        return float(np.sum(_apply_output_head(raw, w, model)))

    grads: dict[str, np.ndarray] = {}
    for name, base in (("lam", lam0), ("circuit", weights0), ("w", w0)):
        g = np.zeros(base.numel())
        for i in range(base.numel()):
            plus, minus = base.clone(), base.clone()
            plus[i] += eps
            minus[i] -= eps
            if name == "lam":
                hi, lo = total(plus, weights0, w0), total(minus, weights0, w0)
            elif name == "circuit":
                hi, lo = total(lam0, plus, w0), total(lam0, minus, w0)
            else:
                hi, lo = total(lam0, weights0, plus), total(lam0, weights0, minus)
            g[i] = (hi - lo) / (2 * eps)
        grads[name] = g
    return grads


def check_gradients(model, states: torch.Tensor, tol: float) -> bool:
    model.zero_grad(set_to_none=True)
    model(states).sum().backward()
    analytic = {
        "lam": model.lam.grad.detach().numpy().ravel(),
        "circuit": model.vqc.weight.grad.detach().numpy().ravel(),
        "w": model.w.grad.detach().numpy().ravel(),
    }
    reference = finite_difference_gradients(model, states)

    print("\n=== gradient check vs central finite differences on the qiskit reference ===")
    ok = True
    for name in ("lam", "circuit", "w"):
        a, r = analytic[name], reference[name]
        # Relative to the gradient's own scale: the circuit group spans a wide
        # range of magnitudes and a bare absolute tolerance would be dominated
        # by whichever entry happens to be largest.
        scale = max(1.0, float(np.max(np.abs(r))))
        err = float(np.max(np.abs(a - r))) / scale
        status = "OK  " if err <= tol else "FAIL"
        print(
            f"  {status} {name:8s} n={a.size:3d}  max rel err {err:.2e}  "
            f"|analytic|={np.linalg.norm(a):.6f}  |reference|={np.linalg.norm(r):.6f}"
        )
        ok = ok and err <= tol
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="torch_sv")
    parser.add_argument("--per-layer-encoding", action="store_true")
    parser.add_argument("--gradient-method", default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="also check torch_sv's fast stage-grouped path against its gate-at-a-time reference",
    )
    parser.add_argument(
        "--check-grad",
        action="store_true",
        help="also check gradients against finite differences (slow: O(n_params) sims)",
    )
    parser.add_argument(
        "--grad-tol",
        type=float,
        default=2e-3,
        help="relative tolerance for the gradient check. Loose on purpose: the "
        "reference is a finite difference in float32, so its own truncation and "
        "round-off error is ~1e-4 -- a tighter bound would flag the reference, "
        "not the backend.",
    )
    args = parser.parse_args()

    method = args.gradient_method or ("param_shift" if args.backend == "qiskit_ml" else "adjoint")

    torch.manual_seed(0)
    model = VQCQFunction(backend=args.backend, gradient_method=method, seed=0,
                         per_layer_encoding=args.per_layer_encoding)

    # Spread across the state space, not clustered near zero: an ordering bug on
    # a near-identity circuit can hide when every input is tiny.
    g = torch.Generator().manual_seed(1)
    states = torch.rand(args.batch, 4, generator=g) * 4.0 - 2.0

    # Give lam and w non-trivial values so a mis-scaling cannot cancel out.
    with torch.no_grad():
        model.lam.copy_(torch.linspace(0.4, 1.6, model.lam.numel()))
        model.w.copy_(torch.tensor([2.0, -3.0]))

    got = model(states).detach().numpy()

    raw_ref = reference_expectations(
        states,
        model.lam,
        model.circuit_weights(),
        model.n_qubits,
        model.n_layers,
        model.reuploading,
        model.observables,
        getattr(model, "per_layer_encoding", False),
    )
    ref = _apply_output_head(raw_ref, model.w, model)

    max_abs = float(np.max(np.abs(got - ref)))
    print(f"backend:      {args.backend} (gradient={method})")
    print(f"observables:  {list(model.observables)}")
    print(f"reuploading:  {model.reuploading}   n_layers: {model.n_layers}")
    print(f"batch:        {args.batch}")
    print(f"max |backend - statevector reference|: {max_abs:.3e}   (tol {args.tol:g})")
    print("\nfirst 3 rows")
    print("  backend:  ", np.array2string(got[:3], precision=6, suppress_small=True))
    print("  reference:", np.array2string(ref[:3], precision=6, suppress_small=True))

    if max_abs > args.tol:
        print(
            f"\nMISMATCH: backend {args.backend!r} does not evaluate the reference circuit.\n"
            f"Most likely causes, in order: (1) the flat weight vector is permuted relative "
            f"to CIRCUIT_WEIGHT_ORDER, (2) Pauli-string endianness is flipped, (3) the "
            f"encoding gates are placed on the wrong layers (reuploading)."
        )
        return 1

    print(f"\nOK -- {args.backend} forward matches the statevector reference to {args.tol:g}")

    if args.check_reference and args.backend == "torch_sv":
        # torch_sv's `simulate` is a performance rewrite of `_simulate_reference`
        # (stage grouping, batched matrix construction, CX runs precomposed into
        # a permutation). Checking them against each other localizes a failure:
        # if both disagree with Qiskit the circuit is wrong, if only the fast one
        # does the optimization is wrong.
        from ..models import torch_statevector as tsv

        scaled = (model.lam * normalize_observation(states)).detach()
        w_ref = model.circuit_weights()
        fast = tsv.simulate(model.vqc._compiled, scaled, w_ref, model.vqc._obs)
        slow = tsv._simulate_reference(model.vqc._compiled, scaled, w_ref, model.vqc._obs)
        diff = float((fast - slow).abs().max())
        n_ops, n_stages = len(model.vqc._compiled.ops), len(model.vqc._compiled.stages)
        print(f"\nfast path vs gate-at-a-time reference: max |diff| {diff:.3e} "
              f"({n_ops} ops -> {n_stages} stages)")
        if diff > args.tol:
            print("MISMATCH: the stage-grouping optimization changed the result.")
            return 1

    if args.check_grad and not check_gradients(model, states, args.grad_tol):
        print(
            f"\nGRADIENT MISMATCH: {args.backend!r} evaluates the right circuit but "
            f"differentiates it wrongly. Forward agreement above rules out an ordering "
            f"or endianness bug, so look at the differentiation path itself."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
