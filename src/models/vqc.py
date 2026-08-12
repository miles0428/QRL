"""Variational Quantum Circuit (VQC) Q-function for QDQN on CartPole-v1.

Circuit (per Chen et al. 2020, https://arxiv.org/abs/1907.00397; recipe follows
Skolik, Jerbi & Dunjko 2022, https://arxiv.org/abs/2103.15084):
  - 4 qubits, one per observation dimension.
  - n_layers variational layers (default 5). Each layer:
      * RY(x[q]) on every qubit -- the encoding. Placed only before layer 0
        unless `reuploading=True`, in which case it repeats at the start of
        every layer (data re-uploading, Perez-Salinas et al. 2020,
        https://doi.org/10.22331/q-2020-02-06-226). v3 default is True: the
        Skolik reference implementation (github.com/askolik/quantum_agents)
        enables it for CartPole, and single encoding leaves the circuit with
        only base-frequency expressivity.
      * RY(theta), RZ(theta) on every qubit -- trainable.
      * CNOT ring q -> (q+1) % n_qubits -- entanglement.
  - Two observables -> two expectation values in [-1, 1], one per CartPole
    action. v3 default is the two-qubit tensor products ZZII / IIZZ rather
    than single-qubit ZIII / IZII, so each observable reads more of the
    register (the paper notes performance depends critically on this choice).

BACKENDS
--------
The same circuit is evaluated through one of three interchangeable backends,
selected by `backend=`. All expose the identical `forward(x) -> [B, n_obs]`
contract, so `trainer.py` -- which must never import qiskit or any quantum
framework -- is unaffected by the choice. Backend selection is entirely
internal to this module.

  "torch_sv"   The v3 training path. The *same Qiskit circuit* -- read gate by
               gate out of `circuit.data`, with observable matrices from
               Qiskit's own SparsePauliOp.to_matrix() -- evaluated as torch
               tensor algebra, so autograd produces exact gradients in one
               reverse pass. 28 ms/grad step here vs 2.16 s on the pre-v3 path.
               See src/models/torch_statevector.py.
  "qtm"        qiskit-torch-module (Meyer et al. 2024, arXiv:2404.06314).
               Batch-parallel adjoint gradients. Was the intended v3 training
               path; its parallelism is multiprocessing.Process fan-out, which
               degenerates to per-call interpreter spawn on Windows. Retained
               as an independent gradient reference for the equivalence check
               and as an arm of the speed-comparison figure.
  "qiskit_ml"  EstimatorQNN + TorchConnector. Retained for two reasons: it is
               the only path that can be driven by a shot-based / noisy Aer
               estimator (see evaluate_finite_shot), and it is the baseline
               arm of the backend speed-comparison figure. No longer the
               training path.

All three consume the identical `QuantumCircuit` from `build_circuit` and the
identical Pauli strings; none of them redefines the circuit. That is what makes
a parameter vector trained on one meaningful when loaded into another.

THREE FAILURE MODES this module exists to avoid (see project README):

FAILURE MODE 1 (output scaling): raw <Z> in [-1, 1], but true CartPole Q-values
at gamma=0.99 reach ~100. Without a trainable output scaling `w`, the model
cannot represent the value function and training flatlines at ~reward 10.
`self.w` is a plain nn.Parameter multiplying the backend's raw output.

FAILURE MODE 2 (do not scale inside the circuit): lam_i * x_i is a product of
an input value and a trainable weight. Both parameter-shift and adjoint
differentiation require parameters to enter gates linearly, so computing that
product *inside* the circuit silently breaks gradients. Both `lam` (input
scaling) and `w` (output scaling) live in this PyTorch module, strictly outside
the backend -- `lam` multiplies the already-normalized observation in plain
torch before it is handed to the backend as ordinary numeric input; the circuit
never sees `lam` itself.

FAILURE MODE 3 (dead input gradient): EstimatorQNN defaults to
input_gradients=False, under which TorchConnector's backward never computes
d(output)/d(input) -- so `lam.grad`, which reaches `lam` only by backpropagating
through the backend's *input*, is silently None while the forward pass looks
completely normal. Fixed by input_gradients=True. The torch_sv and qtm paths
differentiate w.r.t. inputs natively, but the smoke test below checks all three
parameter groups on every backend regardless, because a backend migration is
exactly where that class of bug comes back.

PARAMETER ORDERING is a fourth, backend-specific trap. Qiskit orders circuit
parameters alphabetically by default, which would misalign the flat weight
vector with the gates it is supposed to drive (the qtm paper flags this
explicitly). `build_circuit` returns weight parameters in *circuit* order and
every backend is handed that explicit ordered list, so index i of the weight
vector always drives the same gate on every backend. `CIRCUIT_WEIGHT_ORDER`
documents the layout.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from src.models.base import QFunction, normalize_observation

# v3 defaults, from the Skolik reference implementation
# (github.com/askolik/quantum_agents, run_quantum.py) -- not our own tuning.
DEFAULT_OBSERVABLES: tuple[str, ...] = ("ZZII", "IIZZ")
# v3 intended "qtm" here. qtm's batch parallelism is multiprocessing.Process
# fan-out, which degenerates on Windows (spawn -> a fresh interpreter importing
# torch+qiskit per call); forced sequential it reaches only 1.42 s/grad step
# against a 2.16 s baseline, under the migration's 5x floor. See the backend
# note in configs/qdqn.yaml.
DEFAULT_BACKEND = "torch_sv"
DEFAULT_GRADIENT_METHOD = "adjoint"

BACKENDS = ("torch_sv", "qtm", "qiskit_ml")

# Flat weight-vector layout, per layer, in circuit order:
#   [ry(q0) ry(q1) ... ry(qN-1)  rz(q0) rz(q1) ... rz(qN-1)]
# concatenated over layers. Every backend must honour this exact ordering or
# the weight vector silently drives the wrong gates.
CIRCUIT_WEIGHT_ORDER = "per layer: all RY (qubit-major), then all RZ (qubit-major)"


def build_circuit(
    n_qubits: int = 4,
    n_layers: int = 5,
    reuploading: bool = True,
) -> tuple[QuantumCircuit, list, list]:
    """Build the RY-encoding / RY+RZ-variational / CNOT-ring circuit.

    Returns (circuit, input_params, weight_params). weight_params has
    n_layers * n_qubits * 2 entries, in circuit order (see
    CIRCUIT_WEIGHT_ORDER) -- explicitly *not* qiskit's alphabetical default.
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


def suggested_batch_size(default: int = 16) -> tuple[int, str]:
    """Batch size = 4x physical core count, per the qtm paper's benchmarks.

    Returns (batch_size, provenance) so the caller can log which branch was
    taken. Falls back to `default` when core count cannot be determined.
    """
    physical: int | None = None
    source = ""
    try:
        import psutil  # optional; not in requirements.txt

        physical = psutil.cpu_count(logical=False)
        source = "psutil physical cores"
    except Exception:
        physical = None

    if not physical:
        logical = os.cpu_count()
        if logical:
            # No psutil: assume SMT, i.e. 2 logical per physical. A GUESS, not
            # a measurement -- right on mainstream x86, merely conservative
            # (half the true count) on a non-SMT machine.
            physical = max(1, logical // 2)
            source = f"os.cpu_count()={logical} // 2 (assumed SMT)"

    if not physical:
        return default, f"core detection failed, using default {default}"
    return 4 * physical, f"4 x {physical} ({source})"


def _qiskit_observables(observables) -> list[SparsePauliOp]:
    return [SparsePauliOp(s) for s in observables]


def _initial_weights(n_weights: int, seed: int | None) -> np.ndarray:
    """Small uniform init. Deliberately narrow: a wide random init on a depth-5
    circuit starts in a barren-plateau-flat region and the run never gets going."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.1, 0.1, size=n_weights)


# --------------------------------------------------------------------------
# Backend adapters. Each is an nn.Module exposing a `weight` parameter (the
# flat circuit-weight vector, in CIRCUIT_WEIGHT_ORDER) and a
# forward(x: [B, n_qubits]) -> [B, n_obs] contract.
# --------------------------------------------------------------------------


class _QTMBackend(nn.Module):
    """qiskit-torch-module: batch-parallel adjoint gradients."""

    def __init__(
        self,
        circuit,
        input_params,
        weight_params,
        observables,
        seed=None,
        num_threads_forward: int = 0,
        num_threads_backward: int = 0,
    ):
        super().__init__()
        import qiskit_torch_module as qtm_pkg

        self.qtm = qtm_pkg.QuantumModule(
            circuit=circuit,
            # Passed as an explicit ordered list, never left to qiskit's
            # alphabetical default -- see PARAMETER ORDERING in the module
            # docstring. `weight_params` is in CIRCUIT_WEIGHT_ORDER.
            encoding_params=list(input_params),
            variational_params=list(weight_params),
            observables=_qiskit_observables(observables),
            variational_params_initial="uniform",
            seed_init=seed,
            # FAILURE MODE 3, qtm edition. qtm defaults this to False, under
            # which it never computes d(output)/d(encoding_params) -- and
            # `lam` sits upstream of the encoding parameters, so `lam.grad`
            # would be silently dead while the forward pass looked perfect.
            # This is the same bug class as EstimatorQNN's input_gradients
            # default, reintroduced by the migration. The smoke test asserts it.
            encoding_gradients_flag=True,
            # 0 = use all available threads (qtm's own default).
            num_threads_forward=num_threads_forward,
            num_threads_backward=num_threads_backward,
        )

        # qtm's "uniform" initializer is uniform(0, 2*pi), which is far too wide
        # for a depth-5 circuit -- it starts in a barren-plateau-flat region.
        # Overwrite with the same narrow init the other backends use, so runs
        # started from a given seed begin at the same point in parameter space
        # regardless of backend.
        flat = torch.as_tensor(_initial_weights(len(weight_params), seed), dtype=torch.float32)
        params = list(self.qtm.parameters())
        total = sum(p.numel() for p in params)
        if total != flat.numel():
            raise RuntimeError(
                f"qtm exposed {total} trainable scalars but the circuit has "
                f"{flat.numel()} weight parameters -- the weight vector would not line up"
            )
        with torch.no_grad():
            offset = 0
            for p in params:
                p.copy_(flat[offset : offset + p.numel()].view_as(p))
                offset += p.numel()

    @property
    def weight(self) -> nn.Parameter:
        params = list(self.qtm.parameters())
        if len(params) != 1:
            raise RuntimeError(
                f"expected qtm to expose exactly one trainable parameter tensor, got "
                f"{len(params)}; use qtm.parameters() directly for the optimizer"
            )
        return params[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qtm(x)


class _TorchStatevectorBackend(nn.Module):
    """The Qiskit circuit evaluated as torch tensor algebra; autograd for gradients.

    The circuit object, its gate list, its qubit indices and the observables are
    all still Qiskit's -- `compile_circuit` reads `circuit.data`, and the
    observable matrices come from `SparsePauliOp.to_matrix()`. What changes is
    only *how* the expectation values are computed: 16 complex amplitudes
    propagated through torch ops, differentiated in one reverse pass, instead of
    a circuit dispatched through the primitive layer once per parameter (or per
    batch entry). See src/models/torch_statevector.py for why, with numbers.

    Gradients here are exact, like adjoint/parameter-shift and unlike SPSA.
    `scripts/verify_backend_equivalence.py` checks the forward values against
    Qiskit's own Statevector and the gradients against qiskit-torch-module.
    """

    def __init__(self, circuit, input_params, weight_params, observables, seed=None):
        super().__init__()
        from src.models import torch_statevector as tsv

        # Explicit ordered parameter lists, never circuit.parameters -- that
        # property sorts alphabetically. See PARAMETER ORDERING in the module
        # docstring; `weight_params` is in CIRCUIT_WEIGHT_ORDER.
        self._compiled = tsv.compile_circuit(circuit, list(input_params), list(weight_params))
        # Observable matrices are constant; registered as a buffer so they move
        # with .to()/.cuda() and are saved alongside the module.
        self.register_buffer(
            "_obs", tsv.observable_matrices(_qiskit_observables(observables), circuit.num_qubits)
        )
        self.weight = nn.Parameter(
            torch.as_tensor(_initial_weights(len(weight_params), seed), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from src.models import torch_statevector as tsv

        return tsv.simulate(self._compiled, x, self.weight, self._obs)


class _QiskitMLBackend(nn.Module):
    """EstimatorQNN + TorchConnector -- evaluation path and speed-comparison baseline.

    This is the only backend that accepts an arbitrary estimator, which is what
    makes shot-based / noisy Aer evaluation possible (see evaluate_finite_shot).
    """

    def __init__(
        self,
        circuit,
        input_params,
        weight_params,
        observables,
        gradient_method: str = "param_shift",
        estimator=None,
        seed=None,
    ):
        super().__init__()
        from qiskit.primitives import StatevectorEstimator
        from qiskit_machine_learning.connectors import TorchConnector
        from qiskit_machine_learning.neural_networks import EstimatorQNN

        if estimator is None:
            # Exact statevector for training/timing; Aer is reserved for the
            # finite-shot hook and passed in explicitly.
            estimator = StatevectorEstimator()

        gradient = None
        if gradient_method == "spsa":
            from qiskit_machine_learning.gradients import SPSAEstimatorGradient

            # Stochastic 2-circuit gradient estimate. Retained only as an arm of
            # the speed-comparison figure -- adjoint is both exact and faster.
            gradient = SPSAEstimatorGradient(estimator, epsilon=0.01, seed=seed)
        elif gradient_method in ("param_shift", "parameter_shift"):
            gradient = None  # EstimatorQNN's own default is parameter-shift
        elif gradient_method == "lin_comb":
            from qiskit_machine_learning.gradients import LinCombEstimatorGradient

            gradient = LinCombEstimatorGradient(estimator)
        else:
            raise ValueError(
                f"gradient method {gradient_method!r} is not available on the qiskit_ml "
                f"backend (choose spsa, param_shift, or lin_comb). 'adjoint' requires "
                f"backend='torch_sv' or backend='qtm'."
            )

        qnn = EstimatorQNN(
            circuit=circuit,
            observables=_qiskit_observables(observables),
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
            gradient=gradient,
            # FAILURE MODE 3 -- without this, lam.grad is silently dead.
            input_gradients=True,
        )
        self.qnn = TorchConnector(
            qnn, initial_weights=_initial_weights(qnn.num_weights, seed)
        )

    @property
    def weight(self) -> nn.Parameter:
        return self.qnn.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)


class VQCQFunction(QFunction):
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

        # Outside-the-circuit trainable scalings (Failure Modes 1 and 2 above).
        self.lam = nn.Parameter(torch.ones(n_qubits))
        self.w = nn.Parameter(torch.ones(n_actions))

        circuit, input_params, weight_params = build_circuit(n_qubits, n_layers, reuploading)
        self.circuit = circuit
        self._input_params = input_params
        self._weight_params = weight_params

        if backend == "torch_sv":
            self.vqc = _TorchStatevectorBackend(
                circuit, input_params, weight_params, observables, seed=seed
            )
        elif backend == "qtm":
            self.vqc = _QTMBackend(circuit, input_params, weight_params, observables, seed=seed)
        else:
            self.vqc = _QiskitMLBackend(
                circuit,
                input_params,
                weight_params,
                observables,
                gradient_method=gradient_method,
                estimator=estimator,
                seed=seed,
            )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = normalize_observation(states)  # [B, n_qubits]
        scaled = self.lam * normalized  # elementwise, plain torch -- outside the circuit
        raw_out = self.vqc(scaled)  # [B, n_actions]
        return raw_out * self.w

    # --- backend-neutral weight transfer -------------------------------------
    # Raw state_dict keys differ between backends (TorchConnector carries an
    # extra `_weights` entry alongside `weight`), so cross-backend loading goes
    # through these rather than load_state_dict(). Same-backend checkpoints
    # still load with plain load_state_dict(), as before.

    def circuit_weights(self) -> torch.Tensor:
        return self.vqc.weight.detach().clone()

    def load_circuit_weights(self, weights: torch.Tensor) -> None:
        target = self.vqc.weight
        flat = torch.as_tensor(weights, dtype=target.dtype).reshape(target.shape)
        with torch.no_grad():
            target.copy_(flat)

    def export_weights(self) -> dict:
        """Backend-neutral checkpoint: lam, w, and the ordered circuit weights."""
        return {
            "lam": self.lam.detach().clone(),
            "w": self.w.detach().clone(),
            "circuit_weights": self.circuit_weights(),
            "meta": {
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
            self.w.copy_(torch.as_tensor(blob["w"], dtype=self.w.dtype))
        self.load_circuit_weights(blob["circuit_weights"])


def evaluate_finite_shot(model: VQCQFunction, shots: int) -> VQCQFunction:
    """Hook: reload `model`'s trained parameters onto a shot-based Aer estimator.

    torch_sv and qtm are statevector-only -- no shot noise, no hardware noise
    model -- so finite-shot and noisy evaluation necessarily runs on the retained
    qiskit_ml/Aer path. This is why the EstimatorQNN construction code was kept
    rather than deleted in the v3 migration: training moved off it, but it is the
    only backend here that a noisy estimator can drive.

    Returns a new VQCQFunction on backend="qiskit_ml" carrying `model`'s trained
    lam / w / circuit weights, evaluated with `shots`-shot sampling noise instead
    of an exact statevector (Skolik et al. 2023,
    https://doi.org/10.1140/epjqt/s40507-023-00166-1). Only the swap is
    implemented here -- the sweep across shot counts / evaluation episodes is
    intentionally not implemented yet, per the project's build order.
    """
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

    precision = 1.0 / np.sqrt(shots)  # AerEstimatorV2 exposes default_precision, not shots
    aer_estimator = AerEstimatorV2(options={"default_precision": precision})

    shot_model = VQCQFunction(
        n_qubits=model.n_qubits,
        n_layers=model.n_layers,
        n_actions=model.n_actions,
        reuploading=model.reuploading,
        observables=model.observables,
        backend="qiskit_ml",
        gradient_method="param_shift",
        estimator=aer_estimator,
    )
    # Cross-backend transfer: explicit and ordering-checked, not raw state_dict.
    shot_model.import_weights(model.export_weights())
    return shot_model


def _smoke_test(backend: str, gradient_method: str) -> None:
    """Batch of 8 random states -> [8, 2], with live gradients on all three
    parameter groups. This is the test that caught FAILURE MODE 3, and a backend
    migration is exactly where that class of bug comes back."""
    torch.manual_seed(0)
    model = VQCQFunction(backend=backend, gradient_method=gradient_method, seed=0)

    n_circuit_params = len(model._weight_params)
    n_total_params = sum(p.numel() for p in model.parameters())
    print(f"  circuit weight params: {n_circuit_params}")
    print(f"  total trainable params (lam + circuit + w): {n_total_params}")

    states = torch.rand(8, 4, dtype=torch.float32) * 0.2 - 0.1
    q_values = model(states)
    assert q_values.shape == (8, 2), f"expected [8, 2], got {tuple(q_values.shape)}"

    q_values.sum().backward()

    for name, param in (("lam", model.lam), ("w", model.w), ("circuit", model.vqc.weight)):
        assert param.grad is not None, f"{name} gradient is None on backend={backend}"
        assert param.grad.abs().sum().item() > 0, f"{name} gradient is dead on backend={backend}"
        print(f"  {name} grad norm: {param.grad.norm().item():.6f}")

    print(f"  OK -- backend={backend}: shape [8, 2], all three parameter groups live")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vqc.py smoke test")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=[*BACKENDS, "all"],
        help="which backend to smoke-test ('all' tries each and reports every failure)",
    )
    parser.add_argument("--gradient-method", default=None)
    args = parser.parse_args()

    batch, provenance = suggested_batch_size()
    print(f"suggested batch size: {batch}  [{provenance}]")

    defaults = {"torch_sv": "adjoint", "qtm": "adjoint", "qiskit_ml": "param_shift"}
    targets = list(BACKENDS) if args.backend == "all" else [args.backend]

    failures = []
    for name in targets:
        method = args.gradient_method or defaults[name]
        print(f"\n=== backend={name} gradient={method} ===")
        try:
            _smoke_test(name, method)
        except Exception as exc:  # keep going so 'all' reports every backend
            failures.append((name, exc))
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    if failures:
        print("\nsmoke test FAILED for: " + ", ".join(n for n, _ in failures))
        raise SystemExit(1)
    print("\nvqc.py smoke test OK")
