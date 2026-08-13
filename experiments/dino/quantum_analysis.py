"""'More quantum' analysis of the trained decision head — the Qiskit-substance slides.

Given the trained agent, on a few real game frames it shows:
  1. EXACT ⟨ZZ⟩ → Q-values on torch_sv (what the agent used).
  2. FINITE-SHOT Aer readout (128 / 1024 shots) — the quantum head *sampled like real hardware*;
     the argmax action is (almost always) unchanged → the decision is robust to shot noise.
  3. TRANSPILATION to a real IBM backend's ISA (basis gates + coupling map) — depth / gate counts
     the circuit would take on hardware.

Run:  python -m experiments.dino.quantum_analysis --ckpt results/dino_vqc.pt --frames 5
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import torch

import experiments.dino  # noqa: F401
from experiments.dino.model import load_agent
from experiments.dino.quantum_readout import collect_frames
from src.models.vqc import build_circuit

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIONS = ["RUN", "JUMP", "DUCK"]


def aer_expectations(circuit, observables, input_params, weight_params, enc, w, shots):
    """⟨O⟩ per observable on Aer with finite shots, for one encoded input `enc` and weights `w`.

    qiskit_aer's EstimatorV2 is precision-based; precision = 1/sqrt(shots) reproduces the
    statistical noise of `shots` measurement samples.
    """
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    binding = {**{input_params[i]: float(enc[i]) for i in range(len(input_params))},
               **{weight_params[j]: float(w[j]) for j in range(len(weight_params))}}
    bound = circuit.assign_parameters(binding)
    est = AerEstimator()
    pub = (bound, list(observables))
    res = est.run([pub], precision=1.0 / np.sqrt(int(shots))).result()
    return np.asarray(res[0].data.evs, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Finite-shot + transpilation analysis of the VQC head.")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "results", "dino_vqc.pt"))
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--shots", type=int, nargs="+", default=[128, 1024])
    args = ap.parse_args()

    model = load_agent(args.ckpt)
    nq, na = model.n_qubits, model.n_actions
    frames = collect_frames(model, args.frames)
    circuit, input_params, weight_params = build_circuit(nq, model.n_layers, reuploading=model.reuploading)
    w = model.vqc.weights.detach().double().numpy()
    w_out = model.w.detach().double().numpy()

    with torch.no_grad():
        X = torch.as_tensor(np.stack(frames), dtype=torch.float32)
        enc = (model.lam * model.encoder(X / 255.0)).double().numpy()
        q_exact = model(torch.as_tensor(np.stack(frames), dtype=torch.uint8)).numpy()

    print(f"=== VQC head: exact vs finite-shot Aer readout ({nq} qubits, {model.observable_kind}) ===")
    print("The quantum circuit picks the dino's action; below, does it still pick it under shot noise?\n")
    agree = {s: 0 for s in args.shots}
    for i in range(len(frames)):
        a_exact = int(np.argmax(q_exact[i]))
        line = f"frame {i}: EXACT Q={np.round(q_exact[i],2).tolist()} -> {ACTIONS[a_exact]:4s}"
        for s in args.shots:
            o = aer_expectations(circuit, model.observables, input_params, weight_params, enc[i], w, s)
            q_s = (np.asarray(o) + 1.0) * 0.5 * w_out
            a_s = int(np.argmax(q_s))
            agree[s] += int(a_s == a_exact)
            line += f"  | {s}sh -> {ACTIONS[a_s]:4s}"
        print(line)
    print()
    for s in args.shots:
        print(f"  {s} shots: same action as exact on {agree[s]}/{len(frames)} frames")

    # --- transpilation to a real IBM backend ISA ---
    print("\n=== Transpilation to hardware (what the circuit costs on a real device) ===")
    print(f"logical circuit: qubits {circuit.num_qubits}, depth {circuit.depth()}, "
          f"gates {dict(circuit.count_ops())}")
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        backend = bname = None
        try:
            from qiskit_ibm_runtime.fake_provider import FakeLagosV2      # 7-qubit IBM device model
            backend, bname = FakeLagosV2(), "FakeLagosV2 (7-qubit IBM device model, real noise/coupling)"
        except Exception:
            pass
        if backend is None or backend.num_qubits < circuit.num_qubits:
            from qiskit.providers.fake_provider import GenericBackendV2
            backend = GenericBackendV2(num_qubits=max(circuit.num_qubits, 6))
            bname = f"GenericBackendV2({backend.num_qubits}q)"
        if backend.num_qubits >= circuit.num_qubits:
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            isa = pm.run(circuit)
            print(f"on {bname}: basis {list(backend.operation_names)[:6]}... | "
                  f"depth {isa.depth()}, gates {dict(isa.count_ops())}")
        else:
            print(f"({bname} has {backend.num_qubits} qubits < {circuit.num_qubits}; "
                  "use a >=6-qubit backend to transpile this head.)")
    except Exception as e:
        print(f"(transpilation skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
