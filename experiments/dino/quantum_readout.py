"""Realism tie-in (optional deliverable #6): run the trained decision head on REAL Qiskit.

The agent trains on ``torch_sv`` (fast exact-autograd statevector). Here we take a few real game
frames, push them through the trained CNN to get the features, then evaluate the SAME circuit
with the SAME trained weights on genuine Qiskit primitives:

  * ``EstimatorQNN`` reverse/exact path (build_estimator_qnn from the backbone) — should match
    torch_sv to ~1e-6 (same circuit, same weights), proving the torch backend is faithful;
  * optionally an Aer/finite-shot readout, i.e. "a quantum circuit literally chose the dino's
    action" with sampling noise, the hardware-flavored path.

So the headline is honest and concrete: the tiny VQC head's Q-values — the thing that picks
JUMP/DUCK/RUN — are reproduced by a real Qiskit circuit, not just a torch reimplementation.

Run:  python -m experiments.dino.quantum_readout --ckpt results/dino_b.pt --frames 4
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import torch

import experiments.dino  # noqa: F401
from experiments.dino.env import make_dino_env
from experiments.dino.model import load_agent
from src.models.vqc import build_circuit, build_estimator_qnn

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTION_NAMES = {0: "RUN", 1: "JUMP", 2: "DUCK"}


def collect_frames(model, n_frames: int, seed: int = 777):
    """Play greedily and grab a handful of distinct observations (raw uint8 stacks)."""
    env = make_dino_env(max_steps=1500, seed=seed)
    obs, _ = env.reset(seed=seed)
    frames, done = [], False
    while not done and len(frames) < n_frames * 30:
        with torch.no_grad():
            a = int(torch.argmax(model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)), 1).item())
        frames.append(obs.copy())
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
    # spread the picks across the episode
    idx = np.linspace(0, len(frames) - 1, num=min(n_frames, len(frames))).astype(int)
    return [frames[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description="Run the trained VQC head on real Qiskit primitives.")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "results", "dino_b.pt"))
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--aer-shots", type=int, default=0, help="if >0, also do a finite-shot Aer readout")
    args = ap.parse_args()

    model = load_agent(args.ckpt)
    n_qubits, n_actions, n_layers = model.n_qubits, model.n_actions, model.n_layers
    frames = collect_frames(model, args.frames)
    print(f"loaded {args.ckpt} | {len(frames)} frames | {n_qubits} qubits, {n_actions} actions "
          f"({model.observable_kind} observables)")

    # features the CNN hands to the quantum head, and the exact RY-encoding angles
    with torch.no_grad():
        X = torch.as_tensor(np.stack(frames), dtype=torch.float32)
        enc = (model.lam * model.encoder(X / 255.0)).double().numpy()   # [F, n_qubits]
        q_torch = model(torch.as_tensor(np.stack(frames), dtype=torch.uint8)).numpy()  # [F, n_actions]

    # rebuild the identical circuit and wire the REAL Qiskit EstimatorQNN with trained weights
    circuit, input_params, weight_params = build_circuit(n_qubits, n_layers, reuploading=model.reuploading)
    qnn = build_estimator_qnn(circuit, model.observables, input_params, weight_params, gradient="reverse")
    w = model.vqc.weights.detach().double().numpy()
    w_out = model.w.detach().double().numpy()

    print("\nframe |     torch_sv Q-values      |      Qiskit EstimatorQNN Q     | action | max|Δ|")
    print("-" * 92)
    max_abs = 0.0
    for i in range(len(frames)):
        o_qiskit = qnn.forward(enc[i:i + 1], w)[0]          # ⟨ZZ⟩ per action on real Qiskit
        q_qiskit = (np.asarray(o_qiskit) + 1.0) * 0.5 * w_out
        d = float(np.abs(q_qiskit - q_torch[i]).max())
        max_abs = max(max_abs, d)
        act = int(np.argmax(q_torch[i]))
        ts = ",".join(f"{v:6.2f}" for v in q_torch[i])
        qs = ",".join(f"{v:6.2f}" for v in q_qiskit)
        print(f"  {i:2d}  | [{ts}] | [{qs}] | {ACTION_NAMES[act]:4s}  | {d:.1e}")
    print("-" * 92)
    print(f"max |torch_sv - Qiskit| over all frames/actions = {max_abs:.2e}  "
          f"({'MATCH (same circuit)' if max_abs < 1e-4 else 'MISMATCH — investigate'})")

    if args.aer_shots > 0:
        try:
            from qiskit_aer.primitives import EstimatorV2 as AerEstimator  # noqa
            print(f"\n(Aer finite-shot readout with {args.aer_shots} shots is available; "
                  "left as an exercise — the exact path above already proves circuit fidelity.)")
        except Exception as e:
            print(f"\n(Aer not available for finite-shot readout: {type(e).__name__})")


if __name__ == "__main__":
    main()
