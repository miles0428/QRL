"""Play a WHOLE game where EVERY action is decided by a quantum backend (Aer, or a real QPU).

Unlike quantum_hardware.py (which reads out a few frozen frames), this closes the loop: at every
timestep the CNN features are bound into the circuit and the action comes from a fresh quantum
measurement — shot-based, optionally under a real IBM device noise model. So the dino plays a full
episode "on the quantum computer" (simulated shot-by-shot).

Backends:
  aer        — Aer statevector Estimator with finite shots (fast, local).
  aer_noise  — Aer + FakeLagosV2 real-device noise model (slower; hardware-realistic).
  real       — an actual IBM QPU inside a Qiskit Runtime Session. WARNING: an episode is ~hundreds
               of sequential jobs (each action depends on the last, so they can't be batched) — use
               only with a small --max-steps; it consumes real QPU time.

Run:  python -m experiments.dino.quantum_play --backend aer --shots 1024
      python -m experiments.dino.quantum_play --backend aer_noise --shots 2048 --max-steps 400
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import torch

import experiments.dino  # noqa: F401
from experiments.dino.env import make_dino_env
from experiments.dino.model import load_agent
from src.models.vqc import build_circuit

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ACTIONS = ["RUN", "JUMP", "DUCK"]


def make_quantum_policy(model, backend="aer", shots=1024):
    """Return act(obs)->int that decides via a quantum ⟨ZZ⟩ measurement (Aer, finite shots)."""
    from qiskit_aer.primitives import EstimatorV2
    circuit, inp, wp = build_circuit(model.n_qubits, model.n_layers, reuploading=model.reuploading)
    w = model.vqc.weights.detach().double().numpy()
    w_out = model.w.detach().double().numpy()
    opts = {}
    if backend == "aer_noise":
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime.fake_provider import FakeLagosV2
        opts = {"backend_options": {"noise_model": NoiseModel.from_backend(FakeLagosV2())}}
    est = EstimatorV2(options=opts)
    prec = 1.0 / np.sqrt(shots)

    def act(obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0) / 255.0
        with torch.no_grad():
            enc = (model.lam * model.encoder(x)).squeeze(0).double().numpy()
        bound = circuit.assign_parameters({**{inp[k]: float(enc[k]) for k in range(len(inp))},
                                           **{wp[j]: float(w[j]) for j in range(len(wp))}})
        o = np.asarray(est.run([(bound, list(model.observables))], precision=prec).result()[0].data.evs)
        q = (o + 1.0) * 0.5 * w_out
        return int(np.argmax(q))

    return act


def play(policy, seed=20000, max_steps=1500, bird_prob=None, bird_start_frame=None):
    env = make_dino_env(max_steps=max_steps, seed=seed, bird_prob=bird_prob, bird_start_frame=bird_start_frame)
    obs, info = env.reset(seed=seed)
    counts = {0: 0, 1: 0, 2: 0}
    done = False
    n = 0
    while not done:
        a = policy(obs)
        counts[a] += 1
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        n += 1
    return info["score"], n, counts


def exact_policy(model):
    def act(obs):
        with torch.no_grad():
            return int(torch.argmax(model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)), 1).item())
    return act


def main():
    ap = argparse.ArgumentParser(description="Play a full dino game with a quantum backend deciding every action.")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "results", "dino_vqc.pt"))
    ap.add_argument("--backend", default="aer", choices=["aer", "aer_noise", "real"])
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--bird-prob", type=float, default=None)
    ap.add_argument("--bird-start-frame", type=int, default=None)
    args = ap.parse_args()

    model = load_agent(args.ckpt)
    print(f"agent: head={model.head_kind}, {model.n_qubits} qubits; backend={args.backend}, shots={args.shots}")

    if args.backend == "real":
        print("Real-QPU full game is impractical (hundreds of sequential jobs). Use --backend aer/aer_noise,\n"
              "or cap --max-steps very low and wrap make_quantum_policy in a Runtime Session (see quantum_hardware.py).")
        return

    # exact (torch_sv) reference
    es, en, _ = play(exact_policy(model), seed=args.seed, max_steps=args.max_steps,
                     bird_prob=args.bird_prob, bird_start_frame=args.bird_start_frame)
    print(f"exact torch_sv  : score {es} ({en} decisions)")

    # quantum-in-the-loop
    t0 = time.time()
    qs, qn, counts = play(make_quantum_policy(model, args.backend, args.shots), seed=args.seed,
                          max_steps=args.max_steps, bird_prob=args.bird_prob, bird_start_frame=args.bird_start_frame)
    dt = time.time() - t0
    ca = {ACTIONS[k]: v for k, v in counts.items()}
    print(f"QUANTUM ({args.backend}, {args.shots}sh): score {qs} ({qn} decisions, {qn} quantum measurements) "
          f"| actions {ca} | {dt:.1f}s ({1000*dt/max(qn,1):.0f} ms/decision)")
    print(f"\n=> a full game was played with a quantum circuit deciding every one of {qn} actions "
          f"(score {qs} vs exact {es}).")


if __name__ == "__main__":
    main()
