"""A more complete real-hardware study of the trained VQC decision head.

Over N game frames, compares the head's Q-values / chosen actions across:
  * exact       — torch_sv statevector (ground truth the agent used);
  * aer_noise   — Aer with a FakeLagosV2 real-device noise model (local sim);
  * real_raw    — an actual IBM QPU, no error mitigation (resilience_level=0);
  * real_zne    — the same QPU with Zero-Noise Extrapolation (resilience_level=2).

Metrics per method: action-agreement with exact (how often the greedy action is unchanged) and
mean |ΔQ| (Q-value magnitude error). This shows (a) whether the quantum DECISION survives real
hardware, and (b) whether error mitigation helps the Q MAGNITUDES vs the decisions. The real runs
share one Qiskit Runtime Session (transpile once, two jobs). Saves a table + a bar-chart figure.

Run:  python -m experiments.dino.quantum_hardware_study --frames 16 --shots 4096            # sim only
      python -m experiments.dino.quantum_hardware_study --frames 16 --shots 4096 --real     # + real QPU
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
from experiments.dino.quantum_hardware import _ibm_service, _binding
from src.models.vqc import build_circuit

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "figures")
RES = os.path.join(_ROOT, "results")
ACTIONS = ["RUN", "JUMP", "DUCK"]


def _q_from_evs(evs, w_out):
    return (np.asarray(evs) + 1.0) * 0.5 * w_out


def _metrics(q_method, q_exact):
    q_method, q_exact = np.asarray(q_method), np.asarray(q_exact)
    agree = int(np.sum(np.argmax(q_method, 1) == np.argmax(q_exact, 1)))
    mad = float(np.mean(np.abs(q_method - q_exact)))
    return agree, mad


def run_aer_noise(circuit, observables, inp, wp, enc, w, shots):
    from qiskit_aer.primitives import EstimatorV2
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2
    est = EstimatorV2(options={"backend_options": {"noise_model": NoiseModel.from_backend(FakeLagosV2())}})
    out = []
    for i in range(len(enc)):
        b = _binding(circuit, inp, wp, enc[i], w)
        out.append(np.asarray(est.run([(b, list(observables))], precision=1.0 / np.sqrt(shots)).result()[0].data.evs))
    return out


def run_real(circuit, observables, inp, wp, enc, w, shots, account, backend_name, resiliences):
    """Run all frames on a real QPU at each resilience level. Uses separate JOBS (not a Session,
    which the IBM open plan forbids) — one job per resilience level. Returns {res: evs}."""
    from qiskit_ibm_runtime import EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    svc = _ibm_service(account)
    backend = (svc.least_busy(operational=True, simulator=False, min_num_qubits=circuit.num_qubits)
               if backend_name in (None, "least_busy") else svc.backend(backend_name))
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    pubs = []
    for i in range(len(enc)):
        isa = pm.run(_binding(circuit, inp, wp, enc[i], w))
        pubs.append((isa, [o.apply_layout(isa.layout) for o in observables]))
    results = {}
    for res in resiliences:
        est = EstimatorV2(mode=backend)          # job mode (open-plan allowed)
        est.options.default_shots = shots
        est.options.resilience_level = res
        print(f"  real job on {backend.name}: {len(pubs)} frames, resilience_level={res} ...", flush=True)
        r = est.run(pubs).result()
        results[res] = [np.asarray(r[k].data.evs) for k in range(len(pubs))]
    return results, backend.name


def main():
    ap = argparse.ArgumentParser(description="Complete real-hardware study of the VQC head.")
    ap.add_argument("--ckpt", default=os.path.join(RES, "dino_vqc.pt"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--account", default="default-ibm-quantum-platform")
    ap.add_argument("--backend", default="ibm_marrakesh")
    args = ap.parse_args()

    model = load_agent(args.ckpt)
    nq = model.n_qubits
    frames = collect_frames(model, args.frames)
    circuit, inp, wp = build_circuit(nq, model.n_layers, reuploading=model.reuploading)
    w = model.vqc.weights.detach().double().numpy()
    w_out = model.w.detach().double().numpy()
    with torch.no_grad():
        enc = (model.lam * model.encoder(torch.as_tensor(np.stack(frames), dtype=torch.float32) / 255.0)).double().numpy()
        q_exact = model(torch.as_tensor(np.stack(frames), dtype=torch.uint8)).numpy()

    methods = {}   # label -> list of Q arrays
    methods["aer_noise"] = [_q_from_evs(o, w_out) for o in
                            run_aer_noise(circuit, model.observables, inp, wp, enc, w, args.shots)]
    backend_used = "FakeLagosV2 (sim)"
    if args.real:
        try:
            real, backend_used = run_real(circuit, model.observables, inp, wp, enc, w, args.shots,
                                          args.account, args.backend, resiliences=[0, 2])
            methods["real_raw"] = [_q_from_evs(o, w_out) for o in real[0]]
            methods["real_zne"] = [_q_from_evs(o, w_out) for o in real[2]]
        except Exception as e:
            print(f"(real-hardware part failed: {type(e).__name__}: {e})")

    # ---- report ----
    N = len(frames)
    lines = [f"VQC head hardware study — {N} frames, {args.shots} shots, backend={backend_used}",
             f"{'method':12s} | action-agreement | mean|ΔQ|",
             "-" * 46]
    summary = {}
    for label, qs in methods.items():
        agree, mad = _metrics(np.stack(qs), q_exact)
        summary[label] = (agree, mad)
        lines.append(f"{label:12s} |      {agree:2d}/{N:2d}       |  {mad:.3f}")
    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "dino_hardware_study.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = list(summary.keys())
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
        a1.bar(labels, [summary[l][0] / N * 100 for l in labels], color="#1f77b4")
        a1.set_ylabel("action agreement with exact (%)"); a1.set_ylim(0, 105)
        a1.axhline(100, ls="--", color="gray"); a1.tick_params(axis="x", rotation=20)
        a2.bar(labels, [summary[l][1] for l in labels], color="#d62728")
        a2.set_ylabel("mean |ΔQ| vs exact"); a2.tick_params(axis="x", rotation=20)
        fig.suptitle(f"VQC head on {backend_used} ({N} frames, {args.shots} shots)", fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "dino_hardware_study.png"), dpi=130)
        print("saved figures/dino_hardware_study.png")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__})")


if __name__ == "__main__":
    main()
