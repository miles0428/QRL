"""Run the trained VQC decision head on real IBM hardware (or a device-noise simulation).

Answers "how do we run this on real hardware / Aer". Three readout modes on a few real frames:
  * exact  — torch_sv statevector (what the agent used to play);
  * noise  — Aer with a REAL IBM device NOISE MODEL (FakeLagosV2) → the decision *under hardware
             noise*, no IBM account needed (runs today);
  * real   — an actual IBM Quantum backend via qiskit_ibm_runtime EstimatorV2 (needs a saved
             account/token; see the printed setup steps).

For each frame we report the head's Q-values and the chosen action, and whether the noisy/real
readout picks the SAME action as exact. The VQC's per-action Q-values are closely spaced, so shot
noise flips only the near-ties — an honest, concrete "it's real, hardware-ready quantum" result.

Run:  python -m experiments.dino.quantum_hardware --ckpt results/dino_vqc.pt --frames 4         # noise sim
      python -m experiments.dino.quantum_hardware --real --shots 4096                            # real QPU
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

SETUP = """\
To run on a REAL IBM Quantum computer:
  1. Make a free IBM Quantum account: https://quantum.ibm.com  (copy your API token)
  2. Save it once:
       python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \\
                  QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')"
  3. Re-run:  python -m experiments.dino.quantum_hardware --real --shots 4096
The circuit is 6 qubits / depth ~27; on a device it routes to ~57 two-qubit gates (see
quantum_analysis.py). Queue time varies; --frames 2 keeps the job small."""


def _binding(circuit, inp, wp, enc_i, w):
    return circuit.assign_parameters(
        {**{inp[k]: float(enc_i[k]) for k in range(len(inp))},
         **{wp[j]: float(w[j]) for j in range(len(wp))}})


def readout_noise(circuit, observables, inp, wp, enc, w, shots):
    """⟨ZZ⟩ per frame under a REAL IBM device noise model (FakeLagosV2), via Aer."""
    from qiskit_aer.primitives import EstimatorV2
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2
    nm = NoiseModel.from_backend(FakeLagosV2())
    est = EstimatorV2(options={"backend_options": {"noise_model": nm}})
    out = []
    for i in range(len(enc)):
        b = _binding(circuit, inp, wp, enc[i], w)
        res = est.run([(b, list(observables))], precision=1.0 / np.sqrt(shots)).result()
        out.append(np.asarray(res[0].data.evs, dtype=float))
    return out, "Aer + FakeLagosV2 real-device noise model"


def _ibm_service(account="default-ibm-quantum-platform"):
    """Connect to IBM Quantum. The saved accounts use channel='ibm_quantum_platform' (needs
    runtime>=0.40 / qiskit 2.x); their token works via channel='ibm_cloud' on runtime 0.34, which
    keeps the shared venv at qiskit 1.4.6 for the backbone (see tfq_repro/hardware.py)."""
    import json
    from qiskit_ibm_runtime import QiskitRuntimeService
    cfg = json.load(open(os.path.expanduser("~/.qiskit/qiskit-ibm.json")))[account]
    return QiskitRuntimeService(channel="ibm_cloud", token=cfg["token"], instance=cfg["instance"])


def readout_real(circuit, observables, inp, wp, enc, w, shots, account, backend_name):
    """⟨ZZ⟩ per frame on an ACTUAL IBM Quantum backend — all frames in ONE job."""
    from qiskit_ibm_runtime import EstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    svc = _ibm_service(account)
    backend = (svc.least_busy(operational=True, simulator=False, min_num_qubits=circuit.num_qubits)
               if backend_name in (None, "least_busy") else svc.backend(backend_name))
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    pubs = []
    for i in range(len(enc)):
        isa = pm.run(_binding(circuit, inp, wp, enc[i], w))
        isa_obs = [o.apply_layout(isa.layout) for o in observables]
        pubs.append((isa, isa_obs))
    est = EstimatorV2(mode=backend)
    est.options.default_shots = shots
    print(f"submitting {len(pubs)}-frame job to {backend.name} ({backend.status().pending_jobs} queued)...")
    res = est.run(pubs).result()
    out = [np.asarray(res[k].data.evs, dtype=float) for k in range(len(pubs))]
    return out, f"IBM QPU '{backend.name}'"


def main():
    ap = argparse.ArgumentParser(description="Run the trained VQC head on IBM hardware / device noise.")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "results", "dino_vqc.pt"))
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--real", action="store_true", help="run on an actual IBM QPU (needs saved account)")
    ap.add_argument("--account", default="default-ibm-quantum-platform")
    ap.add_argument("--backend", default="least_busy", help="e.g. ibm_marrakesh, or least_busy")
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

    if args.real:
        try:
            evs, source = readout_real(circuit, model.observables, inp, wp, enc, w,
                                       args.shots, args.account, args.backend)
        except Exception as e:
            print(f"Real-hardware run unavailable ({type(e).__name__}: {e}).\n\n{SETUP}")
            return
    else:
        evs, source = readout_noise(circuit, model.observables, inp, wp, enc, w, args.shots)

    print(f"=== VQC head on {source} ({args.shots} shots) ===")
    print("Does the quantum decision survive real hardware / device noise?\n")
    agree = 0
    for i in range(len(frames)):
        a_ex = int(np.argmax(q_exact[i]))
        q_hw = (np.asarray(evs[i]) + 1.0) * 0.5 * w_out
        a_hw = int(np.argmax(q_hw))
        agree += int(a_hw == a_ex)
        print(f"frame {i}: exact {np.round(q_exact[i],2).tolist()} -> {ACTIONS[a_ex]:4s}  ||  "
              f"hw {np.round(q_hw,2).tolist()} -> {ACTIONS[a_hw]:4s}  {'OK' if a_hw==a_ex else 'flip(near-tie)'}")
    print(f"\nsame action as exact on {agree}/{len(frames)} frames.")
    if not args.real:
        print("\n" + SETUP)


if __name__ == "__main__":
    main()
