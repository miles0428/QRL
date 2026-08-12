"""Time every backend arm under one protocol: N real gradient steps, same
circuit, same batch, same machine.

This exists because the numbers otherwise floating around the project are not
comparable with each other -- the pre-v3 figure came from a profiler run, the
v3 figures from training logs of parallel seeds. Task 5 of the v3 brief wants a
single measurement all arms share, and the result is a headline finding in its
own right: profiling attributed 91% of training time to the framework layer
rather than to anything intrinsic to simulating four qubits.

ENVIRONMENTS. No single environment holds every arm, because qiskit-torch-module
pins qiskit 1.x while qiskit-machine-learning here needs 2.x. Arms that cannot
be imported are skipped and reported as such, so this is run once per
environment and the results merged:

    <qtm env>/python  scripts/benchmark_backends.py --out results/backends_qtm.json
    <ml  env>/python  scripts/benchmark_backends.py --out results/backends_ml.json

Run it on an otherwise idle machine. A 5-seed sweep in the background inflates
every arm, and unevenly.

Usage:
    python scripts/benchmark_backends.py --steps 100 --batch 16
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# (label, backend, gradient_method, steps_override)
# param_shift gets far fewer steps: it costs 2 circuit evaluations per parameter
# per sample, so at 40 parameters a full 100 steps is not worth the wall-clock
# just to confirm it is the slowest arm.
ARMS = [
    ("torch_sv + adjoint (v3 training path)", "torch_sv", "adjoint", None),
    ("qtm + adjoint (sequential)", "qtm", "adjoint", 20),
    ("qiskit-ML + SPSA (pre-v3 baseline)", "qiskit_ml", "spsa", 20),
    ("qiskit-ML + param-shift", "qiskit_ml", "param_shift", 5),
]


def time_arm(backend: str, gradient_method: str, steps: int, batch: int, n_warmup: int = 2) -> dict:
    """Mean wall-clock per forward+backward+step, on random inputs.

    Deliberately not driven by a live environment: a real rollout's episode
    lengths differ between arms (a faster arm trains further in the same number
    of steps and its episodes get longer), which would confound the measurement
    with learning progress. Random inputs cost the circuit exactly the same.
    """
    from src.models.vqc import VQCQFunction

    t0 = time.perf_counter()
    model = VQCQFunction(backend=backend, gradient_method=gradient_method, seed=0)
    construction_s = time.perf_counter() - t0

    optimizer = torch.optim.Adam(
        [
            {"params": [model.lam], "lr": 0.001},
            {"params": list(model.vqc.parameters()), "lr": 0.001},
            {"params": [model.w], "lr": 0.1},
        ]
    )

    def one_step():
        states = torch.rand(batch, 4) * 4.0 - 2.0
        loss = model(states).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(n_warmup):
        one_step()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    elapsed = time.perf_counter() - t0

    return {
        "backend": backend,
        "gradient_method": gradient_method,
        "steps": steps,
        "batch": batch,
        "construction_s": construction_s,
        "total_s": elapsed,
        "s_per_grad_step": elapsed / steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default=None, help="write results as JSON here")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these backend names")
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]} | {platform.platform()}")
    print(f"torch threads: {torch.get_num_threads()} | batch {args.batch}\n")

    results = []
    for label, backend, method, steps_override in ARMS:
        if args.only and backend not in args.only:
            continue
        steps = steps_override or args.steps
        print(f"{label}  ({steps} steps) ... ", end="", flush=True)
        try:
            row = time_arm(backend, method, steps, args.batch)
        except Exception as exc:
            print(f"SKIPPED ({type(exc).__name__}: {str(exc)[:80]})")
            results.append({"backend": backend, "gradient_method": method, "skipped": str(exc)})
            continue
        row["label"] = label
        results.append(row)
        print(f"{row['s_per_grad_step'] * 1000:9.2f} ms/grad step")

    timed = [r for r in results if "s_per_grad_step" in r]
    if timed:
        fastest = min(r["s_per_grad_step"] for r in timed)
        print(f"\n{'arm':40s} {'ms/step':>10s} {'vs fastest':>12s}")
        print("-" * 64)
        for r in sorted(timed, key=lambda r: r["s_per_grad_step"]):
            print(
                f"{r['label']:40s} {r['s_per_grad_step'] * 1000:10.2f} "
                f"{r['s_per_grad_step'] / fastest:11.1f}x"
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "torch_threads": torch.get_num_threads(),
                    "batch": args.batch,
                    "arms": results,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
