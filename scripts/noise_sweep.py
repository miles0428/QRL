"""
Sweep the noise/Rabi ratio and report greedy vs idle at each point.

Locates the difficulty cliff: below it greedy trivially saturates, above it no
policy has enough control authority and greedy is indistinguishable from doing
nothing. The interesting regime for RL is where greedy is clearly better than
idle but far from the 500 ceiling.

Usage:
    python scripts/noise_sweep.py --episodes 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from greedy_baseline import (  # noqa: E402
    GreedyPolicy,
    run_episode,
    sigma_for_noise_rabi,
    summarize,
)
from quantum_spin_cartpole import QuantumSpinCartPoleEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--ratios", default="0.03,0.1,0.25,0.5,0.75,1.0,2.0,4.0,31.62")
    p.add_argument("--out", default="results/noise_sweep.json")
    p.add_argument("--plot", default="figures/noise_sweep.png", help="empty string disables")
    args = p.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    rows = []
    t0 = time.time()

    hdr = (f"{'noise/Rabi':>10} {'sigma_ou':>10} | {'greedy return':>16} {'len':>7} {'surv':>6} "
           f"{'idle%':>6} | {'idle return':>13} {'len':>7} | {'greedy-idle':>11}")
    print(hdr)
    print("-" * len(hdr))

    for ratio in ratios:
        sigma = sigma_for_noise_rabi(ratio)
        stats = {}
        for name in ("greedy", "idle"):
            env = QuantumSpinCartPoleEnv(seed=0, sigma_ou=sigma)
            greedy = GreedyPolicy(env)
            rng = np.random.default_rng(12345)
            eps = [run_episode(env, name, greedy, rng, seed=1000 + k) for k in range(args.episodes)]
            stats[name] = summarize(name, eps)

        g, i = stats["greedy"], stats["idle"]
        print(
            f"{ratio:>10.2f} {sigma:>10.4f} | {g['return_mean']:>9.2f}+-{g['return_std']:<6.1f} "
            f"{g['length_mean']:>7.1f} {g['survival_rate']*100:>5.0f}% {g['idle_frac']*100:>5.1f}% | "
            f"{i['return_mean']:>8.2f}+-{i['return_std']:<4.1f} {i['length_mean']:>7.1f} | "
            f"{g['return_mean']-i['return_mean']:>11.2f}"
        )
        rows.append({"noise_rabi": ratio, "sigma_ou": sigma, "greedy": g, "idle": i})

    print(f"\n{args.episodes} episodes/point, {time.time()-t0:.1f}s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes": args.episodes, "rows": rows}, indent=2))
    print(f"wrote {out}")

    if args.plot:
        plot(rows, Path(args.plot))


def plot(rows, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["noise_rabi"] for r in rows]
    gm = np.array([r["greedy"]["return_mean"] for r in rows])
    gs = np.array([r["greedy"]["return_std"] for r in rows])
    im = np.array([r["idle"]["return_mean"] for r in rows])
    surv = np.array([r["greedy"]["survival_rate"] * 100 for r in rows])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1})

    ax.fill_between(x, np.clip(gm - gs, -10, None), gm + gs, alpha=0.18, color="#1f77b4", lw=0)
    ax.plot(x, gm, "o-", color="#1f77b4", label="greedy (one-step lookahead)")
    ax.plot(x, im, "s--", color="0.5", ms=4, label="always IDLE")
    ax.axhline(500, color="0.8", lw=0.8, ls=":")
    ax.text(x[0], 505, "ceiling ~500", fontsize=8, color="0.45")
    ax.axvspan(0.3, 0.9, color="#d62728", alpha=0.07, lw=0)
    ax.text(0.52, 300, "cliff", fontsize=9, color="#d62728", ha="center")
    ax.set_xscale("log")
    ax.set_ylabel("episode return")
    ax.legend(fontsize=9, loc="center left")
    ax.set_title("QuantumSpinCartPole: greedy vs noise/Rabi ratio")

    ax2.plot(x, surv, "o-", color="#2ca02c")
    ax2.axvspan(0.3, 0.9, color="#d62728", alpha=0.07, lw=0)
    ax2.set_ylabel("greedy survival %")
    ax2.set_xlabel("noise std / Rabi frequency   (log scale)")
    ax2.set_ylim(-5, 105)
    for a in (ax, ax2):
        a.grid(alpha=0.25, lw=0.5)
        a.axvline(0.5, color="#d62728", lw=1.0, ls="--", alpha=0.7)
        a.axvline(31.62, color="k", lw=1.0, ls="-.", alpha=0.5)
    ax2.text(31.62, 50, " as shipped", fontsize=8, rotation=90, va="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
