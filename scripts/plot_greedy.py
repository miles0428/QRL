"""Plot the greedy episode-0 trace saved by greedy_baseline.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACTION_NAMES = ["+X", "-X", "+Y", "-Y", "IDLE"]


def panel(ax_top, ax_bot, payload, title):
    tr = payload["greedy_trace_ep0"]
    if not tr:
        ax_top.set_title(f"{title} (no trace)")
        return
    arr = np.array(tr, dtype=float)
    sx, sy, sz, act = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    t = np.arange(1, len(arr) + 1)

    ax_top.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax_top.plot(t, sz, lw=1.2, label=r"$\langle\sigma_z\rangle$", color="#1f77b4")
    ax_top.plot(t, sx, lw=0.7, alpha=0.6, label=r"$\langle\sigma_x\rangle$", color="#ff7f0e")
    ax_top.plot(t, sy, lw=0.7, alpha=0.6, label=r"$\langle\sigma_y\rangle$", color="#2ca02c")
    ax_top.set_ylim(-1.05, 1.05)
    ax_top.set_ylabel("Bloch component")
    ax_top.set_title(title)
    ax_top.legend(loc="lower right", fontsize=8, ncol=3)

    ax_bot.scatter(t, act, s=4, c=["0.7" if a == 4 else "#d62728" for a in act])
    ax_bot.set_yticks(range(5))
    ax_bot.set_yticklabels(ACTION_NAMES, fontsize=8)
    ax_bot.set_xlabel("step")
    ax_bot.set_ylabel("action")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--default", default="results/greedy_baseline_default.json")
    p.add_argument("--fixed", default="results/greedy_baseline_sigma_fixed.json")
    p.add_argument("--out", default="figures/greedy_baseline.png")
    args = p.parse_args()

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 5.5), sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.22},
    )

    for col, (path, label) in enumerate(
        [(args.default, "as shipped: sigma_ou = 2pi x 1.0  (noise/Rabi = 31.6x)"),
         (args.fixed, "sigma_ou = 2pi x 1.0e-3  (noise/Rabi = 0.03x)")]
    ):
        payload = json.loads(Path(path).read_text())
        s = payload["summary"]["greedy"]
        title = f"{label}\ngreedy: return {s['return_mean']:.2f}, len {s['length_mean']:.1f}, survive {s['survival_rate']*100:.0f}%"
        panel(axes[0, col], axes[1, col], payload, title)

    fig.suptitle("QuantumSpinCartPole-v0 -- greedy one-step-lookahead baseline (episode 0)", fontsize=11)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
