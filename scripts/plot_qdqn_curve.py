"""
QDQN learning curve, against the baselines it has to be read against.

Two x-axes matter here and they disagree, so both are drawn:

  environment steps   what the runs are budgeted on, and what makes the
                      comparison with the classical agents like-for-like.
  gradient steps      what actually updates the model. train_every=10 (from
                      configs/qdqn.yaml, where a circuit gradient step is the
                      dominant wall-clock cost) gives QDQN one update per 10 env
                      steps against the classical runs' one per step, so at
                      equal env steps QDQN has taken 10x fewer.

Reference lines are the final 50-episode evaluations of the controllers on the
same seeds: greedy, the parameter-matched MLP (width 6, 137 params vs the
circuit's 140) and the oversized MLP (width 128, 18,437 params).

Usage:
    python scripts/plot_qdqn_curve.py
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(pattern):
    runs = []
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text())
        c = d.get("curve") or []
        if c:
            runs.append((d.get("seed", "?"),
                         np.array([p["step"] for p in c]),
                         np.array([p["return_mean"] for p in c]),
                         np.array([p["survival_rate"] for p in c])))
    return runs


LOG_LINE = re.compile(
    r"step\s+(\d+).*?eval\s+(-?[\d.]+)\s*\+-\s*[\d.]+\s+len\s+[\d.]+\s+surv\s+(\d+)%")


def load_logs(pattern):
    """
    Parse in-progress runs straight from their logs.

    The training script only writes its JSON at the end, so a run that is still
    going has no results file. Its printed evaluation lines carry the same
    numbers, so a curve can be drawn while the run continues.
    """
    runs = []
    for f in sorted(glob.glob(pattern)):
        steps, rets, survs = [], [], []
        for line in Path(f).read_text(errors="ignore").splitlines():
            m = LOG_LINE.search(line)
            if m:
                steps.append(int(m.group(1)))
                rets.append(float(m.group(2)))
                survs.append(int(m.group(3)) / 100.0)
        if steps:
            seed = re.search(r"seed(\d+)", f)
            runs.append((seed.group(1) if seed else "?",
                         np.array(steps), np.array(rets), np.array(survs)))
    return runs


def mean_curve(runs):
    """Mean over seeds on the steps every run reached."""
    if not runs:
        return None, None, None
    common = min(len(r[1]) for r in runs)
    steps = runs[0][1][:common]
    ret = np.stack([r[2][:common] for r in runs])
    surv = np.stack([r[3][:common] for r in runs])
    return steps, ret, surv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/qdqn_learning_curve.png")
    p.add_argument("--train-every", type=int, default=10)
    # Default is the plain curve, for presenting. The reference lines and the
    # gradient-step axis are analysis aids: useful when judging the result,
    # clutter when showing it.
    p.add_argument("--refs", action=argparse.BooleanOptionalAction, default=False,
                   help="draw greedy / classical-MLP reference lines")
    p.add_argument("--gradient-axis", action=argparse.BooleanOptionalAction,
                   default=False, help="add the gradient-step axis on top")
    p.add_argument("--survival", action=argparse.BooleanOptionalAction, default=False,
                   help="add the survival-rate panel below")
    args = p.parse_args()

    # finished runs first; fall back to parsing the logs of runs still going
    qdqn = load("results/qdqn_sxy_prevact_nr0.35_seed*.json")
    src = "results JSON"
    if not qdqn:
        qdqn = load_logs("logs/qdqn_sxy_prevact_nr0.35_seed*.log")
        src = "in-progress logs"
    if not qdqn:
        qdqn = load("results/qdqn_anim_nr0.35_seed*.json")
        src = "anim run JSON"
    print(f"QDQN curve source: {src}, {len(qdqn)} run(s)")

    if args.survival:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True,
                                      gridspec_kw={"height_ratios": [2.1, 1],
                                                   "hspace": 0.12})
    else:
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        ax2 = None

    # --- reference lines -----------------------------------------------------
    refs = []
    if not args.refs:
        refs = None
    gb = Path("results/greedy_baseline_nr0.35.json") if args.refs else Path("__none__")
    if gb.exists():
        refs.append(("greedy (one-step lookahead)",
                     json.loads(gb.read_text())["summary"]["greedy"]["return_mean"],
                     "#d62728", "-"))
    for pat, name, col in ([] if not args.refs else [
        ("results/dqn_w6_sxy_prevact_nr0.35_seed*.json",
         "classical MLP width-6 (137 params)", "#2ca02c"),
        ("results/dqn_sxy_prevact_nr0.35_seed*.json",
         "classical MLP width-128 (18,437 params)", "#7f7f7f"),
    ]):
        fs = sorted(glob.glob(pat))
        if fs:
            v = np.mean([json.loads(Path(f).read_text())["final"]["return_mean"] for f in fs])
            refs.append((name, v, col, "--"))
    # Stagger the labels: the width-6 and width-128 references land within ~10
    # points of each other and their text would otherwise sit on top of itself.
    for i, (name, v, col, ls) in enumerate(sorted(refs or [], key=lambda r: -r[1])):
        ax.axhline(v, color=col, ls=ls, lw=1.2, alpha=0.85)
        ax.text(0.015 + 0.34 * i, v, f" {name}: {v:.0f}", color=col, fontsize=8,
                ha="left", va="bottom", transform=ax.get_yaxis_transform())

    # --- QDQN curves ---------------------------------------------------------
    steps, ret, surv = mean_curve(qdqn)
    if steps is not None:
        for sd, st, rt, _sv in qdqn:
            ax.plot(st, rt, lw=0.8, alpha=0.4, color="#1f77b4")
        m, sd_ = ret.mean(0), ret.std(0, ddof=1) if ret.shape[0] > 1 else (ret.mean(0), None)
        ax.plot(steps, m, lw=2.2, color="#1f77b4",
                label=f"QDQN 9q x 5L, 140 params (mean of {ret.shape[0]} seeds)")
        if ret.shape[0] > 1:
            ax.fill_between(steps, m - ret.std(0, ddof=1), m + ret.std(0, ddof=1),
                            color="#1f77b4", alpha=0.15, lw=0)
        if ax2 is not None:
            for sd, st, _rt, sv in qdqn:
                ax2.plot(st, sv * 100, lw=0.8, alpha=0.4, color="#1f77b4")
            ax2.plot(steps, surv.mean(0) * 100, lw=2.2, color="#1f77b4")

    ax.set_ylabel("evaluation return (20 episodes)")
    ax.set_title("QDQN on the sxy game, noise/Rabi = 0.35", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(bottom=0)

    if ax2 is not None:
        ax2.set_ylabel("survival %")
        ax2.set_xlabel("environment steps")
        ax2.grid(alpha=0.25, lw=0.5)
        ax2.set_ylim(-3, 103)
    else:
        ax.set_xlabel("environment steps")

    if args.gradient_axis:
        top = ax.secondary_xaxis("top",
                                 functions=(lambda x: x / args.train_every,
                                            lambda x: x * args.train_every))
        top.set_xlabel(f"gradient steps  (train_every={args.train_every})", fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    if steps is not None:
        print(f"\nQDQN, {ret.shape[0]} seeds, to {steps[-1]:,} env steps "
              f"({steps[-1]//args.train_every:,} gradient steps):")
        for i in range(0, len(steps), max(1, len(steps) // 8)):
            print(f"  {steps[i]:>7,} steps: {ret[:, i].mean():7.1f}  "
                  f"({surv[:, i].mean()*100:3.0f}% survive)   per seed " +
                  ", ".join(f"{v:.0f}" for v in ret[:, i]))
        print(f"  {steps[-1]:>7,} steps: {ret[:, -1].mean():7.1f}  "
              f"({surv[:, -1].mean()*100:3.0f}% survive)")


if __name__ == "__main__":
    main()
