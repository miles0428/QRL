"""Learning-curve + diagnostics plots from the training CSV(s).

Run:  python -m experiments.dino.plot --csv results/dino_b.csv [results/dino_a.csv] \
          --labels "trained CNN (B)" "frozen ImageNet (A)" --out figures/dino_learning_curve.png

Top panel: per-episode score + a moving average vs env steps (the learning curve; a random-
policy reference line if --random is given). Bottom: the VQC output-scaling weights ``w`` (should
climb into the tens) and TD loss — the direct evidence the quantum head's value scale is training.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(path):
    cols = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    out = {}
    for k, vals in cols.items():
        arr = []
        for v in vals:
            try:
                arr.append(float(v))
            except (ValueError, TypeError):
                arr.append(np.nan)
        out[k] = np.array(arr)
    return out


def plot(csv_paths, labels=None, out_path=None, random_mean=None, title=None):
    labels = labels or [os.path.basename(p).replace(".csv", "") for p in csv_paths]
    out_path = out_path or os.path.join(_ROOT, "figures", "dino_learning_curve.png")
    datas = [_read(p) for p in csv_paths]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (d, lab) in enumerate(zip(datas, labels)):
        c = colors[i % len(colors)]
        x = d.get("env_steps", np.arange(len(d["score"])))
        ax0.plot(x, d["score"], color=c, alpha=0.25, lw=1)
        ma = d.get("ma20")
        if ma is not None:
            ax0.plot(x, ma, color=c, lw=2.2, label=f"{lab} (ma20)")
    if random_mean is not None:
        ax0.axhline(random_mean, ls="--", color="gray", lw=1.5, label=f"random ({random_mean:.0f})")
    ax0.set_ylabel("episode score (frames survived)")
    ax0.set_xlabel("env steps")
    ax0.set_title(title or "Dino CNN→VQC agent — learning curve")
    ax0.legend(loc="upper left", fontsize=9)
    ax0.grid(alpha=0.3)

    # bottom: w (left axis) + loss (right axis) for the first run
    d0 = datas[0]
    x0 = d0.get("env_steps", np.arange(len(d0["score"])))
    for wk in ("w0", "w1", "w2"):
        if wk in d0:
            ax1.plot(x0, d0[wk], lw=1.6, label=wk)
    ax1.set_ylabel("VQC output scale w")
    ax1.set_xlabel("env steps")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)
    if "mean_loss" in d0:
        axr = ax1.twinx()
        axr.plot(x0, d0["mean_loss"], color="crimson", alpha=0.4, lw=1, label="TD loss")
        axr.set_ylabel("TD loss", color="crimson")
        axr.tick_params(axis="y", colors="crimson")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"saved learning curve -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot dino training learning curve(s).")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--random", type=float, default=None, help="random-policy mean score reference")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    plot(args.csv, labels=args.labels, out_path=args.out, random_mean=args.random, title=args.title)


if __name__ == "__main__":
    main()
