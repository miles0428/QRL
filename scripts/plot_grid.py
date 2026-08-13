"""Comparison figures for the actor/critic grid: reward, loss, and critic quality.

Produces figures/10..13_*.{png,pdf} following the same conventions as
src/plots.py (150 dpi PNG alongside a vector PDF).

Re-runnable at any point: every panel draws whatever seeds are on disk and
labels each series with the number of completed seeds behind it, so a figure
made mid-sweep says so rather than silently showing fewer runs.

COLOR. The categorical order is fixed and assigned per configuration, never
cycled and never reassigned when a series is missing -- a figure regenerated
after more seeds land keeps every existing series the same color. Two of the
light-mode hues sit below 3:1 against the surface, so every series is also
DIRECTLY LABELLED at the end of its line; identity never rests on hue alone.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = "results"
OUTDIR = "figures"

# (config prefix, tag, label, actor kind, critic kind). One row per arm; a row
# with no runs on disk is skipped rather than drawn empty.
SETS = [
    ("qpg",          "", "QPG",      "quantum",   "-"),
    ("mlp_pg",       "", "MLP-PG",   "classical", "-"),
    ("qa2c",         "", "Q2Q",      "quantum",   "quantum"),
    ("q2c",          "", "Q2C",      "quantum",   "classical"),
    ("a2q",          "", "A2Q",      "classical", "quantum"),
    ("mlp_a2c",      "", "A2C",      "classical", "classical"),
    ("qdqn",   "vec10", "QDQN",      "quantum",   "-"),
    ("mlp_baseline", "", "MLP-DQN",  "classical", "-"),
    ("qpg_noreup",   "", "QPG no-reup",  "quantum", "-"),
    ("qa2c_noreup",  "", "Q2Q no-reup",  "quantum", "quantum"),
]

# Validated categorical order (light mode), assigned by configuration key.
# scripts/validate_palette.js: all checks pass for the first four slots in both
# modes; worst adjacent CVD dE 9.1, normal-vision 22.9.
COLORS = {
    "qa2c":          "#2a78d6",  # slot 1 blue    -- Q2Q
    "q2c":           "#eb6834",  # slot 2 orange  -- Q2C
    "a2q":           "#1baf7a",  # slot 3 aqua    -- A2Q
    "mlp_a2c":       "#eda100",  # slot 4 yellow  -- A2C classical
    "qpg":           "#e87ba4",  # slot 5 magenta
    "mlp_pg":        "#008300",  # slot 6 green
    "qdqn_vec10":    "#4a3aa7",  # slot 7 violet
    "mlp_baseline":  "#e34948",  # slot 8 red
    "qpg_noreup":    "#898781",  # muted -- ablations read as "the control"
    "qa2c_noreup":   "#52514e",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SOLVE = "#d03b3b"  # status critical, used only for the threshold rule


def _seeds_for(config: str, tag: str):
    """Every (seed, dataframe, eval-blob) on disk for one arm."""
    prefix = f"{config}_{tag}_" if tag else f"{config}_"
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, f"{prefix}*.csv"))):
        m = re.match(rf"{re.escape(prefix)}(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        ev_path = path.replace(".csv", "_eval.json")
        ev = json.load(open(ev_path)) if os.path.exists(ev_path) else None
        out.append((int(m.group(1)), df, ev))
    return out


def _curve(runs, xcol, ycol, n_bins: int = 40):
    """Median + IQR of ycol against xcol across seeds, on a shared x grid.

    Seeds stop at different episodes, so the grid runs to the SHORTEST seed --
    beyond that point the median would silently be over a shrinking subset, and
    a curve whose sample size changes along its length is the kind of chart that
    misleads without ever being wrong.
    """
    series = []
    for _s, df, _ev in runs:
        sub = df[[xcol, ycol]].dropna()
        if len(sub) >= 2:
            series.append((sub[xcol].to_numpy(dtype=float), sub[ycol].to_numpy(dtype=float)))
    if not series:
        return None
    xmax = min(x[-1] for x, _ in series)
    if xmax <= 0:
        return None
    grid = np.linspace(0, xmax, n_bins)
    stacked = np.array([np.interp(grid, x, y) for x, y in series])
    return {"x": grid, "median": np.median(stacked, axis=0),
            "lo": np.percentile(stacked, 25, axis=0),
            "hi": np.percentile(stacked, 75, axis=0), "n": len(series)}


def load():
    """Read results/ directly -- no intermediate file, so the figures are
    reproducible from a clean checkout with `python scripts/plot_grid.py`."""
    sets = {}
    for config, tag, label, actor, critic in SETS:
        runs = _seeds_for(config, tag)
        if not runs:
            continue
        done = [(s, df, ev) for s, df, ev in runs if ev is not None]
        greedy = [ev["greedy"]["mean_reward"] for _s, _d, ev in done]
        entry = {
            "label": label, "actor": actor, "critic": critic,
            "n_seeds": len(runs), "n_done": len(done),
            "greedy_median": float(np.median(greedy)) if greedy else None,
            "greedy_min": float(np.min(greedy)) if greedy else None,
            "greedy_max": float(np.max(greedy)) if greedy else None,
            "greedy_solves": sum(bool(ev["greedy"]["solved"]) for _s, _d, ev in done),
            "train_solves": sum(bool(ev["training_criterion"]["solved"]) for _s, _d, ev in done),
            "median_wall_s": float(np.median([df.iloc[-1]["wall_clock_s"] for _s, df, _e in done])) if done else None,
            "median_grad": int(np.median([df.iloc[-1]["grad_steps"] for _s, df, _e in done])) if done else None,
            "median_env": int(np.median([df.iloc[-1]["total_env_steps"] for _s, df, _e in done])) if done else None,
            "eval_curve": _curve(runs, "episode", "eval_mean_reward"),
            "reward_curve": _curve(runs, "episode", "avg_reward_100"),
            "loss_curve": _curve(runs, "episode", "mean_loss"),
        }
        cols = runs[0][1].columns
        if "explained_variance" in cols:
            entry["ev_curve"] = _curve(runs, "episode", "explained_variance")
            entry["vloss_curve"] = _curve(runs, "episode", "value_loss")
        if "entropy" in cols:
            entry["entropy_curve"] = _curve(runs, "episode", "entropy")
        sets[f"{config}_{tag}" if tag else config] = entry
    return sets


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _band(ax, s, curve_key, label, key, direct_label=True):
    c = s.get(curve_key)
    if not c:
        return None
    color = COLORS.get(key, MUTED)
    x, med, lo, hi = c["x"], c["median"], c["lo"], c["hi"]
    ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.plot(x, med, color=color, linewidth=2.0,
            label=f"{label}  (n={c['n']})", zorder=3)
    if direct_label:
        ax.annotate(label, xy=(x[-1], med[-1]), xytext=(4, 0),
                    textcoords="offset points", color=color, fontsize=8,
                    va="center", fontweight="bold")
    return med[-1]


def fig_grid_reward(sets):
    """The headline: greedy-eval reward for the four actor/critic combinations."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style(ax)
    order = [("qa2c", "Q2Q"), ("q2c", "Q2C"), ("a2q", "A2Q"), ("mlp_a2c", "A2C")]
    any_drawn = False
    for key, label in order:
        if key in sets and _band(ax, sets[key], "eval_curve", label, key) is not None:
            any_drawn = True
    if not any_drawn:
        plt.close(fig)
        return
    ax.axhline(475, color=SOLVE, linewidth=1.2, linestyle="--", zorder=2)
    ax.annotate("solve threshold 475", xy=(0.01, 475), xycoords=("axes fraction", "data"),
                xytext=(0, 5), textcoords="offset points", color=SOLVE, fontsize=8)
    ax.set_xlabel("episode")
    ax.set_ylabel("greedy evaluation reward (median of seeds, IQR band)")
    ax.set_title("Actor/critic grid on CartPole-v1 — greedy policy quality",
                 color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(f"{OUTDIR}/10_grid_reward.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/10_grid_reward.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_algorithms(sets):
    """Quantum agents across the three algorithms, with classical controls."""
    fig, (ax_q, ax_c) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax in (ax_q, ax_c):
        _style(ax)
        ax.axhline(475, color=SOLVE, linewidth=1.2, linestyle="--", zorder=2)
        ax.set_xlabel("episode")

    for key, label in [("qpg", "QPG"), ("qa2c", "Q2Q"), ("qdqn_vec10", "QDQN")]:
        if key in sets:
            _band(ax_q, sets[key], "eval_curve", label, key)
    for key, label in [("mlp_pg", "MLP-PG"), ("mlp_a2c", "MLP-A2C"), ("mlp_baseline", "MLP-DQN")]:
        if key in sets:
            _band(ax_c, sets[key], "eval_curve", label, key)

    ax_q.set_ylabel("greedy evaluation reward")
    ax_q.set_title("Quantum", color=INK, fontsize=11, loc="left")
    ax_c.set_title("Classical (parameter-matched)", color=INK, fontsize=11, loc="left")
    for ax in (ax_q, ax_c):
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Same circuit, three algorithms — and the classical controls",
                 color=INK, fontsize=12, x=0.09, ha="left")
    fig.savefig(f"{OUTDIR}/11_algorithms.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/11_algorithms.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_critic(sets):
    """Does the critic earn its keep? Explained variance and the value loss."""
    have = [k for k in ("qa2c", "q2c", "a2q", "mlp_a2c") if k in sets and sets[k].get("ev_curve")]
    if not have:
        return
    fig, (ax_ev, ax_vl) = plt.subplots(1, 2, figsize=(13, 5))
    for ax in (ax_ev, ax_vl):
        _style(ax)
        ax.set_xlabel("episode")

    labels = {"qa2c": "Q2Q", "q2c": "Q2C", "a2q": "A2Q", "mlp_a2c": "A2C"}
    for key in have:
        _band(ax_ev, sets[key], "ev_curve", labels[key], key)
        _band(ax_vl, sets[key], "vloss_curve", labels[key], key)

    # ev = 0 is exactly the performance of predicting the batch mean, which is
    # what REINFORCE's baseline already does for free.
    ax_ev.axhline(0.0, color=SOLVE, linewidth=1.2, linestyle="--", zorder=2)
    ax_ev.annotate("0 = no better than the batch mean\n(REINFORCE's free baseline)",
                   xy=(0.02, 0.0), xycoords=("axes fraction", "data"),
                   xytext=(0, 6), textcoords="offset points", color=SOLVE, fontsize=8)
    ax_ev.set_ylabel("critic explained variance")
    ax_ev.set_title("Critic quality", color=INK, fontsize=11, loc="left")
    ax_vl.set_ylabel("value loss (Huber)")
    ax_vl.set_yscale("log")
    ax_vl.set_title("Value loss", color=INK, fontsize=11, loc="left")
    for ax in (ax_ev, ax_vl):
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Does the critic earn the second circuit evaluation it costs?",
                 color=INK, fontsize=12, x=0.09, ha="left")
    fig.savefig(f"{OUTDIR}/12_critic.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/12_critic.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cost(sets):
    """Wall-clock and gradient-step cost of each configuration."""
    keys = [k for k in ("qpg", "qa2c", "q2c", "a2q", "qdqn_vec10",
                        "mlp_pg", "mlp_a2c", "mlp_baseline") if k in sets]
    keys = [k for k in keys if sets[k].get("median_wall_s") is not None]
    if not keys:
        return
    short = {"qpg": "QPG", "qa2c": "Q2Q", "q2c": "Q2C", "a2q": "A2Q",
             "qdqn_vec10": "QDQN", "mlp_pg": "MLP-PG", "mlp_a2c": "MLP-A2C",
             "mlp_baseline": "MLP-DQN"}
    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax in (ax_t, ax_g):
        _style(ax)
        ax.grid(axis="y", visible=False)

    y = np.arange(len(keys))
    for ax, field, xlabel, logscale in (
        (ax_t, "median_wall_s", "median training wall-clock (s)", False),
        (ax_g, "median_grad", "median gradient steps (log)", True),
    ):
        vals = [sets[k][field] for k in keys]
        ax.barh(y, vals, height=0.62,
                color=[COLORS.get(k, MUTED) for k in keys], zorder=3)
        ax.set_yticks(y, [f"{short[k]}  ({sets[k]['n_done']}/{sets[k]['n_seeds']})" for k in keys])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        if logscale:
            ax.set_xscale("log")
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:,.0f}", xy=(v, yi), xytext=(4, 0),
                        textcoords="offset points", va="center",
                        fontsize=8, color=INK)
        ax.margins(x=0.16)

    fig.suptitle("Cost: policy gradient takes far fewer, far more expensive steps",
                 color=INK, fontsize=12, x=0.09, ha="left")
    fig.savefig(f"{OUTDIR}/13_cost.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/13_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_progress(sets):
    """Training reward and training loss, side by side, for every arm.

    A WARNING LIVES ON THIS FIGURE. The two panels are not the same kind of
    quantity. Reward is directly comparable across arms. Loss is NOT: DQN's is a
    TD error (an error -- lower is better), while the policy-gradient and A2C
    numbers are SURROGATE objectives, -(log pi * advantage) plus a value term.
    With advantages normalized per batch that surrogate hovers near zero by
    construction and its magnitude carries almost no information about policy
    quality; it is plotted because it is what "training loss" means for these
    algorithms, not because low is good. The value loss in figure 12 is the
    A2C-side number that does behave like an error.
    """
    keys = [k for k in ("qpg", "qa2c", "q2c", "a2q", "mlp_a2c", "mlp_pg")
            if k in sets and sets[k].get("reward_curve")]
    if not keys:
        return
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax in (ax_r, ax_l):
        _style(ax)
        ax.set_xlabel("episode")

    for key in keys:
        _band(ax_r, sets[key], "reward_curve", sets[key]["label"], key)
        _band(ax_l, sets[key], "loss_curve", sets[key]["label"], key)

    ax_r.axhline(475, color=SOLVE, linewidth=1.2, linestyle="--", zorder=2)
    ax_r.annotate("solve threshold 475", xy=(0.02, 475),
                  xycoords=("axes fraction", "data"), xytext=(0, 5),
                  textcoords="offset points", color=SOLVE, fontsize=8)
    ax_r.set_ylabel("training reward (100-episode average)")
    ax_r.set_title("Reward progress — comparable across arms",
                   color=INK, fontsize=11, loc="left")
    ax_l.set_ylabel("training loss (surrogate objective)")
    ax_l.set_title("Loss progress — NOT comparable across algorithms",
                   color=INK, fontsize=11, loc="left")
    ax_l.annotate("policy-gradient loss is a surrogate, not an error:\n"
                  "with per-batch normalized advantages it sits near 0\n"
                  "by construction. See figure 12 for the value loss.",
                  xy=(0.03, 0.06), xycoords="axes fraction",
                  fontsize=8, color=SOLVE, va="bottom")
    for ax in (ax_r, ax_l):
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Training progress: reward rises, but the loss curve is not the story",
                 color=INK, fontsize=12, x=0.09, ha="left")
    fig.savefig(f"{OUTDIR}/14_progress.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/14_progress.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_entropy(sets):
    """Policy entropy -- the comparable 'progress' signal the loss curve is not.

    Both PG and A2C anneal their own exploration through the trainable inverse
    temperature, so entropy falling from ln(2)=0.693 toward 0 is the honest
    picture of a policy sharpening. Unlike the loss it means the same thing in
    every arm.
    """
    keys = [k for k in ("qpg", "qa2c", "q2c", "a2q", "mlp_a2c", "mlp_pg")
            if k in sets and sets[k].get("entropy_curve")]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    _style(ax)
    for key in keys:
        _band(ax, sets[key], "entropy_curve", sets[key]["label"], key)
    ax.axhline(np.log(2), color=MUTED, linewidth=1.2, linestyle=":", zorder=2)
    ax.annotate("ln 2 = 0.693 — uniform policy over two actions",
                xy=(0.02, np.log(2)), xycoords=("axes fraction", "data"),
                xytext=(0, -12), textcoords="offset points",
                color=MUTED, fontsize=8)
    ax.set_xlabel("episode")
    ax.set_ylabel("policy entropy (nats)")
    ax.set_title("Exploration anneals itself — no epsilon schedule anywhere",
                 color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(f"{OUTDIR}/15_entropy.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/15_entropy.pdf", bbox_inches="tight")
    plt.close(fig)


def print_table(sets):
    """The table view the palette's contrast WARN obliges us to ship."""
    print(f"\n{'config':>10} {'actor':>10} {'critic':>10} {'seeds':>6} "
          f"{'greedy':>8} {'range':>15} {'solve':>7} {'t(s)':>7} {'grad':>7} {'env':>9}")
    for key, s in sets.items():
        if s["greedy_median"] is None:
            continue
        rng = f"[{s['greedy_min']:.0f}, {s['greedy_max']:.0f}]"
        print(f"{key:>10} {s['actor']:>10} {s['critic']:>10} "
              f"{s['n_done']:>3}/{s['n_seeds']:<2} {s['greedy_median']:>8.1f} {rng:>15} "
              f"{s['greedy_solves']:>3}/{s['n_done']:<3} {s['median_wall_s']:>7.0f} "
              f"{s['median_grad']:>7} {s['median_env']:>9,}")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    sets = load()
    fig_grid_reward(sets)
    fig_algorithms(sets)
    fig_critic(sets)
    fig_cost(sets)
    fig_progress(sets)
    fig_entropy(sets)
    print_table(sets)
    print(f"\nwrote figures to {OUTDIR}/10..15_*.png and .pdf")
