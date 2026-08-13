"""
Paired per-episode comparison of DQN against the greedy controllers.

Every policy in this project is evaluated on the same fixed evaluation seeds
(1000..1049), and each seed pins both the initial state and the OU noise
realization. So the episodes are matched pairs and a paired test is available --
which matters here, because the per-episode spread (sd ~ 126) is as large as the
mean, and unpaired error bars overlap almost completely even when one policy is
consistently better.

Usage:
    python scripts/paired_compare.py --noise-rabi 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats


def greedy_returns(nr: float, policy: str = "greedy"):
    p = Path(f"results/greedy_baseline_nr{nr}.json")
    if not p.exists():
        p = Path(f"results/greedy_masked_nr{nr}.json")
    d = json.loads(p.read_text())
    eps = d["per_episode"][policy]
    return np.array([e["return"] for e in sorted(eps, key=lambda e: e["seed"])])


def masked_returns(nr: float, policy: str):
    """None when no masked baseline was run at this noise level."""
    p = Path(f"results/greedy_masked_nr{nr}.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    eps = d["per_episode"][policy]
    return np.array([e["return"] for e in sorted(eps, key=lambda e: e["seed"])])


def dqn_returns(mode: str, nr: float):
    """(n_seeds, n_eval_episodes) matrix of returns."""
    rows, seeds = [], []
    for f in sorted(glob.glob(f"results/dqn_{mode}_nr{nr}_seed*.json")):
        d = json.loads(Path(f).read_text())
        rows.append(d["final"]["returns"])
        seeds.append(d["seed"])
    return np.array(rows), seeds


def report(label, a, b, name_a, name_b):
    """a, b are matched per-episode return vectors."""
    d = a - b
    n = len(d)
    mean_d = d.mean()
    # Wilcoxon: returns here are bimodal (early death vs long survival), so a
    # t-test's normality assumption is not safe.
    try:
        w = stats.wilcoxon(a, b, zero_method="wilcox")
        pw = w.pvalue
    except ValueError:
        pw = float("nan")
    t = stats.ttest_rel(a, b)
    wins = int((d > 0).sum())
    ties = int((d == 0).sum())
    print(f"{label}")
    print(f"    {name_a} {a.mean():8.2f}   {name_b} {b.mean():8.2f}   diff {mean_d:+8.2f}"
          f"  (95% CI {mean_d - 1.96*d.std(ddof=1)/np.sqrt(n):+.2f}"
          f" .. {mean_d + 1.96*d.std(ddof=1)/np.sqrt(n):+.2f})")
    print(f"    per-episode: {name_a} better on {wins}/{n}, ties {ties}"
          f"   Wilcoxon p={pw:.4f}   paired t p={t.pvalue:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise-rabi", type=float, default=0.5)
    args = p.parse_args()
    nr = args.noise_rabi

    g = greedy_returns(nr, "greedy")
    print(f"=== paired comparison at noise/Rabi = {nr}, {len(g)} matched eval episodes ===\n")

    full, seeds = dqn_returns("full", nr)
    if len(full):
        for i, s in enumerate(seeds):
            report(f"DQN full seed{s}  vs  greedy", full[i], g, "DQN", "greedy")
        report("DQN full (mean over seeds)  vs  greedy", full.mean(0), g, "DQN", "greedy")
        print()

    pf = masked_returns(nr, "greedy_pf")
    idle = greedy_returns(nr, "idle")
    for mode in ("masked", "masked_hist"):
        m, seeds = dqn_returns(mode, nr)
        if not len(m):
            continue
        if pf is not None:
            report(f"DQN {mode} (mean over seeds)  vs  masked greedy_pf", m.mean(0), pf, "DQN", "pf")
        report(f"DQN {mode} (mean over seeds)  vs  always IDLE", m.mean(0), idle, "DQN", "idle")
        print()


if __name__ == "__main__":
    main()
