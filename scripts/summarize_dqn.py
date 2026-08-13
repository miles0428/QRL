"""
Aggregate DQN runs across seeds and compare against the greedy baselines.

Usage:
    python scripts/summarize_dqn.py --noise-rabi 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

MODES = ["full", "masked", "masked_hist"]


def load(mode: str, nr: float):
    out = []
    for f in sorted(glob.glob(f"results/dqn_{mode}_nr{nr}_seed*.json")):
        out.append(json.loads(Path(f).read_text()))
    return out


def baselines(nr: float):
    """Greedy reference numbers at the same noise level, same eval seeds."""
    ref = {}
    p = Path(f"results/greedy_baseline_nr{nr}.json")
    if p.exists():
        s = json.loads(p.read_text())["summary"]
        for k in ("greedy", "idle", "random"):
            if k in s:
                ref[f"greedy_baseline:{k}"] = s[k]
    p = Path(f"results/greedy_masked_nr{nr}.json")
    if p.exists():
        s = json.loads(p.read_text())["summary"]
        for k in ("greedy_pf", "greedy_pf_probe"):
            if k in s:
                ref[f"masked:{k}"] = s[k]
    return ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise-rabi", type=float, default=0.5)
    args = p.parse_args()
    nr = args.noise_rabi

    print(f"=== DQN vs greedy at noise/Rabi = {nr} "
          f"(final eval: 50 greedy episodes, seeds 1000-1049) ===\n")

    hdr = f"{'config':<24} {'seeds':>5} {'mean of seed-means':>19} {'best seed':>10} {'worst':>8} {'survive':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for mode in MODES:
        runs = load(mode, nr)
        if not runs:
            continue
        means = np.array([r["final"]["return_mean"] for r in runs])
        surv = np.array([r["final"]["survival_rate"] for r in runs])
        rows[mode] = {
            "n_seeds": len(runs),
            "mean": float(means.mean()),
            "std": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "best": float(means.max()),
            "worst": float(means.min()),
            "survival": float(surv.mean()),
            "per_seed": {r["seed"]: r["final"]["return_mean"] for r in runs},
        }
        r = rows[mode]
        print(f"{'DQN ' + mode:<24} {r['n_seeds']:>5} {r['mean']:>12.2f} +- {r['std']:<4.1f} "
              f"{r['best']:>10.2f} {r['worst']:>8.2f} {r['survival']*100:>7.0f}%")

    print()
    for name, s in baselines(nr).items():
        print(f"{name:<24} {'--':>5} {s['return_mean']:>12.2f} +- {s['return_std']:<4.1f} "
              f"{'':>10} {'':>8} {s['survival_rate']*100:>7.0f}%")

    # headline comparisons
    ref = baselines(nr)
    g = ref.get("greedy_baseline:greedy", {}).get("return_mean")
    pf = ref.get("masked:greedy_pf", {}).get("return_mean")
    print()
    if g is not None and "full" in rows:
        d = rows["full"]["mean"] - g
        print(f"unmasked : DQN {rows['full']['mean']:.2f} vs greedy {g:.2f}  -> {d:+.2f}"
              f"  ({'DQN wins' if d > 0 else 'greedy wins'})")
    if pf is not None:
        for m in ("masked", "masked_hist"):
            if m in rows:
                d = rows[m]["mean"] - pf
                print(f"masked   : DQN {m} {rows[m]['mean']:.2f} vs masked greedy {pf:.2f}"
                      f"  -> {d:+.2f}  ({'DQN wins' if d > 0 else 'greedy wins'})")

    Path("results").mkdir(exist_ok=True)
    Path(f"results/dqn_summary_nr{nr}.json").write_text(
        json.dumps({"noise_rabi": nr, "dqn": rows, "baselines": ref}, indent=2))
    print(f"\nwrote results/dqn_summary_nr{nr}.json")


if __name__ == "__main__":
    main()
