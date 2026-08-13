"""Per-seed summary for one experiment tag, and A/B comparison between two.

Every status check in this project was ad-hoc inline python, which is how a seed
label got mis-parsed out of a filename more than once. This is the same query as
a command.

Reports BOTH solve criteria, never one. They disagree in both directions once
epsilon reaches its floor: the training criterion needs 100 consecutive episodes
averaging >= 475 and so certifies a newly-learned policy slowly, while the greedy
evaluation reads the policy directly but at a single instant and is therefore
exposed to oscillation phase. Observed on the same sweep: one seed passed
training at 475.1 but evaluated at 462.7, another failed training outright yet
finished at a perfect 500.0.

Standard deviation is printed next to every mean because it separates a stable
solve from a lucky window better than the mean does -- 404.6 +- 190.8 is a
policy swinging between 500 and near zero, not a policy at 404.

Usage:
    python scripts/summarize.py vec10
    python scripts/summarize.py vec10 --vs pl10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


def load_tag(tag: str, results_dir: str = "results", config: str = "qdqn") -> list[dict]:
    pattern = os.path.join(results_dir, f"{config}_{tag}_*.csv" if tag else f"{config}_*.csv")
    # The glob already handles an empty tag (the untagged runs, results/{config}_{seed}.csv)
    # but the regex did not: it interpolated to "{config}__(\d+)" and matched nothing, so
    # `summarize.py "" --config qdqn` silently reported 0/0 rather than the untagged sweep.
    # Anchored at the start so an untagged query cannot pick up tagged runs.
    prefix = f"{re.escape(config)}_{re.escape(tag)}_" if tag else f"{re.escape(config)}_"
    rows = []
    for path in sorted(glob.glob(pattern)):
        m = re.match(rf"{prefix}(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        last = df.iloc[-1]
        eval_path = path.replace(".csv", "_eval.json")
        blob = json.load(open(eval_path)) if os.path.exists(eval_path) else None
        rows.append({
            "seed": int(m.group(1)),
            "episodes": int(last["episode"]),
            "env_steps": int(last["total_env_steps"]),
            "wall_clock_s": float(last["wall_clock_s"]),
            "train_avg100": float(last["avg_reward_100"]),
            "train_solved": bool(blob["training_criterion"]["solved"]) if blob else None,
            "solve_ep": (blob["training_criterion"]["episodes_to_solve"] if blob else None),
            "greedy": (blob["greedy"]["mean_reward"] if blob else None),
            "greedy_std": (blob["greedy"]["std_reward"] if blob else None),
            "greedy_solved": (blob["greedy"]["solved"] if blob else None),
            "done": blob is not None,
        })
    return rows


def summarize(tag: str, rows: list[dict]) -> dict:
    print(f"\n=== {tag} ({sum(r['done'] for r in rows)}/{len(rows)} complete) ===")
    print(f"{'seed':>4} {'eps':>5} {'steps':>7} {'t(s)':>5} {'avg100':>7} "
          f"{'train':>6} {'solve@':>6} {'greedy100':>16} {'ok':>5}")
    for r in sorted(rows, key=lambda r: r["seed"]):
        g = f"{r['greedy']:7.1f}+-{r['greedy_std']:6.1f}" if r["done"] else " " * 14
        print(f"{r['seed']:>4} {r['episodes']:>5} {r['env_steps']:>7} {r['wall_clock_s']:>5.0f} "
              f"{r['train_avg100']:>7.1f} {str(r['train_solved']):>6} "
              f"{str(r['solve_ep'] or '-'):>6} {g:>16} {str(r['greedy_solved']):>5}")

    done = [r for r in rows if r["done"]]
    if not done:
        return {}
    greedy = np.array([r["greedy"] for r in done])
    stats = {
        "n": len(done),
        "greedy_median": float(np.median(greedy)),
        "greedy_q1": float(np.percentile(greedy, 25)),
        "greedy_q3": float(np.percentile(greedy, 75)),
        # Mean of the per-seed spreads: how stable a solved policy is, distinct
        # from how much the seeds disagree with each other.
        "mean_within_seed_std": float(np.mean([r["greedy_std"] for r in done])),
        "greedy_solves": sum(bool(r["greedy_solved"]) for r in done),
        "train_solves": sum(bool(r["train_solved"]) for r in done),
    }
    print(f"  greedy median {stats['greedy_median']:.1f}  "
          f"IQR [{stats['greedy_q1']:.1f}, {stats['greedy_q3']:.1f}]  "
          f"mean within-seed std {stats['mean_within_seed_std']:.1f}")
    print(f"  solves: greedy {stats['greedy_solves']}/{stats['n']}, "
          f"training {stats['train_solves']}/{stats['n']}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag")
    p.add_argument("--vs", default=None, help="second tag to compare against")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--config", default="qdqn")
    args = p.parse_args()

    a = summarize(args.tag, load_tag(args.tag, args.results_dir, args.config))
    if not args.vs:
        return
    b = summarize(args.vs, load_tag(args.vs, args.results_dir, args.config))

    if a and b:
        print(f"\n=== {args.tag} vs {args.vs} ===")
        for key, label in (("greedy_median", "greedy median"),
                           ("mean_within_seed_std", "mean within-seed std"),
                           ("greedy_solves", "greedy solves")):
            print(f"  {label:22s} {a[key]:8.1f}  ->  {b[key]:8.1f}   ({b[key] - a[key]:+.1f})")
        print("\n  5 seeds is too few for this difference to be significant on its own.\n"
              "  Read it as a direction to confirm, not a result -- and weigh the\n"
              "  within-seed std alongside the median, since the open problem is\n"
              "  stability rather than peak performance.")


if __name__ == "__main__":
    main()
