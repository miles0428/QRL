"""Aggregate every run in results/ into one comparison table.

Answers the two questions the v3 work exists to answer:

  1. throughput -- did the backend migration actually buy wall-clock, measured
     on real training rather than a microbenchmark.
  2. learning  -- does the reconfigured agent (reuploading, ZZII/IIZZ,
     target_update_every=1) improve, and how does it compare to the classical
     MLP through the identical trainer.

Reports MEDIAN and IQR across seeds, never mean +- std and never a single-run
curve: DQN seed variance in this setting is large enough that one run is not
evidence. Where seeds have reached different episode counts (a sweep still in
flight), aggregation truncates to the shortest so every reported quantile is
computed over the same set of seeds -- a median over a shrinking seed pool
would drift upward for free as the slower seeds drop out.

Runs on partial results by design; safe to call while a sweep is going.

Usage:
    python scripts/benchmark_report.py
    python scripts/benchmark_report.py --results-dir results --baseline-ref 3da70a2
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SOLVE_THRESHOLD = 475.0
SOLVE_WINDOW = 100

# The v3 brief's profiled pre-v3 cost: 413 s over 191 gradient steps, measured
# in isolation. Kept as a named constant because the pre-v3 CSV's own average
# wall-clock disagrees with it by an order of magnitude (that run covered two
# episodes, so its average is mostly startup), and the two must not be conflated.
PROFILED_BASELINE_S = 2.16


def load_run(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if len(df) and "episode_reward" in df.columns else None


def git_show(ref: str, path: str) -> pd.DataFrame | None:
    """Read a results CSV out of a git commit, for the pre-v3 baseline."""
    import io

    try:
        blob = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return load_run_from_text(blob)


def load_run_from_text(text: str) -> pd.DataFrame | None:
    import io

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return None
    return df if len(df) and "episode_reward" in df.columns else None


def throughput_row(name: str, backend: str, df: pd.DataFrame) -> dict:
    """Per-run wall-clock summary.

    s/grad step is taken from the LAST row's cumulative wall_clock and
    grad_steps, so it is the whole run's average including environment stepping
    and action selection -- not an isolated forward/backward microbenchmark.
    Those two differ by ~40% here and the honest number for "how long will a
    sweep take" is this one.
    """
    last = df.iloc[-1]
    grad_steps = float(last.get("grad_steps", 0) or 0)
    wall = float(last.get("wall_clock_s", 0) or 0)
    return {
        "run": name,
        "backend": backend,
        "episodes": int(last["episode"]) + 1,
        "env_steps": int(last.get("total_env_steps", 0) or 0),
        "grad_steps": int(grad_steps),
        "wall_clock_s": wall,
        "s_per_grad_step": wall / grad_steps if grad_steps else float("nan"),
        "best_avg100": float(df["avg_reward_100"].max()),
        "final_avg100": float(last["avg_reward_100"]),
    }


def solve_episode(df: pd.DataFrame) -> tuple[int | None, int | None]:
    """(episodes_to_solve, env_steps_to_solve) under the project criterion:
    mean reward >= 475 over 100 consecutive episodes."""
    if len(df) < SOLVE_WINDOW:
        return None, None
    rolling = df["episode_reward"].rolling(SOLVE_WINDOW).mean()
    hit = rolling[rolling >= SOLVE_THRESHOLD]
    if hit.empty:
        return None, None
    idx = hit.index[0]
    return int(df.loc[idx, "episode"]) + 1, int(df.loc[idx, "total_env_steps"])


def scaling_row(name: str, df: pd.DataFrame) -> dict | None:
    """Final values of the output scaling w and input scaling lam.

    w is the load-bearing diagnostic: <Z> is bounded in [-1, 1] but Q* at
    gamma=0.99 is ~100, so an agent whose w stays near its init of 1 cannot
    represent the value function at all and its reward curve is flat near 10
    no matter what else is true.
    """
    last = df.iloc[-1]
    if not isinstance(last.get("w"), str):
        return None
    w = [float(v) for v in last["w"].split(",")]
    lam = [float(v) for v in str(last["lam"]).split(",")]
    return {
        "run": name,
        "w_final": ", ".join(f"{v:.1f}" for v in w),
        "w_max": max(w),
        "lam_final": ", ".join(f"{v:.3f}" for v in lam),
    }


def load_final_eval(csv_path: Path) -> dict | None:
    """The end-of-training greedy evaluation written beside a run's CSV."""
    path = csv_path.with_name(csv_path.stem + "_eval.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def aggregate_eval(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Median and IQR of the GREEDY eval reward across seeds.

    This, not the training curve, is what policy quality should be read from.
    The training columns are recorded under epsilon-greedy exploration and are
    bounded well below the agent's actual ability: measured on a 30-episode
    check, training avg100 was 33.7 while the greedy policy scored 153.2 at the
    same episode.

    Seeds evaluate on the same episode grid (eval_every is a config value), so
    rows are joined on episode index and truncated to the shortest seed.
    """
    series = {}
    for name, df in runs.items():
        if "eval_mean_reward" not in df.columns:
            continue
        ev = df[["episode", "eval_mean_reward"]].dropna()
        if len(ev):
            series[name] = ev.set_index("episode")["eval_mean_reward"]
    if not series:
        return pd.DataFrame()

    joined = pd.concat(series, axis=1).dropna()
    if joined.empty:
        return pd.DataFrame()
    q1, med, q3 = (joined.quantile(q, axis=1) for q in (0.25, 0.5, 0.75))
    return pd.DataFrame(
        {
            "episode": joined.index,
            "median_greedy": med.to_numpy(),
            "q1": q1.to_numpy(),
            "q3": q3.to_numpy(),
            "iqr": (q3 - q1).to_numpy(),
        }
    )


def aggregate_seeds(runs: dict[str, pd.DataFrame], every: int = 250) -> pd.DataFrame:
    """Median and IQR of episode reward across seeds, on a common episode grid."""
    if not runs:
        return pd.DataFrame()
    n = min(len(df) for df in runs.values())
    stacked = np.stack([df["episode_reward"].to_numpy()[:n] for df in runs.values()])

    rows = []
    for ep in list(range(0, n, every)) + ([n - 1] if n - 1 not in range(0, n, every) else []):
        # Smooth over a 100-episode window before quantiling: a single episode's
        # reward is far too noisy to read a trend from, and the solve criterion
        # is defined on the 100-episode mean anyway.
        lo = max(0, ep - SOLVE_WINDOW + 1)
        window = stacked[:, lo : ep + 1].mean(axis=1)
        q1, med, q3 = np.percentile(window, [25, 50, 75])
        rows.append({"episode": ep, "median_avg100": med, "q1": q1, "q3": q3, "iqr": q3 - q1})
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame, floatfmt: str = "%.3f") -> str:
    if df.empty:
        return "  (none)"
    return df.to_string(index=False, float_format=lambda v: floatfmt % v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--baseline-ref",
        default="3da70a2",
        help="git ref holding the pre-v3 (qiskit-ML + SPSA) results for comparison",
    )
    parser.add_argument(
        "--baseline-csv",
        default="results/qdqn_prev3_0.csv",
        help="pre-v3 (qiskit-ML + SPSA) results CSV, kept under its own name so the v3 sweep "
        "cannot overwrite it -- it is the only surviving record of the old backend's "
        "wall-clock. The pre-v3 run was still going when it was committed, so this file "
        "usually holds more episodes than the commit does; whichever source has more is used.",
    )
    parser.add_argument("--every", type=int, default=250, help="episode stride in the curve table")
    args = parser.parse_args()

    rdir = Path(args.results_dir)

    qdqn = {}
    for path in sorted(rdir.glob("qdqn_[0-9].csv")):
        df = load_run(path)
        if df is not None:
            qdqn[path.stem] = df

    mlp = {}
    for path in sorted(rdir.glob("mlp_baseline_[0-9].csv")):
        df = load_run(path)
        if df is not None:
            mlp[path.stem] = df

    # TensorFlow Quantum arm. Same CSV schema on purpose (see tfq/train.py), so
    # it needs no special-casing here. NOTE when reading the tables: this arm has
    # 94 trainable parameters against the Qiskit arm's 46, and mlp_baseline's 44
    # is parameter-matched to the Qiskit arm -- so mlp-vs-tfq is not a
    # parameter-matched comparison.
    tfq_runs = {}
    for path in sorted(rdir.glob("tfq_[0-9].csv")):
        df = load_run(path)
        if df is not None:
            tfq_runs[path.stem] = df

    # Prefer whichever pre-v3 source got further: the commit captured the run
    # mid-flight, so the working copy is usually longer.
    candidates = [
        git_show(args.baseline_ref, "results/qdqn_0.csv"),
        load_run(Path(args.baseline_csv)),
    ]
    candidates = [c for c in candidates if c is not None]
    baseline = max(candidates, key=len) if candidates else None

    print("=" * 78)
    print("THROUGHPUT  -- s/grad step is the whole-run average (env stepping included)")
    print("=" * 78)
    rows = []
    if baseline is not None:
        rows.append(throughput_row("qdqn pre-v3 (baseline)", "qiskit_ml+SPSA", baseline))
    for name, df in qdqn.items():
        rows.append(throughput_row(f"qdqn v3 {name.split('_')[-1]}", "torch_sv", df))
    for name, df in tfq_runs.items():
        rows.append(throughput_row(f"tfq {name.split('_')[-1]}", "tfq/cirq", df))
    for name, df in mlp.items():
        rows.append(throughput_row(f"mlp {name.split('_')[-1]}", "classical", df))
    tp = pd.DataFrame(rows)
    print(fmt(tp))

    if baseline is not None and qdqn:
        base_s = tp.iloc[0]["s_per_grad_step"]
        v3 = tp[tp["backend"] == "torch_sv"]["s_per_grad_step"]
        if len(v3) and np.isfinite(base_s):
            v3_s = float(v3.median())
            print(f"\nv3 (torch_sv): {v3_s:.4f} s/grad step, median over {len(v3)} seeds")
            print("speedup, against two different anchors -- they disagree, so both are given:")
            print(
                f"  vs {PROFILED_BASELINE_S:.2f} s   {PROFILED_BASELINE_S / v3_s:6.1f}x   "
                f"the v3 brief's PROFILED figure (413 s / 191 steps, isolated). "
                f"This is the\n{'':16}defensible number: it is what a controlled profile of the "
                f"old path measured."
            )
            print(
                f"  vs {base_s:.2f} s  {base_s / v3_s:6.1f}x   the pre-v3 run's OWN wall-clock "
                f"({tp.iloc[0]['wall_clock_s']:.0f} s / {tp.iloc[0]['grad_steps']} steps).\n"
                f"{'':16}Reported as measured, but do not headline it: that run covered 2 "
                f"episodes,\n{'':16}so the average is dominated by startup and whatever else "
                f"shared the machine."
            )
            print(
                "\nAlso not matched conditions in the other direction: the v3 seeds above ran 5-up\n"
                "in parallel, so each is slightly slower than it would be alone (0.023 s solo).\n"
                "For a clean single-protocol comparison run scripts/benchmark_backends.py."
            )

    print("\n" + "=" * 78)
    print(f"LEARNING  -- median & IQR of the 100-episode mean reward, across {len(qdqn)} seeds")
    print("=" * 78)
    agg = aggregate_seeds(qdqn, every=args.every)
    print(fmt(agg, "%.1f"))
    if qdqn:
        n = min(len(df) for df in qdqn.values())
        print(f"\n(truncated to the shortest seed: {n} episodes; sweep may still be running)")

    print("\n" + "=" * 78)
    print(f"GREEDY EVAL  -- median & IQR of epsilon=0 rollouts, across {len(qdqn)} seeds")
    print("=" * 78)
    ev = aggregate_eval(qdqn)
    ev_tfq = aggregate_eval(tfq_runs)
    if not ev_tfq.empty:
        print("-- qiskit arm (torch_sv, 46 params) --")
    if ev.empty:
        print("  (no eval columns -- run was produced before the greedy-eval hook, or eval_every=0)")
    else:
        stride = max(1, len(ev) // 20)
        print(fmt(pd.concat([ev.iloc[::stride], ev.tail(1)]).drop_duplicates(), "%.1f"))
    if not ev_tfq.empty:
        print("\n-- tfq arm (cirq, 94 params) --")
        stride = max(1, len(ev_tfq) // 20)
        print(fmt(pd.concat([ev_tfq.iloc[::stride], ev_tfq.tail(1)]).drop_duplicates(), "%.1f"))
    print(
        "\nRead policy quality from THIS table, not the one above: the training columns are\n"
        "recorded under exploration and understate the agent (measured: 33.7 training vs\n"
        "153.2 greedy at the same episode)."
    )

    print("\n" + "=" * 78)
    print(f"SOLVE  -- mean reward >= {SOLVE_THRESHOLD:g} over {SOLVE_WINDOW} episodes")
    print("=" * 78)
    solve_rows = []
    for label, group, paths in (
        ("qdqn v3", qdqn, rdir.glob("qdqn_[0-9].csv")),
        ("tfq", tfq_runs, rdir.glob("tfq_[0-9].csv")),
        ("mlp", mlp, rdir.glob("mlp_baseline_[0-9].csv")),
    ):
        by_stem = {p.stem: p for p in paths}
        for name, df in group.items():
            ep, steps = solve_episode(df)
            final_eval = load_final_eval(by_stem[name]) if name in by_stem else None
            greedy = final_eval["greedy"] if final_eval else None
            solve_rows.append(
                {
                    "run": f"{label} {name.split('_')[-1]}",
                    # Training criterion: measured under exploration, understates.
                    "train_solved": ep is not None,
                    "train_eps_to_solve": ep if ep is not None else -1,
                    "env_steps_to_solve": steps if steps is not None else -1,
                    "best_train_avg100": float(df["avg_reward_100"].max()),
                    # Greedy criterion: the standard CartPole-v1 one, the headline.
                    "greedy_mean": greedy["mean_reward"] if greedy else float("nan"),
                    "greedy_solved": greedy["solved"] if greedy else False,
                }
            )
    print(fmt(pd.DataFrame(solve_rows), "%.1f"))
    print(
        "(-1 = not solved within the episodes run so far)\n"
        "train_* is the exploring-episode criterion and understates the agent; greedy_* is\n"
        "the standard CartPole-v1 criterion and is the number to headline. Both are shown\n"
        "so neither can be silently swapped for the other."
    )

    print("\n" + "=" * 78)
    print("SCALING PARAMETERS  -- w must climb from 1 toward ~100 (Q* at gamma=0.99)")
    print("=" * 78)
    srows = [r for name, df in qdqn.items() if (r := scaling_row(f"qdqn v3 {name.split('_')[-1]}", df))]
    if baseline is not None and (r := scaling_row("qdqn pre-v3", baseline)):
        srows.insert(0, r)
    print(fmt(pd.DataFrame(srows), "%.1f"))


if __name__ == "__main__":
    main()
