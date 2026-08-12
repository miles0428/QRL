"""Multi-seed sweep — runs one training subprocess per seed, in parallel.

    python scripts/sweep_seeds.py --config configs/qdqn.yaml --seeds 0 1 2 3 4
    python scripts/sweep_seeds.py --config configs/qdqn.yaml --seeds 0 1 2 3 4 --max-parallel 5

Each seed is a separate `scripts/train.py` process (writes its own {name}_{seed}.csv
and .log). We parallelize at the SEED level with subprocesses -- robust on Windows and
free of the multiprocessing-spawn fragility that in-process quantum parallelism hits.
Single-run curves are not evidence in this field, so >=5 seeds is the norm; aggregation
(median + IQR) happens in plots.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(config: str, seed: int, results_dir: str, max_episodes, progress_every: int) -> tuple[int, int]:
    cmd = [sys.executable, os.path.join(REPO, "scripts", "train.py"),
           "--config", config, "--seed", str(seed),
           "--results-dir", results_dir, "--quiet-warnings",
           "--progress-every", str(progress_every)]
    if max_episodes is not None:
        cmd += ["--max-episodes", str(max_episodes)]
    # Child writes its own CSV + .log; we don't capture stdout (follow the .log files).
    proc = subprocess.run(cmd, cwd=REPO)
    return seed, proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a multi-seed training sweep in parallel.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--max-parallel", type=int, default=None,
                    help="max concurrent seeds (default: min(#seeds, cpu_count//2))")
    args = ap.parse_args()

    os.makedirs(os.path.join(REPO, args.results_dir), exist_ok=True)
    default_par = min(len(args.seeds), max(1, (os.cpu_count() or 2) // 2))
    max_parallel = args.max_parallel or default_par

    print(f"sweep: config={args.config} | seeds={args.seeds} | "
          f"max_parallel={max_parallel} | cores={os.cpu_count()}", flush=True)
    print(f"follow live: tail -f {args.results_dir}/<name>_<seed>.log", flush=True)

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = [ex.submit(run_one, args.config, s, args.results_dir,
                          args.max_episodes, args.progress_every) for s in args.seeds]
        for fut in as_completed(futs):
            seed, rc = fut.result()
            status = "ok" if rc == 0 else f"FAILED(rc={rc})"
            print(f"  seed {seed} finished: {status} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    print(f"sweep done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
