"""Multi-seed sweep.  [IMPLEMENTED AT CHECKPOINT 5]

    python scripts/sweep_seeds.py --config configs/qdqn.yaml --seeds 0 1 2 3 4

Runs >=5 seeds (single-run curves are not evidence in this field), writing one
results/{name}_{seed}.csv per seed. Aggregation (median + IQR) happens in plots.py.
Prints a wall-clock projection before launching the full sweep.
"""
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("scripts/sweep_seeds.py is implemented at Checkpoint 5.")
