"""Single-run entrypoint.

    python scripts/train.py --config configs/qdqn.yaml --seed 0
    python scripts/train.py --config configs/mlp_baseline.yaml --seed 0 --max-episodes 500

Loads a YAML config, prints resolved versions, seeds everything (BEFORE building the
model so weight init is reproducible), builds the model via the factory, and runs the
model-agnostic trainer, writing results/{name}_{seed}.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config          # noqa: E402
from src.models import build_model          # noqa: E402
from src.seeds import set_global_seeds      # noqa: E402
from src.trainer import train               # noqa: E402
from src.versions import print_versions     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a (Q)DQN agent on CartPole-v1.")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=None, help="override trainer.max_episodes")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--quiet-warnings", action="store_true", help="silence qiskit V1-deprecation noise")
    args = ap.parse_args()

    if args.quiet_warnings:
        warnings.filterwarnings("ignore")

    print_versions()
    config = load_config(args.config)
    if args.max_episodes is not None:
        config["trainer"]["max_episodes"] = args.max_episodes

    set_global_seeds(args.seed)              # seed BEFORE build_model -> reproducible weight init
    model = build_model(config)
    name = config["name"]
    print(f"model: {name} | trainable params: {model.num_trainable_params()} | seed: {args.seed}")

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, f"{name}_{args.seed}.csv")

    summary = train(model, config, args.seed, results_path, progress_every=args.progress_every)
    print("summary:", summary)


if __name__ == "__main__":
    main()
