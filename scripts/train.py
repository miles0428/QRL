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
from src.logutil import TeeLogger           # noqa: E402
from src.models import build_model          # noqa: E402
from src.seeds import set_global_seeds      # noqa: E402
from src.trainer import train               # noqa: E402
from src.versions import format_versions    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a (Q)DQN agent on CartPole-v1.")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=None, help="override trainer.max_episodes")
    ap.add_argument("--n-layers", type=int, default=None, help="override model.n_layers (VQC)")
    ap.add_argument("--train-every", type=int, default=None, help="override trainer.train_every")
    ap.add_argument("--learning-starts", type=int, default=None, help="override trainer.learning_starts")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--quiet-warnings", action="store_true", help="silence qiskit V1-deprecation noise")
    args = ap.parse_args()

    if args.quiet_warnings:
        warnings.filterwarnings("ignore")

    config = load_config(args.config)
    if args.max_episodes is not None:
        config["trainer"]["max_episodes"] = args.max_episodes
    if args.n_layers is not None:
        config["model"]["n_layers"] = args.n_layers
    if args.train_every is not None:
        config["trainer"]["train_every"] = args.train_every
    if args.learning_starts is not None:
        config["trainer"]["learning_starts"] = args.learning_starts

    name = config["name"]
    # Tag the run name with overrides so parallel experiments write distinct files.
    if args.n_layers is not None:
        name = f"{name}_L{args.n_layers}"
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, f"{name}_{args.seed}.csv")
    log_path = os.path.join(args.results_dir, f"{name}_{args.seed}.log")

    # Tee everything (versions banner, model info, per-episode progress) to a flushed
    # log file next to the CSV, so training can be followed live:
    #   Git Bash:   tail -f results/{name}_{seed}.log
    #   PowerShell: Get-Content results/{name}_{seed}.log -Wait
    # The per-episode CSV (results/{name}_{seed}.csv) is likewise flushed every episode.
    with TeeLogger(log_path) as log:
        log(format_versions())
        set_global_seeds(args.seed)          # seed BEFORE build_model -> reproducible weight init
        model = build_model(config)
        log(f"model: {name} | trainable params: {model.num_trainable_params()} | seed: {args.seed}")
        log(f"logging to: {log_path}  |  csv: {results_path}")

        summary = train(model, config, args.seed, results_path,
                        log=log, progress_every=args.progress_every)
        log(f"summary: {summary}")


if __name__ == "__main__":
    main()
