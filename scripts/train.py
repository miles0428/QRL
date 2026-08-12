"""CLI entry point: train one (config, seed) run and write results/{name}_{seed}.csv.

This script -- not src/trainer.py -- is where it's okay to know whether the
model is quantum: it reads the config, decides which model class to build,
and assembles the (possibly multi-group) optimizer. trainer.py never sees any
of that.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.seeds import set_seed
from src.trainer import train


def print_resolved_versions() -> None:
    packages = [
        "torch",
        "qiskit",
        "qiskit-aer",
        "qiskit-machine-learning",
        "gymnasium",
        "numpy",
        "matplotlib",
        "pandas",
        "pyyaml",
    ]
    print(f"python: {sys.version.split()[0]}")
    for pkg in packages:
        try:
            print(f"{pkg}: {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")


def build_model_and_optimizer(config: dict, seed: int):
    if config["model"] == "vqc":
        from src.models.vqc import VQCQFunction

        model = VQCQFunction(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            reuploading=config.get("reuploading", False),
            seed=seed,
        )
        optimizer = torch.optim.Adam(
            [
                {"params": [model.lam], "lr": config["lr_lam"]},
                {"params": model.vqc.parameters(), "lr": config["lr_vqc"]},
                {"params": [model.w], "lr": config["lr_w"]},
            ]
        )
    elif config["model"] == "mlp":
        from src.models.mlp import MLPQFunction

        model = MLPQFunction(hidden=config["hidden"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"unknown model type: {config['model']}")

    return model, optimizer


def main():
    parser = argparse.ArgumentParser(description="Train a QDQN/MLP agent on CartPole-v1")
    parser.add_argument("--config", type=str, required=True, help="path to a configs/*.yaml file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--max-episodes", type=int, default=None, help="override config's max_episodes")
    parser.add_argument("--max-wall-clock-s", type=float, default=None, help="safety cutoff, disabled by default")
    args = parser.parse_args()

    print_resolved_versions()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Seed *before* constructing the model so parameter initialization is
    # reproducible too -- trainer.train()'s own set_seed() call happens after
    # the model already exists and can't retroactively fix this.
    set_seed(args.seed)
    model, optimizer = build_model_and_optimizer(config, args.seed)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nconfig: {config['name']} | seed: {args.seed} | trainable params: {n_params}")

    results_path = f"{args.results_dir}/{config['name']}_{args.seed}.csv"
    max_episodes = args.max_episodes if args.max_episodes is not None else config["max_episodes"]

    result = train(
        model,
        optimizer,
        results_path=results_path,
        seed=args.seed,
        env_id=config["env_id"],
        max_episodes=max_episodes,
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        buffer_capacity=config["buffer_capacity"],
        min_buffer_size=config["min_buffer_size"],
        target_update_every=config["target_update_every"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay_steps=config["epsilon_decay_steps"],
        solve_threshold=config["solve_threshold"],
        solve_window=config["solve_window"],
        max_steps_per_episode=config["max_steps_per_episode"],
        max_wall_clock_s=args.max_wall_clock_s,
    )

    print("\n=== training summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    torch.save(model.state_dict(), results_path.replace(".csv", "_final.pt"))
    print(f"saved final model weights to {results_path.replace('.csv', '_final.pt')}")


if __name__ == "__main__":
    main()
