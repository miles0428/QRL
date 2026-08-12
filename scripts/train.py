"""CLI entry point: train one (config, seed) run and write results/{name}_{seed}.csv.

This script -- not src/trainer.py -- is where it's okay to know whether the
model is quantum: it reads the config, decides which model class to build,
picks the backend, and assembles the (possibly multi-group) optimizer.
trainer.py never sees any of that.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.seeds import set_seed
from src.trainer import train

# Must match src.trainer.train()'s eval_seed_base default, so the final greedy
# evaluation is scored on the same start states as the periodic curve -- and so
# neither overlaps the training seeds.
EVAL_SEED_BASE = 10_000

# qiskit-machine-learning is absent from the qtm environment (it pins
# qiskit 1.x). Missing entries are reported rather than raising: which ones are
# present is itself part of the run's provenance.
VERSION_PACKAGES = [
    "torch",
    "qiskit",
    "qiskit-aer",
    "qiskit-machine-learning",
    "qiskit-algorithms",
    "qiskit-torch-module",
    "gymnasium",
    "numpy",
    "matplotlib",
    "pandas",
    "pyyaml",
]


def print_resolved_versions() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"executable: {sys.executable}")
    for pkg in VERSION_PACKAGES:
        try:
            print(f"{pkg}: {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg}: not installed")


def resolve_batch_size(config: dict) -> int:
    """Apply the auto batch-size rule and log which value was used and why."""
    batch_size = config["batch_size"]
    if not config.get("auto_batch_size"):
        print(f"batch size: {batch_size} (from config; auto_batch_size disabled)")
        return batch_size

    from src.models.vqc import suggested_batch_size

    detected, provenance = suggested_batch_size(default=batch_size)
    print(f"batch size: {detected} (auto: {provenance}; config value {batch_size} overridden)")
    return detected


def build_model_and_optimizer(config: dict, seed: int):
    if config["model_type"] == "vqc":
        from src.models.vqc import VQCQFunction

        model = VQCQFunction(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            reuploading=config["reuploading"],
            observables=config["observables"],
            backend=config["backend"],
            gradient_method=config["gradient_method"],
            output_rescaling=config["output_rescaling"],
            seed=seed,
        )
        optimizer = torch.optim.Adam(
            [
                {"params": [model.lam], "lr": config["lr_lam"]},
                {"params": list(model.vqc.parameters()), "lr": config["lr_vqc"]},
                {"params": [model.w], "lr": config["lr_w"]},
            ]
        )
    elif config["model_type"] == "mlp":
        from src.models.mlp import MLPQFunction

        model = MLPQFunction(hidden=config["hidden"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"unknown model type: {config['model_type']}")

    return model, optimizer


def train_from_config(config: dict, seed: int, results_path: str, **overrides):
    """Single place that maps a normalized config onto trainer.train()'s kwargs,
    shared by this script and scripts/sweep_seeds.py."""
    kwargs = dict(
        env_id=config["env_id"],
        max_episodes=config["max_episodes"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        buffer_capacity=config["buffer_capacity"],
        min_buffer_size=config["min_buffer_size"],
        target_update_every=config["target_update_every"],
        steps_per_update=config["steps_per_update"],
        n_envs=config["n_envs"],
        loss_fn=config["loss_fn"],
        epsilon_schedule=config["epsilon_schedule"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_decay_steps=config["epsilon_decay_steps"],
        solve_threshold=config["solve_threshold"],
        solve_window=config["solve_window"],
        max_steps_per_episode=config["max_steps_per_episode"],
        eval_every=config["eval_every"],
        eval_episodes=config["eval_episodes"],
    )
    kwargs.update(overrides)

    set_seed(seed)
    model, optimizer = build_model_and_optimizer(config, seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nconfig: {config['name']} | seed: {seed} | trainable params: {n_params}")

    result = train(model, optimizer, results_path=results_path, seed=seed, **kwargs)
    return model, result


def main():
    parser = argparse.ArgumentParser(description="Train a QDQN/MLP agent on CartPole-v1")
    parser.add_argument("--config", type=str, required=True, help="path to a configs/*.yaml file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--max-episodes", type=int, default=None, help="override config's episodes")
    parser.add_argument("--backend", type=str, default=None, help="override config's gradient.backend")
    parser.add_argument("--batch-size", type=int, default=None, help="override config/auto batch size")
    parser.add_argument("--max-wall-clock-s", type=float, default=None, help="safety cutoff, disabled by default")
    parser.add_argument("--tag", type=str, default=None, help="suffix for the results filename")
    parser.add_argument("--n-envs", type=int, default=None, help="environments stepped in lockstep; >1 batches action selection")
    parser.add_argument("--eval-every", type=int, default=None, help="episodes between greedy evaluations; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=None, help="rollouts per periodic greedy evaluation")
    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=None,
        help="rollouts in the end-of-training greedy evaluation; 0 skips it",
    )
    args = parser.parse_args()

    print_resolved_versions()

    config = load_config(args.config)
    if args.max_episodes is not None:
        config["max_episodes"] = args.max_episodes
    if args.backend is not None:
        config["backend"] = args.backend
    for flag, key in (
        (args.n_envs, "n_envs"),
        (args.eval_every, "eval_every"),
        (args.eval_episodes, "eval_episodes"),
        (args.final_eval_episodes, "final_eval_episodes"),
    ):
        if flag is not None:
            config[key] = flag

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
        print(f"batch size: {args.batch_size} (--batch-size override)")
    else:
        config["batch_size"] = resolve_batch_size(config)

    if config["model_type"] == "vqc":
        print(
            f"backend: {config['backend']} | gradient: {config['gradient_method']} | "
            f"reuploading: {config['reuploading']} | observables: {config['observables']} | "
            f"target_update_every: {config['target_update_every']}"
        )

    tag = f"_{args.tag}" if args.tag else ""
    results_path = f"{args.results_dir}/{config['name']}{tag}_{args.seed}.csv"

    # train_from_config seeds *before* constructing the model, so parameter
    # initialization is reproducible too -- trainer.train()'s own set_seed()
    # call happens after the model exists and can't retroactively fix that.
    model, result = train_from_config(
        config,
        args.seed,
        results_path,
        max_wall_clock_s=args.max_wall_clock_s,
    )

    print("\n=== training summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Headline solve claim. The `solved` field above is the TRAINING criterion,
    # measured on epsilon-greedy episodes, and understates the agent -- at
    # epsilon=0.13 roughly one action in eight is random, which topples the pole
    # regardless of policy quality. The standard CartPole-v1 criterion is
    # defined on greedy play, so it is measured here, once, over enough episodes
    # to be worth quoting. Both numbers are reported; neither replaces the other.
    if config["final_eval_episodes"]:
        from src.evaluate import evaluate

        print(f"\n=== final greedy evaluation ({config['final_eval_episodes']} episodes) ===")
        final_eval = evaluate(
            model,
            n_episodes=config["final_eval_episodes"],
            env_id=config["env_id"],
            seed=EVAL_SEED_BASE,
            solve_threshold=config["solve_threshold"],
        )
        print(
            f"greedy mean reward: {final_eval['mean_reward']:.1f} "
            f"+- {final_eval['std_reward']:.1f}  "
            f"| solved (greedy): {final_eval['solved']}  "
            f"| solved (training avg100): {result['solved']}"
        )

        eval_path = results_path.replace(".csv", "_eval.json")
        with open(eval_path, "w") as fh:
            json.dump(
                {
                    "config": config["name"],
                    "seed": args.seed,
                    "results_path": results_path,
                    "greedy": {k: v for k, v in final_eval.items()},
                    "training_criterion": {
                        "solved": result["solved"],
                        "episodes_to_solve": result["episodes_to_solve"],
                        "env_steps_to_solve": result["env_steps_to_solve"],
                    },
                    "episodes_run": result["episodes_run"],
                    "total_env_steps": result["total_env_steps"],
                    "grad_steps": result["grad_steps"],
                },
                fh,
                indent=2,
            )
        print(f"saved final greedy evaluation to {eval_path}")

    checkpoint_path = results_path.replace(".csv", "_final.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved final model weights to {checkpoint_path}")

    if config["model_type"] == "vqc":
        # Backend-neutral copy alongside the state_dict: raw state_dict keys
        # differ between backends, so this is what makes a qtm-trained agent
        # loadable into the qiskit_ml/Aer finite-shot evaluation path.
        portable_path = results_path.replace(".csv", "_weights.pt")
        torch.save(model.export_weights(), portable_path)
        print(f"saved backend-neutral weights to {portable_path}")


if __name__ == "__main__":
    main()
