"""CLI entry point for policy-gradient runs: train one (config, seed) and write
results/{name}_{seed}.csv.

The PG counterpart of scripts/train.py, and structured the same way: this is
where it is okay to know whether the model is quantum. It reads the config,
picks the model class and backend, and assembles the (possibly multi-group)
optimizer. src/pg_trainer.py never sees any of that.

Output files are named and shaped exactly as the DQN script's, so
scripts/summarize.py and scripts/sweep_seeds.py work on qpg runs unchanged:
  results/{name}_{seed}.csv          per-episode log (PG_CSV_HEADER)
  results/{name}_{seed}_eval.json    final greedy evaluation + solve claims
  results/{name}_{seed}_final.pt     state_dict
  results/{name}_{seed}_weights.pt   backend-neutral weights (quantum only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import algo_for, load_config
from src.pg_trainer import train_pg
from src.seeds import set_seed
from scripts.train import EVAL_SEED_BASE, print_resolved_versions


def build_model_and_optimizer(config: dict, seed: int):
    if config["model_type"] == "vqc_policy":
        from src.models.vqc_policy import VQCPolicy

        model = VQCPolicy(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            reuploading=config["reuploading"],
            observables=config["observables"],
            backend=config["backend"],
            gradient_method=config["gradient_method"],
            beta_init=config["beta_init"],
            trainable_beta=config["trainable_beta"],
            per_layer_encoding=config["per_layer_encoding"],
            seed=seed,
        )
        # Three groups on three scales, mirroring the DQN script: the circuit
        # weights move slowly, the head (beta) fast. amsgrad for the same reason
        # as there -- REINFORCE gradients are heavy-tailed, and amsgrad stops the
        # effective step size growing back after a spike.
        groups = [
            {"params": [model.lam], "lr": config["lr_lam"]},
            {"params": list(model.vqc.parameters()), "lr": config["lr_vqc"]},
        ]
        if config["trainable_beta"]:
            groups.append({"params": [model.beta_raw], "lr": config["lr_beta"]})
        optimizer = torch.optim.Adam(groups, amsgrad=config["amsgrad"])
    elif config["model_type"] == "mlp_policy":
        from src.models.mlp import MLPPolicy

        model = MLPPolicy(hidden=config["hidden"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], amsgrad=config["amsgrad"])
    else:
        raise ValueError(
            f"{config['model_type']!r} is a DQN model type; train it with scripts/train.py"
        )

    return model, optimizer


def train_pg_from_config(config: dict, seed: int, results_path: str, **overrides):
    """Single place mapping a normalized config onto train_pg()'s kwargs."""
    kwargs = dict(
        env_id=config["env_id"],
        max_episodes=config["max_episodes"],
        gamma=config["gamma"],
        n_envs=config["n_envs"],
        episodes_per_update=config["episodes_per_update"],
        baseline=config["baseline"],
        normalize_advantages=config["normalize_advantages"],
        entropy_coef=config["entropy_coef"],
        max_grad_norm=config["max_grad_norm"],
        solve_threshold=config["solve_threshold"],
        solve_window=config["solve_window"],
        max_steps_per_episode=config["max_steps_per_episode"],
        eval_every=config["eval_every"],
        eval_episodes=config["eval_episodes"],
    )
    kwargs.update(overrides)

    # Seed BEFORE constructing the model, so parameter initialization is
    # reproducible too -- train_pg()'s own set_seed() runs after the model
    # exists and cannot retroactively fix that.
    set_seed(seed)
    model, optimizer = build_model_and_optimizer(config, seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nconfig: {config['name']} | seed: {seed} | trainable params: {n_params}")

    result = train_pg(model, optimizer, results_path=results_path, seed=seed, **kwargs)
    return model, result


def main():
    parser = argparse.ArgumentParser(
        description="Train a policy-gradient (REINFORCE + baseline) agent on CartPole-v1"
    )
    parser.add_argument("--config", type=str, required=True, help="path to a configs/*.yaml file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--max-episodes", type=int, default=None, help="override config's episodes")
    parser.add_argument("--backend", type=str, default=None, help="override config's gradient.backend")
    parser.add_argument("--max-wall-clock-s", type=float, default=None, help="safety cutoff, disabled by default")
    parser.add_argument("--tag", type=str, default=None, help="suffix for the results filename")
    parser.add_argument("--n-envs", type=int, default=None, help="environments stepped in lockstep")
    parser.add_argument("--episodes-per-update", type=int, default=None,
                        help="batch size in episodes; rounded up to a whole number of n_envs rounds")
    parser.add_argument("--entropy-coef", type=float, default=None, help="entropy bonus weight")
    parser.add_argument("--baseline", type=str, default=None, choices=["batch_mean", "none"])
    parser.add_argument("--eval-every", type=int, default=None, help="episodes between greedy evaluations; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=None, help="rollouts per periodic greedy evaluation")
    parser.add_argument("--final-eval-episodes", type=int, default=None,
                        help="rollouts in the end-of-training greedy evaluation; 0 skips it")
    args = parser.parse_args()

    print_resolved_versions()

    config = load_config(args.config)
    if algo_for(config) != "pg":
        raise SystemExit(
            f"config {args.config} declares model type {config['model_type']!r}, which is a "
            f"DQN model. Use scripts/train.py for it."
        )

    if args.max_episodes is not None:
        config["max_episodes"] = args.max_episodes
    if args.backend is not None:
        config["backend"] = args.backend
    for flag, key in (
        (args.n_envs, "n_envs"),
        (args.episodes_per_update, "episodes_per_update"),
        (args.entropy_coef, "entropy_coef"),
        (args.baseline, "baseline"),
        (args.eval_every, "eval_every"),
        (args.eval_episodes, "eval_episodes"),
        (args.final_eval_episodes, "final_eval_episodes"),
    ):
        if flag is not None:
            config[key] = flag

    if config["model_type"] == "vqc_policy":
        print(
            f"backend: {config['backend']} | gradient: {config['gradient_method']} | "
            f"reuploading: {config['reuploading']} | observables: {config['observables']} | "
            f"beta_init: {config['beta_init']} (trainable: {config['trainable_beta']})"
        )
    print(
        f"algo: REINFORCE | baseline: {config['baseline']} | "
        f"normalize_advantages: {config['normalize_advantages']} | "
        f"episodes_per_update: {config['episodes_per_update']} | n_envs: {config['n_envs']}"
    )

    tag = f"_{args.tag}" if args.tag else ""
    results_path = f"{args.results_dir}/{config['name']}{tag}_{args.seed}.csv"

    model, result = train_pg_from_config(
        config,
        args.seed,
        results_path,
        max_wall_clock_s=args.max_wall_clock_s,
    )

    print("\n=== training summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Headline solve claim, on the same footing as the DQN script's. The
    # `solved` field above is the TRAINING criterion, measured on episodes
    # sampled from the policy, which understates a policy that is still
    # stochastic. The standard CartPole-v1 criterion is defined on greedy play,
    # so it is measured here, once, over enough episodes to be worth quoting.
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
                    "algo": "reinforce",
                    "seed": args.seed,
                    "results_path": results_path,
                    "greedy": dict(final_eval),
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

    if config["model_type"] == "vqc_policy":
        portable_path = results_path.replace(".csv", "_weights.pt")
        torch.save(model.export_weights(), portable_path)
        print(f"saved backend-neutral weights to {portable_path}")


if __name__ == "__main__":
    main()
