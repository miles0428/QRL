"""CLI entry point for A2C runs: train one (config, seed) and write
results/{name}_{seed}.csv.

The A2C counterpart of scripts/train.py and scripts/train_pg.py, structured the
same way: this is where it is okay to know whether the models are quantum. It
reads the config, builds the actor and the critic, and assembles the combined
optimizer. src/a2c_trainer.py never sees any of that.

Output files match the other two scripts, so summarize.py and sweep_seeds.py
work unchanged. The critic gets its own checkpoint alongside the actor's:
  results/{name}_{seed}.csv             per-episode log (A2C_CSV_HEADER)
  results/{name}_{seed}_eval.json       final greedy evaluation + solve claims
  results/{name}_{seed}_final.pt        actor state_dict
  results/{name}_{seed}_critic.pt       critic state_dict
  results/{name}_{seed}_weights.pt      backend-neutral actor+critic (quantum only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.a2c_trainer import train_a2c
from src.config import algo_for, load_config
from src.seeds import set_seed
from scripts.train import EVAL_SEED_BASE, print_resolved_versions

# model_type -> (actor kind, critic kind). The four cells of the actor/critic
# grid in Koelle et al. 2024 (arXiv:2401.07043), whose naming this keeps: the
# first letter is the ACTOR, the second the CRITIC.
#
#   mlp_a2c  = A2C  classical actor, classical critic
#   q2c      = Q2C  quantum   actor, classical critic
#   a2q      = A2Q  classical actor, quantum   critic
#   vqc_a2c  = Q2Q  quantum   actor, quantum   critic
#
# `vqc_a2c` and `mlp_a2c` keep their original names rather than being renamed to
# q2q/a2c: runs already on disk are keyed by config name, and renaming would
# orphan them.
A2C_KINDS = {
    "vqc_a2c": ("vqc", "vqc"),
    "mlp_a2c": ("mlp", "mlp"),
    "q2c": ("vqc", "mlp"),
    "a2q": ("mlp", "vqc"),
}


def build_models_and_optimizer(config: dict, seed: int):
    """Actor, critic, and ONE optimizer carrying both.

    The loss is combined (policy + value_coef * value), so a single backward and
    a single step cover both networks -- but they still need different learning
    rates per parameter group, for the same reasons they do on the DQN side.

    The actor and the critic are chosen INDEPENDENTLY, via A2C_KINDS. That is
    the whole 2x2 grid Koelle et al. 2024 (arXiv:2401.07043) study, and their
    naming is kept: the first letter is the actor, the second the critic.
    """
    actor_kind, critic_kind = A2C_KINDS[config["model_type"]]

    if actor_kind == "vqc":
        from src.models.vqc_policy import VQCPolicy

        actor = VQCPolicy(
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
        groups = [
            {"params": [actor.lam], "lr": config["lr_lam"]},
            {"params": list(actor.vqc.parameters()), "lr": config["lr_vqc"]},
        ]
        if config["trainable_beta"]:
            groups.append({"params": [actor.beta_raw], "lr": config["lr_beta"]})
    else:
        from src.models.mlp import MLPPolicy

        actor = MLPPolicy(hidden=config["hidden"])
        groups = [{"params": list(actor.parameters()), "lr": config["lr"]}]

    if critic_kind == "vqc":
        from src.models.vqc_value import VQCValue, value_scale

        # The critic's output weight starts at the return scale rather than at 1.
        # An A2C run takes ~60 gradient steps, and Adam moves a parameter by
        # about its learning rate per step, so a head starting at 1 cannot reach
        # V* ~ 99 inside the run -- it would stay near-constant and quietly turn
        # GAE's baseline back into REINFORCE's. Derived from the config, not
        # hardcoded; `critic_w_init` overrides it if set.
        w_init = config["critic_w_init"]
        if w_init is None:
            w_init = value_scale(config["gamma"], config["max_steps_per_episode"])
            print(f"critic w_init: {w_init:.2f} "
                  f"(auto: (1-gamma^T)/(1-gamma) at gamma={config['gamma']}, "
                  f"T={config['max_steps_per_episode']})")

        # Separate circuit, separate weights -- when both are quantum they share
        # the ansatz but not a single parameter. seed+1 so the critic does not
        # start life as an identical copy of the actor's circuit.
        critic = VQCValue(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            reuploading=config["reuploading"],
            observable=config["critic_observable"],
            backend=config["backend"],
            gradient_method=config["gradient_method"],
            output_rescaling=config["output_rescaling"],
            per_layer_encoding=config["per_layer_encoding"],
            seed=seed + 1,
            w_init=w_init,
        )
        groups += [
            {"params": [critic.lam], "lr": config["lr_critic_lam"]},
            {"params": list(critic.vqc.parameters()), "lr": config["lr_critic_vqc"]},
            # The critic's output weight does the same job w has in VQCQFunction,
            # so it gets the same much larger learning rate.
            {"params": [critic.w], "lr": config["lr_critic_w"]},
        ]
    else:
        from src.models.mlp import MLPValue
        from src.models.vqc_value import value_scale

        # Same scale initialization the quantum critic gets, for the same reason
        # and from the same derivation -- otherwise the classical critic starts
        # near 0, cannot reach V* in ~60 gradient steps, and the grid would be
        # comparing initializations rather than function approximators.
        v_init = config["critic_w_init"]
        if v_init is None:
            v_init = value_scale(config["gamma"], config["max_steps_per_episode"])
            print(f"critic value_init: {v_init:.2f} "
                  f"(auto: (1-gamma^T)/(1-gamma) at gamma={config['gamma']}, "
                  f"T={config['max_steps_per_episode']})")
        critic = MLPValue(hidden=config["critic_hidden"], value_init=v_init)
        # `w` gets the large learning rate its quantum counterpart gets, and the
        # body gets the ordinary one -- same split, same reasons.
        body = [p for n, p in critic.named_parameters() if n != "w"]
        groups += [
            {"params": body, "lr": config["lr_critic"]},
            {"params": [critic.w], "lr": config["lr_critic_w"]},
        ]

    optimizer = torch.optim.Adam(groups, amsgrad=config["amsgrad"])
    return actor, critic, optimizer


def train_a2c_from_config(config: dict, seed: int, results_path: str, **overrides):
    """Single place mapping a normalized config onto train_a2c()'s kwargs."""
    kwargs = dict(
        env_id=config["env_id"],
        max_episodes=config["max_episodes"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        n_envs=config["n_envs"],
        episodes_per_update=config["episodes_per_update"],
        normalize_advantages=config["normalize_advantages"],
        entropy_coef=config["entropy_coef"],
        value_coef=config["value_coef"],
        value_loss_fn=config["value_loss_fn"],
        max_grad_norm=config["max_grad_norm"],
        solve_threshold=config["solve_threshold"],
        solve_window=config["solve_window"],
        max_steps_per_episode=config["max_steps_per_episode"],
        eval_every=config["eval_every"],
        eval_episodes=config["eval_episodes"],
    )
    kwargs.update(overrides)

    set_seed(seed)
    actor, critic, optimizer = build_models_and_optimizer(config, seed)
    n_actor = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"\nconfig: {config['name']} | seed: {seed} | "
          f"trainable params: {n_actor + n_critic} (actor {n_actor} + critic {n_critic})")

    result = train_a2c(actor, critic, optimizer, results_path=results_path, seed=seed, **kwargs)
    return actor, critic, result


def main():
    parser = argparse.ArgumentParser(
        description="Train an advantage actor-critic (A2C + GAE) agent on CartPole-v1"
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
    parser.add_argument("--gae-lambda", type=float, default=None, help="GAE lambda; 1.0 is Monte-Carlo, 0.0 is one-step TD")
    parser.add_argument("--value-coef", type=float, default=None, help="weight on the value loss")
    parser.add_argument("--entropy-coef", type=float, default=None, help="entropy bonus weight")
    parser.add_argument("--eval-every", type=int, default=None, help="episodes between greedy evaluations; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=None, help="rollouts per periodic greedy evaluation")
    parser.add_argument("--final-eval-episodes", type=int, default=None,
                        help="rollouts in the end-of-training greedy evaluation; 0 skips it")
    args = parser.parse_args()

    print_resolved_versions()

    config = load_config(args.config)
    if algo_for(config) != "a2c":
        raise SystemExit(
            f"config {args.config} declares model type {config['model_type']!r}, which is not "
            f"an A2C model. Use scripts/train.py (DQN) or scripts/train_pg.py (REINFORCE)."
        )

    if args.max_episodes is not None:
        config["max_episodes"] = args.max_episodes
    if args.backend is not None:
        config["backend"] = args.backend
    for flag, key in (
        (args.n_envs, "n_envs"),
        (args.episodes_per_update, "episodes_per_update"),
        (args.gae_lambda, "gae_lambda"),
        (args.value_coef, "value_coef"),
        (args.entropy_coef, "entropy_coef"),
        (args.eval_every, "eval_every"),
        (args.eval_episodes, "eval_episodes"),
        (args.final_eval_episodes, "final_eval_episodes"),
    ):
        if flag is not None:
            config[key] = flag

    actor_kind, critic_kind = A2C_KINDS[config["model_type"]]
    print(f"actor: {actor_kind} | critic: {critic_kind}")
    if "vqc" in (actor_kind, critic_kind):
        line = f"backend: {config['backend']} | gradient: {config['gradient_method']}"
        if actor_kind == "vqc":
            line += f" | actor observables: {config['observables']}"
        if critic_kind == "vqc":
            line += f" | critic observable: {config['critic_observable']}"
        print(line)
    print(
        f"algo: A2C | gae_lambda: {config['gae_lambda']} | value_coef: {config['value_coef']} | "
        f"value_loss: {config['value_loss_fn']} | "
        f"episodes_per_update: {config['episodes_per_update']} | n_envs: {config['n_envs']}"
    )

    tag = f"_{args.tag}" if args.tag else ""
    results_path = f"{args.results_dir}/{config['name']}{tag}_{args.seed}.csv"

    actor, critic, result = train_a2c_from_config(
        config,
        args.seed,
        results_path,
        max_wall_clock_s=args.max_wall_clock_s,
    )

    print("\n=== training summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    if config["final_eval_episodes"]:
        from src.evaluate import evaluate

        print(f"\n=== final greedy evaluation ({config['final_eval_episodes']} episodes) ===")
        # The critic plays no part in evaluation -- it exists only to reduce the
        # variance of the actor's gradient. Greedy play is argmax over the
        # actor's logits, identical to the PG and DQN paths.
        final_eval = evaluate(
            actor,
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
                    "algo": "a2c",
                    "gae_lambda": config["gae_lambda"],
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
    torch.save(actor.state_dict(), checkpoint_path)
    print(f"saved final actor weights to {checkpoint_path}")

    critic_path = results_path.replace(".csv", "_critic.pt")
    torch.save(critic.state_dict(), critic_path)
    print(f"saved final critic weights to {critic_path}")

    # Backend-neutral copies for whichever side is quantum -- the classical side
    # has no such notion, so a mixed run saves only the quantum half.
    portable = {}
    if actor_kind == "vqc":
        portable["actor"] = actor.export_weights()
    if critic_kind == "vqc":
        portable["critic"] = critic.export_weights()
    if portable:
        portable_path = results_path.replace(".csv", "_weights.pt")
        torch.save(portable, portable_path)
        print(f"saved backend-neutral weights ({', '.join(portable)}) to {portable_path}")


if __name__ == "__main__":
    main()
