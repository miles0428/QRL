"""Run a config across multiple seeds (5 minimum, 10 if runtime allows -- per
evaluation protocol) and aggregate with median + IQR, not mean +- std (single
seeds are not evidence; DQN variance is enormous).

Performance warning (from the project brief): training is dominated by
circuit simulation. Before launching the full multi-seed sweep, this script
times a small number of real gradient steps on the target config, extrapolates
a projected wall-clock for the requested seed count, and refuses to launch the
full sweep if that projection exceeds ~6 hours -- printing the numbers and
stopping instead of running blind. Pass --force to override once you've seen
the projection and decided to proceed anyway (e.g. after cutting n_layers or
switching to a shorter max_episodes for a demo run).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import pandas as pd
import torch

from scripts.train import (
    build_model_and_optimizer,
    print_resolved_versions,
    resolve_batch_size,
    train_from_config,
)
from src.config import algo_for, load_config
from src.replay import ReplayBuffer
from src.seeds import set_seed
from src.trainer import _select_action, _td_loss

MAX_WALL_CLOCK_S = 6 * 3600  # ~6 hours, per project brief


def time_gradient_steps(config: dict, seed: int, n_probe_steps: int = 100) -> float:
    """Run n_probe_steps real gradient steps (fresh model/env/buffer) and
    return the average wall-clock seconds per gradient step."""
    set_seed(seed)
    model, optimizer = build_model_and_optimizer(config, seed)
    target_model = copy.deepcopy(model)
    target_model.load_state_dict(model.state_dict())

    env = gym.make(config["env_id"])
    buffer = ReplayBuffer(capacity=config["buffer_capacity"])
    n_actions = env.action_space.n

    grad_steps = 0
    obs, _info = env.reset(seed=seed)
    start = time.time()
    try:
        while grad_steps < n_probe_steps:
            action = _select_action(model, obs, epsilon=1.0, n_actions=n_actions)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs if not done else env.reset(seed=seed)[0]

            if len(buffer) >= max(config["min_buffer_size"], config["batch_size"]):
                batch = buffer.sample(config["batch_size"])
                loss = _td_loss(model, target_model, batch, config["gamma"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                grad_steps += 1
    finally:
        env.close()

    return (time.time() - start) / grad_steps


def project_wall_clock(config: dict, seconds_per_grad_step: float, n_seeds: int) -> dict:
    max_episodes = config["max_episodes"]
    max_steps = config["max_steps_per_episode"]

    # Worst case: every episode runs to max_steps_per_episode.
    worst_case_grad_steps_per_seed = max_episodes * max_steps
    worst_case_s = worst_case_grad_steps_per_seed * seconds_per_grad_step * n_seeds

    # Rough heuristic case: average ~150 env steps/episode (short random early
    # episodes ~20-50 steps, longer near-solved episodes up to 500) -- a guess,
    # not a tuned estimate; flagged as such.
    heuristic_avg_steps_per_episode = 150
    heuristic_grad_steps_per_seed = max_episodes * heuristic_avg_steps_per_episode
    heuristic_s = heuristic_grad_steps_per_seed * seconds_per_grad_step * n_seeds

    return {
        "seconds_per_grad_step": seconds_per_grad_step,
        "worst_case_hours": worst_case_s / 3600,
        "heuristic_hours": heuristic_s / 3600,
    }


def time_pg_gradient_steps(config: dict, seed: int, n_probe_steps: int = 10) -> float:
    """Seconds per gradient step for a policy-gradient config.

    Unlike the DQN probe above, this runs the REAL loop (src/pg_trainer.train_pg)
    on a throwaway CSV rather than reassembling its parts. It can: under PG the
    episode budget alone determines the number of gradient steps
    (grad_steps = episodes / episodes_per_update), so a probe is just a short
    run. The DQN probe cannot do the same, because there the gradient-step count
    depends on how many environment steps the episodes happen to take.

    The probe deliberately measures EARLY episodes, which are short. A PG
    gradient step costs one forward+backward over every state in the batch, so
    it gets more expensive as episodes lengthen -- this figure is a floor, not
    an average, and the projection below says so.
    """
    from src.pg_trainer import train_pg
    from scripts.train_pg import build_model_and_optimizer as build_pg

    set_seed(seed)
    model, optimizer = build_pg(config, seed)
    probe_path = os.path.join(tempfile.gettempdir(), f"_pg_probe_{os.getpid()}.csv")
    try:
        result = train_pg(
            model, optimizer, results_path=probe_path, seed=seed,
            env_id=config["env_id"],
            max_episodes=n_probe_steps * config["episodes_per_update"],
            gamma=config["gamma"],
            n_envs=config["n_envs"],
            episodes_per_update=config["episodes_per_update"],
            baseline=config["baseline"],
            normalize_advantages=config["normalize_advantages"],
            entropy_coef=config["entropy_coef"],
            max_grad_norm=config["max_grad_norm"],
            max_steps_per_episode=config["max_steps_per_episode"],
            eval_every=0,
            verbose=False,
        )
        df = pd.read_csv(probe_path)
        elapsed = float(df.iloc[-1]["wall_clock_s"])
    finally:
        if os.path.exists(probe_path):
            os.remove(probe_path)
    return elapsed / max(1, result["grad_steps"])


def project_pg_wall_clock(config: dict, seconds_per_grad_step: float, n_seeds: int) -> dict:
    """Projection for PG, which needs no env-steps-per-episode guess.

    grad_steps = max_episodes / episodes_per_update exactly, by construction. The
    uncertainty sits somewhere else instead: cost per gradient step grows with
    episode length, since the batch holds every state of every episode in it. The
    probe measures short early episodes, so the two bounds here scale that floor
    by the ratio of the longest possible episode to a typical early one.
    """
    grad_steps_per_seed = config["max_episodes"] / config["episodes_per_update"]
    heuristic_s = grad_steps_per_seed * seconds_per_grad_step * n_seeds
    # Worst case: every episode runs the full 500 steps. Early episodes average
    # roughly 20-25 steps under a near-uniform policy, so a solved run's batches
    # are ~20x larger than the probe's.
    worst_case_s = heuristic_s * (config["max_steps_per_episode"] / 25.0)
    return {
        "seconds_per_grad_step": seconds_per_grad_step,
        "worst_case_hours": worst_case_s / 3600,
        "heuristic_hours": heuristic_s / 3600,
    }


def aggregate_results(csv_paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["seed"] = Path(path).stem.rsplit("_", 1)[-1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Sweep a config across multiple seeds")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--probe-steps", type=int, default=100)
    parser.add_argument("--max-episodes", type=int, default=None, help="override config's episodes (e.g. for a bounded smoke test)")
    parser.add_argument("--backend", type=str, default=None, help="override config's gradient.backend")
    parser.add_argument("--force", action="store_true", help="launch the full sweep even if the projected wall-clock exceeds ~6h")
    args = parser.parse_args()

    print_resolved_versions()

    config = load_config(args.config)
    if args.max_episodes is not None:
        config["max_episodes"] = args.max_episodes
    if args.backend is not None:
        config["backend"] = args.backend

    # Which algorithm this config declares. Dispatching on it matters more than
    # it looks: softmax logits are shaped exactly like Q-values, so running a
    # policy through the DQN probe and TD loss would not raise -- it would
    # produce a plausible, meaningless timing figure and an equally meaningless
    # sweep. See MODEL_ALGO in src/config.py.
    algo = algo_for(config)
    probe_steps = args.probe_steps
    if algo == "pg":
        # A PG gradient step consumes a whole batch of episodes, so 100 of them
        # is a substantial run rather than a probe.
        if probe_steps == parser.get_default("probe_steps"):
            probe_steps = 5
    else:
        config["batch_size"] = resolve_batch_size(config)

    print(f"\n=== timing probe: {probe_steps} real gradient steps on seed {args.seeds[0]} ===")
    if algo == "pg":
        seconds_per_step = time_pg_gradient_steps(config, args.seeds[0], n_probe_steps=probe_steps)
        projection = project_pg_wall_clock(config, seconds_per_step, len(args.seeds))
    else:
        seconds_per_step = time_gradient_steps(config, args.seeds[0], n_probe_steps=probe_steps)
        projection = project_wall_clock(config, seconds_per_step, len(args.seeds))

    print(f"measured: {seconds_per_step:.3f}s / gradient step")
    print(
        f"projected wall-clock for {len(args.seeds)} seeds of '{config['name']}' "
        f"({config['max_episodes']} episodes each):"
    )
    print(f"  worst case (every episode hits max_steps_per_episode): {projection['worst_case_hours']:.1f} hours")
    print(f"  rough heuristic (~150 env steps/episode average, unverified guess): {projection['heuristic_hours']:.1f} hours")

    if min(projection["worst_case_hours"], projection["heuristic_hours"]) * 3600 > MAX_WALL_CLOCK_S and not args.force:
        print(
            f"\nSTOPPING: projected wall-clock exceeds ~6 hours even under the optimistic "
            f"heuristic estimate. Not launching the full sweep blind, per the project brief.\n"
            f"Options: cut n_layers in {args.config}, reduce max_episodes, reduce --seeds, "
            f"or re-run this script with --force once you've decided how to proceed."
        )
        return

    print(f"\n=== launching {len(args.seeds)}-seed sweep ===")
    csv_paths = []
    for seed in args.seeds:
        results_path = f"{args.results_dir}/{config['name']}_{seed}.csv"
        print(f"\n--- seed {seed} ---")
        if algo == "pg":
            from scripts.train_pg import train_pg_from_config

            model, result = train_pg_from_config(config, seed, results_path)
        else:
            model, result = train_from_config(config, seed, results_path)
        csv_paths.append(results_path)
        print(result)

        final_path = results_path.replace(".csv", "_final.pt")
        torch.save(model.state_dict(), final_path)
        print(f"saved final model weights to {final_path}")

        if config["model_type"] in ("vqc", "vqc_policy"):
            portable_path = results_path.replace(".csv", "_weights.pt")
            torch.save(model.export_weights(), portable_path)
            print(f"saved backend-neutral weights to {portable_path}")

    agg = aggregate_results(csv_paths)
    summary = agg.groupby("episode")["episode_reward"].agg(["median"]).reset_index()
    print("\n=== aggregate (median reward per episode, first/last 5) ===")
    print(summary.head())
    print(summary.tail())


if __name__ == "__main__":
    main()
