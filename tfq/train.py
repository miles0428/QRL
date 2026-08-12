"""DQN training for the TFQ arm -- the TFQ tutorial's Section 3 loop.

The loop below follows the tutorial exactly: Huber loss, a gradient step every
10 environment steps, target sync every 30 environment steps, epsilon decayed
once per episode after it ends, three optimizers with per-group learning rates
and amsgrad. Those are the reference implementation's values, not our tuning,
and several of them are precisely where the Qiskit arm diverges (see 報告.md
section G) -- so keeping them faithful is the point of this file.

What is NOT from the tutorial is the instrumentation. This writes the SAME CSV
schema as src/trainer.py, including the greedy-evaluation columns, so
scripts/benchmark_report.py reads both arms with no special-casing and the
comparison is like for like:

    episode, total_env_steps, episode_reward, avg_reward_100, epsilon,
    mean_loss, wall_clock_s, grad_steps, w, lam,
    eval_mean_reward, eval_std_reward

Two instrumentation choices worth stating:

  - Solve criterion is OURS (mean reward >= 475 over 100 consecutive episodes),
    not the tutorial's (mean of last 10 >= 500). Comparing arms requires one
    criterion, and the tutorial's 10-episode window is too short to be a solve.
  - A periodic greedy (epsilon=0) evaluation is added. Training reward is
    recorded under exploration and understates the policy; on the Qiskit arm the
    gap was 33.7 vs 153.2 at the same episode. Without this the two arms would
    be compared on a metric that hides policy collapse.

Runs in WSL:
    cd /mnt/c/Users/tseng/quantum\\ hackthon\\ 2026/qdqn-cartpole
    ~/tfq/.venv/bin/python tfq/train.py --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tfq.model import W_IN, W_OUT, W_VAR, build_q_model, parameter_group_report

CSV_HEADER = [
    "episode", "total_env_steps", "episode_reward", "avg_reward_100", "epsilon",
    "mean_loss", "wall_clock_s", "grad_steps", "w", "lam",
    "eval_mean_reward", "eval_std_reward",
]

EVAL_SEED_BASE = 10_000  # matches src/trainer.py, so both arms evaluate on the same start states


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def greedy_action(model, state) -> int:
    q = model([tf.convert_to_tensor([np.asarray(state, dtype=np.float32)])])
    return int(tf.argmax(q[0]).numpy())


def run_greedy_rollouts(model, n_episodes, env_id, seed, max_steps_per_episode=500):
    """Fully greedy (epsilon=0) rollouts. Mirrors src/evaluate.py so the two arms
    are evaluated identically, on the same seeded start states."""
    env = gym.make(env_id)
    rewards = []
    try:
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            for _ in range(max_steps_per_episode):
                obs, reward, terminated, truncated, _ = env.step(greedy_action(model, obs))
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
    finally:
        env.close()
    return rewards


def param_to_str(variable) -> str:
    return ",".join(f"{v:.6f}" for v in np.asarray(variable).ravel())


def main() -> None:
    p = argparse.ArgumentParser(description="TFQ QDQN on CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=2000, help="tutorial default: 2000")
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--steps-per-update", type=int, default=10)
    p.add_argument("--steps-per-target-update", type=int, default=30)
    p.add_argument("--buffer", type=int, default=10_000)
    p.add_argument("--eps-init", type=float, default=1.0)
    p.add_argument("--eps-min", type=float, default=0.01)
    p.add_argument("--eps-decay", type=float, default=0.99)
    p.add_argument("--lr-in", type=float, default=0.001)
    p.add_argument("--lr-var", type=float, default=0.001)
    p.add_argument("--lr-out", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--final-eval-episodes", type=int, default=100)
    p.add_argument("--solve-threshold", type=float, default=475.0)
    p.add_argument("--solve-window", type=int, default=100)
    p.add_argument("--max-steps-per-episode", type=int, default=500)
    p.add_argument("--max-wall-clock-s", type=float, default=None)
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    print(f"tensorflow {tf.__version__} | python {sys.version.split()[0]}")
    set_seed(args.seed)

    n_actions = 2
    model = build_q_model(args.n_qubits, args.n_layers, n_actions)
    model_target = build_q_model(args.n_qubits, args.n_layers, n_actions)
    model_target.set_weights(model.get_weights())

    print(parameter_group_report(model))
    # The three-optimizer setup indexes trainable_variables positionally. If the
    # order ever changed, the output-scaling lr (0.1) would silently be applied
    # to the circuit weights (0.001) -- it would train, just wrongly.
    shapes = [tuple(v.shape) for v in model.trainable_variables]
    expected_var = (1, 3 * (args.n_layers + 1) * args.n_qubits)
    expected_in = (args.n_qubits * args.n_layers,)
    expected_out = (1, n_actions)
    assert shapes[W_VAR] == expected_var, f"variational slot: {shapes[W_VAR]} != {expected_var}"
    assert shapes[W_IN] == expected_in, f"input-scaling slot: {shapes[W_IN]} != {expected_in}"
    assert shapes[W_OUT] == expected_out, f"output-scaling slot: {shapes[W_OUT]} != {expected_out}"
    print("parameter-group slot assignment verified")

    optimizer_in = tf.keras.optimizers.Adam(learning_rate=args.lr_in, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=args.lr_var, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=args.lr_out, amsgrad=True)

    huber = tf.keras.losses.Huber()

    def q_learning_update(states, actions, rewards, next_states, done):
        states = tf.convert_to_tensor(states)
        next_states = tf.convert_to_tensor(next_states)
        rewards = tf.convert_to_tensor(rewards)
        done = tf.convert_to_tensor(done)

        future_rewards = model_target([next_states])
        target_q = rewards + args.gamma * tf.reduce_max(future_rewards, axis=1) * (1.0 - done)
        masks = tf.one_hot(actions, n_actions)

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_variables)
            q_values = model([states])
            q_taken = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = huber(target_q, q_taken)

        grads = tape.gradient(loss, model.trainable_variables)
        for optimizer, idx in ((optimizer_in, W_IN), (optimizer_var, W_VAR), (optimizer_out, W_OUT)):
            optimizer.apply_gradients([(grads[idx], model.trainable_variables[idx])])
        return float(loss.numpy())

    env = gym.make(args.env_id)
    replay: deque = deque(maxlen=args.buffer)

    tag = f"_{args.tag}" if args.tag else ""
    results_path = f"{args.results_dir}/tfq{tag}_{args.seed}.csv"
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    epsilon = args.eps_init
    total_env_steps = 0
    grad_steps = 0
    reward_window: deque = deque(maxlen=args.solve_window)
    solved_at_episode = solved_at_env_steps = None
    start_time = time.time()

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            episode_reward = 0.0
            losses = []

            for _ in range(args.max_steps_per_episode):
                if np.random.random() > epsilon:
                    action = greedy_action(model, obs)
                else:
                    action = np.random.choice(n_actions)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # Raw observation, no normalization -- the tutorial relies on
                # tanh(lambda * x) inside the PQC instead. DIVERGES FROM src/.
                replay.append((np.asarray(obs, dtype=np.float32), action, float(reward),
                               np.asarray(next_obs, dtype=np.float32), float(done)))
                obs = next_obs
                episode_reward += reward
                total_env_steps += 1

                if total_env_steps % args.steps_per_update == 0 and len(replay) >= args.batch_size:
                    # Index-based sampling: np.random.choice cannot sample from a
                    # deque of tuples, and sampling with replacement over indices
                    # is what the tutorial's np.random.choice does anyway.
                    idx = np.random.choice(len(replay), size=args.batch_size)
                    batch = [replay[i] for i in idx]
                    losses.append(q_learning_update(
                        np.asarray([b[0] for b in batch]),
                        np.asarray([b[1] for b in batch]),
                        np.asarray([b[2] for b in batch], dtype=np.float32),
                        np.asarray([b[3] for b in batch]),
                        np.asarray([b[4] for b in batch], dtype=np.float32),
                    ))
                    grad_steps += 1

                if total_env_steps % args.steps_per_target_update == 0:
                    model_target.set_weights(model.get_weights())

                if done:
                    break

            reward_window.append(episode_reward)
            avg_reward_100 = sum(reward_window) / len(reward_window)
            mean_loss = sum(losses) / len(losses) if losses else ""
            wall_clock_s = time.time() - start_time

            eval_mean = eval_std = ""
            if args.eval_every and (episode + 1) % args.eval_every == 0:
                eval_t0 = time.time()
                ev = run_greedy_rollouts(model, args.eval_episodes, args.env_id,
                                         EVAL_SEED_BASE, args.max_steps_per_episode)
                eval_mean, eval_std = float(np.mean(ev)), float(np.std(ev))
                start_time += time.time() - eval_t0  # keep wall_clock_s training-only

            writer.writerow([
                episode, total_env_steps, episode_reward, avg_reward_100, epsilon,
                mean_loss, wall_clock_s, grad_steps,
                param_to_str(model.trainable_variables[W_OUT]),
                param_to_str(model.trainable_variables[W_IN]),
                eval_mean, eval_std,
            ])
            f.flush()

            # Epsilon decays once per episode, after it ends -- as in the tutorial.
            epsilon = max(epsilon * args.eps_decay, args.eps_min)

            if (episode + 1) % 10 == 0:
                note = f" | greedy {eval_mean:.1f}" if eval_mean != "" else ""
                print(f"episode {episode + 1}/{args.episodes} | reward {episode_reward:.1f} | "
                      f"avg100 {avg_reward_100:.1f} | epsilon {epsilon:.3f} | "
                      f"grad_steps {grad_steps} | wall_clock {wall_clock_s:.1f}s{note}")

            if len(reward_window) == args.solve_window and avg_reward_100 >= args.solve_threshold:
                solved_at_episode, solved_at_env_steps = episode + 1, total_env_steps
                print(f"SOLVED at episode {solved_at_episode} (avg100={avg_reward_100:.1f})")
                break

            if args.max_wall_clock_s is not None and wall_clock_s >= args.max_wall_clock_s:
                print(f"stopping: max_wall_clock_s={args.max_wall_clock_s} reached")
                break

    env.close()

    print(f"\nepisodes_run: {episode + 1} | total_env_steps: {total_env_steps} | "
          f"grad_steps: {grad_steps} | solved (training avg100): {solved_at_episode is not None}")

    if args.final_eval_episodes:
        print(f"\n=== final greedy evaluation ({args.final_eval_episodes} episodes) ===")
        ev = run_greedy_rollouts(model, args.final_eval_episodes, args.env_id,
                                 EVAL_SEED_BASE, args.max_steps_per_episode)
        mean_reward, std_reward = float(np.mean(ev)), float(np.std(ev))
        greedy_solved = mean_reward >= args.solve_threshold
        print(f"greedy mean reward: {mean_reward:.1f} +- {std_reward:.1f}  "
              f"| solved (greedy): {greedy_solved}  "
              f"| solved (training avg100): {solved_at_episode is not None}")

        with open(results_path.replace(".csv", "_eval.json"), "w") as fh:
            json.dump({
                "config": "tfq", "seed": args.seed, "results_path": results_path,
                "greedy": {
                    "n_episodes": args.final_eval_episodes, "mean_reward": mean_reward,
                    "std_reward": std_reward, "solved": greedy_solved,
                    "solve_threshold": args.solve_threshold, "rewards": ev,
                },
                "training_criterion": {
                    "solved": solved_at_episode is not None,
                    "episodes_to_solve": solved_at_episode,
                    "env_steps_to_solve": solved_at_env_steps,
                },
                "episodes_run": episode + 1, "total_env_steps": total_env_steps,
                "grad_steps": grad_steps,
            }, fh, indent=2)

    model.save_weights(results_path.replace(".csv", "_final.weights.h5"))
    print(f"saved weights to {results_path.replace('.csv', '_final.weights.h5')}")


if __name__ == "__main__":
    main()
