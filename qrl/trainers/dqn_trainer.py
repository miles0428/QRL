"""Model-agnostic DQN training loop.

Hard architectural rule: this module must not import anything from qiskit.
It only ever touches `model` through the generic nn.Module interface
(forward(), parameters(), state_dict()) plus optional getattr() lookups for
attributes that MAY exist (`w`, `lam`) purely for CSV logging -- it never
checks *what kind* of model it was handed. The optimizer (with however many
parameter groups / learning rates the model calls for) is built by the
caller (scripts/train.py) and passed in already-configured; this is what
lets a 3-parameter-group VQC and a single-group MLP share this exact loop.
"""

from __future__ import annotations

import copy
import csv
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from qrl.trainers.evaluate import run_greedy_rollouts
from qrl.trainers.replay import ReplayBuffer
from qrl.trainers.seeds import set_seed

CSV_HEADER = [
    "episode",
    "total_env_steps",
    "episode_reward",
    "avg_reward_100",
    "epsilon",
    "mean_loss",
    "wall_clock_s",
    "grad_steps",
    "w",
    "lam",
    # Greedy (epsilon=0) evaluation, blank except on evaluation episodes. See
    # the eval_every docstring in train() for why these columns exist at all.
    "eval_mean_reward",
    "eval_std_reward",
]


def epsilon_at(
    schedule: str,
    episode: int,
    total_env_steps: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    epsilon_decay_steps: int,
) -> float:
    """Exploration rate under either supported schedule.

    "exponential_episodes" (v3 default) is the Skolik reference schedule:
    epsilon is multiplied by `epsilon_decay` once per *episode* and floored at
    `epsilon_end`. At decay=0.99 it reaches 0.01 after ~458 episodes. Note this
    is episode-indexed, so it is unaffected by how long each episode runs --
    which matters here, because episode length grows by an order of magnitude
    as the agent improves.

    "linear_steps" is the pre-v3 schedule: linear in *environment* steps over
    `epsilon_decay_steps`. Kept so old configs reproduce exactly.
    """
    if schedule == "exponential_episodes":
        return max(epsilon_end, epsilon_start * (epsilon_decay**episode))
    if schedule == "linear_steps":
        progress = max(0.0, 1.0 - total_env_steps / epsilon_decay_steps)
        return epsilon_end + (epsilon_start - epsilon_end) * progress
    raise ValueError(f"unknown epsilon schedule: {schedule!r}")


def _select_actions_batched(model: nn.Module, states: np.ndarray, epsilon: float, n_actions: int) -> np.ndarray:
    """Epsilon-greedy actions for a batch of states in ONE forward pass.

    This is the whole point of vectorized environments here. Simulating a
    4-qubit circuit is dispatch-bound, not arithmetic-bound, so the forward cost
    is nearly flat in batch size -- measured on this machine: 6.52 ms at batch 1,
    6.81 ms at batch 8, 6.80 ms at batch 16. Stepping 16 environments together
    therefore costs about what one did, taking action selection from 6.5 ms per
    environment step to 0.43 ms.

    Random actions still get drawn for every environment (not only the ones that
    need one) so the consumption of the RNG stream does not depend on the greedy
    actions, keeping a run reproducible from its seed.
    """
    explore = np.random.random(len(states)) < epsilon
    random_actions = np.random.randint(0, n_actions, size=len(states))
    if explore.all():
        return random_actions
    with torch.no_grad():
        q_values = model(torch.as_tensor(states, dtype=torch.float32))
        greedy = q_values.argmax(dim=1).numpy()
    return np.where(explore, random_actions, greedy)


def _param_to_str(model: nn.Module, name: str) -> str:
    param = getattr(model, name, None)
    if param is None:
        return ""
    return ",".join(f"{v:.6f}" for v in param.detach().cpu().numpy().ravel())


def _select_action(model: nn.Module, state: np.ndarray, epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def _td_loss(policy_model, target_model, batch, gamma: float, loss_fn: str = "mse") -> torch.Tensor:
    states, actions, rewards, next_states, dones = batch
    states_t = torch.as_tensor(states, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32)
    dones_t = torch.as_tensor(dones, dtype=torch.float32)

    q_values = policy_model(states_t)
    q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_model(next_states_t)
        next_q_max = next_q.max(dim=1).values
        td_target = rewards_t + gamma * next_q_max * (1.0 - dones_t)

    if loss_fn == "huber":
        # Linear beyond |error| = 1 instead of quadratic, so a single large TD
        # error cannot dominate the batch. The reference implementation uses this
        # and we did not; with Q-values reaching ~100 and observed mean losses in
        # the hundreds, MSE was letting outliers drive the update.
        return nn.functional.smooth_l1_loss(q_taken, td_target)
    if loss_fn == "mse":
        return nn.functional.mse_loss(q_taken, td_target)
    raise ValueError(f"unknown loss_fn {loss_fn!r}; expected 'huber' or 'mse'")


def _train_vectorized(
    model, optimizer, results_path, checkpoint_path, seed, env_id, max_episodes, gamma, batch_size,
    buffer_capacity, min_buffer_size, n_envs, steps_per_update, loss_fn,
    target_update_every, epsilon_schedule, epsilon_start, epsilon_end, epsilon_decay,
    epsilon_decay_steps, solve_threshold, solve_window, max_steps_per_episode,
    max_wall_clock_s, eval_every, eval_episodes, eval_seed_base, print_every, verbose,
) -> dict:
    """train() with n_envs environments stepped in lockstep.

    Exists purely for wall-clock: action selection is one forward per *tick*
    rather than one per environment step, and that forward is dispatch-bound so
    it barely notices the extra batch. See _select_actions_batched.

    AUTORESET. gymnasium 1.x SyncVectorEnv uses next-step autoreset, verified
    against 1.3.0 rather than assumed: the observation returned on a terminating
    step is the true final observation (so that transition is real and gets
    stored), and the FOLLOWING step returns the reset observation with reward 0
    and terminated False. That following transition is an artifact and must not
    enter the replay buffer -- storing it would teach the agent that the state
    which just ended an episode leads to a fresh pole with zero reward.
    `just_reset` tracks it.

    Episodes complete asynchronously across environments, so one CSV row is
    written per completed episode in completion order, and `episode` counts
    completions rather than ticks.
    """
    set_seed(seed)
    env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)])

    target_model = copy.deepcopy(model)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    buffer = ReplayBuffer(capacity=buffer_capacity)
    n_actions = int(env.single_action_space.n)

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    total_env_steps = grad_steps = episode = 0
    reward_window: deque = deque(maxlen=solve_window)
    solved_at_episode = solved_at_env_steps = solved_at_wall_clock = None
    last_eval_mean: float | None = None
    best_eval_mean = float("-inf")
    next_eval_at = eval_every

    ep_reward = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=int)
    just_reset = np.zeros(n_envs, dtype=bool)
    recent_losses: deque = deque(maxlen=200)

    obs, _info = env.reset(seed=[seed + i for i in range(n_envs)])
    start_time = time.time()

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        try:
            while episode < max_episodes:
                epsilon = epsilon_at(epsilon_schedule, episode, total_env_steps,
                                     epsilon_start, epsilon_end, epsilon_decay, epsilon_decay_steps)
                actions = _select_actions_batched(model, obs, epsilon, n_actions)
                next_obs, rewards, terminated, truncated, _info = env.step(actions)
                dones = np.logical_or(terminated, truncated)

                for i in range(n_envs):
                    if just_reset[i]:
                        continue  # reset artifact, not a real transition
                    buffer.push(obs[i], int(actions[i]), float(rewards[i]),
                                next_obs[i], float(dones[i]))
                    ep_reward[i] += rewards[i]
                    ep_len[i] += 1
                    total_env_steps += 1

                    if len(buffer) >= max(min_buffer_size, batch_size) and \
                            total_env_steps % steps_per_update == 0:
                        loss = _td_loss(model, target_model, buffer.sample(batch_size), gamma, loss_fn)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        grad_steps += 1
                        loss_value = loss.item()
                        if np.isnan(loss_value):
                            raise RuntimeError(f"NaN loss at episode {episode}, step {total_env_steps}")
                        recent_losses.append(loss_value)
                        if grad_steps % target_update_every == 0:
                            target_model.load_state_dict(model.state_dict())

                just_reset = dones.copy()
                obs = next_obs

                for i in np.flatnonzero(dones):
                    reward_window.append(ep_reward[i])
                    avg_reward_100 = sum(reward_window) / len(reward_window)
                    wall_clock_s = time.time() - start_time

                    eval_mean: float | str = ""
                    eval_std: float | str = ""
                    if eval_every and episode + 1 >= next_eval_at:
                        next_eval_at += eval_every
                        eval_t0 = time.time()
                        ev = run_greedy_rollouts(model, n_episodes=eval_episodes, env_id=env_id,
                                                 seed=eval_seed_base,
                                                 max_steps_per_episode=max_steps_per_episode)
                        eval_mean, eval_std = float(np.mean(ev)), float(np.std(ev))
                        last_eval_mean = eval_mean
                        best_eval_mean = max(best_eval_mean, eval_mean)
                        start_time += time.time() - eval_t0  # training-only clock

                    writer.writerow([
                        episode, total_env_steps, float(ep_reward[i]), avg_reward_100, epsilon,
                        (sum(recent_losses) / len(recent_losses)) if recent_losses else "",
                        wall_clock_s, grad_steps,
                        _param_to_str(model, "w"), _param_to_str(model, "lam"),
                        eval_mean, eval_std,
                    ])
                    f.flush()

                    if verbose and (episode + 1) % print_every == 0:
                        note = f" | greedy {eval_mean:.1f}" if eval_mean != "" else ""
                        print(f"episode {episode + 1}/{max_episodes} | reward {ep_reward[i]:.1f} | "
                              f"avg100 {avg_reward_100:.1f} | epsilon {epsilon:.3f} | "
                              f"grad_steps {grad_steps} | wall_clock {wall_clock_s:.1f}s{note}")

                    ep_reward[i] = 0.0
                    ep_len[i] = 0
                    episode += 1

                    if len(reward_window) == solve_window and avg_reward_100 >= solve_threshold:
                        solved_at_episode, solved_at_env_steps = episode, total_env_steps
                        solved_at_wall_clock = wall_clock_s
                        if verbose:
                            print(f"SOLVED at episode {episode} (avg100={avg_reward_100:.1f})")
                        break

                if solved_at_episode is not None:
                    break
                if max_wall_clock_s is not None and time.time() - start_time >= max_wall_clock_s:
                    if verbose:
                        print(f"stopping: max_wall_clock_s={max_wall_clock_s} reached")
                    break
        finally:
            env.close()

    torch.save(
        {"policy": model.state_dict(), "target": target_model.state_dict()},
        checkpoint_path,
    )

    return {
        "results_path": results_path,
        "checkpoint_path": checkpoint_path,
        "episodes_run": episode,
        "total_env_steps": total_env_steps,
        "grad_steps": grad_steps,
        "solved": solved_at_episode is not None,
        "episodes_to_solve": solved_at_episode,
        "env_steps_to_solve": solved_at_env_steps,
        "wall_clock_to_solve_s": solved_at_wall_clock,
        "last_eval_mean_reward": last_eval_mean,
        "best_eval_mean_reward": None if best_eval_mean == float("-inf") else best_eval_mean,
        "n_envs": n_envs,
    }


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    results_path: str,
    checkpoint_path: str,
    seed: int,
    env_id: str = "CartPole-v1",
    max_episodes: int = 2000,
    gamma: float = 0.99,
    batch_size: int = 16,
    buffer_capacity: int = 10_000,
    min_buffer_size: int = 16,
    n_envs: int = 1,  # >1 collects experience from that many envs in lockstep
    steps_per_update: int = 1,  # environment steps between gradient steps
    loss_fn: str = "mse",  # "mse" | "huber"
    target_update_every: int = 1,  # gradient steps, per spec (not episodes)
    epsilon_schedule: str = "exponential_episodes",
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.99,  # per-episode multiplier, exponential_episodes
    epsilon_decay_steps: int = 20_000,  # environment steps, linear_steps only
    solve_threshold: float = 475.0,
    solve_window: int = 100,
    max_steps_per_episode: int = 500,
    max_wall_clock_s: float | None = None,
    eval_every: int = 50,
    eval_episodes: int = 5,
    eval_seed_base: int = 10_000,
    print_every: int = 10,
    verbose: bool = True,
) -> dict:
    """Run the DQN loop and write a per-episode CSV log to results_path.

    Given the same seed, this function reproduces the same sequence of env
    resets, action-selection coin flips, and replay-buffer samples. For a
    byte-identical CSV end to end, the caller must *also* seed model
    construction with the same seed (see scripts/train.py) -- this function's
    own set_seed() call happens after `model` already exists, so it cannot
    retroactively make the model's initial weights reproducible.

    GREEDY EVALUATION (`eval_every`, in episodes; 0 disables). `episode_reward`
    and `avg_reward_100` are measured on *exploring* episodes, so they are
    bounded well below the agent's actual ability whenever epsilon is
    appreciable: at epsilon=0.13 about one action in eight is random, which on
    CartPole is enough to topple the pole long before 500 steps no matter how
    good the policy is. A run can therefore look stuck near 200 while its greedy
    policy is already near-optimal. Every `eval_every` episodes this runs
    `eval_episodes` fully greedy rollouts and logs their mean/std, which is the
    curve to read for policy quality -- and the standard CartPole-v1 solve
    criterion is defined on greedy play, not on exploring play.

    The evaluation does not perturb training. `run_greedy_rollouts` builds its
    own environment and seeds it explicitly, and greedy action selection draws
    no randomness, so the `random` and `torch` streams driving exploration and
    replay sampling are untouched -- a run with eval_every=0 and a run with
    eval_every=N produce identical training columns.

    `eval_seed_base` offsets evaluation start states away from the training
    seeds, and is deliberately *fixed* across evaluations: reusing the same
    start states every time means successive points on the eval curve differ
    only by the policy, not by which initial conditions happened to be drawn.
    """
    if n_envs > 1:
        return _train_vectorized(**locals())

    set_seed(seed)
    env = gym.make(env_id)

    target_model = copy.deepcopy(model)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    buffer = ReplayBuffer(capacity=buffer_capacity)
    n_actions = env.action_space.n

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    total_env_steps = 0
    grad_steps = 0
    reward_window: deque = deque(maxlen=solve_window)
    solved_at_episode = None
    solved_at_env_steps = None
    solved_at_wall_clock = None
    last_eval_mean: float | None = None
    best_eval_mean = float("-inf")

    start_time = time.time()

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        try:
            for episode in range(max_episodes):
                obs, _info = env.reset(seed=seed + episode)
                episode_reward = 0.0
                episode_losses: list[float] = []

                for _step in range(max_steps_per_episode):
                    epsilon = epsilon_at(
                        epsilon_schedule,
                        episode,
                        total_env_steps,
                        epsilon_start,
                        epsilon_end,
                        epsilon_decay,
                        epsilon_decay_steps,
                    )
                    action = _select_action(model, obs, epsilon, n_actions)
                    next_obs, reward, terminated, truncated, _info = env.step(action)
                    done = terminated or truncated

                    buffer.push(obs, action, reward, next_obs, float(done))
                    obs = next_obs
                    episode_reward += reward
                    total_env_steps += 1

                    # steps_per_update > 1 trains on every Nth environment step
                    # rather than every one. The reference uses 10. Beyond being
                    # the reference value it is the single largest lever on
                    # wall-clock here, since a gradient step costs far more than
                    # an environment step when the Q-function is a circuit.
                    if (
                        len(buffer) >= max(min_buffer_size, batch_size)
                        and total_env_steps % steps_per_update == 0
                    ):
                        batch = buffer.sample(batch_size)
                        loss = _td_loss(model, target_model, batch, gamma, loss_fn)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        grad_steps += 1

                        loss_value = loss.item()
                        if np.isnan(loss_value):
                            raise RuntimeError(
                                f"NaN loss at episode {episode}, env step {total_env_steps}. "
                                "Check epsilon_decay_steps (exploration), target_update_every "
                                "(training stability), and initial weight scale (barren plateau)."
                            )
                        episode_losses.append(loss_value)

                        if grad_steps % target_update_every == 0:
                            target_model.load_state_dict(model.state_dict())

                    if done:
                        break

                reward_window.append(episode_reward)
                avg_reward_100 = sum(reward_window) / len(reward_window)
                mean_loss = sum(episode_losses) / len(episode_losses) if episode_losses else ""
                wall_clock_s = time.time() - start_time

                eval_mean: float | str = ""
                eval_std: float | str = ""
                if eval_every and (episode + 1) % eval_every == 0:
                    eval_t0 = time.time()
                    eval_rewards = run_greedy_rollouts(
                        model,
                        n_episodes=eval_episodes,
                        env_id=env_id,
                        seed=eval_seed_base,
                        max_steps_per_episode=max_steps_per_episode,
                    )
                    eval_mean = float(np.mean(eval_rewards))
                    eval_std = float(np.std(eval_rewards))
                    last_eval_mean = eval_mean
                    if eval_mean > best_eval_mean:
                        best_eval_mean = eval_mean
                    # Push start_time forward by however long evaluation took, so
                    # wall_clock_s stays a training-only clock. Otherwise turning
                    # evaluation on would inflate the s/grad-step figures that the
                    # whole backend comparison rests on.
                    start_time += time.time() - eval_t0

                writer.writerow(
                    [
                        episode,
                        total_env_steps,
                        episode_reward,
                        avg_reward_100,
                        epsilon,
                        mean_loss,
                        wall_clock_s,
                        grad_steps,
                        _param_to_str(model, "w"),
                        _param_to_str(model, "lam"),
                        eval_mean,
                        eval_std,
                    ]
                )
                f.flush()

                if verbose and (episode + 1) % print_every == 0:
                    eval_note = f" | greedy {eval_mean:.1f}" if eval_mean != "" else ""
                    print(
                        f"episode {episode + 1}/{max_episodes} | reward {episode_reward:.1f} | "
                        f"avg100 {avg_reward_100:.1f} | epsilon {epsilon:.3f} | "
                        f"grad_steps {grad_steps} | wall_clock {wall_clock_s:.1f}s{eval_note}"
                    )

                if len(reward_window) == solve_window and avg_reward_100 >= solve_threshold:
                    solved_at_episode = episode + 1
                    solved_at_env_steps = total_env_steps
                    solved_at_wall_clock = wall_clock_s
                    if verbose:
                        print(f"SOLVED at episode {solved_at_episode} (avg100={avg_reward_100:.1f})")
                    break

                if max_wall_clock_s is not None and wall_clock_s >= max_wall_clock_s:
                    if verbose:
                        print(f"stopping: max_wall_clock_s={max_wall_clock_s} reached at episode {episode + 1}")
                    break
        finally:
            env.close()

    torch.save(
        {"policy": model.state_dict(), "target": target_model.state_dict()},
        checkpoint_path,
    )

    return {
        "results_path": results_path,
        "checkpoint_path": checkpoint_path,
        "episodes_run": episode + 1,
        "total_env_steps": total_env_steps,
        "grad_steps": grad_steps,
        # "solved" here is the TRAINING criterion: avg reward over the last
        # `solve_window` *exploring* episodes. It understates the agent, because
        # those episodes are played with epsilon > 0. The headline claim should
        # come from the final greedy evaluation in scripts/train.py instead.
        "solved": solved_at_episode is not None,
        "episodes_to_solve": solved_at_episode,
        "env_steps_to_solve": solved_at_env_steps,
        "wall_clock_to_solve_s": solved_at_wall_clock,
        "last_eval_mean_reward": last_eval_mean,
        "best_eval_mean_reward": None if best_eval_mean == float("-inf") else best_eval_mean,
    }
