"""Model-agnostic DQN training loop.

HARD ARCHITECTURAL RULE: this file MUST NOT import anything from qiskit. It only ever
sees a `models/base.QFunction` (forward + param_groups + loggable_scalars). The target
network is built with `copy.deepcopy`, which is model-agnostic (verified to work on the
VQC's qiskit objects). If this file could tell the model were quantum, the design is wrong.

DQN specifics (README "Trainer spec"):
  * uniform replay, batch 16, gamma 0.99
  * hard target update every `target_update_interval` gradient steps
  * loss = MSE(Q(s,a), r + gamma*(1-done)*max_a' Q_target(s',a'))
  * epsilon-greedy, linear 1.0 -> 0.01 over the first `decay_env_steps` env steps
  * `terminated` (pole fell) bootstraps as done; `truncated` (500-step time limit) does NOT
  * writes results/{name}_{seed}.csv per episode with the model's loggable_scalars appended
"""
from __future__ import annotations

import copy
import csv
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .models.base import QFunction  # interface only; NEVER import qiskit here
from .replay import ReplayBuffer
from .seeds import make_rng, set_global_seeds


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def select_action(model: QFunction, obs: np.ndarray, epsilon: float,
                  rng: np.random.Generator, n_actions: int) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    with torch.no_grad():
        q = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    return int(torch.argmax(q, dim=1).item())


def dqn_update(model, target, buffer, batch_size, gamma, optimizer, grad_clip) -> float:
    batch = buffer.sample(batch_size)  # all `batch_size` states go through ONE forward call
    q_sa = model(batch.states).gather(1, batch.actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        max_next = target(batch.next_states).max(dim=1).values
        y = batch.rewards + gamma * (1.0 - batch.dones) * max_next
    loss = F.mse_loss(q_sa, y)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    optimizer.step()
    return float(loss.detach())


def train(model: QFunction, config: dict, seed: int, results_path: str,
          log=print, progress_every: int = 10) -> dict:
    tcfg, ecfg = config["trainer"], config["eval"]
    # Re-seed at loop start so the run is byte-identical regardless of RNG draws during
    # model construction; env + replay sampling + eps/action draws all derive from `seed`.
    set_global_seeds(seed)
    rng = make_rng(seed)
    env = gym.make("CartPole-v1")

    gamma = float(tcfg["gamma"])
    batch_size = int(tcfg["batch_size"])
    learning_starts = int(tcfg["learning_starts"])
    train_every = int(tcfg["train_every"])
    target_update = int(tcfg["target_update_interval"])
    grad_clip = tcfg.get("grad_clip_norm")
    max_episodes = int(tcfg["max_episodes"])
    e = tcfg["epsilon"]
    eps_start, eps_end, eps_decay = float(e["start"]), float(e["end"]), int(e["decay_env_steps"])
    solve_reward = float(ecfg["solve_reward"])
    solve_window = int(ecfg["solve_window"])

    target = copy.deepcopy(model)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.param_groups())
    buffer = ReplayBuffer(int(tcfg["replay_capacity"]), model.obs_dim, rng)

    scalar_cols = list(model.loggable_scalars().keys())
    columns = ["episode", "env_steps", "reward", "ma100", "epsilon",
               "mean_loss", "wall_clock_s"] + scalar_cols
    fcsv = open(results_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fcsv, fieldnames=columns)
    writer.writeheader()

    reward_window: deque[float] = deque(maxlen=solve_window)
    env_steps = grad_steps = 0
    solved_at_ep = solved_at_steps = None
    t0 = time.time()

    obs, _ = env.reset(seed=seed)  # seed env once; subsequent resets use the seeded RNG
    ma100 = float("nan")
    for episode in range(1, max_episodes + 1):
        if episode > 1:
            obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        losses: list[float] = []
        epsilon = linear_epsilon(env_steps, eps_start, eps_end, eps_decay)
        while not done:
            epsilon = linear_epsilon(env_steps, eps_start, eps_end, eps_decay)
            action = select_action(model, obs, epsilon, rng, model.n_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.push(obs, action, float(reward), next_obs, bool(terminated))
            obs = next_obs
            ep_reward += float(reward)
            env_steps += 1
            if len(buffer) >= learning_starts and env_steps % train_every == 0:
                losses.append(dqn_update(model, target, buffer, batch_size, gamma, optimizer, grad_clip))
                grad_steps += 1
                if grad_steps % target_update == 0:
                    target.load_state_dict(model.state_dict())
            done = terminated or truncated

        reward_window.append(ep_reward)
        ma100 = float(np.mean(reward_window))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        row = {"episode": episode, "env_steps": env_steps, "reward": ep_reward,
               "ma100": ma100, "epsilon": epsilon, "mean_loss": mean_loss,
               "wall_clock_s": round(time.time() - t0, 3)}
        row.update(model.loggable_scalars())
        writer.writerow(row)
        fcsv.flush()

        if episode == 1 or episode % progress_every == 0:
            extra = ""
            if scalar_cols:
                sc = model.loggable_scalars()
                extra = f" | w {sc.get('w0', 0):.2f},{sc.get('w1', 0):.2f}"
            log(f"ep {episode:4d} | steps {env_steps:6d} | R {ep_reward:5.0f} | "
                f"ma100 {ma100:6.1f} | eps {epsilon:.3f} | loss {mean_loss:.4f}{extra}")

        if len(reward_window) == solve_window and ma100 >= solve_reward:
            solved_at_ep, solved_at_steps = episode, env_steps
            log(f"SOLVED at episode {episode} ({env_steps} env steps), ma100={ma100:.1f}")
            break

    fcsv.close()
    env.close()
    return {
        "solved": solved_at_ep is not None,
        "episodes_to_solve": solved_at_ep,
        "env_steps_to_solve": solved_at_steps,
        "final_ma100": ma100,
        "n_params": model.num_trainable_params(),
        "results_path": results_path,
    }
