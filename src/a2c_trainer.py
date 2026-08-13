"""Model-agnostic advantage actor-critic (A2C) with GAE(lambda).

Same architectural rule as src/trainer.py and src/pg_trainer.py: no qiskit
import. The actor and critic are touched only through nn.Module, and the
optimizer -- carrying however many parameter groups the two models call for --
is built by the caller (scripts/train_a2c.py).

WHAT THIS SHARES WITH THE REINFORCE LOOP

Collection is imported outright from src/pg_trainer.py: `_collect_round`,
`_sample_actions`, `_beta_to_str`. The synchronized-round scheme and its
on-policy guarantee are described there and are unchanged here -- all n_envs
environments reset together, run until every one terminates, and no gradient
step happens mid-round.

WHAT REPLACES WHAT

REINFORCE subtracts the batch mean of the returns. A2C subtracts a LEARNED
V(s), which is a per-state baseline rather than a per-batch constant, and
estimates the advantage with GAE(lambda) rather than the full Monte-Carlo
return:

    delta_t = r_t + gamma * V(s_{t+1}) * nonterminal_t - V(s_t)
    A_t     = delta_t + gamma * lambda * nonterminal_t * A_{t+1}

lambda interpolates between the one-step TD estimate (lambda=0: low variance,
biased by whatever the critic gets wrong) and the Monte-Carlo return
(lambda=1: unbiased, high variance -- REINFORCE's estimator). The critic is
regressed onto A_t + V(s_t), the same targets the advantages were built from.

TRUNCATION IS BOOTSTRAPPED, TERMINATION IS NOT

This is the correctness detail that separates a working CartPole critic from a
subtly broken one, and it does not exist in the REINFORCE version. An episode
that ends because the pole fell has a true return of 0 from the final state. An
episode that ends at the 500-step limit does NOT -- the pole is still up and the
state is worth a lot. A trained CartPole agent truncates EVERY episode, so
treating truncation as terminal teaches the critic that the best states it ever
reaches are worthless, exactly inverting the signal at the point it matters
most. `_collect_round` records the two cases separately and `compute_gae` uses
V(final_state) as the bootstrap on truncation and 0.0 on termination.

COST

One critic forward per gradient step on top of the actor's -- the separate-critic
choice, as opposed to reading a third observable off the actor's own circuit.
Both are batched: the whole batch of episodes is ONE [T, n_qubits] simulate call
per network. See src/pg_trainer.py for why that batching is what makes policy
gradient affordable on a simulated circuit at all.
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from src.evaluate import run_greedy_rollouts
from src.pg_trainer import PG_CSV_HEADER, _beta_to_str, _collect_round
from src.seeds import set_seed
from src.trainer import _param_to_str

# PG's header plus two critic diagnostics. Appended, never substituted, for the
# same reason PG_CSV_HEADER appends: scripts/summarize.py and src/plots.py index
# by column name, so trailing columns are invisible to them.
#
# `explained_variance` is the one number that says whether the critic is doing
# anything: 1 - Var(target - V) / Var(target). At 0 the critic is no better than
# predicting the batch mean, which is REINFORCE's baseline -- so a run sitting
# near 0 is evidence the actor-critic machinery is not earning its cost.
A2C_CSV_HEADER = PG_CSV_HEADER + ["value_loss", "explained_variance"]


def compute_gae(
    rewards: list[float],
    values: np.ndarray,
    bootstrap_value: float,
    truncated: bool,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) advantages and value targets for ONE episode.

    `values` is V(s_0..s_{T-1}); `bootstrap_value` is V(s_T), used only when the
    episode was truncated rather than terminated. Returns (advantages, targets)
    where targets = advantages + values, both length T.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    # nonterminal_t is 0 exactly where the NEXT state is genuinely terminal.
    # Inside the episode the next state is always real; at the last step it is
    # terminal only if the episode terminated rather than being truncated.
    last_nonterminal = 1.0 if truncated else 0.0
    next_value = bootstrap_value if truncated else 0.0

    running = 0.0
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            nonterminal, v_next = last_nonterminal, next_value
        else:
            nonterminal, v_next = 1.0, float(values[t + 1])
        delta = rewards[t] + gamma * v_next * nonterminal - float(values[t])
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[t] = running
    return advantages, advantages + values.astype(np.float32)


def _a2c_losses(
    actor,
    critic,
    batch,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
    entropy_coef: float,
    value_coef: float,
    value_loss_fn: str,
):
    """Combined actor-critic loss over a batch of complete episodes.

    Returns (loss, diagnostics). Two batched forwards: the critic sees every
    state plus each episode's final state (needed for the truncation bootstrap),
    the actor sees only the real states.
    """
    # --- critic forward over states + per-episode final state -----------------
    chunks, spans, offset = [], [], 0
    for ep in batch:
        T = len(ep["rewards"])
        chunks.append(ep["states"])
        chunks.append(ep["final_state"][None, :])
        spans.append((offset, T))
        offset += T + 1

    all_states = torch.as_tensor(np.concatenate(chunks), dtype=torch.float32)
    values_all = critic(all_states)  # [sum(T_i + 1)]
    values_np = values_all.detach().numpy()

    adv_parts, target_parts, value_parts = [], [], []
    for ep, (off, T) in zip(batch, spans):
        adv, target = compute_gae(
            ep["rewards"], values_np[off:off + T], float(values_np[off + T]),
            ep["truncated"], gamma, gae_lambda,
        )
        adv_parts.append(adv)
        target_parts.append(target)
        value_parts.append(values_all[off:off + T])  # keeps the graph

    advantages = torch.as_tensor(np.concatenate(adv_parts), dtype=torch.float32)
    targets = torch.as_tensor(np.concatenate(target_parts), dtype=torch.float32)
    values = torch.cat(value_parts)

    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    # --- actor forward over the real states only ------------------------------
    states = torch.as_tensor(np.concatenate([ep["states"] for ep in batch]), dtype=torch.float32)
    actions = torch.as_tensor(np.concatenate([ep["actions"] for ep in batch]), dtype=torch.int64)
    logits = actor(states)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()

    policy_loss = -(log_probs * advantages).mean()
    if value_loss_fn == "huber":
        # Same reasoning as the DQN side's loss_fn: targets reach ~99 early in a
        # run while the head still outputs ~0.5, and MSE lets those first
        # enormous errors dominate every batch they appear in.
        value_loss = nn.functional.smooth_l1_loss(values, targets)
    elif value_loss_fn == "mse":
        value_loss = nn.functional.mse_loss(values, targets)
    else:
        raise ValueError(f"unknown value_loss_fn {value_loss_fn!r}; expected 'huber' or 'mse'")

    loss = policy_loss + value_coef * value_loss
    if entropy_coef:
        loss = loss - entropy_coef * entropy.mean()

    target_var = float(targets.var(unbiased=False).item())
    explained = 0.0 if target_var < 1e-8 else float(
        1.0 - (targets - values).detach().var(unbiased=False).item() / target_var
    )
    return loss, {
        "entropy": float(entropy.mean().item()),
        "value_loss": float(value_loss.item()),
        "explained_variance": explained,
    }


def train_a2c(
    actor: nn.Module,
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    results_path: str,
    seed: int,
    env_id: str = "CartPole-v1",
    max_episodes: int = 2000,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_envs: int = 8,
    episodes_per_update: int = 8,  # rounded UP to a whole number of rounds
    normalize_advantages: bool = True,
    entropy_coef: float = 0.0,
    value_coef: float = 0.5,
    value_loss_fn: str = "huber",  # "huber" | "mse"
    max_grad_norm: float | None = None,
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
    """Run A2C with GAE and write a per-episode CSV to results_path.

    Returns the same summary dict as src/trainer.py::train and
    src/pg_trainer.py::train_pg, so the sweep and summary tooling consumes any
    of the three.

    `optimizer` must already carry BOTH the actor's and the critic's parameters
    (scripts/train_a2c.py builds it): the loss is combined, so one backward and
    one step cover both networks. Reproducibility works as in the other loops --
    the caller must seed model construction as well as passing `seed` here.
    """
    set_seed(seed)
    env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)])

    rounds_per_update = max(1, -(-episodes_per_update // n_envs))  # ceil
    episodes_per_update = rounds_per_update * n_envs

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    total_env_steps = grad_steps = episode = 0
    reward_window: deque = deque(maxlen=solve_window)
    recent_losses: deque = deque(maxlen=50)
    solved_at_episode = solved_at_env_steps = solved_at_wall_clock = None
    last_eval_mean: float | None = None
    best_eval_mean = float("-inf")
    next_eval_at = eval_every
    diag = {"entropy": "", "value_loss": "", "explained_variance": ""}
    stop = False

    start_time = time.time()

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(A2C_CSV_HEADER)
        try:
            while episode < max_episodes and not stop:
                batch: list[dict] = []
                for _r in range(rounds_per_update):
                    batch.extend(
                        _collect_round(
                            actor, env, n_envs,
                            seed=seed + 1000 * (grad_steps + 1) + 7 * _r,
                            max_steps_per_episode=max_steps_per_episode,
                        )
                    )

                loss, d = _a2c_losses(
                    actor, critic, batch, gamma, gae_lambda, normalize_advantages,
                    entropy_coef, value_coef, value_loss_fn,
                )
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(actor.parameters()) + list(critic.parameters()), max_grad_norm
                    )
                optimizer.step()
                grad_steps += 1

                loss_value = loss.item()
                if np.isnan(loss_value):
                    raise RuntimeError(f"NaN loss at episode {episode}, grad step {grad_steps}")
                recent_losses.append(loss_value)
                diag = {k: f"{v:.6f}" for k, v in d.items()}

                for ep in batch:
                    ep_reward = float(sum(ep["rewards"]))
                    total_env_steps += len(ep["rewards"])
                    reward_window.append(ep_reward)
                    avg_reward_100 = sum(reward_window) / len(reward_window)
                    wall_clock_s = time.time() - start_time

                    eval_mean: float | str = ""
                    eval_std: float | str = ""
                    if eval_every and episode + 1 >= next_eval_at:
                        next_eval_at += eval_every
                        eval_t0 = time.time()
                        ev = run_greedy_rollouts(actor, n_episodes=eval_episodes, env_id=env_id,
                                                 seed=eval_seed_base,
                                                 max_steps_per_episode=max_steps_per_episode)
                        eval_mean, eval_std = float(np.mean(ev)), float(np.std(ev))
                        last_eval_mean = eval_mean
                        best_eval_mean = max(best_eval_mean, eval_mean)
                        start_time += time.time() - eval_t0  # training-only clock

                    writer.writerow([
                        episode, total_env_steps, ep_reward, avg_reward_100,
                        "",  # epsilon: no exploration schedule under A2C either
                        (sum(recent_losses) / len(recent_losses)) if recent_losses else "",
                        wall_clock_s, grad_steps,
                        _param_to_str(actor, "w"), _param_to_str(actor, "lam"),
                        eval_mean, eval_std,
                        diag["entropy"], _beta_to_str(actor),
                        diag["value_loss"], diag["explained_variance"],
                    ])
                    f.flush()

                    if verbose and (episode + 1) % print_every == 0:
                        note = f" | greedy {eval_mean:.1f}" if eval_mean != "" else ""
                        print(f"episode {episode + 1}/{max_episodes} | reward {ep_reward:.1f} | "
                              f"avg100 {avg_reward_100:.1f} | entropy {d['entropy']:.3f} | "
                              f"vloss {d['value_loss']:.2f} | ev {d['explained_variance']:.3f} | "
                              f"grad_steps {grad_steps} | wall_clock {wall_clock_s:.1f}s{note}")

                    episode += 1

                    if len(reward_window) == solve_window and avg_reward_100 >= solve_threshold:
                        solved_at_episode, solved_at_env_steps = episode, total_env_steps
                        solved_at_wall_clock = wall_clock_s
                        if verbose:
                            print(f"SOLVED at episode {episode} (avg100={avg_reward_100:.1f})")
                        stop = True
                        break
                    if episode >= max_episodes:
                        stop = True
                        break

                if not stop and max_wall_clock_s is not None and \
                        time.time() - start_time >= max_wall_clock_s:
                    if verbose:
                        print(f"stopping: max_wall_clock_s={max_wall_clock_s} reached")
                    stop = True
        finally:
            env.close()

    return {
        "results_path": results_path,
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
        "episodes_per_update": episodes_per_update,
    }


if __name__ == "__main__":
    # --- GAE against hand-computed values ------------------------------------
    # Terminated episode, gamma=1, lambda=1: A_t is the Monte-Carlo return minus
    # V(s_t), which is REINFORCE's estimator. Zero critic -> A_t = return-to-go.
    adv, tgt = compute_gae([1.0, 1.0, 1.0], np.zeros(3, dtype=np.float32),
                           bootstrap_value=0.0, truncated=False, gamma=1.0, gae_lambda=1.0)
    assert np.allclose(adv, [3.0, 2.0, 1.0]), adv
    assert np.allclose(tgt, [3.0, 2.0, 1.0]), tgt

    # A perfect critic on a terminated episode leaves zero advantage everywhere.
    v = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    adv, tgt = compute_gae([1.0, 1.0, 1.0], v, 0.0, False, gamma=1.0, gae_lambda=1.0)
    assert np.allclose(adv, 0.0, atol=1e-6), adv
    assert np.allclose(tgt, v, atol=1e-6), tgt

    # TRUNCATION MUST BOOTSTRAP. Same episode, same critic, truncated instead of
    # terminated with V(s_T)=50: the final step is now worth far more, and
    # treating it as terminal would understate every advantage in the episode.
    adv_trunc, _ = compute_gae([1.0, 1.0, 1.0], v, 50.0, True, gamma=1.0, gae_lambda=1.0)
    assert np.allclose(adv_trunc, 50.0, atol=1e-6), adv_trunc
    assert (adv_trunc > adv).all(), (adv_trunc, adv)

    # lambda=0 is the one-step TD estimate: A_t depends only on r_t and V(s_t+1).
    adv0, _ = compute_gae([1.0, 1.0, 1.0], np.zeros(3, dtype=np.float32),
                          0.0, False, gamma=1.0, gae_lambda=0.0)
    assert np.allclose(adv0, [1.0, 1.0, 1.0]), adv0

    # --- end to end on the classical pair ------------------------------------
    from src.models.mlp import MLPPolicy, MLPValue

    set_seed(0)
    actor, critic = MLPPolicy(), MLPValue()
    optimizer = torch.optim.Adam(
        [{"params": actor.parameters(), "lr": 0.01},
         {"params": critic.parameters(), "lr": 0.05}]
    )
    result = train_a2c(
        actor, critic, optimizer, results_path="results/_a2c_smoke.csv", seed=0,
        max_episodes=16, n_envs=4, episodes_per_update=8, eval_every=0, verbose=False,
    )
    assert result["episodes_run"] == 16, result
    assert result["grad_steps"] == 2, result

    import pandas as pd

    df = pd.read_csv("results/_a2c_smoke.csv")
    assert list(df.columns) == A2C_CSV_HEADER, df.columns
    assert len(df) == 16, len(df)
    assert df["value_loss"].notna().all(), "value_loss has gaps"
    assert df["explained_variance"].notna().all(), "explained_variance has gaps"
    assert df["epsilon"].isna().all(), "epsilon should be blank under A2C"
    os.remove("results/_a2c_smoke.csv")
    print("a2c_trainer.py smoke test OK")
