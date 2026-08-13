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
from .replay import RainbowReplayBuffer, ReplayBuffer
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


def dqn_update(model, target, buffer, batch_size, gamma, optimizer, grad_clip,
               double: bool = False) -> float:
    """One DQN gradient step (vanilla or Double).

    Double DQN (van Hasselt et al. 2016) decouples action SELECTION (online net) from
    action EVALUATION (target net) in the bootstrap, reducing the DQN max-operator
    overestimation bias. It is the first composable Rainbow flag; ``double=False``
    reproduces the original vanilla target byte-for-byte.
    """
    batch = buffer.sample(batch_size)  # all `batch_size` states go through ONE forward call
    q_sa = model(batch.states).gather(1, batch.actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        if double:
            next_actions = model(batch.next_states).argmax(dim=1, keepdim=True)  # ONLINE selects
            next_q = target(batch.next_states).gather(1, next_actions).squeeze(1)  # TARGET evaluates
        else:
            next_q = target(batch.next_states).max(dim=1).values
        y = batch.rewards + gamma * (1.0 - batch.dones) * next_q
    loss = F.mse_loss(q_sa, y)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    optimizer.step()
    return float(loss.detach())


class NStepAccumulator:
    """Turns a stream of 1-step transitions into n-step transitions for the Rainbow buffer.

    Holds the last ``n`` transitions of the current episode; once ``n`` are buffered it
    emits ``(s_t, a_t, R_t^(n), s_{t+n}, done, gamma**n)`` where ``R_t^(n)=sum_{k<n}
    gamma^k r`` (truncated at a real terminal). ``flush`` drains the tail at episode end,
    emitting shorter (k<n) transitions with the correct ``gamma**k``. n=1 reproduces the
    ordinary 1-step target exactly.
    """

    def __init__(self, n: int, gamma: float):
        self.n = int(n)
        self.gamma = float(gamma)
        self._buf: deque = deque()   # (s, a, r, s2, terminated)

    def push(self, s, a, r, s2, terminated):
        self._buf.append((s, a, float(r), s2, bool(terminated)))
        out = []
        if len(self._buf) >= self.n:
            out.append(self._emit(self.n))
            self._buf.popleft()
        if terminated:                # true terminal: drain everything, bootstrap masked
            out.extend(self.flush())
        return out

    def _emit(self, k: int):
        s, a, _, _, _ = self._buf[0]
        R, g = 0.0, 1.0
        for j in range(k):
            _, _, r_j, s2_j, term_j = self._buf[j]
            R += g * r_j
            g *= self.gamma
            if term_j:                # real terminal within the window: mask the bootstrap
                return (s, a, R, s2_j, True, self.gamma ** (j + 1))
        # no terminal: bootstrap from s_{t+k}
        return (s, a, R, self._buf[k - 1][3], False, self.gamma ** k)

    def flush(self):
        """Emit the remaining (<n) transitions when the episode ends."""
        out = []
        while self._buf:
            out.append(self._emit(len(self._buf)))
            self._buf.popleft()
        return out


def rainbow_update(model, target, batch, gamma, optimizer, grad_clip, double: bool):
    """One Rainbow value-update step: Double target, n-step bootstrap, PER IS-weights.

    Uses the pre-accumulated n-step return and per-sample discount ``gamma**k`` from the
    RainbowReplayBuffer, weights the Huber/MSE loss by importance-sampling weights, and
    returns ``(loss, |TD error| per sample)`` so the caller can refresh PER priorities.
    """
    q_sa = model(batch.states).gather(1, batch.actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        if double:
            next_a = model(batch.boot_states).argmax(dim=1, keepdim=True)
            next_q = target(batch.boot_states).gather(1, next_a).squeeze(1)
        else:
            next_q = target(batch.boot_states).max(dim=1).values
        y = batch.returns + batch.n_gammas * (1.0 - batch.dones) * next_q
    td = q_sa - y
    loss = (batch.weights * td.pow(2)).mean()      # IS-weighted squared TD error
    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    optimizer.step()
    return float(loss.detach()), td.detach().abs().cpu().numpy()


def distributional_update(model, target, batch, optimizer, grad_clip, double: bool):
    """One distributional (C51) Rainbow update: projected categorical Bellman + cross-entropy.

    Uses the n-step return + per-sample discount from the RainbowReplayBuffer. The model
    exposes ``dist(states)->(B,A,atoms)``, the support ``z``, ``dz``, ``v_min/v_max``,
    ``n_atoms``. Returns ``(loss, per-sample CE)`` for PER priorities. Stays qiskit-free."""
    z = model.z
    n_atoms, v_min, v_max, dz = model.n_atoms, model.v_min, model.v_max, model.dz
    B = batch.states.shape[0]
    device = z.device
    p = model.dist(batch.states)                                       # (B, A, atoms)
    a_idx = batch.actions.view(B, 1, 1).expand(B, 1, n_atoms)
    log_p_sa = torch.log(p.gather(1, a_idx).squeeze(1).clamp_min(1e-8))  # (B, atoms)

    with torch.no_grad():
        next_p = target.dist(batch.boot_states)                        # (B, A, atoms)
        sel_p = model.dist(batch.boot_states) if double else next_p    # Double: online selects
        next_a = (sel_p * z).sum(2).argmax(1)                          # (B,)
        next_p_a = next_p.gather(1, next_a.view(B, 1, 1).expand(B, 1, n_atoms)).squeeze(1)  # (B, atoms)
        Tz = (batch.returns.view(B, 1)
              + batch.n_gammas.view(B, 1) * (1.0 - batch.dones.view(B, 1)) * z.view(1, -1))
        Tz = Tz.clamp(v_min, v_max)
        b = (Tz - v_min) / dz                                          # (B, atoms) fractional index
        l = b.floor().long()
        u = b.ceil().long()
        # keep l<u where b is integer (else mass vanishes), respecting the support boundaries
        l[(u > 0) & (l == u)] -= 1
        u[(l < (n_atoms - 1)) & (l == u)] += 1
        m = torch.zeros(B, n_atoms, device=device)
        offset = (torch.arange(B, device=device) * n_atoms).view(B, 1)
        m.view(-1).index_add_(0, (l + offset).view(-1), (next_p_a * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (next_p_a * (b - l.float())).view(-1))

    ce = -(m * log_p_sa).sum(1)                                        # (B,) cross-entropy
    loss = (batch.weights * ce).mean()
    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    optimizer.step()
    return float(loss.detach()), ce.detach().abs().cpu().numpy()


def build_spsa_state(model: QFunction, spsa_cfg: dict) -> dict:
    """Build SPSA optimizer state. Reuses the model's param-group learning rates as
    per-parameter gain SCALES, so the output scaling `w` still moves ~100x faster than
    the circuit params (the three-lr structure is preserved for the gradient-free path).
    """
    # Per-group scales must be RELATIVE (Spall's diagonal gain scaling), not the
    # absolute Adam lrs -- otherwise the base gain `a` is folded in twice and the
    # variational block (lr 1e-3) barely moves. Normalize by the smallest group lr so
    # the ratio is variational:input:output = 1:1:100 (matching Skolik's 1e-3/1e-3/1e-1).
    lr_by_id = {id(p): float(g["lr"]) for g in model.param_groups() for p in g["params"]}
    raw = [lr_by_id.get(id(p), 1.0) for p in model.parameters()]
    base = min(raw) if raw else 1.0
    scales = [r / base for r in raw]
    return {
        "k": 0,
        "a": float(spsa_cfg.get("a", 0.05)),        # GUESS -- tuned in the ablation
        "c": float(spsa_cfg.get("c", 0.1)),          # GUESS
        "alpha": float(spsa_cfg.get("alpha", 0.602)),  # Spall recommended
        "gamma": float(spsa_cfg.get("gamma", 0.101)),  # Spall recommended
        "A": float(spsa_cfg.get("A", 100.0)),
        "resamplings": int(spsa_cfg.get("resamplings", 1)),
        "scales": scales,
    }


def spsa_update(model, target, buffer, batch_size, gamma, spsa) -> float:
    """One SPSA (gradient-free) DQN update: FORWARD-ONLY (no autograd backward).

    Estimates the gradient of the TD loss from 2*resamplings forward passes with a
    single simultaneous +-1 perturbation of ALL parameters. Crucially the minibatch AND
    the target values `y` are held FIXED across the +/- evaluations, so the finite
    difference is not corrupted by different data. Fast here precisely because it avoids
    the expensive Qiskit reverse-gradient backward.
    """
    params = list(model.parameters())
    batch = buffer.sample(batch_size)
    with torch.no_grad():
        y = batch.rewards + gamma * (1.0 - batch.dones) * target(batch.next_states).max(dim=1).values

    k = spsa["k"]
    ck = spsa["c"] / (k + 1) ** spsa["gamma"]
    ak = spsa["a"] / (k + 1 + spsa["A"]) ** spsa["alpha"]
    orig = [p.detach().clone() for p in params]

    def loss_forward() -> float:
        with torch.no_grad():
            q = model(batch.states).gather(1, batch.actions.view(-1, 1)).squeeze(1)
            return float(F.mse_loss(q, y))

    accum = [torch.zeros_like(p) for p in params]
    last_loss = 0.0
    R = spsa["resamplings"]
    for _ in range(R):
        deltas = [(torch.randint(0, 2, p.shape, dtype=torch.float32) * 2 - 1) for p in params]
        with torch.no_grad():
            for p, o, d in zip(params, orig, deltas):
                p.copy_(o + ck * d)
        lp = loss_forward()
        with torch.no_grad():
            for p, o, d in zip(params, orig, deltas):
                p.copy_(o - ck * d)
        lm = loss_forward()
        ghat = (lp - lm) / (2.0 * ck)          # scalar SPSA gradient magnitude
        for a_acc, d in zip(accum, deltas):
            a_acc.add_(ghat * d)                # g_i = ghat * delta_i  (1/delta_i = delta_i)
        last_loss = 0.5 * (lp + lm)

    with torch.no_grad():
        for p, o, g_acc, scale in zip(params, orig, accum, spsa["scales"]):
            p.copy_(o - ak * scale * (g_acc / R))
    spsa["k"] += 1
    return last_loss


def train(model: QFunction, config: dict, seed: int, results_path: str,
          log=print, progress_every: int = 10, on_best=None) -> dict:
    """Model-agnostic DQN/Rainbow training loop.

    ``on_best(model, episode, best_avg10, ma100)`` (optional) is invoked whenever the
    peak-avg10 improves, so a caller can persist a grabbable checkpoint mid-run (used for
    the early Session-B handoff without waiting for the run to end). It stays qiskit-free:
    the trainer only calls the callback; the caller owns the checkpoint format.
    """
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
    eps_start, eps_end = float(e["start"]), float(e["end"])
    eps_decay = int(e.get("decay_env_steps", 20000))
    # Two schedules: "linear" over env steps (original), or "multiplicative" per EPISODE
    # (eps *= decay_per_episode each episode) -- the Skolik/TFQ recipe, which reaches
    # eps_end at a fixed episode count regardless of (variable) episode length.
    eps_mode = str(e.get("mode", "linear")).lower()
    eps_decay_pe = float(e.get("decay_per_episode", 0.99))
    solve_reward = float(ecfg["solve_reward"])
    solve_window = int(ecfg["solve_window"])

    target = copy.deepcopy(model)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    # --- Rainbow components (composable flags; all default OFF -> vanilla DQN) ---
    rcfg = config.get("rainbow", {}) or {}
    double = bool(rcfg.get("double", False))
    n_step = int(rcfg.get("n_step", 1))
    per_cfg = rcfg.get("per", False)
    per_on = bool(per_cfg) and per_cfg is not False
    per_cfg = per_cfg if isinstance(per_cfg, dict) else {}
    per_alpha = float(per_cfg.get("alpha", 0.5))
    per_eps = float(per_cfg.get("eps", 1e-3))
    per_beta0 = float(per_cfg.get("beta_start", 0.4))
    per_beta1 = float(per_cfg.get("beta_end", 1.0))
    per_beta_steps = int(per_cfg.get("beta_anneal_steps", 20000))
    noisy_cfg = rcfg.get("noisy", False)
    noisy_on = bool(noisy_cfg) and noisy_cfg is not False
    noisy_cfg = noisy_cfg if isinstance(noisy_cfg, dict) else {}
    shots_start = int(noisy_cfg.get("shots_start", 8))
    shots_end = int(noisy_cfg.get("shots_end", 1024))
    shots_anneal_ep = int(noisy_cfg.get("anneal_episodes", 400))
    distributional = bool(rcfg.get("distributional", False))
    # Rainbow buffer whenever multi-step OR PER OR distributional is on (needs n-step
    # returns + bootstrap states); Double/Dueling/noisy are orthogonal.
    use_rainbow = per_on or n_step > 1 or distributional
    if noisy_on and not hasattr(model, "set_exploration_shots"):
        raise ValueError("rainbow.noisy requires a model exposing set_exploration_shots() "
                         "(the shot-noise NoisyNet hook). Use the VQC/Dueling value model.")

    opt_cfg = config.get("optimizer", {"type": "adam"})
    opt_type = str(opt_cfg.get("type", "adam")).lower()
    if opt_type == "adam":
        optimizer = torch.optim.Adam(model.param_groups())
        spsa = None
    elif opt_type == "spsa":
        optimizer = None
        spsa = build_spsa_state(model, opt_cfg.get("spsa", {}))
    else:
        raise ValueError(f"Unknown optimizer {opt_type!r}. Expected 'adam' or 'spsa'.")
    log(f"optimizer: {opt_type}"
        + (f" (a={spsa['a']}, c={spsa['c']}, A={spsa['A']}, resamplings={spsa['resamplings']})"
           if spsa else ""))
    if rcfg:
        log(f"rainbow flags: double={double} n_step={n_step} per={per_on} noisy={noisy_on} "
            f"distributional={distributional} (rainbow_buffer={use_rainbow})")
    if noisy_on:
        log(f"  noisy exploration: shots {shots_start}->{shots_end} over {shots_anneal_ep} ep "
            f"(epsilon disabled)")

    capacity = int(tcfg["replay_capacity"])
    if use_rainbow:
        buffer = RainbowReplayBuffer(capacity, model.obs_dim, rng, alpha=per_alpha, eps=per_eps)
        nstep_acc = NStepAccumulator(n_step, gamma)
    else:
        buffer = ReplayBuffer(capacity, model.obs_dim, rng)
        nstep_acc = None

    scalar_cols = list(model.loggable_scalars().keys())
    columns = ["episode", "env_steps", "reward", "ma100", "epsilon",
               "mean_loss", "wall_clock_s"] + scalar_cols
    fcsv = open(results_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fcsv, fieldnames=columns)
    writer.writeheader()

    reward_window: deque[float] = deque(maxlen=solve_window)
    avg10_window: deque[float] = deque(maxlen=10)   # short window for best-checkpoint tracking
    avg10_solve_reward = float(ecfg.get("avg10_solve_reward", solve_reward))  # TFQ-style criterion
    avg10_solved_at = None
    best_avg10 = float("-inf")
    best_state = copy.deepcopy(model.state_dict())  # DQN forgets: keep the peak weights
    env_steps = grad_steps = 0
    solved_at_ep = solved_at_steps = None
    t0 = time.time()

    obs, _ = env.reset(seed=seed)  # seed env once; subsequent resets use the seeded RNG
    ma100 = float("nan")
    epsilon = eps_start                     # running value (used directly in multiplicative mode)
    for episode in range(1, max_episodes + 1):
        if episode > 1:
            obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        losses: list[float] = []
        if noisy_on:
            # quantum-native exploration: anneal the shot budget instead of epsilon.
            frac = min(episode / max(shots_anneal_ep, 1), 1.0)
            shots = int(round(shots_start * (shots_end / shots_start) ** frac))
            epsilon = 0.0
        elif eps_mode == "linear":
            epsilon = linear_epsilon(env_steps, eps_start, eps_end, eps_decay)
        while not done:
            if noisy_on:
                model.set_exploration_shots(shots)               # perturb <O> with finite-shot noise
                action = select_action(model, obs, 0.0, rng, model.n_actions)
                model.set_exploration_shots(None)                # updates/eval stay exact
            else:
                if eps_mode == "linear":     # per-env-step schedule
                    epsilon = linear_epsilon(env_steps, eps_start, eps_end, eps_decay)
                action = select_action(model, obs, epsilon, rng, model.n_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if use_rainbow:
                for tr in nstep_acc.push(obs, action, float(reward), next_obs, bool(terminated)):
                    buffer.push(*tr)          # (s, a, R, s_boot, done, gamma**k)
            else:
                buffer.push(obs, action, float(reward), next_obs, bool(terminated))
            obs = next_obs
            ep_reward += float(reward)
            env_steps += 1
            if len(buffer) >= learning_starts and env_steps % train_every == 0:
                if use_rainbow:
                    beta = min(per_beta1, per_beta0 + (per_beta1 - per_beta0) * grad_steps / max(per_beta_steps, 1))
                    rbatch = buffer.sample(batch_size, per=per_on, beta=beta)
                    if distributional:
                        loss, td = distributional_update(model, target, rbatch, optimizer, grad_clip, double)
                    else:
                        loss, td = rainbow_update(model, target, rbatch, gamma, optimizer, grad_clip, double)
                    if per_on:
                        buffer.update_priorities(rbatch.indices, td)
                    losses.append(loss)
                elif spsa is None:
                    losses.append(dqn_update(model, target, buffer, batch_size, gamma,
                                             optimizer, grad_clip, double=double))
                else:
                    losses.append(spsa_update(model, target, buffer, batch_size, gamma, spsa))
                grad_steps += 1
                if grad_steps % target_update == 0:
                    target.load_state_dict(model.state_dict())
            done = terminated or truncated

        if use_rainbow:                       # drain the n-step tail (handles truncation too)
            for tr in nstep_acc.flush():
                buffer.push(*tr)

        if not noisy_on and eps_mode != "linear":   # multiplicative per-episode decay (Skolik/TFQ)
            epsilon = max(epsilon * eps_decay_pe, eps_end)

        reward_window.append(ep_reward)
        ma100 = float(np.mean(reward_window))
        avg10_window.append(ep_reward)
        avg10 = float(np.mean(avg10_window))
        if len(avg10_window) == 10 and avg10 > best_avg10:
            best_avg10 = avg10
            best_state = copy.deepcopy(model.state_dict())
            if on_best is not None:
                on_best(model, episode, best_avg10, ma100)
        if avg10_solved_at is None and len(avg10_window) == 10 and avg10 >= avg10_solve_reward:
            avg10_solved_at = episode                          # TFQ-style solve (avg-last-10)
            log(f"AVG10-SOLVE (avg-last-10>={avg10_solve_reward:.0f}) at episode {episode}, "
                f"avg10={avg10:.1f} ({env_steps} env steps)")
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
            best_avg10 = max(best_avg10, avg10)
            best_state = copy.deepcopy(model.state_dict())  # solved weights are the best
            log(f"SOLVED at episode {episode} ({env_steps} env steps), ma100={ma100:.1f}")
            break

    fcsv.close()
    env.close()
    return {
        "solved": solved_at_ep is not None,
        "episodes_to_solve": solved_at_ep,
        "env_steps_to_solve": solved_at_steps,
        "final_ma100": ma100,
        "best_avg10": (None if best_avg10 == float("-inf") else best_avg10),
        "best_state": best_state,          # peak-avg10 state_dict (load before eval/checkpoint)
        "n_params": model.num_trainable_params(),
        "results_path": results_path,
    }
