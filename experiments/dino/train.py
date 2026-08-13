"""Double-QDQN training for the CNN→VQC dino agent — on the headless clone, via torch_sv.

House rule "do not reimplement DQN": the DQN math is the backbone's. This loop only supplies
the image env + image replay buffer and then calls the reused pieces:
  * ``src.trainer.dqn_update``  — one Double-DQN gradient step (identical to CartPole),
  * ``src.trainer.select_action`` / ``linear_epsilon`` — epsilon-greedy + schedule,
  * ``experiments.common.save_ckpt`` — the backbone-neutral checkpoint (Session B can read it).
The backbone ``train()`` itself hardcodes ``CartPole-v1`` + a flat replay buffer, so it can't be
called directly for images; everything reusable underneath it IS reused.

Dino has no fixed solve threshold — it's survival maximization — so we log per-episode score
(== frames survived) and track the best model by a short moving average. ``terminated`` (crash)
masks the bootstrap; ``truncated`` (max_steps) does NOT (backbone contract).

Run:  python -m experiments.dino.train --encoder trainable_cnn --steps 80000 --name dino_b
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from collections import deque

import numpy as np
import torch

import experiments.dino  # noqa: F401  (QRL-root sys.path bootstrap)
from experiments.common import save_ckpt
from experiments.dino.env import make_dino_env
from experiments.dino.model import DinoQFunction
from experiments.dino.replay import ImageReplayBuffer, ImageRainbowReplayBuffer
from src.seeds import make_rng, set_global_seeds
from src.trainer import dqn_update, linear_epsilon, select_action, NStepAccumulator, rainbow_update

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(_ROOT, "results")
FIG_DIR = os.path.join(_ROOT, "figures")


def default_config() -> dict:
    # NOTE: with frame-skip=4, one "env step" here == one DECISION == up to 4 game frames.
    return dict(
        max_env_steps=30000,       # decisions (~120k game frames; pixel-DQN is sample-hungry)
        max_episodes=100000,       # safety cap; max_env_steps is the real budget
        max_steps=2000,            # per-episode truncation (in game frames)
        replay_capacity=12000,
        learning_starts=800,
        batch_size=32,
        train_every=4,
        gamma=0.99,
        target_update=500,         # in gradient steps
        grad_clip=10.0,
        double=True,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=9000,   # more exploration to discover clears
        # Rainbow value components (brief: "PER + multi-step help here"); default OFF -> plain Double-QDQN.
        n_step=1,            # >1 -> n-step returns (faster credit assignment for the delayed clear bonus)
        per=False,           # True -> prioritized experience replay (over-sample rare critical transitions)
        per_alpha=0.5, per_beta_start=0.4, per_beta_end=1.0, per_beta_steps=15000,
    )


def evaluate(model, n_episodes: int = 20, seed: int = 10_000, max_steps: int = 2000,
             bird_prob: float | None = None, bird_start_frame: int | None = None,
             variable_jump: bool = False, full_mode: bool = False) -> dict:
    """Greedy (epsilon=0) eval over fixed seeds → score stats (mean/median/best + list)."""
    was_training = model.training
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(n_episodes):
            env = make_dino_env(max_steps=max_steps, seed=seed + i,
                                bird_prob=bird_prob, bird_start_frame=bird_start_frame,
                                variable_jump=variable_jump, full_mode=full_mode)
            obs, info = env.reset(seed=seed + i)
            done = False
            while not done:
                q = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                a = int(torch.argmax(q, dim=1).item())
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
            scores.append(info["score"])
    if was_training:
        model.train()
    s = np.array(scores, dtype=float)
    return {"mean": float(s.mean()), "median": float(np.median(s)),
            "best": float(s.max()), "min": float(s.min()), "scores": scores}


def random_baseline(n_episodes: int = 20, seed: int = 10_000, max_steps: int = 2000,
                    bird_prob: float | None = None, bird_start_frame: int | None = None,
                    variable_jump: bool = False, full_mode: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(n_episodes):
        env = make_dino_env(max_steps=max_steps, seed=seed + i,
                            bird_prob=bird_prob, bird_start_frame=bird_start_frame,
                            variable_jump=variable_jump, full_mode=full_mode)
        obs, info = env.reset(seed=seed + i)
        n_act = env.action_space.n            # honor the env's action count (4 in full mode, else 3)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(int(rng.integers(n_act)))
            done = term or trunc
        scores.append(info["score"])
    s = np.array(scores, dtype=float)
    return {"mean": float(s.mean()), "median": float(np.median(s)),
            "best": float(s.max()), "scores": scores}


def _save(model, ckpt_path, ctor, seed, ev, pc, cfg):
    meta = {"experiment": "dino", "seed": seed, "eval": ev, "param_counts": pc, "config": cfg,
            "constructor": ctor}   # exact kwargs to rebuild DinoQFunction in load_agent
    try:
        save_ckpt(model, ckpt_path, meta=meta)          # backbone-neutral (needs circuit weights)
    except ValueError:
        # classical-head diagnostic has no circuit → plain state_dict checkpoint (still load_agent-able)
        import torch
        torch.save({"format": "dino-classical-v1", "state_dict": model.state_dict(), "meta": meta},
                   ckpt_path)


def train(encoder: str = "trainable_cnn", n_qubits: int = 6, observable: str = "zz",
          n_layers: int = 3, head: str = "vqc", entangler: str = "cx", seed: int = 0, name: str = "dino",
          w_init: float = 10.0, lam_init: float = 1.0, cfg: dict | None = None, log=print,
          bird_prob: float | None = None, bird_start_frame: int | None = None,
          variable_jump: bool = False, full_mode: bool = False, n_actions: int = 3) -> dict:
    cfg = {**default_config(), **(cfg or {})}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # FULL mode always needs the 4-action head; make it explicit even if --n-actions wasn't passed.
    if full_mode:
        n_actions = 4
    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    ckpt_path = os.path.join(RESULTS_DIR, f"{name}.pt")
    log_path = os.path.join(RESULTS_DIR, f"{name}.log")
    ctor = dict(n_qubits=n_qubits, n_actions=n_actions, n_layers=n_layers, encoder=encoder,
                observable=observable, head=head, entangler=entangler)

    # tee every console line into a persistent results/{name}.log (a permanent training record)
    _logf = open(log_path, "w", encoding="utf-8")
    _user_log = log

    def log(msg):
        _user_log(msg)
        _logf.write(str(msg) + "\n")
        _logf.flush()

    set_global_seeds(seed)
    rng = make_rng(seed)

    model = DinoQFunction(n_qubits=n_qubits, n_actions=n_actions, n_layers=n_layers, encoder=encoder,
                          observable=observable, head=head, entangler=entangler,
                          w_init=w_init, lam_init=lam_init, seed=seed)
    model.train()
    pc = model.param_counts()
    log(f"[{name}] encoder={encoder} obs={observable} qubits={n_qubits} layers={n_layers} | "
        f"params: CNN(trainable) {pc['cnn_trainable']:,} vs VQC {pc['vqc_circuit']} "
        f"(+lam {pc['lam']}, w {pc['w']}) | total {pc['total_trainable']:,}")

    target = copy.deepcopy(model)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.param_groups())
    # Rainbow value path (n-step and/or PER) reuses the backbone NStepAccumulator + rainbow_update;
    # only the buffer stores image states. Off by default -> plain Double-QDQN via dqn_update.
    n_step = int(cfg.get("n_step", 1))
    per_on = bool(cfg.get("per", False))
    use_rainbow = per_on or n_step > 1
    if use_rainbow:
        buffer = ImageRainbowReplayBuffer(cfg["replay_capacity"], (4, 84, 84), rng,
                                          alpha=float(cfg.get("per_alpha", 0.5)))
        nstep_acc = NStepAccumulator(n_step, cfg["gamma"])
        log(f"[{name}] Rainbow value path: n_step={n_step} per={per_on}")
    else:
        buffer = ImageReplayBuffer(cfg["replay_capacity"], (4, 84, 84), rng)
        nstep_acc = None

    env = make_dino_env(max_steps=cfg["max_steps"], seed=seed,
                        bird_prob=bird_prob, bird_start_frame=bird_start_frame,
                        variable_jump=variable_jump, full_mode=full_mode)

    columns = ["episode", "env_steps", "score", "ma20", "epsilon", "mean_loss",
               "grad_steps", "wall_clock_s", "w0", "w1", "w2", "lam_mean"]
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fcsv, fieldnames=columns)
    writer.writeheader()

    score_window: deque[float] = deque(maxlen=20)
    best_ma = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    env_steps = grad_steps = 0
    episode = 0
    t0 = time.time()

    while env_steps < cfg["max_env_steps"] and episode < cfg["max_episodes"]:
        episode += 1
        obs, _ = env.reset(seed=seed if episode == 1 else None)  # ep1 fixed; later eps vary (det.)
        done = False
        score = 0
        losses: list[float] = []
        while not done:
            epsilon = linear_epsilon(env_steps, cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"])
            action = select_action(model, obs, epsilon, rng, model.n_actions)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if use_rainbow:                     # accumulate into n-step transitions, then store
                for tr in nstep_acc.push(obs, action, float(reward), next_obs, bool(terminated)):
                    buffer.push(*tr)            # (s, a, R, s_boot, done, gamma**k)
            else:
                buffer.push(obs, action, float(reward), next_obs, bool(terminated))  # only crash masks bootstrap
            obs = next_obs
            score = info["score"]
            env_steps += 1
            if len(buffer) >= cfg["learning_starts"] and env_steps % cfg["train_every"] == 0:
                if use_rainbow:
                    beta = min(cfg["per_beta_end"], cfg["per_beta_start"] + (cfg["per_beta_end"] -
                               cfg["per_beta_start"]) * grad_steps / max(cfg["per_beta_steps"], 1))
                    rbatch = buffer.sample(cfg["batch_size"], per=per_on, beta=beta)
                    loss, td = rainbow_update(model, target, rbatch, cfg["gamma"], optimizer,
                                              cfg["grad_clip"], double=cfg["double"])
                    if per_on:
                        buffer.update_priorities(rbatch.indices, td)
                else:
                    loss = dqn_update(model, target, buffer, cfg["batch_size"], cfg["gamma"],
                                      optimizer, cfg["grad_clip"], double=cfg["double"])
                losses.append(loss)
                grad_steps += 1
                if grad_steps % cfg["target_update"] == 0:
                    target.load_state_dict(model.state_dict())
            done = terminated or truncated
        if use_rainbow:                         # drain the n-step tail at episode end (handles truncation)
            for tr in nstep_acc.flush():
                buffer.push(*tr)

        score_window.append(score)
        ma20 = float(np.mean(score_window))
        if len(score_window) >= 10 and ma20 > best_ma:
            best_ma = ma20
            best_state = copy.deepcopy(model.state_dict())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        sc = model.loggable_scalars()
        writer.writerow({"episode": episode, "env_steps": env_steps, "score": score,
                         "ma20": ma20, "epsilon": round(epsilon, 4), "mean_loss": mean_loss,
                         "grad_steps": grad_steps, "wall_clock_s": round(time.time() - t0, 1),
                         "w0": sc.get("w0"), "w1": sc.get("w1"), "w2": sc.get("w2"),
                         "lam_mean": sc.get("lam_mean")})
        fcsv.flush()

        if episode == 1 or episode % 10 == 0:
            log(f"[{name}] ep {episode:4d} | steps {env_steps:6d}/{cfg['max_env_steps']} | "
                f"score {score:4d} | ma20 {ma20:6.1f} | eps {epsilon:.3f} | "
                f"loss {mean_loss:8.3f} | w {sc.get('w0',0):.2f},{sc.get('w1',0):.2f},{sc.get('w2',0):.2f} | "
                f"{round(time.time()-t0)}s")

        # periodic checkpoint of the best-so-far weights, so a partial/interrupted run is usable
        if episode % 25 == 0 and best_ma > float("-inf"):
            snap = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            _save(model, ckpt_path, ctor, seed,
                  {"best_ma20": best_ma, "partial": True, "env_steps": env_steps}, pc, cfg)
            model.load_state_dict(snap)

    fcsv.close()
    env.close()

    # restore best weights before final eval + checkpoint (DQN forgets its peak)
    model.load_state_dict(best_state)
    ev = evaluate(model, n_episodes=20, max_steps=cfg["max_steps"],
                  bird_prob=bird_prob, bird_start_frame=bird_start_frame,
                  variable_jump=variable_jump, full_mode=full_mode)
    log(f"[{name}] EVAL(best) mean {ev['mean']:.1f} median {ev['median']:.1f} best {ev['best']:.0f} "
        f"(min {ev['min']:.0f})")

    _save(model, ckpt_path, ctor, seed, ev, pc, cfg)
    log(f"[{name}] saved checkpoint -> {ckpt_path}")
    log(f"[{name}] logs: {csv_path} (per-episode) | {log_path} (console)")
    _logf.close()
    return {"name": name, "csv": csv_path, "ckpt": ckpt_path, "log": log_path, "eval": ev,
            "best_ma20": best_ma, "episodes": episode, "env_steps": env_steps, "param_counts": pc}


def main():
    # optional thread cap (set DINO_THREADS to run several ablations in parallel without oversubscribing)
    _t = os.environ.get("DINO_THREADS")
    if _t:
        torch.set_num_threads(int(_t))
    ap = argparse.ArgumentParser(description="Train the CNN→VQC dino agent (Double-QDQN, torch_sv).")
    ap.add_argument("--encoder", default="trainable_cnn", choices=["trainable_cnn", "pretrained"])
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--observable", default="zz", choices=["zz", "z"])
    ap.add_argument("--head", default="vqc", choices=["vqc", "classical"])
    ap.add_argument("--entangler", default="cx", choices=["cx", "cz", "none"])
    ap.add_argument("--steps", type=int, default=30000, help="env-step budget")
    ap.add_argument("--n-step", type=int, default=1, help=">1 enables n-step returns (Rainbow)")
    ap.add_argument("--per", action="store_true", help="enable prioritized experience replay (Rainbow)")
    ap.add_argument("--eps-end", type=float, default=0.05, help="final epsilon floor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="dino_b")
    # OPT-IN difficulty: default None keeps the cacti-only behavior (BIRD_PROB=0.0 in dino_game).
    # Pass --bird-prob >0 to enable BIRDS-REQUIRE-DUCK (jump cacti AND duck birds).
    ap.add_argument("--bird-prob", type=float, default=None,
                    help="opt-in: per-spawn bird probability (default None = cacti-only)")
    ap.add_argument("--bird-start-frame", type=int, default=None,
                    help="opt-in: first game frame birds may spawn (default None = module default)")
    # OPT-IN VARIABLE-JUMP mode: actions become {RUN, SMALL_JUMP, BIG_JUMP}; SHORT/TALL cacti,
    # small-jump clears short only, big-jump clears both but costs a small reward penalty.
    ap.add_argument("--variable-jump", action="store_true",
                    help="opt-in: VARIABLE-JUMP mode (pick SMALL vs BIG jump for cactus height)")
    # OPT-IN FULL mode: 4 actions {RUN,SMALL_JUMP,BIG_JUMP,DUCK}; mix of SHORT/TALL cacti + birds.
    ap.add_argument("--full-mode", action="store_true",
                    help="opt-in: FULL mode (SHORT/TALL cacti + duck-only birds, 4 actions)")
    ap.add_argument("--n-actions", type=int, default=3,
                    help="number of actions/Q-outputs (default 3; forced to 4 by --full-mode)")
    args = ap.parse_args()
    res = train(encoder=args.encoder, n_qubits=args.n_qubits, n_layers=args.n_layers,
                observable=args.observable, head=args.head, entangler=args.entangler,
                seed=args.seed, name=args.name,
                cfg={"max_env_steps": args.steps, "n_step": args.n_step, "per": args.per,
                     "eps_end": args.eps_end},
                bird_prob=args.bird_prob, bird_start_frame=args.bird_start_frame,
                variable_jump=args.variable_jump, full_mode=args.full_mode,
                n_actions=args.n_actions)
    print("DONE:", {k: res[k] for k in ("name", "episodes", "env_steps")}, "eval:", res["eval"]["mean"])


if __name__ == "__main__":
    main()
