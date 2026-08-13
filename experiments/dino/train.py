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
from experiments.dino.replay import ImageReplayBuffer
from src.seeds import make_rng, set_global_seeds
from src.trainer import dqn_update, linear_epsilon, select_action

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
    )


def evaluate(model, n_episodes: int = 20, seed: int = 10_000, max_steps: int = 2000) -> dict:
    """Greedy (epsilon=0) eval over fixed seeds → score stats (mean/median/best + list)."""
    was_training = model.training
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(n_episodes):
            env = make_dino_env(max_steps=max_steps, seed=seed + i)
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


def random_baseline(n_episodes: int = 20, seed: int = 10_000, max_steps: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(n_episodes):
        env = make_dino_env(max_steps=max_steps, seed=seed + i)
        obs, info = env.reset(seed=seed + i)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(int(rng.integers(3)))
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
          n_layers: int = 3, head: str = "vqc", seed: int = 0, name: str = "dino",
          w_init: float = 10.0, lam_init: float = 1.0, cfg: dict | None = None, log=print) -> dict:
    cfg = {**default_config(), **(cfg or {})}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    ckpt_path = os.path.join(RESULTS_DIR, f"{name}.pt")
    log_path = os.path.join(RESULTS_DIR, f"{name}.log")
    ctor = dict(n_qubits=n_qubits, n_actions=3, n_layers=n_layers, encoder=encoder,
                observable=observable, head=head)

    # tee every console line into a persistent results/{name}.log (a permanent training record)
    _logf = open(log_path, "w", encoding="utf-8")
    _user_log = log

    def log(msg):
        _user_log(msg)
        _logf.write(str(msg) + "\n")
        _logf.flush()

    set_global_seeds(seed)
    rng = make_rng(seed)

    model = DinoQFunction(n_qubits=n_qubits, n_actions=3, n_layers=n_layers, encoder=encoder,
                          observable=observable, head=head, w_init=w_init, lam_init=lam_init, seed=seed)
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
    buffer = ImageReplayBuffer(cfg["replay_capacity"], (4, 84, 84), rng)

    env = make_dino_env(max_steps=cfg["max_steps"], seed=seed)

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
            buffer.push(obs, action, float(reward), next_obs, bool(terminated))  # only crash masks bootstrap
            obs = next_obs
            score = info["score"]
            env_steps += 1
            if len(buffer) >= cfg["learning_starts"] and env_steps % cfg["train_every"] == 0:
                loss = dqn_update(model, target, buffer, cfg["batch_size"], cfg["gamma"],
                                  optimizer, cfg["grad_clip"], double=cfg["double"])
                losses.append(loss)
                grad_steps += 1
                if grad_steps % cfg["target_update"] == 0:
                    target.load_state_dict(model.state_dict())
            done = terminated or truncated

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
    ev = evaluate(model, n_episodes=20, max_steps=cfg["max_steps"])
    log(f"[{name}] EVAL(best) mean {ev['mean']:.1f} median {ev['median']:.1f} best {ev['best']:.0f} "
        f"(min {ev['min']:.0f})")

    _save(model, ckpt_path, ctor, seed, ev, pc, cfg)
    log(f"[{name}] saved checkpoint -> {ckpt_path}")
    log(f"[{name}] logs: {csv_path} (per-episode) | {log_path} (console)")
    _logf.close()
    return {"name": name, "csv": csv_path, "ckpt": ckpt_path, "log": log_path, "eval": ev,
            "best_ma20": best_ma, "episodes": episode, "env_steps": env_steps, "param_counts": pc}


def main():
    ap = argparse.ArgumentParser(description="Train the CNN→VQC dino agent (Double-QDQN, torch_sv).")
    ap.add_argument("--encoder", default="trainable_cnn", choices=["trainable_cnn", "pretrained"])
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--observable", default="zz", choices=["zz", "z"])
    ap.add_argument("--head", default="vqc", choices=["vqc", "classical"])
    ap.add_argument("--steps", type=int, default=30000, help="env-step budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="dino_b")
    args = ap.parse_args()
    res = train(encoder=args.encoder, n_qubits=args.n_qubits, n_layers=args.n_layers,
                observable=args.observable, head=args.head, seed=args.seed, name=args.name,
                cfg={"max_env_steps": args.steps})
    print("DONE:", {k: res[k] for k in ("name", "episodes", "env_steps")}, "eval:", res["eval"]["mean"])


if __name__ == "__main__":
    main()
