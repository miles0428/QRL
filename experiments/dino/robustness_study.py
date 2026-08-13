"""Robustness of the trained quantum agent to reaction-time noise and game speed.

Takes a trained agent (default the flagship cacti VQC) and, WITHOUT retraining, sweeps two
perturbations, plotting a score-vs-perturbation degradation curve:

  * REACTION DELAY — the chosen action only takes effect ``delay`` game-frames later (a random-ish
    actuator/human latency on top of the frame-skip). Tests how brittle the agent's timing is.
  * GAME SPEED — the game runs faster (like the real Chrome-Dino accelerating): SPEED_START/MAX are
    scaled by ``speed_mult``. The agent was trained at mult 1.0, so higher is out-of-distribution.

Neither touches the game file (speed is monkey-patched at runtime; delay is applied in the eval
loop), so this coexists with any in-progress env edits.

Run:  python -m experiments.dino.robustness_study --ckpt results/dino_vqc.pt
"""
from __future__ import annotations

import argparse
import os
from collections import deque

import numpy as np
import torch

import experiments.dino  # noqa: F401
import experiments.dino.dino_game as dg
from experiments.dino.dino_game import DinoGame
from experiments.dino.env import FRAME_SKIP, N_STACK
from experiments.dino.model import load_agent

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "figures")
RES = os.path.join(_ROOT, "results")


def _greedy(model, stack):
    with torch.no_grad():
        return int(torch.argmax(model(torch.as_tensor(np.stack(stack), dtype=torch.float32).unsqueeze(0)), 1).item())


def eval_run(model, n_ep=20, seed0=40000, max_steps=1500, delay=0, speed_mult=1.0,
             bird_prob=None, bird_start_frame=None):
    """Greedy eval with an optional reaction DELAY (game-frames) and SPEED multiplier."""
    orig_start, orig_max = dg.SPEED_START, dg.SPEED_MAX
    dg.SPEED_START = orig_start * speed_mult
    dg.SPEED_MAX = orig_max * speed_mult
    scores = []
    try:
        for i in range(n_ep):
            game = DinoGame(seed=seed0 + i, bird_prob=bird_prob, bird_start_frame=bird_start_frame)
            game.reset(seed=seed0 + i)
            stack = deque([game.render_gray84()] * N_STACK, maxlen=N_STACK)
            pending = deque()          # (fire_frame, action) — models reaction latency
            cur = 0
            for t in range(max_steps):
                if t % FRAME_SKIP == 0:
                    pending.append((t + delay, _greedy(model, stack)))
                while pending and pending[0][0] <= t:
                    cur = pending.popleft()[1]
                if game.step(cur):
                    break
                stack.append(game.render_gray84())
            scores.append(game.score)
    finally:
        dg.SPEED_START, dg.SPEED_MAX = orig_start, orig_max   # always restore
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser(description="Reaction-time + speed robustness of the quantum agent.")
    ap.add_argument("--ckpt", default=os.path.join(RES, "dino_vqc.pt"))
    ap.add_argument("--n-ep", type=int, default=20)
    ap.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8])
    ap.add_argument("--speeds", type=float, nargs="+", default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
    args = ap.parse_args()

    model = load_agent(args.ckpt)
    print(f"robustness of {os.path.basename(args.ckpt)} (head={model.head_kind})\n")

    print("=== reaction delay (game-frames) ===")
    delay_curve = []
    for d in args.delays:
        m, s = eval_run(model, n_ep=args.n_ep, delay=d)
        delay_curve.append((d, m, s))
        print(f"  delay {d}: score {m:6.0f} +/- {s:.0f}")

    print("\n=== game speed multiplier ===")
    speed_curve = []
    for sp in args.speeds:
        m, s = eval_run(model, n_ep=args.n_ep, speed_mult=sp)
        speed_curve.append((sp, m, s))
        print(f"  speed x{sp}: score {m:6.0f} +/- {s:.0f}")

    # --- figure + csv ---
    os.makedirs(FIG, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        d, dm, ds = zip(*delay_curve)
        a1.errorbar(d, dm, yerr=ds, marker="o", capsize=3, color="#1f77b4")
        a1.set_xlabel("reaction delay (game frames)"); a1.set_ylabel("greedy score"); a1.grid(alpha=0.3)
        a1.set_title("Robustness to reaction-time noise")
        sp, sm, ss = zip(*speed_curve)
        a2.errorbar(sp, sm, yerr=ss, marker="o", capsize=3, color="#d62728")
        a2.axvline(1.0, ls="--", color="gray", lw=1, label="trained speed")
        a2.set_xlabel("game speed x (real Dino accelerates)"); a2.set_ylabel("greedy score")
        a2.grid(alpha=0.3); a2.legend(); a2.set_title("Robustness to game speed")
        fig.suptitle(f"Quantum agent robustness ({os.path.basename(args.ckpt)})", fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "dino_robustness.png"), dpi=130)
        print("\nsaved figures/dino_robustness.png")
    except Exception as e:
        print(f"(figure skipped: {type(e).__name__})")
    with open(os.path.join(RES, "dino_robustness.csv"), "w", encoding="utf-8") as f:
        f.write("kind,level,score_mean,score_std\n")
        for d, m, s in delay_curve:
            f.write(f"delay,{d},{m:.1f},{s:.1f}\n")
        for sp, m, s in speed_curve:
            f.write(f"speed,{sp},{m:.1f},{s:.1f}\n")
    print("saved results/dino_robustness.csv")


if __name__ == "__main__":
    main()
