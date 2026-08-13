"""Record the CNN ↔ quantum-circuit ↔ env INTERACTION as one synchronized animation.

Each frame of the output shows the full data-flow at that instant, side by side:
  ENV (game frame)  →  CNN input [4×84×84]  →  6 CNN features (= encoding angles λ·f)
                    →  quantum circuit ⟨ZZ⟩ per action  →  Q-values → chosen action.

So a viewer literally watches the screen get compressed to 6 numbers, fed into the quantum
circuit as rotation angles, measured as ⟨ZZ⟩, and turned into the dino's action — every step.

Run:  python -m experiments.dino.visualize_pipeline --ckpt results/dino_vqc.pt --out figures/dino_pipeline.gif
"""
from __future__ import annotations

import argparse
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import experiments.dino  # noqa: F401
from experiments.dino.dino_game import DinoGame
from experiments.dino.env import FRAME_SKIP, N_STACK
from experiments.dino.model import load_agent

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "figures")
ACTIONS = ["RUN", "JUMP", "DUCK"]
ACOL = {"RUN": "#7f7f7f", "JUMP": "#2ca02c", "DUCK": "#1f77b4"}


def _internals(model, stack):
    """Run the head and return (features f, ⟨ZZ⟩ o, Q-values, action) for one obs."""
    x = torch.as_tensor(np.stack(stack), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        f = model.encoder(x / 255.0)                       # [1,6] in [-1,1]
        enc = model.lam * f                                # encoding angles
        o = model.vqc(enc)                                 # [1,3] ⟨ZZ⟩
        q = (o + 1.0) * 0.5 * model.w                      # [1,3] Q
    return (f.squeeze(0).numpy(), enc.squeeze(0).numpy(),
            o.squeeze(0).numpy(), q.squeeze(0).numpy(), int(np.argmax(q.numpy())))


def _panel(fig, gs, game, obs_frame, f, enc, o, q, action, score):
    fig.clf()
    fig.suptitle("ENV  →  CNN (perception)  →  QUANTUM CIRCUIT (decision)  →  ACTION",
                 fontsize=13, fontweight="bold")
    # ENV
    ax = fig.add_subplot(gs[0, :2]); ax.imshow(game); ax.axis("off")
    ax.set_title(f"ENV — score {score}   |   VQC head → {ACTIONS[action]}",
                 color=ACOL[ACTIONS[action]], fontweight="bold", loc="left")
    # CNN input
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(obs_frame, cmap="gray"); ax.axis("off")
    ax.set_title("CNN input\n[4×84×84]", fontsize=9)
    # CNN features / encoding angles
    ax = fig.add_subplot(gs[1, 0]); ax.barh(range(len(enc)), enc, color="#1f77b4")
    ax.set_yticks(range(len(enc))); ax.set_yticklabels([f"q{i}" for i in range(len(enc))], fontsize=8)
    ax.set_xlim(-3.2, 3.2); ax.axvline(0, color="k", lw=0.5)
    ax.set_title("CNN → 6 features\n(λ·f = RY encoding angles)", fontsize=9)
    # quantum measurement
    ax = fig.add_subplot(gs[1, 1]); ax.bar(ACTIONS, o, color=[ACOL[a] for a in ACTIONS])
    ax.set_ylim(-1.05, 1.05); ax.axhline(0, color="k", lw=0.5)
    ax.set_title("quantum circuit →\n⟨ZZ⟩ per action", fontsize=9); ax.tick_params(labelsize=8)
    # Q-values / decision
    ax = fig.add_subplot(gs[1, 2])
    bars = ax.bar(ACTIONS, q, color=[ACOL[a] for a in ACTIONS])
    bars[action].set_edgecolor("red"); bars[action].set_linewidth(3)
    ax.set_title("Q = (⟨ZZ⟩+1)/2·w\n→ argmax = action", fontsize=9); ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def record(ckpt, out, seed=20000, max_frames=360, every=2, fps=15,
           bird_prob=None, bird_start_frame=None, log=print):
    import imageio
    model = load_agent(ckpt)
    log(f"loaded {ckpt} (head={model.head_kind})")
    game = DinoGame(seed=seed, bird_prob=bird_prob, bird_start_frame=bird_start_frame)
    game.reset(seed=seed)
    stack = deque([game.render_gray84()] * N_STACK, maxlen=N_STACK)
    f, enc, o, q, action = _internals(model, stack)
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
    frames = []
    for t in range(max_frames):
        if t % FRAME_SKIP == 0:
            f, enc, o, q, action = _internals(model, stack)
        if t % every == 0:
            _panel(fig, gs, game.render_rgb(), np.stack(stack)[-1], f, enc, o, q, action, game.score)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if game.step(action):
            break
        stack.append(game.render_gray84())
    plt.close(fig)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    log(f"saved {out} ({len(frames)} frames, final score {game.score})")
    try:
        imageio.mimsave(os.path.splitext(out)[0] + ".mp4", frames, fps=fps, codec="libx264")
        log("saved mp4 too")
    except Exception as e:
        log(f"(mp4 skipped: {type(e).__name__})")
    return out


def main():
    ap = argparse.ArgumentParser(description="Record the CNN↔quantum↔env interaction animation.")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "results", "dino_vqc.pt"))
    ap.add_argument("--out", default=os.path.join(FIG, "dino_pipeline.gif"))
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--max-frames", type=int, default=360)
    ap.add_argument("--bird-prob", type=float, default=None)
    ap.add_argument("--bird-start-frame", type=int, default=None)
    args = ap.parse_args()
    record(args.ckpt, args.out, seed=args.seed, max_frames=args.max_frames,
           bird_prob=args.bird_prob, bird_start_frame=args.bird_start_frame)


if __name__ == "__main__":
    main()
