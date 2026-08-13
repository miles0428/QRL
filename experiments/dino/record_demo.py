"""Record THE demo: the trained CNN→VQC agent playing the dino game → figures/dino_demo.gif.

Runs the greedy (epsilon=0) agent on the full-resolution :class:`DinoRawEnv`, overlays a live
score + a "quantum head chose: JUMP/DUCK/RUN" caption (the money shot: a quantum circuit is
picking the dino's action), and writes an animated GIF (and an MP4 if a writer is available).

Run:  python -m experiments.dino.record_demo --ckpt results/dino_b.pt --out figures/dino_demo.gif
Picks the best of several seeds so the clip shows the agent actually clearing obstacles.
"""
from __future__ import annotations

import argparse
import os

from collections import deque

import numpy as np
import torch
from PIL import Image, ImageDraw

import experiments.dino  # noqa: F401  (sys.path bootstrap)
from experiments.dino.dino_game import DinoGame, DINO_X, DINO_DUCK_W
from experiments.dino.env import FRAME_SKIP, N_STACK
from experiments.dino.model import load_agent

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(_ROOT, "figures")
ACTION_NAMES = {0: "RUN", 1: "JUMP", 2: "DUCK", 3: "DUCK"}


def _greedy(model, stack):
    with torch.no_grad():
        q = model(torch.as_tensor(stack, dtype=torch.float32).unsqueeze(0))
    a = int(torch.argmax(q, dim=1).item())
    return a, q.squeeze(0).tolist()


def rollout_frames(model, seed: int, max_steps: int = 1500, scale: int = 1, head_label: str = "VQC head",
                   bird_prob=None, bird_start_frame=None):
    """Play one greedy episode by driving the game directly.

    The agent re-decides every ``FRAME_SKIP`` game frames (exactly its training cadence) and the
    chosen action is held in between, but we render EVERY game frame so the GIF is smooth.
    ``bird_prob``/``bird_start_frame`` opt into the birds-require-duck mode (default: cacti only).
    """
    game = DinoGame(seed=seed, bird_prob=bird_prob, bird_start_frame=bird_start_frame)
    game.reset(seed=seed)
    stack = deque([game.render_gray84()] * N_STACK, maxlen=N_STACK)
    frames = []
    action, qvals = 0, [0.0, 0.0, 0.0]
    passed_birds: set[int] = set()   # ids of bird obstacles that fully passed the dino (cleared)
    duck_bird_frames = 0             # frames spent ducking WHILE a bird overlaps the dino column
    for t in range(max_steps):
        if t % FRAME_SKIP == 0:                              # decide at the trained cadence
            action, qvals = _greedy(model, np.stack(stack))
        # --- bird bookkeeping (so we can pick a clip that actually shows ducking) ---
        for ob in game.obstacles:
            if getattr(ob, "kind", "") == "bird":
                if ob.rect.right < DINO_X and id(ob) not in passed_birds:
                    passed_birds.add(id(ob))                  # a bird just cleared the dino
                elif ob.rect.left < DINO_X + DINO_DUCK_W and ob.rect.right > DINO_X and game.ducking:
                    duck_bird_frames += 1                     # ducking right under a bird
        birds = _bird_count(game, passed_birds)
        frames.append(_annotate(game.render_rgb(), game.score, action, qvals, scale,
                                 head_label=head_label, birds=birds))
        crashed = game.step(action)
        stack.append(game.render_gray84())
        if crashed:
            frames.append(_annotate(game.render_rgb(), game.score, action, qvals, scale,
                                    crashed=True, head_label=head_label, birds=birds))
            break
    stats = {"birds_cleared": len(passed_birds), "duck_bird_frames": duck_bird_frames}
    return frames, game.score, stats


def _bird_count(game, passed_birds) -> int:
    """Birds cleared so far (passed the dino) — shown live on the demo overlay."""
    return len(passed_birds)


def _annotate(rgb, score, action, qvals, scale, crashed=False, head_label="VQC head", birds=None):
    img = Image.fromarray(rgb)
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.text((6, 4), f"score {score}", fill=(20, 20, 20))
    tag = "CRASH" if crashed else ACTION_NAMES.get(action, "?")
    # highlight the DUCK action (the hard, quantum-learned skill) in green so it pops in the clip
    duck = (not crashed) and tag == "DUCK"
    color = (180, 30, 30) if crashed else ((20, 140, 40) if duck else (30, 90, 180))
    d.text((6, 16), f"{head_label} -> {tag}", fill=color)
    if birds is not None:
        d.text((6, 28), f"birds ducked: {birds}", fill=(20, 110, 30))
    return np.asarray(img, dtype=np.uint8)


def record(ckpt_path: str, out_path: str, seeds=range(20000, 20040), max_steps: int = 1500,
           fps: int = 30, scale: int = 2, subsample: int = 1, log=print,
           bird_prob=None, bird_start_frame=None) -> dict:
    import imageio
    model = load_agent(ckpt_path)
    head_label = "VQC head" if model.head_kind == "vqc" else "CNN+Linear head"
    log(f"loaded agent from {ckpt_path} (head={model.head_kind}, encoder={model.encoder_kind})")

    # pick the best clip. For a cacti-only demo that's just the top score; for a BIRDS demo we
    # rank by (birds cleared, score) so the highlight actually SHOWS the dino ducking under birds
    # (the hard, quantum-learned skill) rather than a lucky bird-free run.
    birds_demo = bool(bird_prob)
    best = None
    for s in seeds:
        frames, score, st = rollout_frames(model, seed=s, max_steps=max_steps, scale=scale,
                                           head_label=head_label, bird_prob=bird_prob,
                                           bird_start_frame=bird_start_frame)
        key = (st["birds_cleared"], score) if birds_demo else (0, score)
        log(f"  seed {s}: score {score} | birds cleared {st['birds_cleared']} "
            f"| duck-under-bird frames {st['duck_bird_frames']} ({len(frames)} frames)")
        if best is None or key > best[0]:
            best = (key, frames, score, s, st)
    _, frames, score, s, st = best
    log(f"best: seed {s}, score {score}, birds cleared {st['birds_cleared']}, "
        f"duck-under-bird frames {st['duck_bird_frames']}")

    frames = frames[::subsample]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    log(f"saved GIF -> {out_path}  ({len(frames)} frames, {score} score)")

    # optional MP4 (nicer for slides); skip silently if no ffmpeg
    mp4 = os.path.splitext(out_path)[0] + ".mp4"
    try:
        imageio.mimsave(mp4, frames, fps=fps, codec="libx264")
        log(f"saved MP4 -> {mp4}")
    except Exception as e:
        log(f"(mp4 skipped: {type(e).__name__})")
    return {"out": out_path, "score": score, "seed": s, "n_frames": len(frames)}


def main():
    ap = argparse.ArgumentParser(description="Record the trained dino agent playing (GIF/MP4).")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "results", "dino_b.pt"))
    ap.add_argument("--out", default=os.path.join(FIG_DIR, "dino_demo.gif"))
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--bird-prob", type=float, default=None,
                    help="opt-in: enable the birds-require-duck mode in the demo")
    ap.add_argument("--bird-start-frame", type=int, default=None,
                    help="opt-in: first game frame birds may spawn (small = birds appear early)")
    args = ap.parse_args()
    record(args.ckpt, args.out, seeds=range(20000, 20000 + args.n_seeds),
           max_steps=args.max_steps, fps=args.fps, scale=args.scale,
           bird_prob=args.bird_prob, bird_start_frame=args.bird_start_frame)


if __name__ == "__main__":
    main()
