"""Env de-risk (house-rule priority #1): API + shapes + fps + a scripted-oracle check.

Run:  python -m experiments.dino.sanity_env
Confirms: Gymnasium API, observation [4,84,84] uint8, steps/sec, and that a hand-written
"jump the cactus / duck the bird" oracle survives far longer than random -- i.e. the game is
winnable AND the two obstacle types genuinely require different actions (perception matters).
Also saves a sample frame to figures/dino_sample_frame.png.
"""
from __future__ import annotations

import os
import time

import numpy as np

from experiments.dino.dino_game import DINO_X, JUMP, DUCK, NONE
from experiments.dino.env import make_dino_env, DinoRawEnv

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def oracle_action(game) -> int:
    """Look at the nearest obstacle ahead of the dino; jump a cactus, duck a bird."""
    ahead = [o for o in game.obstacles if o.rect.right > DINO_X]
    if not ahead:
        return NONE
    nxt = min(ahead, key=lambda o: o.rect.x)
    dist = nxt.rect.x - (DINO_X + 24)
    if nxt.kind == "bird":
        # hold duck through the whole bird window
        return DUCK if dist < 90 else NONE
    else:  # cactus: jump so the apex lines up with the obstacle
        return JUMP if 0 < dist < 70 else NONE


def rollout(policy, n_episodes=20, seed0=0, max_steps=2000):
    scores = []
    for ep in range(n_episodes):
        env = make_dino_env(max_steps=max_steps, seed=seed0 + ep)
        obs, info = env.reset(seed=seed0 + ep)
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8, (obs.shape, obs.dtype)
        done = False
        while not done:
            a = policy(env.game)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        scores.append(info["score"])
    return np.array(scores)


def main():
    # --- API + shape + fps on a random policy ---
    env = make_dino_env(seed=0)
    obs, info = env.reset(seed=0)
    print(f"obs shape {obs.shape} dtype {obs.dtype} | action space {env.action_space}")
    rng = np.random.default_rng(0)
    t0 = time.time()
    steps = 0
    for _ in range(3000):
        a = int(rng.integers(3))
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        if term or trunc:
            env.reset()
    dt = time.time() - t0
    print(f"env speed: {steps/dt:,.0f} steps/sec ({steps} steps in {dt:.2f}s)")

    # --- random vs scripted-oracle (does the game reward correct actions?) ---
    # cap oracle episodes so the check stays quick (it otherwise survives to max_steps)
    rand = rollout(lambda g: int(rng.integers(3)), n_episodes=12, max_steps=800)
    orc = rollout(oracle_action, n_episodes=12, max_steps=800)
    print(f"random  score: mean {rand.mean():6.1f}  median {np.median(rand):6.1f}  "
          f"best {rand.max():4.0f}")
    print(f"oracle  score: mean {orc.mean():6.1f}  median {np.median(orc):6.1f}  "
          f"best {orc.max():4.0f}")
    assert orc.mean() > 3 * rand.mean() + 20, "oracle should beat random by a wide margin"

    # --- save a sample raw frame ---
    raw = DinoRawEnv(seed=1)
    raw.reset(seed=1)
    for _ in range(120):
        raw.step(NONE)
    try:
        from PIL import Image
        fig_dir = os.path.join(_ROOT, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        out = os.path.join(fig_dir, "dino_sample_frame.png")
        Image.fromarray(raw.render()).save(out)
        print(f"saved sample frame -> {out}")
    except Exception as e:  # non-fatal
        print(f"(frame save skipped: {e})")

    print("ENV SANITY OK")


if __name__ == "__main__":
    main()
