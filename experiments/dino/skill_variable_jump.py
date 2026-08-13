"""Skill evidence for the VARIABLE-JUMP mode: does the trained agent pick the RIGHT jump size?

Run:  python -m experiments.dino.skill_variable_jump --ckpt results/vj_classical.pt --episodes 20

Runs the greedy policy over N episodes on the ``variable_jump=True`` env and, per cactus HEIGHT
(SHORT vs TALL), counts which jump the agent used to get past it: SMALL_JUMP, BIG_JUMP, or RUN
(no jump). A skilled agent SMALL_JUMPs short cacti (cheap, sufficient) and BIG_JUMPs tall ones
(a small jump can't clear them). Attribution: each obstacle is bucketed by the take-off action
taken while it was the nearest cactus in the dino's jump-trigger zone; if the dino never jumped
for it, it's a RUN.

Works for any head (classical or vqc) — it drives the env with the checkpoint's greedy argmax.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import experiments.dino  # noqa: F401
from experiments.dino.dino_game import DINO_X, DINO_STAND_W, SMALL_JUMP, BIG_JUMP
from experiments.dino.env import make_dino_env
from experiments.dino.model import load_agent

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRIGGER = 90   # x-distance (px) within which a jump is attributed to the nearest cactus ahead


def _nearest_cactus(game):
    ahead = [o for o in game.obstacles if o.rect.right > DINO_X and o.kind.startswith("cactus")]
    if not ahead:
        return None
    return min(ahead, key=lambda o: o.rect.x)


def analyze(model, n_episodes=20, seed0=20000, max_steps=1200):
    # counts[height][action] where height in {short,tall}, action in {run,small,big}
    counts = {"short": {"run": 0, "small": 0, "big": 0},
              "tall": {"run": 0, "small": 0, "big": 0}}
    scores = []
    model.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            env = make_dino_env(max_steps=max_steps, seed=seed0 + ep, variable_jump=True)
            obs, info = env.reset(seed=seed0 + ep)
            game = env.game
            # remember, per obstacle id, the take-off jump used for it (small/big) while nearest
            takeoff: dict[int, str] = {}
            heights: dict[int, str] = {}    # id -> "short"|"tall"
            counted: set[int] = set()
            done = False
            while not done:
                q = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                a = int(torch.argmax(q, dim=1).item())
                # attribute this action's take-off (if a jump starts this frame-skip block) to the
                # nearest cactus in the trigger zone. We snapshot BEFORE stepping (dino grounded).
                nxt = _nearest_cactus(game)
                will_takeoff = (not game.airborne) and a in (SMALL_JUMP, BIG_JUMP)
                if nxt is not None:
                    dist = nxt.rect.x - (DINO_X + DINO_STAND_W)
                    hid = id(nxt)
                    heights[hid] = "tall" if nxt.kind == "cactus_tall" else "short"
                    if will_takeoff and 0 < dist < _TRIGGER and hid not in takeoff:
                        takeoff[hid] = "big" if a == BIG_JUMP else "small"
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
                # tally any obstacle that has now been fully cleared (passed the dino), once
                for oid, h in list(heights.items()):
                    if oid in counted:
                        continue
                    # obstacle is gone from the field once cleared+despawned; approximate "cleared"
                    # by it no longer being ahead AND having been seen in the trigger zone
                # (final tally done after the loop from cleared set below)
                # mark obstacles that left the field
                present = {id(o) for o in game.obstacles}
                for oid, h in heights.items():
                    if oid not in present and oid not in counted:
                        counted.add(oid)
                        act = takeoff.get(oid, "run")
                        counts[h][act] += 1
            # end-of-episode: tally any obstacles still tracked but cleared/gone
            present = {id(o) for o in game.obstacles}
            for oid, h in heights.items():
                if oid not in counted:
                    counted.add(oid)
                    act = takeoff.get(oid, "run")
                    counts[h][act] += 1
            scores.append(info["score"])
    return counts, np.array(scores, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Per-cactus-height action counts for VARIABLE-JUMP.")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "results", "vj_classical.pt"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed0", type=int, default=20000)
    args = ap.parse_args()

    model = load_agent(args.ckpt)
    counts, scores = analyze(model, n_episodes=args.episodes, seed0=args.seed0,
                             max_steps=args.max_steps)

    print(f"SKILL EVIDENCE (VARIABLE-JUMP)  ckpt={os.path.basename(args.ckpt)}  "
          f"episodes={args.episodes}")
    print(f"greedy score: mean {scores.mean():.1f}  median {np.median(scores):.1f}  "
          f"best {scores.max():.0f}  min {scores.min():.0f}")
    for h in ("short", "tall"):
        c = counts[h]
        tot = c["run"] + c["small"] + c["big"]
        if tot == 0:
            print(f"  {h.upper():5s} cacti: (none encountered)")
            continue
        print(f"  {h.upper():5s} cacti (n={tot:3d}): "
              f"SMALL_JUMP {c['small']:3d} ({100*c['small']/tot:4.0f}%)  "
              f"BIG_JUMP {c['big']:3d} ({100*c['big']/tot:4.0f}%)  "
              f"RUN {c['run']:3d} ({100*c['run']/tot:4.0f}%)")
    # a skilled agent: mostly SMALL on short, mostly BIG on tall
    st = counts["short"]; tl = counts["tall"]
    st_tot = max(1, st["small"] + st["big"] + st["run"])
    tl_tot = max(1, tl["small"] + tl["big"] + tl["run"])
    print(f"\n  short->SMALL rate {100*st['small']/st_tot:.0f}% | "
          f"tall->BIG rate {100*tl['big']/tl_tot:.0f}%")


if __name__ == "__main__":
    main()
