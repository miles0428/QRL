"""Sanity check for the OPT-IN VARIABLE-JUMP mode (pick the right jump strength).

Run:  python -m experiments.dino.sanity_variable_jump

Confirms, on the ``variable_jump=True`` env:
  * a scripted ORACLE (SMALL_JUMP for SHORT cacti, BIG_JUMP for TALL cacti) survives far longer
    than a RANDOM policy -> the mode is winnable and both jump strengths are usable;
  * a SMALL-JUMP-ONLY policy (always small-jumps every cactus) crashes early because a small jump
    cannot clear a TALL cactus -> the BIG_JUMP is genuinely required (not cosmetic);
  * an ALWAYS-BIG-JUMP policy survives but earns LESS shaped reward per obstacle than the oracle,
    because each avoidable big jump is penalized -> "always big" is sub-optimal.

Deterministic (seeded RNG per episode), headless, no window.
"""
from __future__ import annotations

import numpy as np

from experiments.dino.dino_game import DINO_X, DINO_STAND_W, RUN, SMALL_JUMP, BIG_JUMP
from experiments.dino.env import make_dino_env

# jump slightly before the cactus reaches the dino; wide enough that both jump sizes clear in time
_TRIGGER = 55


def _nearest_ahead(game):
    ahead = [o for o in game.obstacles if o.rect.right > DINO_X]
    if not ahead:
        return None
    return min(ahead, key=lambda o: o.rect.x)


def oracle_action(game) -> int:
    """SMALL_JUMP a short cactus, BIG_JUMP a tall one, timed by distance."""
    nxt = _nearest_ahead(game)
    if nxt is None:
        return RUN
    dist = nxt.rect.x - (DINO_X + DINO_STAND_W)
    if not (0 < dist < _TRIGGER):
        return RUN
    return BIG_JUMP if nxt.kind == "cactus_tall" else SMALL_JUMP


def small_only_action(game) -> int:
    """Always small-jump every cactus (should die on the first TALL cactus)."""
    nxt = _nearest_ahead(game)
    if nxt is None:
        return RUN
    dist = nxt.rect.x - (DINO_X + DINO_STAND_W)
    return SMALL_JUMP if 0 < dist < _TRIGGER else RUN


def big_only_action(game) -> int:
    """Always big-jump every cactus (survives, but is penalized on short cacti)."""
    nxt = _nearest_ahead(game)
    if nxt is None:
        return RUN
    dist = nxt.rect.x - (DINO_X + DINO_STAND_W)
    return BIG_JUMP if 0 < dist < _TRIGGER else RUN


def rollout(policy, n_episodes=20, seed0=0, max_steps=1200):
    scores, rewards = [], []
    for ep in range(n_episodes):
        env = make_dino_env(max_steps=max_steps, seed=seed0 + ep, variable_jump=True)
        obs, info = env.reset(seed=seed0 + ep)
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8, (obs.shape, obs.dtype)
        done = False
        total_r = 0.0
        while not done:
            a = policy(env.game)
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            done = term or trunc
        scores.append(info["score"])
        rewards.append(total_r)
    return np.array(scores, dtype=float), np.array(rewards, dtype=float)


def main():
    rng = np.random.default_rng(0)
    N, MS = 20, 1200

    rand_s, rand_r = rollout(lambda g: int(rng.integers(3)), n_episodes=N, max_steps=MS)
    orc_s, orc_r = rollout(oracle_action, n_episodes=N, max_steps=MS)
    small_s, small_r = rollout(small_only_action, n_episodes=N, max_steps=MS)
    big_s, big_r = rollout(big_only_action, n_episodes=N, max_steps=MS)

    def line(tag, s, r):
        print(f"{tag:16s} score: mean {s.mean():6.1f}  median {np.median(s):6.1f}  "
              f"best {s.max():4.0f}  min {s.min():4.0f} | shaped_reward mean {r.mean():7.2f}")

    print(f"VARIABLE-JUMP sanity  (N={N} episodes, max_steps={MS})")
    line("random", rand_s, rand_r)
    line("oracle(S/B)", orc_s, orc_r)
    line("small-jump-only", small_s, small_r)
    line("big-jump-only", big_s, big_r)

    # 1) oracle beats random by a wide margin -> mode is winnable, both jumps usable
    assert orc_s.mean() > 3 * rand_s.mean() + 20, "oracle should crush random"
    # 2) small-jump-only dies FAR earlier than the oracle -> big jump is genuinely needed for TALL
    assert small_s.mean() < 0.5 * orc_s.mean(), \
        "small-jump-only should fail on tall cacti (big jump is required)"
    # 3) always-big survives but earns LESS shaped reward than the oracle -> big-jump is costly,
    #    so 'always big' is sub-optimal (the intended incentive to pick the right jump size)
    assert big_r.mean() < orc_r.mean(), \
        "always-big-jump should earn less shaped reward than the oracle (penalty makes it costly)"

    print("VARIABLE-JUMP SANITY OK  "
          "(oracle >> random; small-only fails on tall cacti; always-big is penalized/sub-optimal)")


if __name__ == "__main__":
    main()
