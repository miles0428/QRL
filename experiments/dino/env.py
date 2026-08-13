"""Gymnasium wrapper + Atari-style preprocessing for the headless dino game.

Two public pieces:

  * :class:`DinoRawEnv`   -- Gymnasium env exposing the raw RGB frame (for the demo GIF).
  * :func:`make_dino_env` -- the training env: grayscale -> resize 84x84 -> frame-stack 4,
                             observation ``[4, 84, 84]`` uint8, actions ``Discrete(3)``.

Preprocessing is written explicitly (not via gymnasium's built-in wrappers) so the output
is a guaranteed ``[4,84,84]`` uint8 array regardless of the installed gymnasium version --
one less version trap for the demo. Grayscale uses luminance weights; resize uses PIL
(cv2 is not installed on this box).

Reward is ``+1`` per surviving frame (score == frames survived), episode ``terminated`` on
crash, ``truncated`` at ``max_steps``. That per-frame reward with gamma=0.99 gives a bounded
return (~<=100 for long runs), matching the backbone VQC's trainable output-scale head ``w``.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dino_game import DinoGame, NONE, JUMP, DUCK  # noqa: F401  (re-exported action ids)

FRAME_SIZE = 84
N_STACK = 4
N_ACTIONS = 3

# Shaped reward (the learning signal; reported `score` stays == frames survived):
#   a small per-frame alive bonus + a strong per-obstacle-cleared bonus. The clear bonus is
#   what makes "jump/duck at the right moment" pay off — without it, every action gets +1/frame
#   until a crash, so the critical pre-obstacle transitions are drowned out and DQN learns
#   "all actions are equally good" (observed: greedy policy stuck at random-level score).
ALIVE_R = 0.01
CLEAR_R = 1.0
FRAME_SKIP = 4          # action repeat (Nature-DQN): coarser control + 4x fewer VQC forwards


class DinoRawEnv(gym.Env):
    """Raw-frame dino env (RGB observation). Used to render the demo GIF at full size."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, max_steps: int = 2000, seed: int | None = None,
                 bird_prob: float | None = None, bird_start_frame: int | None = None):
        super().__init__()
        self.max_steps = int(max_steps)
        self.game = DinoGame(seed=seed, bird_prob=bird_prob, bird_start_frame=bird_start_frame)
        from .dino_game import WIDTH, HEIGHT
        self.observation_space = spaces.Box(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._steps = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._steps = 0
        return self.game.render_rgb(), {"score": 0}

    def step(self, action: int):
        crashed = self.game.step(int(action))
        self._steps += 1
        terminated = bool(crashed)
        truncated = self._steps >= self.max_steps
        reward = 0.0 if crashed else 1.0
        obs = self.game.render_rgb()
        return obs, reward, terminated, truncated, {"score": self.game.score}

    def render(self):
        return self.game.render_rgb()


class DinoImageEnv(gym.Env):
    """Training env: grayscale + resize + frame-stack -> observation ``[4,84,84]`` uint8.

    Wraps the same :class:`DinoGame`. On reset the first processed frame is repeated to
    fill the stack; each step appends the newest processed frame (oldest drops out), so the
    stack carries motion information (obstacle approach speed, jump phase).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, max_steps: int = 2000, seed: int | None = None, frame_skip: int = FRAME_SKIP,
                 bird_prob: float | None = None, bird_start_frame: int | None = None):
        super().__init__()
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.game = DinoGame(seed=seed, bird_prob=bird_prob, bird_start_frame=bird_start_frame)
        self.observation_space = spaces.Box(0, 255, (N_STACK, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._frames: deque[np.ndarray] = deque(maxlen=N_STACK)
        self._steps = 0

    def _obs(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)  # (4,84,84) uint8

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._steps = 0
        frame = self.game.render_gray84()
        self._frames.clear()
        for _ in range(N_STACK):
            self._frames.append(frame)
        return self._obs(), {"score": 0}

    def step(self, action: int):
        action = int(action)
        total_r = 0.0
        terminated = truncated = False
        for _ in range(self.frame_skip):                 # action repeat (frame-skip)
            prev_score, prev_cleared = self.game.score, self.game.obstacles_cleared
            crashed = self.game.step(action)
            self._steps += 1
            self._frames.append(self.game.render_gray84())
            total_r += ALIVE_R * (self.game.score - prev_score) \
                + CLEAR_R * (self.game.obstacles_cleared - prev_cleared)
            if crashed:
                terminated = True
                break
            if self._steps >= self.max_steps:
                truncated = True
                break
        return self._obs(), total_r, terminated, truncated, \
            {"score": self.game.score, "cleared": self.game.obstacles_cleared}

    def render(self):
        return self.game.render_rgb()


def make_dino_env(max_steps: int = 2000, seed: int | None = None,
                  frame_skip: int = FRAME_SKIP, bird_prob: float | None = None,
                  bird_start_frame: int | None = None) -> DinoImageEnv:
    """Factory for the training env ([4,84,84] uint8 observation, Discrete(3) actions).

    ``bird_prob``/``bird_start_frame`` are OPT-IN difficulty: leaving them ``None`` keeps the
    current cacti-only default (``BIRD_PROB=0.0``); passing ``bird_prob>0`` enables the harder
    BIRDS-REQUIRE-DUCK mode (jump cacti AND duck birds).
    """
    return DinoImageEnv(max_steps=max_steps, seed=seed, frame_skip=frame_skip,
                        bird_prob=bird_prob, bird_start_frame=bird_start_frame)
