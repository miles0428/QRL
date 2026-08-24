"""A headless, fast, pure-pygame T-Rex ("Chrome Dino") clone.

This is the PRIMARY training environment (house rules: de-risk the browser first). No
display window, no browser, no chromedriver -- it draws to an off-screen ``pygame.Surface``
and reads pixels with ``pygame.surfarray``, so it steps thousands of frames/second.

The game is intentionally designed so BOTH actions matter and perception is required:

  * CACTUS  (sits on the ground)      -> must JUMP; ducking/standing collide  ("jump-only")
  * BIRD    (a tall band in the air)  -> must DUCK; jumping/standing collide   ("duck-only")

The bird band spans from above the jump apex down to just above the crouched dino, so a
well-timed jump cannot clear it -- only ducking passes underneath. That makes the two
obstacle types strictly require different actions, which is what forces the CNN->VQC agent
to actually look at the screen instead of spamming one button.

Coordinates use pygame's convention (y increases downward). All spawns come from an
injected ``random.Random`` so a given seed replays identically (deterministic env).

Rendering is dark-on-light (black dino/obstacles/ground on white) for high contrast after
grayscale, which is what the CNN sees.
"""
from __future__ import annotations

import os

# Headless SDL: no window is ever opened (safe on servers / CI / this box).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random

import pygame

# --- world geometry (see module docstring for the jump/duck collision reasoning) -------
WIDTH, HEIGHT = 600, 150
GROUND_Y = 128                      # y of the ground line; dino/obstacles rest their base here

# The agent's OBSERVATION is cropped to this (x0,y0,x1,y1) action region before the 84x84 resize:
# it zooms in on the dino + approaching obstacles + ground, so the (otherwise tiny) obstacles are
# clearly visible to the CNN. The demo GIF still renders the full 600x150 frame. Set to None to
# observe the whole frame.
OBS_CROP = (16, 30, 350, 146)

DINO_X = 50
DINO_STAND_W, DINO_STAND_H = 24, 46
DINO_DUCK_W, DINO_DUCK_H = 36, 24

GRAVITY = 0.9
JUMP_V = -12.5                      # apex ~ v^2/2g ~ 87px -> long airtime => a WIDE "safe jump" window

# --- VARIABLE-JUMP mode constants (opt-in; see DinoGame.variable_jump) -------------------
# A separate "pick the right jump strength" skill. In this mode the 3 action slots mean
# {0:RUN, 1:SMALL_JUMP, 2:BIG_JUMP}. Two jump velocities give two apex heights (verified by
# collision sim): SMALL clears a SHORT cactus but its apex (~50px) is BELOW a TALL cactus top,
# so it collides; BIG (apex ~68px) clears BOTH. A per-BIG_JUMP reward penalty (env-side) makes
# "always big-jump" sub-optimal, so the optimal policy is SMALL for short cacti / BIG for tall.
SMALL_JUMP_V = -10.0               # apex ~50.6px, airtime ~22f -> clears SHORT (26px), NOT TALL (58px)
BIG_JUMP_V = -13.0                 # apex ~88px  , airtime ~29f -> clears BOTH short and tall
VJ_SHORT_H = 26                    # SHORT cactus height (small jump suffices)
VJ_TALL_H = 58                     # TALL  cactus height (needs the big jump; small jump collides)
VJ_CACTUS_W_RANGE = (12, 18)       # same widths both heights (only height distinguishes them)
VJ_TALL_PROB = 0.40                # fraction of cacti that are TALL (rest SHORT); tuned for learnability
# Cactus TIMING is deliberately more irregular in this mode: wider gap randomness.
VJ_GAP_MIN, VJ_GAP_BASE = 150, 220 # min gap + a wide random span => clearly irregular spacing

# obstacle bands
CACTUS_H_RANGE = (26, 38)
CACTUS_W_RANGE = (12, 18)           # narrower cacti => easier to clear with imperfect timing
BIRD_TOP, BIRD_BOTTOM = 50, 100     # tall band: catches the jump apex, passes only when ducked
BIRD_W = 34

# --- FULL mode constants (opt-in; see DinoGame.full_mode) --------------------------------
# FULL mode COMBINES variable-jump cacti (SHORT/TALL) AND birds into ONE 4-action task:
#   {0:RUN, 1:SMALL_JUMP, 2:BIG_JUMP, 3:DUCK}. It reuses the variable-jump physics (SMALL/BIG
# jump velocities, SHORT/TALL cactus heights) AND the duck posture. The bird must stay DUCK-ONLY
# even though a BIG_JUMP is now available, so the full-mode bird band is placed LOWER/deeper than
# the legacy bird band: a BIG_JUMP's lowest airborne dino-bottom is ~40.5px, so with
# FULL_BIRD_TOP=34 the dino body always overlaps the band mid-jump (can't slip over it), while
# FULL_BIRD_BOTTOM=100 sits above the crouch top (GROUND_Y-DINO_DUCK_H = 104) so ducking passes
# and below the stand top (GROUND_Y-DINO_STAND_H = 82) so standing/jumping collide. These are
# SEPARATE constants from BIRD_TOP/BIRD_BOTTOM so the legacy bird mode is unchanged.
FULL_BIRD_TOP, FULL_BIRD_BOTTOM = 34, 100
# per-obstacle-type spawn mix in FULL mode (must sum to <=1; remainder falls through to SHORT).
# Tuned for LEARNABILITY of all 3 skills: birds are made fairly common so the (harder-to-discover)
# DUCK skill gets enough reinforcement -- with birds rare, DQN converges to a "jump every cactus"
# local optimum and never learns to duck. TALL cacti kept a minority so BIG_JUMP isn't the default.
FULL_BIRD_PROB = 0.40             # fraction of obstacles that are birds (the duck skill)
FULL_TALL_PROB = 0.28             # fraction (of the non-bird cacti) that are TALL (big-jump); rest SHORT
FULL_START_FRAME = 120            # brief pure-cactus intro before birds/tall mix in (learnability)

SPEED_START = 4.5                   # slower => wider timing window in frames => reliably learnable
SPEED_GAIN = 0.0007                 # gentle ramp
SPEED_MAX = 9.0

GAP_MIN, GAP_BASE = 150, 240        # wide random span => clearly irregular spacing (not rhythmic)
WARMUP_FRAMES = 80
# Core demo is the JUMP task (cacti only): pure jump-timing is what the CNN->VQC reliably learns.
# Birds (the duck skill) are much harder to learn in a small step budget, so they are OFF by
# default here and documented as future work. Set BIRD_PROB>0 to re-enable the duck obstacle.
BIRD_PROB = 0.0
BIRD_START_FRAME = 900

# colors
WHITE = (247, 247, 247)
BLACK = (30, 30, 30)
GRAY = (120, 120, 120)

# actions
NONE, JUMP, DUCK = 0, 1, 2
# In VARIABLE-JUMP mode the same 3 slots are reinterpreted (no duck, no birds):
RUN, SMALL_JUMP, BIG_JUMP = 0, 1, 2
# In FULL mode there are 4 slots: {0:RUN, 1:SMALL_JUMP, 2:BIG_JUMP, 3:DUCK} (variable-jump cacti
# AND birds combined). FULL_DUCK reuses slot 3 (the 4th action) — note DUCK==2 above is the
# legacy/bird-mode duck; in FULL mode the duck posture is action index 3.
FULL_DUCK = 3


class Obstacle:
    __slots__ = ("kind", "rect")

    def __init__(self, kind: str, rect: pygame.Rect):
        self.kind = kind          # "cactus" | "bird"
        self.rect = rect


class DinoGame:
    """Pure game logic + off-screen rendering. Frame-stepped by :class:`DinoEnv`.

    Difficulty is OPT-IN and backward-compatible: ``bird_prob`` / ``bird_start_frame``
    default to the module constants (``BIRD_PROB=0.0`` -> the current cacti-only task), so
    ``DinoGame(seed=...)`` behaves exactly as before. Pass ``bird_prob>0`` to enable the
    harder BIRDS-REQUIRE-DUCK mode where the agent must JUMP cacti AND DUCK birds.

    A third, independent OPT-IN mode is ``variable_jump`` (default ``False``). When on, the 3
    action slots mean ``{0:RUN, 1:SMALL_JUMP, 2:BIG_JUMP}`` (NO duck, NO birds): cacti come in
    two heights (SHORT / TALL); a SMALL_JUMP clears a SHORT cactus but is too low for a TALL one
    (the dino hits it), while a BIG_JUMP clears both. Cactus timing is also more irregular. The
    "big-jump is costly" incentive (so always-big is not optimal) is applied env-side as a small
    per-BIG_JUMP reward penalty (see ``DinoImageEnv``). ``variable_jump`` is orthogonal to the
    bird mode; the two are not meant to be combined.

    A fourth OPT-IN mode is ``full_mode`` (default ``False``), which COMBINES the variable-jump
    cacti AND the birds into a single 4-action task ``{0:RUN, 1:SMALL_JUMP, 2:BIG_JUMP, 3:DUCK}``:
    obstacles are a random mix of SHORT cactus (SMALL_JUMP), TALL cactus (BIG_JUMP) and BIRD
    (DUCK). It reuses the SMALL/BIG jump physics and the duck posture; the bird band is lowered
    (``FULL_BIRD_TOP/FULL_BIRD_BOTTOM``) so it stays duck-only even though a BIG_JUMP exists. The
    BIG_JUMP reward penalty still applies (so "always big-jump" is sub-optimal). ``full_mode``
    supersedes ``variable_jump`` when both are set.
    """

    def __init__(self, seed: int | None = None, bird_prob: float | None = None,
                 bird_start_frame: int | None = None, variable_jump: bool = False,
                 full_mode: bool = False):
        pygame.init()
        # One reusable off-screen surface (never shown). All draws land here.
        self._surf = pygame.Surface((WIDTH, HEIGHT))
        self.rng = random.Random(seed)
        # per-instance difficulty (defaults preserve the module-level cacti-only behavior)
        self.bird_prob = float(BIRD_PROB if bird_prob is None else bird_prob)
        self.bird_start_frame = int(BIRD_START_FRAME if bird_start_frame is None else bird_start_frame)
        self.variable_jump = bool(variable_jump)   # opt-in "pick the right jump strength" mode
        # opt-in FULL mode: variable-jump cacti (SHORT/TALL) + birds, 4 actions {RUN,SMALL,BIG,DUCK}.
        # In FULL mode the SMALL/BIG jump physics are used (like variable_jump) AND the duck posture
        # is available; obstacles are a random mix of SHORT/TALL cacti and (duck-only) birds.
        self.full_mode = bool(full_mode)
        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self.dino_base = float(GROUND_Y)     # y of dino's feet
        self.dino_vy = 0.0
        self.airborne = False
        self.ducking = False
        self.speed = SPEED_START
        self.frame = 0
        self.score = 0
        self.obstacles_cleared = 0           # obstacles the dino has gotten past (reward-shaping signal)
        self._counted: set[int] = set()      # ids of obstacles already counted as cleared
        self.obstacles: list[Obstacle] = []
        self._spawn_x = WIDTH + 40           # next spawn happens once the field is clear enough
        self.big_jumps = 0                   # count of BIG_JUMP take-offs (variable-jump reward penalty)
        self._did_big_jump = False           # set True on the frame a BIG_JUMP takes off (env reads it)
        return

    # -- geometry helpers ----------------------------------------------------
    def _dino_rect(self) -> pygame.Rect:
        if self.ducking and not self.airborne:
            w, h = DINO_DUCK_W, DINO_DUCK_H
        else:
            w, h = DINO_STAND_W, DINO_STAND_H
        top = int(self.dino_base) - h
        return pygame.Rect(DINO_X, top, w, h)

    def _maybe_spawn(self):
        # spawn when the rightmost obstacle has moved far enough left
        if self.variable_jump or self.full_mode:   # wider gap randomness => clearly irregular spacing
            gap = VJ_GAP_MIN + int(self.speed * 5) + self.rng.randint(0, VJ_GAP_BASE)
        else:
            gap = GAP_MIN + int(self.speed * 5) + self.rng.randint(0, GAP_BASE)
        if self.obstacles:
            last = self.obstacles[-1].rect
            if last.right > WIDTH - gap:
                return
        if self.frame < WARMUP_FRAMES:
            return
        if self.full_mode:
            # FULL mode: a random MIX of SHORT cactus / TALL cactus / BIRD, on the widened gap.
            # A short pure-cactus intro (frame < FULL_START_FRAME) eases the start (no birds/tall yet).
            u = self.rng.random()
            intro = self.frame < FULL_START_FRAME
            if (not intro) and u < FULL_BIRD_PROB:          # BIRD -> duck-only band (see FULL_BIRD_*)
                r = pygame.Rect(WIDTH, FULL_BIRD_TOP, BIRD_W, FULL_BIRD_BOTTOM - FULL_BIRD_TOP)
                self.obstacles.append(Obstacle("bird", r))
            else:
                # TALL cactus (needs BIG jump) vs SHORT cactus (SMALL jump suffices)
                tall = (not intro) and (self.rng.random() < FULL_TALL_PROB)
                h = VJ_TALL_H if tall else VJ_SHORT_H
                w = self.rng.randint(*VJ_CACTUS_W_RANGE)
                r = pygame.Rect(WIDTH, GROUND_Y - h, w, h)
                self.obstacles.append(Obstacle("cactus_tall" if tall else "cactus_short", r))
            return
        if self.variable_jump:
            # VARIABLE-JUMP mode: cacti only, in two fixed heights (SHORT / TALL). Tag the height
            # on the obstacle kind so eval/skill-analysis can bucket actions by cactus size.
            tall = self.rng.random() < VJ_TALL_PROB
            h = VJ_TALL_H if tall else VJ_SHORT_H
            w = self.rng.randint(*VJ_CACTUS_W_RANGE)
            r = pygame.Rect(WIDTH, GROUND_Y - h, w, h)
            self.obstacles.append(Obstacle("cactus_tall" if tall else "cactus_short", r))
            return
        # birds only after a pure-cactus intro, and rarer (duck is the harder skill)
        bird_ok = self.frame > self.bird_start_frame
        if bird_ok and self.rng.random() < self.bird_prob:
            r = pygame.Rect(WIDTH, BIRD_TOP, BIRD_W, BIRD_BOTTOM - BIRD_TOP)
            self.obstacles.append(Obstacle("bird", r))
        else:
            h = self.rng.randint(*CACTUS_H_RANGE)
            w = self.rng.randint(*CACTUS_W_RANGE)
            r = pygame.Rect(WIDTH, GROUND_Y - h, w, h)
            self.obstacles.append(Obstacle("cactus", r))

    # -- one physics step ----------------------------------------------------
    def step(self, action: int) -> bool:
        """Advance one frame. Returns True if the dino crashed (episode terminates)."""
        self.frame += 1
        self._did_big_jump = False

        if self.full_mode:
            # actions mean {0:RUN, 1:SMALL_JUMP, 2:BIG_JUMP, 3:DUCK}. Variable-jump SMALL/BIG jump
            # physics PLUS a grounded duck posture (for the duck-only bird). Ducking only takes
            # effect on the ground; a jump take-off requires being grounded.
            self.ducking = (action == FULL_DUCK) and not self.airborne
            if not self.airborne and action in (SMALL_JUMP, BIG_JUMP):
                self.airborne = True
                self.dino_vy = BIG_JUMP_V if action == BIG_JUMP else SMALL_JUMP_V
                if action == BIG_JUMP:
                    self.big_jumps += 1
                    self._did_big_jump = True
            if self.airborne:
                self.dino_vy += GRAVITY
                self.dino_base += self.dino_vy
                if self.dino_base >= GROUND_Y:
                    self.dino_base = float(GROUND_Y)
                    self.dino_vy = 0.0
                    self.airborne = False
        elif self.variable_jump:
            # actions mean {0:RUN, 1:SMALL_JUMP, 2:BIG_JUMP}; no duck posture in this mode.
            self.ducking = False
            if not self.airborne and action in (SMALL_JUMP, BIG_JUMP):
                self.airborne = True
                self.dino_vy = BIG_JUMP_V if action == BIG_JUMP else SMALL_JUMP_V
                if action == BIG_JUMP:
                    self.big_jumps += 1
                    self._did_big_jump = True
            if self.airborne:
                self.dino_vy += GRAVITY
                self.dino_base += self.dino_vy
                if self.dino_base >= GROUND_Y:
                    self.dino_base = float(GROUND_Y)
                    self.dino_vy = 0.0
                    self.airborne = False
        else:
            # ducking is a per-frame posture (must be held); pressing DUCK mid-air fast-falls
            self.ducking = (action == DUCK)
            if action == JUMP and not self.airborne:
                self.airborne = True
                self.dino_vy = JUMP_V
            if self.airborne:
                self.dino_vy += GRAVITY
                if action == DUCK:
                    self.dino_vy += GRAVITY          # fast-fall (classic dino duck-in-air)
                self.dino_base += self.dino_vy
                if self.dino_base >= GROUND_Y:
                    self.dino_base = float(GROUND_Y)
                    self.dino_vy = 0.0
                    self.airborne = False

        # move world
        self.speed = min(SPEED_MAX, self.speed + SPEED_GAIN)
        for ob in self.obstacles:
            ob.rect.x -= int(round(self.speed))
            # count an obstacle the moment the dino has gotten fully past it (reward shaping)
            if id(ob) not in self._counted and ob.rect.right < DINO_X:
                self._counted.add(id(ob))
                self.obstacles_cleared += 1
        self.obstacles = [o for o in self.obstacles if o.rect.right > -5]
        self._maybe_spawn()

        # collision
        dino = self._dino_rect()
        crashed = any(dino.colliderect(o.rect) for o in self.obstacles)
        if not crashed:
            self.score += 1
        return crashed

    # -- rendering -----------------------------------------------------------
    def render_surface(self) -> pygame.Surface:
        """Draw the current state to the off-screen surface and return it (RGB)."""
        s = self._surf
        s.fill(WHITE)
        pygame.draw.line(s, GRAY, (0, GROUND_Y + 1), (WIDTH, GROUND_Y + 1), 2)
        for ob in self.obstacles:
            if ob.kind.startswith("cactus"):     # "cactus" | "cactus_short" | "cactus_tall"
                pygame.draw.rect(s, BLACK, ob.rect)
            else:  # bird: a wide filled body + two wing wedges so it reads as a flier
                pygame.draw.rect(s, BLACK, ob.rect)
        dino = self._dino_rect()
        pygame.draw.rect(s, BLACK, dino)
        # a tiny "eye" so the dino is legible in the demo GIF (not needed for training)
        pygame.draw.rect(s, WHITE, (dino.right - 7, dino.top + 4, 3, 3))
        return s

    def render_rgb(self):
        """Return the current frame as an (H, W, 3) uint8 numpy array (full 600x150; for the GIF)."""
        import numpy as np
        surf = self.render_surface()
        # surfarray is (W, H, 3); transpose to (H, W, 3)
        arr = pygame.surfarray.array3d(surf)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render_gray84(self):
        """Fast observation path: (84, 84) uint8 grayscale, all in pygame/numpy (no PIL).

        Downscale in C via ``pygame.transform.smoothscale`` (anti-aliased so thin obstacles
        survive the shrink), then luminance-average the channels. ~5-10x faster per step than
        the PIL round-trip, which matters because this runs on every env step during training.
        """
        import numpy as np
        surf = self.render_surface()
        if OBS_CROP is not None:                                 # zoom into the action region
            x0, y0, x1, y1 = OBS_CROP
            surf = surf.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        small = pygame.transform.smoothscale(surf, (84, 84))     # C-fast anti-aliased downscale
        arr = pygame.surfarray.array3d(small)                    # (W=84, H=84, 3), indexed [x, y]
        gray = arr.mean(axis=2).T                                # -> (H, 84, W 84) = [y, x]
        return gray.astype(np.uint8)
