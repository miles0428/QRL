"""DINO game environment suite"""
import gymnasium as gym
from games.dino.env import DinoImageEnv

# Dynamic gymnasium registration (YC entity decoupling principle)
# qrl only depends on the standard gymnasium.Env interface, not this module
gym.register(
    id="DinoRun-v0",
    entry_point="games.dino.env:make_dino_env",
    max_episode_steps=2000,
)

__all__ = ["DinoImageEnv"]
