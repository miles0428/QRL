"""Quantum Spin CartPole environment suite"""
import gymnasium as gym
from games.spin_cartpole.env import QuantumSpinCartPoleEnv

# Dynamic gymnasium registration (YC entity decoupling principle)
# qrl only depends on the standard gymnasium.Env interface, not this module
gym.register(
    id="QuantumSpinCartPole-v0",
    entry_point="games.spin_cartpole.env:QuantumSpinCartPoleEnv",
    max_episode_steps=500,
)

__all__ = ["QuantumSpinCartPoleEnv"]
