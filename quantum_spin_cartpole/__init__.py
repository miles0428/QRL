"""
Quantum Spin-CartPole: A Gymnasium environment for SU(2) quantum spin control.
"""
from gymnasium.envs.registration import register, registry

from .env import QuantumSpinCartPoleEnv
from .trajectory import SpinTrajectory

# Register at import time so gym.make("QuantumSpinCartPole-v0") works. Guarded
# because a re-import (or a re-registration by a test) would otherwise raise.
# max_episode_steps mirrors the env's own truncation limit: env.py:297 truncates
# on self.t_max, whose default is constants.T_MAX = 500.
if "QuantumSpinCartPole-v0" not in registry:
    register(
        id="QuantumSpinCartPole-v0",
        entry_point="quantum_spin_cartpole.env:QuantumSpinCartPoleEnv",
        max_episode_steps=500,
    )

__all__ = ["QuantumSpinCartPoleEnv", "SpinTrajectory"]
__version__ = "0.1.0"
