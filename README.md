# Quantum Spin-CartPole — QRL Branch

A Gymnasium v0.29+ environment for quantum spin-1/2 control via reinforcement learning, integrated into the QRL framework.

## Overview

This branch (`quantum-game`) contains the Quantum Spin-CartPole environment, implementing an SU(2) quantum spin-1/2 system as a standard Gymnasium RL environment. The spin evolves under a laboratory-frame Hamiltonian with:

- **Static Zeeman term**: H₀ = (ħ ω₀/2) σ_z
- **OU dephasing noise**: H_noise(t) = (ħ δΔ(t)/2) σ_z
- **x/y coherent drives**: H_drive_{x,y}(t) = (ħ Ω_R u_{x,y}/2) cos(ω₀ t) σ_{x,y}

The RL agent controls the spin by selecting among 5 actions (+X, -X, +Y, -Y, IDLE).

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from quantum_spin_cartpole import QuantumSpinCartPoleEnv

env = gym.make("QuantumSpinCartPole-v0")
obs, info = env.reset(seed=42)

for t in range(1000):
    action = env.action_space.sample()  # or your policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Project Structure

```
QRL/quantum_spin_cartpole/
├── __init__.py
├── constants.py   # Physical hyperparameters
├── noise.py        # Ornstein-Uhlenbeck sampler
├── hamiltonian.py  # Hamiltonian factory
└── env.py          # Main Gymnasium environment
QRL/tests/
├── test_hamiltonian.py
├── test_noise.py
└── test_env.py
```

## Physics

- **Observation**: ⟨S_x⟩, ⟨S_z⟩ (Bloch x/y components), shape (2,)
- **Action**: Discrete(5) — +X, -X, +Y, -Y, IDLE
- **Reward**: Fidelity to |0⟩ (spin-up) minus control penalty
- **Termination**: Spin entering southern hemisphere (⟨S_z⟩ < 0)

## Tests

```bash
pytest
```

## Requirements

- Python 3.10+
- qutip
- gymnasium 0.29.1
- numpy
