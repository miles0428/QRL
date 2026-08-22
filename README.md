# QRL — Quantum Reinforcement Learning Framework

## Overview

QRL is a reinforcement learning framework that bridges classical deep learning (CNN, MLP) with quantum Variational Quantum Circuits (VQC). It supports three training algorithms — DQN, A2C, and PG — and provides a flexible, configuration-driven workflow for experiments.

## Features

- **Hybrid Classical–Quantum Models**: CNN encoder + VQC for image-based tasks (e.g., Dino Run); pure VQC for low-dimensional tasks (e.g., QuantumSpinCartPole).
- **Quantum Backend**: Torch-based statevector simulation via [Qiskit](https://qiskit.org/) + [QuTiP](https://qutip.org/).
- **Trainers**: DQN (with Double-DQN support), A2C, and Policy Gradient (REINFORCE).
- **Environments**: Pre-built wrappers for Dino Run and QuantumSpinCartPole, plus compatibility with [Gymnasium](https://gymnasium.farama.org/) classic tasks.
- **Configuration-Driven**: All experiment hyperparameters are declared in YAML config files — no code changes needed to switch models or trainers.

## Architecture

```
                           configs/*.yaml
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  scripts/run_experiment.py │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │        games/            │
                    │  DinoRun-v0              │
                    │  QuantumSpinCartPole-v0   │
                    │  CartPole-v1 (Gymnasium)  │
                    └────────────┬────────────┘
                                 │ state encoding
                    ┌────────────▼────────────┐
                    │   qrl/encoders/          │
                    │  (Angle, PauliZ, CNN…)   │
                    └────────────┬────────────┘
                                 │ encoded observations
          ┌──────────────────────┴───────────────────────┐
          │              qrl/models/                     │
          │  ┌─────────────────┐  ┌───────────────────┐  │
          │  │  Classical Head  │  │  Quantum Backend  │  │
          │  │  CNN / MLP       │  │  TorchStatevectorQNN  │  │
          │  │  (qrl/backends/) │  │  (qiskit + qutip)  │  │
          │  └─────────────────┘  └───────────────────┘  │
          └──────────────────────┬───────────────────────┘
                                 │ Q-values / action logits
                    ┌────────────▼────────────┐
                    │     qrl/trainers/       │
                    │   dqn  │  a2c  │  pg    │
                    └─────────────────────────┘
                                 │
                                 ▼
                              result/
```

## Supported Trainer × Model Combinations

| Config File              | Environment              | Encoder | Model        | Trainer |
|--------------------------|--------------------------|---------|--------------|---------|
| `dino_cnn_dqn.yaml`      | DinoRun-v0               | CNN     | CNN + VQC    | DQN     |
| `dino_cnn_a2c.yaml`      | DinoRun-v0               | CNN     | CNN + VQC    | A2C     |
| `dino_cnn_pg.yaml`       | DinoRun-v0               | CNN     | CNN + VQC    | PG      |
| `qdqn_quantum.yaml`      | QuantumSpinCartPole-v0   | sxy     | VQC          | DQN     |
| `qa2c_quantum.yaml`      | QuantumSpinCartPole-v0   | sxy     | VQC          | A2C     |
| `qpg_quantum.yaml`       | QuantumSpinCartPole-v0   | sxy     | VQC          | PG      |
| `mlp_a2c.yaml`          | CartPole-v1              | MLP     | MLP          | A2C     |
| `mlp_pg.yaml`           | CartPole-v1              | MLP     | MLP          | PG      |

## Installation

```bash
pip install -r requirements.txt
```

> **Note**: `qutip-qip==0.4.2` is pinned for compatibility with the quantum backend. Using a different version may cause import errors.

## Usage

```bash
# Run any experiment from configs/
PYTHONPATH=. python scripts/run_experiment.py --config configs/<experiment>.yaml

# Examples
PYTHONPATH=. python scripts/run_experiment.py --config configs/dino_cnn_dqn.yaml
PYTHONPATH=. python scripts/run_experiment.py --config configs/qdqn_quantum.yaml
PYTHONPATH=. python scripts/run_experiment.py --config configs/mlp_a2c.yaml
```

Results are written to the `result/` directory.

## Project Structure

```
.
├── configs/           # Experiment YAML files
├── experiments/       # Experiment scripts
├── games/             # Game environments (DinoRun, QuantumSpinCartPole)
├── qrl/
│   ├── backends/      # Quantum backend (TorchStatevectorQNN)
│   ├── encoders/      # State encoders (Angle, PauliZ, CNN, …)
│   ├── models/         # Model definitions (VQC, CNN+VQC, MLP)
│   └── trainers/      # Trainer implementations (dqn, a2c, pg)
├── result/            # Training output
├── scripts/           # Entry point (run_experiment.py)
└── tests/             # Unit tests
```

## Notes & Tips

- **Frame-skip**: Dino Run uses a frame-skip mechanism to speed up training; ensure your config enables it for reasonable training speed.
- **Quantum simulation speed**: Statevector simulation scales exponentially with qubit count. The 9-qubit `QuantumSpinCartPole-v0` is lightweight; larger circuits will be significantly slower.
- **PYTHONPATH**: The root of the repository must be on `PYTHONPATH` so that `qrl.*` imports resolve correctly.
- **GPU**: `torch>=2.0` with CUDA is recommended for CNN-based experiments; the quantum backend runs on CPU.
