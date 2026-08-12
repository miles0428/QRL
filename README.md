# QDQN on CartPole-v1 — Variational Quantum Circuits for Deep RL

A **quantum deep Q-network (QDQN)** that solves `CartPole-v1`, where the Q-function
approximator is a **variational quantum circuit (VQC)** instead of a neural network.
This is the *Fundamental* deliverable of "Variational Quantum Circuits for Deep
Reinforcement Learning in Classic Control".

> Status: **Checkpoint 1 (scaffold + environment).** The trainer, VQC, MLP baseline,
> and figures are implemented at later checkpoints — see [Build order](#build-order).
> Files not yet implemented raise a clear `NotImplementedError` naming their checkpoint.

## Design principles

1. **The trainer is model-agnostic.** `src/trainer.py` never imports qiskit. It only
   sees a `models/base.QFunction` (`forward(states) -> [B, n_actions]` +
   `param_groups()`). Swapping the ansatz is a new `QFunction` subclass plus one line
   in `src/models/__init__.py` — the trainer does not change. If the trainer knew the
   model were quantum, the design would be wrong.
2. **The classical baseline is parameter-count matched.** Comparing a 17k-param MLP to
   a ~46-param VQC and claiming quantum parameter-efficiency is not a result. The MLP
   uses a hidden width chosen to land within ±20% of the VQC's trainable count.
3. **Reproducible.** Same seed → byte-identical results CSV. Resolved dependency
   versions are printed at the top of every run log.

### The two failure modes this architecture is built around

- **Output scaling (Failure Mode 1).** `⟨Z⟩ ∈ [-1, 1]`, but true CartPole Q-values at
  γ=0.99 reach ~100. A **trainable output scaling** `w` (an `nn.Parameter`, multiplied
  onto the expectation values) is mandatory; without it training flatlines at ~10
  reward. Figure 3 (`w` climbing from 1 toward the tens) is the direct visual evidence.
- **Scalings live in PyTorch, not the circuit (Failure Mode 2).** `λ_i · x_i` is a
  product of an input parameter and a weight parameter; parameter-shift requires
  parameters to enter gates linearly, so putting the scaling inside the circuit
  **silently breaks gradients**. Both the input scaling `λ` and output scaling `w` are
  `nn.Parameter`s outside `TorchConnector`:

  ```
  s ─arctan─► λ ⊙ s ─► TorchConnector(EstimatorQNN) ─► [⟨Z₀⟩,⟨Z₁⟩] ─► ⊙ w ─► Q(s,·)
             ↑ nn.Parameter                                             ↑ nn.Parameter
  ```

### Circuit (see `configs/qdqn.yaml`)

- 4 qubits (one per observation dimension), `n_layers` (default 5) variational layers.
- Each layer: `RY(x[q])` encoding on every qubit; trainable `RY(θ), RZ(θ)` on every
  qubit; CNOT ring `q → (q+1) % 4`.
- `reuploading: false` (default) encodes once before layer 0; `reuploading: true`
  re-encodes every layer (advanced ablation — a config flag, not a rewrite).
- Observables `Z⊗I⊗I⊗I` and `I⊗Z⊗I⊗I` → 2 expectation values in `[-1, 1]`.
- **Gradient:** adjoint `ReverseEstimatorGradient` (exact, simulator-only, linear in
  n_params) with a shot-free statevector estimator for training. `estimator=` and
  `gradient=` are independent — a statevector estimator alone does **not** reduce
  param-shift's `2·n_params` circuit evaluations; both must be set. `gradient:
  paramshift` stays available for the later finite-shot / hardware path. Aer is used
  only for the finite-shot evaluation hook, never for training.

### Normalization

Cart position ÷ 2.4, pole angle ÷ 0.2095 (their termination bounds). Both velocities
are unbounded → squashed with `arctan` (never raw division). Normalization happens
inside `forward()`; the replay buffer stores **raw** observations.

### Three parameter groups, three learning rates

Owned by the model (`QFunction.param_groups()`), read from config:

| group | parameter | lr | why |
|---|---|---|---|
| input scaling | `λ` (4) | 1e-3 | trainable arctan input scaling |
| variational | VQC angles (40) | 1e-3 | the RY/RZ circuit params |
| output scaling | `w` (2) | **1e-1** | must train ~100× faster to reach Q~100 |

## Parameter-matching argument (baseline fairness)

- VQC trainable params: `5 layers × 4 qubits × 2 angles = 40` variational `+ 4` input
  scalings (`λ`) `+ 2` output scalings (`w`) = **46**.
- MLP with one hidden layer of width `h` (`4 → h → 2`, with biases):
  `params = (4h + h) + (2h + 2) = 7h + 2`. `h = 6 → 44` params, within ±20% of 46.
- The exact MLP count is computed and asserted at construction and printed to the run
  log. Config: `configs/mlp_baseline.yaml` (`hidden_dims: [6]`).

## Stack

Python 3.11+, `torch` (training/autograd), `qiskit` + `qiskit-machine-learning`
(`EstimatorQNN` wrapped in `TorchConnector`), `qiskit-algorithms`
(`ReverseEstimatorGradient`), `qiskit-aer` (finite-shot eval hook only), `gymnasium`
(`CartPole-v1`, **not** the deprecated `gym`), `numpy`, `matplotlib`, `pandas`,
`imageio` (rollout GIF). Exact resolved versions are pinned in `requirements.txt`.

## Setup

```bash
# from the repo root (this folder)
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (Git Bash):    source .venv/Scripts/activate
# macOS/Linux:           source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.versions          # print the resolved environment
```

## Reproduce

```bash
python scripts/train.py       --config configs/qdqn.yaml --seed 0     # single run
python scripts/sweep_seeds.py --config configs/qdqn.yaml --seeds 0 1 2 3 4
python scripts/sweep_seeds.py --config configs/mlp_baseline.yaml --seeds 0 1 2 3 4
python scripts/make_figures.py                                        # all figures -> figures/
```
_(Entrypoints are wired at the checkpoints noted in each file.)_

## Repo layout

```
QRL/  (repo root — the qdqn-cartpole project)
├── README.md
├── requirements.txt
├── configs/{qdqn,mlp_baseline}.yaml
├── src/
│   ├── models/{base,vqc,mlp}.py   # base.py = QFunction ABC; __init__.py = factory
│   ├── replay.py  trainer.py  evaluate.py  seeds.py  plots.py  versions.py
├── scripts/{train,sweep_seeds,make_figures}.py
├── tests/test_vqc.py
├── results/   # one CSV per (config, seed)
└── figures/
```

## Evaluation protocol

- **Solve criterion (used everywhere):** mean reward ≥ 475 over 100 consecutive episodes.
- Report **episodes-to-solve** and **env-steps-to-solve**, not just final reward.
- ≥ 5 seeds (10 if runtime allows); aggregate with **median + IQR**, not mean ± std.
- Final greedy evaluation: 100 episodes at ε = 0.

## References

### Papers

| Topic | Reference |
|---|---|
| Original VQC-for-deep-RL | Chen, S. Y.-C. et al. (2020) — https://arxiv.org/abs/1907.00397 |
| The QDQN recipe followed here (output/input scaling, ablations) | Skolik, A., Jerbi, S., Dunjko, V. (2022), *Quantum agents in the gym*, Quantum 6, 720 — https://arxiv.org/abs/2103.15084 |
| Data re-uploading (`reuploading` flag) | Pérez-Salinas, A. et al. (2020), Quantum 4, 226 — https://doi.org/10.22331/q-2020-02-06-226 |
| Quantum policies / actor-critic direction | Jerbi, S. et al. (2021), NeurIPS 34 — https://arxiv.org/abs/2103.05577 |
| Finite-shot / hardware-noise behaviour (later shot sweep) | Skolik, A. et al. (2023), *Robustness of QRL under hardware errors*, EPJ Quantum Technology 10:8 — https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-023-00166-1 |
| Environment API | Towers, M. et al. (2024), *Gymnasium* — https://arxiv.org/abs/2407.17032 |

### Code references

| What to take from it | Link |
|---|---|
| `TorchConnector` + `EstimatorQNN` usage — primary API reference | https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html |
| PyTorch RL loop structure on CartPole (episode loop, optimizer pattern) | https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py |
| VQC-RL reference implementation in TF (architecture only) | https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning |

> **Note on the Actor-Critic-pytorch reference:** useful shape for the torch loop, but
> it is A2C not DQN, targets `CartPole-v0`, and uses the **old gym API**. Gymnasium
> returns `(obs, info)` from `reset()` and a 5-tuple
> `(obs, reward, terminated, truncated, info)` from `step()`. Take the structure, not
> the API calls; and never call `env.render()` inside the training loop.

## Build order

1. ✅ Scaffold, `requirements.txt`, venv, print versions, confirm imports.
2. `models/vqc.py` + two tests (smoke: shapes + all-three-group grads non-zero;
   agreement: reverse vs param-shift to ~1e-6). **Must pass before proceeding.**
3. `replay.py` + `trainer.py` + `models/mlp.py`; verify the trainer solves CartPole
   with the MLP first (isolates trainer bugs from quantum bugs).
4. Wire the VQC into the trainer; single seed; watch `w` grow.
5. `sweep_seeds.py`, full runs, `plots.py`, all figures.
6. Finalize `README.md`.

## License

MIT
