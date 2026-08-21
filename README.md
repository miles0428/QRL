<title>Quantum Spin-CartPole + QDQN</title>

# Quantum Spin-CartPole — environment, baselines, and a quantum DQN

A Gymnasium environment for SU(2) spin-1/2 control, a set of controllers to
measure it against, and the QDQN (variational-circuit Q-function) from the
`qdqn-cartpole` branch wired onto it.

The agent drives a spin with four coherent pulses (+X, −X, +Y, −Y) or idles,
against triaxial Ornstein–Uhlenbeck noise, and is scored on how long it keeps
the spin in the northern hemisphere and how close to the pole it holds it.

## Install

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install .                                       # or: pip install -e ".[dev]"
pytest -q                                           # 59 tests
```

`pip install .` now installs the env package proper, so `import
quantum_spin_cartpole` resolves from any working directory. The `scripts/`
runners still expect the repo root on the path — for those, keep using
`export PYTHONPATH=$PWD` and install the agent-side extras
(`torch`, `qiskit`) yourself; they are not dependencies of the env package.

Using it as a registered Gymnasium env:

```python
import quantum_spin_cartpole          # required -- this import performs the registration
import gymnasium as gym

env = gym.make("QuantumSpinCartPole-v0")   # -> (6,) Discrete(5)
```

The explicit import is not optional. Gymnasium dropped entry-point-based env
discovery in 0.29, the version this package pins, so `register()` runs from
`quantum_spin_cartpole/__init__.py` at import time and `gym.make` cannot see the
id before the package has been imported once.

Two setup traps worth knowing:

- **qutip 5 *and* qutip-qip are both required.** `processor.py` calls
  `ModelProcessor(num_qubits=1)`; on qutip 4.7 `qutip.qip` resolves to the
  in-tree legacy class whose signature is `N=`, and `env._expectation` calls
  `.real` on a product that qutip 4 returns as a `Qobj`. `qutip-qip` is now a
  declared dependency, so a plain `pip install .` gets it.
- **The `gymnasium==0.29.1` pin is exact on purpose.** The rest of the repo
  pins `gymnasium==1.3.0`; the 0.29 -> 1.x transition changed the env API and
  the 59 tests here were written against 0.29.1.

## The game

| | |
|---|---|
| Observation | `[⟨σx⟩, ⟨σy⟩, ⟨σz⟩, Δ⟨σx⟩, Δ⟨σy⟩, Δ⟨σz⟩]` |
| Actions | `Discrete(5)` — +X, −X, +Y, −Y, IDLE |
| Reward | fidelity − `c_ctrl` for any non-idle action, −5 on termination |
| Termination | `⟨σz⟩ < 0` (southern hemisphere) |
| Truncation | 500 steps |

**Difficulty is set by one number**: the ratio of the noise field's stationary
standard deviation to the Rabi frequency, `NOISE_RABI_RATIO` in
`constants.py`. Measured with a one-step-lookahead controller, 20 episodes per
point:

| noise/Rabi | greedy return | survival | idle |
|---|---|---|---|
| 0.03 | 491 | 100% | 86 |
| 0.10 | 491 | 100% | 23 |
| 0.25 | 477 | 95% | 3 |
| **0.35 (default)** | **400** | **68%** | **2** |
| 0.50 | 138 | 0% | −1 |
| 0.75 | 31 | 0% | −1 |
| ≥ 2.0 | ≈ idle | 0% | — |

The cliff between 0.25 and 0.5 is sharp. The default is **0.35**: below 0.25 the
task is saturated and a learned policy cannot show any advantage; above ~0.5 no
policy has the control authority to matter.

> Earlier revisions shipped `SIGMA_OU = 2π × 1.0`, which is 1000× too large —
> `OMEGA_R` encodes 100 MHz as `2π × 100.0e-3` (GHz units, as rad/ns requires),
> so 1 MHz is `2π × 1.0e-3`. That put the noise at 31.6× the Rabi frequency,
> where greedy, random and always-idle all score ≈ −2 over 3.6 steps and are
> statistically indistinguishable. `constants.py` now derives `SIGMA_OU` from
> the ratio instead.

### Noise structure

The noise is **not** white. `noise.py` implements
`δ(t+1) = δ(t) − θ·δ(t)·Δt + σ·√Δt·η`, which at θ=0.05, Δt=1 is exactly an
**AR(1) process with φ = 0.95**. Measured autocorrelation over 200k steps lands
on `0.95^lag` to three decimals (lag 1: 0.950, 10: 0.599, 20: 0.357, 40: 0.128);
the three axes are independent. With a 20-step correlation time against a
1-step decision period, the noise field is a slowly drifting unknown constant
rather than per-step jitter — estimable online, though not memorizable. None of
the greedy controllers exploit this; their lookahead assumes zero noise.

## Observation modes

`--obs {family}[_prevact|_hist]`, families:

| family | keeps | note |
|---|---|---|
| `full` | all 6 | Markov in the spin state |
| `masked` | `sz, Δsz` | azimuth removed — cannot tell +X from +Y |
| `sxy` | `sx, sy, Δsx, Δsy` | see below |
| `sx` | `sx, Δsx` | genuinely hides the reward |

`_prevact` appends the one-hot of the action that produced the observation
(+5 dims). `_hist` stacks 8 such frames.

**`sxy` is not an information mask.** While an episode is live, `reset` samples
the upper hemisphere and `sz<0` terminates, so `sz = +√(1−sx²−sy²)` exactly
(verified to 4e-14 on every observation an agent acts on). Nothing is hidden in
the Shannon sense; the agent must merely *learn* the nonlinear relation rather
than being handed it. That penalises learners and is free for anything that
already knows the physics. To hide `sz` for real, either use `sx`, or set
`terminate_on_violation=False` so the sign is genuinely ambiguous.

## Controllers

```bash
python scripts/greedy_baseline.py --episodes 50        # physics-aware lookahead
python scripts/greedy_learned.py --collect-terminate   # same, but model fit from data
python scripts/noise_sweep.py --episodes 20            # difficulty curve + figure
```

- **`greedy_baseline.py`** — one-step lookahead through the exact propagator.
  Handed `Ω_R`, `dt`, the Hamiltonian form and the reward formula. Not
  clairvoyant: it never sees the OU realization. Its analytic propagator is
  checked against the env's own qutip `ModelProcessor` at startup (5.5e-11).
- **`greedy_learned.py`** — the same lookahead with **no physics knowledge**:
  per-action dynamics by least squares and a reward MLP, both fit from ~20k
  random transitions. Use `--collect-terminate` to match the fitting
  distribution to deployment; without it, `sz` roams both hemispheres, the
  projected transition becomes two-valued, and no model class can fit it.
- **`greedy_masked.py`** — masked-observation greedy with a particle filter over
  the hidden azimuth.

## Training

```bash
python scripts/train_dqn.py --obs full_prevact --seed 0     # classical DQN
python scripts/train_ppo.py --obs full_prevact --seed 0     # PPO
python scripts/train_dqn.py --model vqc --obs sxy_prevact   # QDQN
```

`--model vqc` swaps in the variational-circuit Q-function from `qdqn/`, copied
verbatim from the `qdqn-cartpole` branch (`src/models/vqc.py`,
`src/models/torch_statevector.py`); only `normalize_observation` and the imports
differ. It runs through the same loop, replay, evaluation protocol and seed
bands as the classical runs, so only the Q-function changes. Trainer settings
follow `configs/qdqn.yaml`: batch 16, a gradient step every 10 env steps, target
sync every 3 gradient steps, and three learning rates (`w` 0.1, circuit weights
and `lam` 0.001).

On `sxy_prevact`: 9 qubits (one per observation dimension), 5 layers,
re-uploading and per-layer encoding on, five ZZ observables one per action —
140 trainable parameters (90 circuit, 45 `lam`, 5 `w`).

> **`train_every=10` means equal env-step budgets are not equal learning
> budgets.** QDQN takes 15k gradient steps per 150k env steps against the
> classical runs' 150k. Account for this before reading any quantum-vs-classical
> comparison.

## Evaluation

Every policy here is scored with ε=0 on the **same fixed seeds (1000–1049)**,
each pinning both the initial state and the OU realization, so episodes are
matched pairs:

```bash
python scripts/paired_compare.py --noise-rabi 0.35
```

This matters. Per-episode standard deviation is comparable to the mean, so
unpaired error bars overlap almost completely even where one policy is
consistently better — at 0.35, PPO trails greedy by only 5% on the mean yet
greedy wins 43 of 50 matched episodes.

Checkpoint selection uses a **disjoint** seed band (`--select-seed0`, default
2000). Selecting the best checkpoint on the seeds you then report inflates the
result, and worse for longer runs: measured +10.7 at 15 evaluations and +21.8 at
40, enough to manufacture an improvement that is not there.

## Results at noise/Rabi = 0.35

Unmasked (50 eval episodes, 3 seeds), paired against greedy:

| policy | return | diff | wins | Wilcoxon p |
|---|---|---|---|---|
| greedy (physics) | 399.6 | — | — | — |
| greedy (learned model) | 388.9 | −10.6 | 25/50 | 0.66 |
| PPO `full_prevact` | 379.5 | −20.1 | 7/50 | 0.0001 |
| DQN `full_prevact` | 378.6 | −21.0 | 7/50 | 0.0001 |
| PPO `full` | 374.9 | −24.7 | 5/50 | 0.0000 |
| DQN `full` | 359.8 | −39.8 | 4/50 | 0.0000 |
| DQN `full_hist` | 320.0 | −79.6 | 5/50 | 0.0000 |

On the `sxy` game (all of these see no `sz`; the two rows above that read the
full Bloch vector are **not** comparable):

| policy | return | survival |
|---|---|---|
| DQN `sxy` | 378.0 ± 12.6 | 62% |
| PPO `sxy_prevact` | 377.4 ± 13.9 | 65% |
| PPO `sxy` | 375.7 ± 9.4 | 62% |
| greedy learned `sxy` | 364.9 ± 13.6 | 57% |
| DQN `sxy_prevact` | 358.0 ± 36.8 | 57% |

Three things this measures that are worth keeping in mind:

1. **The physics knowledge is worth almost nothing.** Stripping it costs greedy
   under 3%, and 20k random transitions buy it back — one seventh of the DQN
   training budget.
2. **Prev-action beats frame stacking.** `full_prevact` (11 dims) roughly halves
   the gap to greedy and reaches its score ~4× sooner than plain `full`, while
   `full_hist` (88 dims) is the worst config tested. The spin state is already
   Markov; stacking old states only dilutes the input and, with a replay buffer,
   fills it with action-histories from stale policies.
3. **Return is dominated by survival, not control precision.** Greedy's per-step
   reward is 0.9455 and a matched DQN seed's is 0.9433 — nearly identical —
   while the returns differ by 19 points purely through when episodes end.

## Layout

```
quantum_spin_cartpole/   the environment (constants, noise, hamiltonian, processor, env)
qdqn/                    VQC Q-function + torch statevector backend, from qdqn-cartpole
scripts/                 controllers, training, sweeps, paired statistics
tests/                   59 tests
results/  figures/  logs/
```
