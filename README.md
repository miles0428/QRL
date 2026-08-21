# QDQN on CartPole-v1 (Fundamental Track)

Quantum Deep Q-Network: the Q-function approximator is a variational quantum circuit
(VQC) instead of a neural network, trained with standard DQN machinery (experience
replay, target network, epsilon-greedy exploration) on Gymnasium's `CartPole-v1`.
Includes a parameter-count-matched classical MLP baseline for comparison.

Challenge: *"Variational Quantum Circuits for Deep Reinforcement Learning in Classic
Control"* -- Fundamental deliverable.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .           # or: pip install -r requirements.txt
```

`pip install .` installs the package as `qdqn_cartpole` and puts seven console
scripts on your PATH. **Run them from the repo root.** Every script addresses
`results/` and `figures/` relative to the current working directory, so invoking
one from elsewhere writes into a *new* `results/` there and `qdqn-summarize`
then reports no runs against a repo full of data. Each command prints its
resolved paths to stderr on startup, so a wrong cwd shows up in the first line
of the log rather than after the run.

| console script | module |
|---|---|
| `qdqn-train` | `qdqn_cartpole.cli.train` |
| `qdqn-sweep` | `qdqn_cartpole.cli.sweep_seeds` |
| `qdqn-figures` | `qdqn_cartpole.cli.make_figures` |
| `qdqn-summarize` | `qdqn_cartpole.cli.summarize` |
| `qdqn-benchmark` | `qdqn_cartpole.cli.benchmark_backends` |
| `qdqn-benchmark-report` | `qdqn_cartpole.cli.benchmark_report` |
| `qdqn-verify-backend` | `qdqn_cartpole.cli.verify_backend_equivalence` |

`tfq/` is deliberately *not* part of the package: it pins an incompatible
TensorFlow Quantum stack and is installed separately via `tfq/install_tfq.sh`.

Developed and tested against the versions pinned in `requirements.txt` (Python 3.11.15,
`qiskit==2.5.1`, `qiskit-aer==0.17.2`, `qiskit-machine-learning==0.9.0`, `torch==2.13.0`,
`gymnasium==1.3.0`). Every run of `qdqn_cartpole/cli/train.py` and `qdqn_cartpole/cli/sweep_seeds.py` prints
the resolved versions of all dependencies at the top of its log.

## Repo layout

```
qdqn-cartpole/
├── configs/
│   ├── qdqn.yaml           # quantum agent hyperparameters
│   └── mlp_baseline.yaml   # classical control hyperparameters
├── qdqn_cartpole/
│   ├── models/
│   │   ├── base.py         # QFunction ABC + shared observation normalization
│   │   ├── vqc.py          # VQC Q-function (TorchConnector)
│   │   └── mlp.py          # classical baseline, param-count matched
│   ├── replay.py           # experience replay buffer
│   ├── trainer.py          # model-agnostic DQN loop -- does NOT import qiskit
│   ├── evaluate.py         # greedy rollouts + solve criterion
│   ├── seeds.py            # reproducibility
│   ├── plots.py            # all 8 figure-generation functions
│   └── cli/                # the seven console-script entry points
│       ├── train.py         # train one (config, seed)
│       ├── sweep_seeds.py   # train across multiple seeds, with a performance gate
│       └── make_figures.py  # generate all figures from results/ + checkpoints
├── tfq/                     # TensorFlow Quantum arm -- separate env, not packaged
├── results/                 # results/{config}_{seed}.csv, results/{config}_{seed}_final.pt
└── figures/                 # 8 required figures, PNG (150dpi) + PDF
```

**Hard architectural rule**: `qdqn_cartpole/trainer.py` never imports `qiskit`. It only touches
`model` through the generic `nn.Module` interface (`forward()`, `.parameters()`,
`.state_dict()`) plus optional `getattr(model, "w"/"lam", None)` lookups for CSV logging.
Swapping the model for a different ansatz means writing a new `qdqn_cartpole/models/*.py` and
pointing a config at it -- `trainer.py` and `evaluate.py` don't change.

## Model spec

**Circuit** (`qdqn_cartpole/models/vqc.py`): 4 qubits, `n_layers` variational layers (default 5).
Each layer: `RY(x[q])` encoding on every qubit (only before layer 0, unless
`reuploading: true` in the config, in which case it repeats every layer -- data
re-uploading, Pérez-Salinas et al. 2020), then trainable `RY(theta)`, `RZ(theta)` on
every qubit, then a CNOT ring `q -> (q+1) % 4`. Observables `Z⊗I⊗I⊗I` and `I⊗Z⊗I⊗I` give
two expectation values in `[-1, 1]`, one per CartPole action.

**Two failure modes** this architecture exists to avoid:

1. **Output scaling.** `⟨Z⟩ ∈ [-1, 1]`, but true CartPole Q-values at `γ=0.99` reach
   ~100. Without a trainable output scaling, the model cannot represent the value
   function and training flatlines at ~reward 10. Fixed with
   `self.w = nn.Parameter(torch.ones(n_actions))`, returning `raw_out * self.w`.
2. **Don't scale inputs inside the circuit.** `lam_i * x_i` is a product of an input
   value and a trainable weight; parameter-shift requires parameters to enter gates
   linearly, so computing that product *inside* the circuit silently breaks gradients.
   Both `lam` (input scaling) and `w` (output scaling) live in the PyTorch module,
   strictly outside `TorchConnector`.

   **A third failure mode we hit and fixed while building this** (not called out in the
   original brief, but caught by the mandated smoke test below): `EstimatorQNN` defaults
   to `input_gradients=False`. Left at that default, `TorchConnector`'s backward pass
   never computes a gradient with respect to its *input* tensor -- only its weights --
   so `lam`'s gradient (which only reaches `lam` by backpropagating through the QNN's
   input) is silently `None`/zero even though the forward pass looks completely normal.
   Fixed by passing `input_gradients=True` when constructing `EstimatorQNN` in
   `qdqn_cartpole/models/vqc.py`. This is exactly the class of bug the brief's smoke test is
   designed to catch, and it did.

**Normalization** (`qdqn_cartpole/models/base.py::normalize_observation`, called inside every
model's `forward()` -- the replay buffer stores raw observations): cart position and
pole angle are divided by their termination bounds (2.4 and 0.2095 rad); both
velocities are formally unbounded, so they go through `arctan` instead.

**Classical baseline** (`qdqn_cartpole/models/mlp.py`): a 4 -> hidden -> 2 MLP. The VQC's default
config (4 qubits, `n_layers=5`) has 40 circuit weights + 4 `lam` + 2 `w` = **46**
trainable parameters. An MLP with bias has `7*hidden + 2` parameters; `hidden=6` gives
**44** (4.3% off, well within the required ±20%). A 2×128 MLP would have ~17k parameters
-- comparing that against the VQC's 46 and calling it "quantum parameter-efficiency"
would not be a real result, which is why both models here are deliberately tiny.

## Trainer spec (`qdqn_cartpole/trainer.py`)

- Replay buffer, capacity 10,000, `(s, a, r, s', done)`, uniform sampling, batch size 16
  (small on purpose -- every forward pass is a circuit simulation for the VQC).
- Target network, **hard update every 30 gradient steps** (not episodes).
- Loss: `MSE(Q(s,a), r + γ·(1-done)·max_a' Q_target(s',a'))`, `γ=0.99`.
- Epsilon-greedy: `1.0 -> 0.01`, **linear decay over the first 20,000 environment steps**
  (not episode-indexed decay).
- Train for 2000 episodes or until solved, whichever comes first.
- **Solve criterion, used everywhere**: mean reward ≥ 475 over 100 consecutive episodes.
- Every run writes `results/{config}_{seed}.csv` with, per episode: episode index, total
  env steps, episode reward, 100-episode moving average, epsilon, mean TD loss,
  wall-clock seconds, cumulative gradient steps, and the current values of `w` and `lam`
  (empty for models without those attributes, e.g. the MLP).
- **Reproducibility**: `qdqn_cartpole/seeds.py::set_seed()` seeds python/numpy/torch;
  `env.reset(seed=seed + episode)` seeds the environment. Verified: given the same seed
  and a model whose construction was seeded the same way (see `qdqn_cartpole/cli/train.py`'s
  ordering -- `set_seed()` runs *before* the model is constructed), every CSV column is
  byte-identical across two independent runs except `wall_clock_s`, which is inherently
  non-deterministic wall-clock timing and expected to differ.

Optimizer construction (three parameter groups, three learning rates for the VQC) lives
in `qdqn_cartpole/cli/train.py`, not `trainer.py` -- that's the boundary that keeps the hard
architectural rule (`trainer.py` never imports `qiskit`) satisfiable: the optimizer is
built by whoever knows what kind of model this is, then handed to `trainer.train()`
already configured.

```python
optimizer = torch.optim.Adam([
    {"params": [model.lam],          "lr": 1e-3},  # input scaling
    {"params": model.vqc.parameters(), "lr": 1e-3},  # variational circuit
    {"params": [model.w],            "lr": 1e-1},  # output scaling -- much larger
])
```

## Running

```bash
# Train one (config, seed):
qdqn-train --config configs/mlp_baseline.yaml --seed 0
qdqn-train --config configs/qdqn.yaml --seed 0

# Sweep multiple seeds (5 minimum, 10 if runtime allows):
qdqn-sweep --config configs/mlp_baseline.yaml --seeds 0 1 2 3 4

# Final greedy evaluation (epsilon=0, 100 episodes) of a trained model:
python -m qdqn_cartpole.evaluate  # see qdqn_cartpole/evaluate.py::evaluate() for programmatic use

# Generate all 8 figures from whatever results/ + checkpoints exist:
qdqn-figures
```

### Performance warning -- read this before running the full QDQN sweep

Training is dominated by circuit simulation, and gradients go through
parameter-shift (2 circuit evaluations per trainable weight -- 80 extra evaluations per
gradient step for the default 40-weight, 5-layer circuit, on top of the forward passes).
**`qdqn_cartpole/cli/sweep_seeds.py` times a real batch of gradient steps before launching anything**
and refuses to launch the full multi-seed sweep if the projected wall-clock exceeds ~6
hours, printing the numbers instead of running blind. Measured on this machine:

| | |
|---|---|
| seconds per gradient step (VQC, batch=16, `n_layers=5`, 40 weights) -- measured, `qdqn-sweep --config configs/qdqn.yaml --seeds 0 1 2 3 4` | 21.6s |
| projected wall-clock, 5 seeds × 2000 episodes, worst case | **~30,000 hours** (~3.4 years) |
| projected wall-clock, 5 seeds × 2000 episodes, optimistic heuristic (~150 env steps/episode avg) | **~9,000 hours** (~1 year) |

(An earlier isolated micro-benchmark outside the actual training loop measured ~14.5s/step; the
number above is from the real `sweep_seeds.py` timing probe -- an actual env-interaction loop,
not a synthetic batch -- and is the one that matters.)

**This is not runnable as configured, on this hardware, in hackathon time.** Both
estimates are printed by `qdqn-sweep --config configs/qdqn.yaml` itself
(along with the exact measured per-step cost from a real timing probe), and the script
stops there rather than launching the sweep -- per the project brief's explicit
instruction to stop and report rather than run blind. The classical MLP baseline has no
such problem (~0.01s/gradient step; a 5-seed, 300-episode sweep runs in well under an
hour) and its full pipeline (training, checkpointing, CSV logging, figures) has been run
for real as part of building this repo.

Options, in order of how much they change the science:
- **Reduce `n_layers`** in `configs/qdqn.yaml` (fewer weights -> fewer parameter-shift
  evaluations per step; roughly linear in `n_layers`).
- **Reduce `max_episodes`** for a demo-scale run rather than a publication-scale one.
- **Reduce `--seeds`** below 5 (weakens the statistical claim -- single-seed curves in
  this field are not evidence per the evaluation protocol, so this should be a last
  resort).
- **Investigate a faster gradient path** -- the measured cost is dominated by qiskit's
  default per-shift-circuit dispatch overhead, not the actual 4-qubit statevector
  simulation (which should take microseconds); a batched/vectorized gradient primitive
  would likely help far more than any of the above and hasn't been investigated yet.

`qdqn-sweep --force` overrides the stop once you've seen the projection and
decided how to proceed.

## Evaluation protocol

- Solve criterion: mean reward ≥ 475 over 100 consecutive episodes (used identically in
  training's early-stop check and in `qdqn_cartpole/evaluate.py`'s final greedy evaluation).
- Report **episodes-to-solve** and **env-steps-to-solve**, both logged in the training
  summary and derivable from any `results/*.csv`.
- Aggregate across seeds with **median and IQR**, not mean ± std (`qdqn_cartpole/plots.py`).
- Final evaluation: 100 greedy (`epsilon=0`) episodes via `qdqn_cartpole/evaluate.py::evaluate()`.

## Figures (`figures/`, PNG 150dpi + PDF, generated by `qdqn_cartpole/cli/make_figures.py`)

1. Learning curves -- reward vs. episode, median across seeds, IQR shaded, both models,
   horizontal line at 475.
2. Sample efficiency -- same, x-axis is environment steps (seed curves interpolated onto
   a common step grid before aggregating).
3. Scaling parameter evolution -- `w[0]`, `w[1]`, each `lam[i]` vs. training step. Direct
   visual evidence for Failure Mode 1: `w` should climb from 1 toward the tens.
4. Circuit diagram (`qc.draw('mpl')`).
5. Q-value landscape -- fix `x=x_dot=0`, sweep `theta` × `theta_dot` on a 100×100 grid,
   plot `Q(s, left) - Q(s, right)` with the decision boundary overlaid, untrained vs.
   trained side by side.
6. Loss and epsilon -- twin-axis diagnostic.
7. Rollout animation -- one greedy episode, GIF. Rendered with matplotlib's
   `PillowWriter` rather than `imageio` (Pillow is already a matplotlib dependency here;
   this avoids adding a new package for an identical result).
8. Parameter count vs. final performance -- scatter, one point per (model, seed). The
   headline comparison figure.

`make_figures.py` skips any figure whose inputs (result CSVs / `*_final.pt`
checkpoints) don't exist yet rather than failing outright, so it's safe to run against
partial results while a sweep is still in progress.

## `evaluate_finite_shot` hook

`qdqn_cartpole/models/vqc.py::evaluate_finite_shot(model, shots)` swaps a trained VQC's estimator
for a shot-based `qiskit_aer.primitives.EstimatorV2` (shots converted to the equivalent
`default_precision = 1/sqrt(shots)`, since that's the option this estimator actually
exposes) and returns a new model with the same trained weights, ready for finite-shot
evaluation. **Only the swap is implemented** -- the sweep across shot counts and
evaluation episodes is intentionally left as future work, per the build order. `qiskit-aer`
is reserved for this evaluation hook; training always uses the exact
`qiskit.primitives.StatevectorEstimator`, never a shot-based estimator.

## Build order (as actually followed)

1. Scaffold, `requirements.txt`, confirm imports, print resolved versions. ✅
2. `qdqn_cartpole/models/vqc.py` alone, smoke test: random batch of 8 states in, assert output
   shape `[8, 2]`, assert gradients are non-`None` and non-zero for **all three**
   parameter groups (`lam`, `vqc`, `w`). This is exactly what caught the
   `input_gradients=False` bug described above -- the smoke test failed on the first
   run, with `lam.grad` dead, before the fix. ✅
3. `qdqn_cartpole/replay.py` + `qdqn_cartpole/trainer.py` + `qdqn_cartpole/models/mlp.py`. Verified the trainer
   actually trains and logs correctly with the MLP first (fast, so this validated the
   loop, CSV reproducibility, and target-network/epsilon mechanics without waiting on
   quantum simulation). ✅
4. Wired the VQC into the trainer with the same loop. Confirmed via the mandatory
   gradient smoke test and a short run that `w` moves off its 1.0 initialization. Ran
   the required 100-gradient-step timing probe (measured on a 20-step sample, ~14.5s per
   step) and stopped before launching the full 5-seed sweep, per the performance
   warning above. ✅
5. `qdqn_cartpole/cli/sweep_seeds.py`, `qdqn_cartpole/plots.py`, `qdqn_cartpole/cli/make_figures.py` -- built and
   exercised against the MLP's real (fast) multi-seed run; the QDQN-side figures that
   need a completed sweep (full learning curves, parameter-count-vs-performance for the
   VQC across seeds) are wired up and will populate once a QDQN sweep is actually run at
   a wall-clock budget you choose from the options above.
6. This README.

## References

**Papers**
- Chen, S. Y.-C. et al. (2020). *Variational quantum circuits for deep reinforcement
  learning.* https://arxiv.org/abs/1907.00397 -- original VQC-for-deep-RL.
- Skolik, A., Jerbi, S., & Dunjko, V. (2022). *Quantum agents in the gym.* Quantum 6, 720.
  https://arxiv.org/abs/2103.15084 -- the QDQN recipe this follows; output/input scaling
  and ablations.
- Pérez-Salinas, A. et al. (2020). *Data re-uploading for a universal quantum
  classifier.* Quantum 4, 226. https://doi.org/10.22331/q-2020-02-06-226 -- for the
  `reuploading` config flag.
- Jerbi, S. et al. (2021). NeurIPS 34. https://arxiv.org/abs/2103.05577 -- quantum
  policies / actor-critic direction.
- Skolik, A. et al. (2023). *Robustness of QRL under hardware errors.* EPJ Quantum
  Technology 10:8.
  https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-023-00166-1
  -- finite-shot and hardware-noise behaviour, for the later shot sweep.
- Towers, M. et al. (2024). *Gymnasium.* https://arxiv.org/abs/2407.17032 -- environment
  API.

**Code references**
- [TorchConnector + EstimatorQNN tutorial](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html)
  -- primary API reference.
- [Gymnasium docs](https://gymnasium.farama.org/) -- `reset()` returns `(obs, info)`,
  `step()` returns the 5-tuple `(obs, reward, terminated, truncated, info)` (not the
  deprecated `gym` 4-tuple API).

## Working style notes

- Hyperparameters in `configs/*.yaml` are starting points from the Skolik et al. recipe,
  flagged in comments as guesses, not tuned results.
- No installs outside `requirements.txt` were needed -- `pyyaml` (for `configs/*.yaml`)
  and `Pillow` (for the rollout GIF, via matplotlib's `PillowWriter` instead of
  `imageio`) were already present in the target environment.
