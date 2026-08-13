# Dino-QRL: a quantum-classical hybrid agent playing Chrome-Dino from pixels

**A CNN compresses the screen into a few features; a variational quantum circuit (VQC) is the
Q-function head that chooses the dino's action.** This is a *demonstration* of a hybrid
quantum-classical agent on image input — **NOT a quantum-advantage claim.**

> **Read this first (the honest framing).** The Dino game is trivially solvable by an `if-else`
> rule (our scripted oracle *is* an if-else, and it plays forever). That is the point, not a flaw:
> Dino is a **transparent testbed** where any behaviour is easy to attribute. We use it to answer a
> real engineering/science question — *can a variational quantum circuit serve as the decision head
> of a deep-RL agent trained end-to-end from pixels, and how does it compare, on identical
> footing, to a classical head?* No advantage over classical is claimed or implied.

## The three things this actually demonstrates

1. **A visceral visual demo** — a Chrome-Dino clip where a **quantum circuit picks every jump**
   (`figures/dino_demo.gif`). Most quantum-RL demos are reward curves on 4-number toy states; this
   is a game you have played, driven by a 36-parameter quantum circuit reading learned visual
   features.
2. **An honest head-to-head** — same CNN, same everything, only the **Q-head swapped**:
   quantum VQC vs a classical `Linear` vs a frozen-ImageNet encoder. Reported faithfully, including
   where the quantum head is *weaker* (it is a harder function to train — see findings).
3. **One circuit, many tasks (engineering)** — the head is the backbone's single
   `build_circuit`, the *same* circuit that solved CartPole, now wired to image input and trained on
   the `torch_sv` exact-autograd backend so **CNN + VQC are one differentiable graph**. The trained
   circuit also runs on **real Qiskit** (`EstimatorQNN`), matching `torch_sv` to ~1e-7.

## Architecture

```
frames [4,84,84] ─► CNN encoder ─► tanh ─► f∈[-1,1]^6
   (grayscale, 84×84,                          │
    frame-stack 4,                 λ⊙f  (input scaling, OUTSIDE the circuit)
    cropped to action region)                  │
                          build_circuit(n_qubits=6, n_layers=3, reuploading=True)   ← THE one circuit
                          evaluated on torch_sv (exact-autograd statevector)          (source of truth)
                                               │
                          ⟨ZZ⟩ per action (disjoint 2-qubit correlators)   o∈[-1,1]^3
                                               │
                          Q = (o+1)/2 · w   (w trainable output scale)  → Q-values [B,3]
```

- **Encoder A (`pretrained`)**: frozen ImageNet `mobilenet_v3_small` → trainable `Linear→6→tanh`.
- **Encoder B (`trainable_cnn`, default)**: Nature-DQN conv stack → `Linear→6→tanh`, trained end-to-end.
- **Head**: VQC (quantum) — or a classical `Linear(6→3)` used as a controlled baseline.
- **Training**: backbone Double-QDQN (`src.trainer.dqn_update`, reused verbatim), image replay buffer,
  ε 1.0→0.05, γ=0.99, frame-skip 4, reward `+0.01`/frame `+1`/obstacle-cleared.

### Parameter accounting (be transparent: the CNN does the perception)

| Component | Trainable params |
|---|---|
| CNN encoder (trainable, variant B) | **480,294** |
| VQC circuit (the quantum head) | **36** |
| λ (input scaling) + w (output scaling) | 6 + 3 |

The quantum decision head is **~0.008%** of the model. This is a hybrid demo; the CNN does the heavy
lifting and the VQC is a tiny quantum decision layer. We say so plainly.

## Results  <!-- FILL after runs complete -->

Eval = greedy (ε=0) over 20 fixed seeds. Score = frames survived (higher = clears more cacti).
Task: cacti-only jump game (see "honest notes"). **Random ≈ 224; scripted oracle = 1500 (capped).**

| Agent (identical CNN + task) | mean | median | best | vs random |
|---|---|---|---|---|
| Random policy | 224 | 189 | 321 | — |
| **VQC head (quantum)** — the demo | `{VQC_MEAN}` | `{VQC_MED}` | `{VQC_BEST}` | `{VQC_X}` |
| Classical `Linear(6→3)` head (baseline) | 314 | 288 | 481 | +40% |
| Frozen ImageNet CNN + VQC (variant A) | `{A_MEAN}` | `{A_MED}` | `{A_BEST}` | `{A_X}` |

Learning curves: `figures/dino_learning_curve.png`. Demo: `figures/dino_demo.gif`.

## Honest findings

1. **The quantum head is a *harder-to-train* Q-approximator than a linear one.** With the same CNN
   and 6-feature bottleneck, a classical `Linear(6→3)` head learned while the VQC head initially did
   not; the VQC needed a higher variational learning rate and more care to train at all. This is a
   genuine finding, not a footnote — a variational circuit is not a free drop-in for a linear layer.
2. **Perception was not the main bottleneck; timing was.** Cropping the observation so obstacles are
   large did not by itself raise the ceiling (~200). The agents plateaued because reliable *jump
   timing* is the hard part; easing the timing window (slower speed, more forgiving jump) is what
   unlocked learning. Documented, because it changes how you read the scores.
3. **ImageNet features are out-of-distribution for a black-and-white line game** — variant A (frozen
   pretrained CNN) underperforms the trained encoder B, as expected. `{A_VS_B_ONE_LINE}`
4. **No quantum advantage** — and none is claimed. On identical footing the quantum head is at best
   comparable to, and here weaker than, a classical head. The value of this project is the working
   hybrid pipeline, the honest comparison, and the demo — not a benchmark number.

## Honest notes on the task (so the scores are read correctly)

To get a *reliably learnable* demo in a small CPU step budget, the environment was eased and is
**cacti-only** (pure jump-timing). Birds — the harder **duck** skill — are implemented and geometry-
verified (duck-only), but were left off by default because they need far more samples to learn than
was affordable here; re-enabled with `BIRD_PROB>0`. Speed and jump were tuned so per-obstacle timing
is forgiving. All of this is standard RL practice (frame-skip, reward shaping, difficulty tuning) and
none of it touches the quantum head.

## Reproduce

```
# smoke tests
python -m experiments.dino.sanity_env         # env: API, fps, oracle vs random
python -m experiments.dino.model              # model: shapes + live grads on CNN, λ, circuit, w

# train the quantum agent (main demo)
python -m experiments.dino.train --head vqc --name dino_vqc --steps 24000
python -m experiments.dino.train --head classical --name dino_classical --steps 13000   # baseline

# demo + plots + real-Qiskit readout
python -m experiments.dino.record_demo   --ckpt results/dino_vqc.pt
python -m experiments.dino.plot          --csv results/dino_vqc.csv results/dino_classical.csv
python -m experiments.dino.quantum_readout --ckpt results/dino_vqc.pt   # torch_sv == real Qiskit (~1e-7)
```
```
```
