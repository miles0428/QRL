# Dino QRL demo — progress log

Priority order from the brief; get a *playing* agent first, log any skips here.

## ✅ 1. Headless pygame dino env + Gymnasium wrapper + preprocessing
- `dino_game.py`: pure-pygame T-Rex clone, headless (SDL dummy driver), deterministic
  (seeded `random.Random`). CACTUS = jump-only, BIRD (tall band) = duck-only, so both
  actions matter and perception is required.
- `env.py`: `DinoImageEnv` (Gymnasium API) → obs `[4,84,84]` uint8 (grayscale→smoothscale
  84×84→frame-stack 4), `Discrete(3)` actions, reward +1/frame, terminate on crash,
  truncate at `max_steps`. `DinoRawEnv` exposes the full 600×150 RGB frame for the GIF.
- Preprocessing runs in pygame/numpy (no PIL on the hot path) → **~2000 steps/sec**.
- Sanity (`sanity_env.py`): obs shape OK; **random mean ≈178 vs scripted-oracle 800 (capped)**
  → game winnable AND correct-action-dependent. Sample frame: `figures/dino_sample_frame.png`.

## ✅ 2. CNN→VQC model (variant B **and** A) + smoke test
- `model.py`: `DinoQFunction(QFunction)` — obeys the backbone interface, so the reused
  `dqn_update`/`select_action` treat it exactly like the CartPole VQC.
  - Encoder A (`pretrained`): FROZEN ImageNet mobilenet_v3_small → trainable Linear→n_qubits→tanh.
  - Encoder B (`trainable_cnn`, default): Nature-DQN conv → Linear → n_qubits → tanh.
  - Quantum head: backbone `build_circuit(n_qubits, n_layers=5, reuploading=True)` on `torch_sv`,
    λ⊙f RY-encoding, **disjoint ZZ correlators** per action (6q,3a → (0,1)(2,3)(4,5)),
    output head `Q=(o+1)/2·w` (returns are ≥0, so non-negative head fits).
- **Smoke test passes both encoders**: out `[B,3]`, grads ALIVE on {cnn, lam, vqc, w}, one graph.
- Param transparency (honest framing): **CNN 882,598 vs VQC 60** params (B); A has 3,462
  trainable (frozen backbone). The CNN does the perception; the VQC is the tiny quantum head.

## 🔄 3. Image replay + Double-QDQN training on the clone — in progress
- `replay.py`: `ImageReplayBuffer` (uint8, ~0.56 GB @10k) returning the backbone `Batch`. ✅
- `train.py`: loop reuses backbone `dqn_update(double=True)` + `select_action` + `linear_epsilon`
  + `set_global_seeds`/`make_rng` + `experiments.common.save_ckpt`. Only the image env + image
  buffer are new (the DQN math is NOT reimplemented). Periodic best-checkpoint saving. ✅

### Tuning journey (honest record — two runs killed and re-tooled)
- **Run 1** (n_layers=5, 60k steps, +1/frame reward): too slow (**torch_sv VQC forward ~30ms is
  the bottleneck**, dispatch-bound on ~120 gate ops — CNN is only ~10ms). Killed.
- **Run 2** (n_layers=3, faster ε-decay): ~4x faster VQC but **greedy policy stayed at random-level
  (~177) even after ε decayed** → a real learning problem, not just "too early". Killed.
- **Diagnosis**: dino is a *sparse-timing* task — every frame gives +1 regardless of action, so the
  only action-discriminating transitions are the few frames before an obstacle; uniform replay
  drowns them and DQN learns "all actions equal" → constant argmax → dies at obstacle 1.
- **Fixes applied (all standard RL; documented for honesty, quantum head unchanged):**
  1. **Frame-skip = 4** (action repeat, Nature-DQN): coarser control + committed jumps, **and 4× fewer
     VQC forwards per game frame** (big speedup). One "env step" = one decision = up to 4 frames.
  2. **Reward shaping**: `+0.01`/frame alive `+1.0` per obstacle *cleared* → direct contrast that
     rewards jumping/ducking at the right moment. (Reported `score` stays == frames survived.)
  3. Eased env slightly: birds rarer/later (learn jump-timing first), gentler speed ramp.
- **Runs 3-4** (frame-skip + shaped reward, denser obstacles): greedy policy STILL stuck at
  random-level (~170) after ε decayed. Not learning.

### Root-cause diagnosis (controlled experiments, not guessing)
- **Q1: VQC head or perception?** Added a `head="classical"` mode (same CNN + same 6-feature
  bottleneck, plain `Linear(6→3)` instead of the VQC). Result: **classical head learns weakly
  (greedy mean ~200, clears 1-2 obstacles); VQC head stays at random (~170).** → the VQC is a
  *harder-to-train* approximator here (an honest finding), AND something caps the whole pipeline.
- **Q2: is it perception (tiny obstacles)?** Cropped the observation to the action region
  (obstacles now large/clear — `figures/dino_obs_cropped.png`). Cropped classical **still ~200**.
  → perception was NOT the (main) bottleneck.
- **Q3: what caps it at ~200?** Scores were bimodal (die at obstacle 1, or clear 1-2) and even
  random exploration never passed ~300 → the agent can't learn *reliable jump timing* (each jump
  ~70% success → can't chain many). **The bottleneck is timing precision, not perception.**

### Fix that worked: make the task cleanly learnable (documented, honest)
- **Cacti-only** (birds/duck off — the duck skill is much harder in a small budget; noted as future
  work), **slower** speed (4.5, wider timing window in frames), **more forgiving jump** (higher/
  longer airtime), **narrower cacti**, kept the obs crop. Oracle 1500, random ~224.
- **Classical head on the eased task now LEARNS**: greedy ma20 climbed 198 → **~304**, individual
  episodes up to **768** (clears ~13 cacti). The pipeline is learnable.
- **Now training the VQC (quantum) head on the same eased task** (`dino_vqc`) — the main demo.
  Verdict on whether the quantum head learns it: pending (watcher at step 13000).

## ⏳ 4. Demo GIF + learning curve + RESULT.md
## ⏳ 5. Ablation A (frozen pretrained CNN) vs B (trained encoder)

### Environment notes / skips
- `torchvision` was not installed on the box → installed CPU build 0.21.0 (matches torch
  2.6.0) so **variant A is available**.
- `cv2` not installed → preprocessing uses pygame/numpy instead (no functional loss).
- `selenium`/chromedriver not installed → real `gym-chrome-dino` browser demo is the
  fallback per house rules; primary demo will be recorded on the pygame clone.
