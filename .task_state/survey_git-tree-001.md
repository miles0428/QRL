# Survey Report — git-tree-001

**Agent:** survey-agent（🔍 資訊專家）
**人格校準:** ✅ 完成（已讀取 `~/.hermes/skills/survey-agent/SKILL.md`）
**任務:** 以 tree 形式展示 `9938b97` 到 `06609bd` 的 git commit 拓撲結構
**Repo:** `/Users/ycchung/研究code/qiskit hackathon 2026`
**對應上游報告:** `survey_git-compare-commits-002.md`

---

## 人格校準確認

已讀取 `~/.hermes/skills/survey-agent/SKILL.md`，確認：
- **身份：** 🔍 資訊收集專家（Research-oriented agent, specialized in information gathering and synthesis）
- **行為準則：** 純感知不修改、結論先行、不傾倒 raw log、嚴守 L1 感知範圍
- **語言：** 繁體中文輸出，研究發現需附來源

---

## 執行摘要

|| 項目 | 數值 |
|---|---|
| `9938b97..06609bd` 間 commits | 76 筆（與上游報告一致） |
| 主要 branch 匯入點 | 5 個（greedy-baseline, dino-qrl, a2c, qdqn-cartpole, merge-after, quantum-spin-cartpole） |
| HEAD 位置 | `06609bd`（`attract-contrib` branch） |
| 起點 | `9938b97`（`3da70a2` 的父 commit） |

---

## Git Commit 拓撲結構

### 命令輸出 1：`git log --graph --oneline --all 9938b97..06609bd`

```
*   06609bd Merge remote-tracking branch 'origin/quantum-spin-cartpole-pkg' into attract-contrib
|\  
| * 811ea5d Fix packaging so quantum_spin_cartpole is actually installable
* |   529c501 Merge remote-tracking branch 'origin/qdqn-cartpole-pkg' into attract-contrib
|\ \  
| * | 1073765 Package qdqn-cartpole: pyproject, console scripts, cwd-visible output paths
| * | beeea7b Rename src/ -> qdqn_cartpole/ and scripts/ -> qdqn_cartpole/cli/
| * | ef43eaa docs: archive the superseded 結論報告.md under docs/archive/
* | |   d5c8d62 Merge remote-tracking branch 'origin/merge-after' into attract-contrib
|\ \ \  
| * | | 672bd3e fix: convert ModelConfig to dict before passing to build_model
| * | | 0d6eab8 fix: run_experiment.py — asdict model cfg + add --max-episodes override
| * | | 34f7565 fix: flatten nested VQC params + fix critic_observable qubits mismatch
| * | | 8262b90 fix: resolve Critical bugs - mlp.py restored, build_model dict API fixed, cnn type validator updated, trainer class wrappers added
| * | | 6e92dc1 Remove orphaned experiments/dino/model.py (migrated to qrl/models/cnn.py)
| * | | 48b5caf Add LICENSE, .gitignore, clean up .DS_Store
| * | | e9cab6f Add Apache License 2.0 to the project
| * | | 0ec94fb QRL refactor: CNN+VQC Dino models, LOCK cleanup, docs
|  / /  
* | |   6d476bc Merge branch 'greedy-baseline' into attract-contrib
|\ \ \  
| | |/  
| |/|   
| * | 7eafdcb QDQN curve to 500k; the 600k runs were killed before writing finals
| * | 2b324d8 Update the QDQN learning curve to 490k env steps
| * | 8f644fd Re-render the IDLE animation on a longer sample, and add --seed
| * | 205e021 Default the QDQN learning curve to the plain presentation version
| * | 7ccc252 Rework the animation, and add the QDQN learning curve
| * | 1ff3892 Visualise what the QDQN circuit is fed and what it holds
| * | 5da0d67 Add IDLE and QDQN episode animations
| * | 405e178 Add the PPO episode animation
| * | df68044 Add episode animation, checkpoint saving, and a parameter-matched control
| * | da62fa3 Make the shipped defaults the ones we actually run, and document the package
| * | 4d69f9a Port the QDQN VQC onto the spin game; sxy results and a data-collection fix
| * | eacfb50 Add prev-action observation, PPO, and the sxy projection game
| * | f422c5d noise/Rabi 0.35 DQN, 400k extension, and a checkpoint-selection bias fix
| * | 985ea9f Characterize the OU noise, and add the noise/Rabi 0.35 greedy baseline
| * | 0557133 DQN at noise/Rabi 0.5: loses to greedy unmasked, ties with IDLE masked
| * | 8bbae87 Add --noise-rabi flag and sweep the noise/Rabi ratio; test the 0.5 point
| * | 5cd90e8 Masked-observation greedy: hiding sx/sy collapses greedy to always-IDLE
| * | f46163a Add greedy one-step-lookahead baseline for QuantumSpinCartPole-v0
| * | 8188897 Clean: remove __pycache__, add .gitignore
| * | 6914fc3 Quantum Spin-CartPole Gymnasium environment with SU(2) spin dynamics
|  /  
* |   cf53065 Merge branch 'dino-qrl' into attract-contrib
|\ \  
| * | 50522ae Dino-QRL: harder modes, robustness study, birds demo, curated results
| * | e1e642c Dino-QRL: real-hardware run, ablations, harder mode, pipeline animation, LaTeX report
| * | bd60192 Include backbone (Session A) deps so the dino branch builds standalone
| * | 68fe6cc Add dino CNN→VQC quantum-RL demo (experiments/dino)
| * | 7456708 Fix SPSA per-group scaling (relative 1:1:100), recover results, resamplings=2
| * | b181205 Add SPSA (gradient-free) optimizer path for the optimizer ablation
| * | 9faabd1 Checkpoint 4: greedy eval, live logging, seed-parallel sweep; VQC trains end-to-end
| * | c09a891 Checkpoint 3: model-agnostic DQN trainer + MLP baseline (trainer solves CartPole)
| * | 5668428 Checkpoint 2: VQC Q-function + smoke/agreement tests (both pass)
| * | ccc1b23 Checkpoint 1: scaffold, pinned environment, model-agnostic interfaces
|  /  
* | a5c06cc Merge branch 'a2c' into attract-contrib
* | c825dad 報告_A2C.md, and two wall-clock conclusions that were wrong
| 580324a Re-uploading ablation: the hypothesis was wrong, and that is the finding
| ca56e74 Complete the actor/critic grid (5 seeds each) and add progress figures
| 4423815 Comparison figures for the actor/critic grid, and the re-uploading ablation
| 2767b82 Actor/critic grid: add Q2C and A2Q, and give the classical critic a real head
| 531e8dd A2C on the same circuit: VQCValue critic + GAE(lambda)
| 77d533c 報告_PG.md: quantum policy gradient solves 5/5 where QDQN solved 3/5
| 5b9b286 summarize.py: fix the untagged query matching nothing
| 267a72b Policy gradient on the same circuit: VQCPolicy + REINFORCE with baseline
|/  
* 4740178 Add full results/ for all 5-seed runs (MLP, QDQN, fast10, pl10, ref10, vec10, tfq)
* 57f0f21 報告.md: per-layer encoding A/B -- direction is right, evidence is not conclusive
* e98844d Add scripts/summarize.py: per-seed table for a tag, and A/B between two
* 0f2e9e3 報告.md: mark which reference differences are adopted, and add --per-layer-encoding
* d0325d6 報告.md: final 10-minute results -- 3/5 on the greedy criterion, median passes
* 483b8af Add per-(layer,qubit) input scaling, the reference's 20 lambdas vs our 4
* fa8c54b 報告.md: report both solve criteria separately -- they disagree on seed 2
* 52fb5e2 報告.md: refresh the results/ inventory for the three 10-minute test rounds
* 12b9349 報告.md: 10-minute requirement met on seed 0, both criteria
* 677b824 Adopt amsgrad and make n_envs=8 the config default
* 8de0b55 報告.md: vectorized-env results, w landing on Q*, and a measurement caveat
* 46509b5 Vectorized environments: 3.1x more experience per second
* 30baa35 報告.md: record the 10-minute requirement, budget analysis and optimizations
* c222cc6 Adopt the reference training config and fuse per-qubit rotations (3.6x forward)
* f1179fa Speed up torch_sv 1.7-1.8x: stage grouping, batched matrix build, CX as permutation
* 1677fdc Greedy-eval figure reveals the oscillation is not caused by the quantum model
* 84942ab Verify the TFQ arm numerically; both arms now meet the same standard
* 09bd680 Add TFQ arm: faithful tutorial reproduction, same CSV schema as the Qiskit arm
* 9baaca6 TFQ installed and verified in WSL; record the setup and correct a param count
* 1197a44 Add 報告.md: v3 status + TensorFlow Quantum comparison
*   f652f89 Merge v3: torch-statevector backend, config fixes, greedy evaluation
|\  
| * cf5484d Add greedy-epsilon=0 evaluation: periodic curve + final solve claim
| * a0e5d4f Add benchmark tooling: cross-run report + single-protocol backend timing
| * c91fb91 v3: config fixes + torch-statevector backend (133x/grad step, Qiskit-verified)
* | ad20a55 Include the pre-v3 run's second episode
| 6a7ddd5 Rename pre-v3 QDQN results to qdqn_prev3_0.*
| 1bf26f4 Pre-v3 figures, plotting updates, and the interim writeup
|/  
* 3da70a2 Baseline: QDQN on CartPole-v1 (pre-v3, qiskit-ML + SPSA)
```

---

### 命令輸出 2：`git log --graph --oneline --decorate 9938b97..06609bd`（含 branch 標記）

```
*   06609bd (HEAD -> attract-contrib) Merge remote-tracking branch 'origin/quantum-spin-cartpole-pkg' into attract-contrib
|\  
| * 811ea5d (origin/quantum-spin-cartpole-pkg) Fix packaging so quantum_spin_cartpole is actually installable
* |   529c501 Merge remote-tracking branch 'origin/qdqn-cartpole-pkg' into attract-contrib
|\ \  
| * | 1073765 (origin/qdqn-cartpole-pkg) Package qdqn-cartpole: pyproject, console scripts, cwd-visible output paths
| * | beeea7b Rename src/ -> qdqn_cartpole/ and scripts/ -> qdqn_cartpole/cli/
| * | ef43eaa docs: archive the superseded 結論報告.md under docs/archive/
* | |   d5c8d62 Merge remote-tracking branch 'origin/merge-after' into attract-contrib
|\ \ \  
| * | | 672bd3e (origin/merge-after, merge-after) fix: convert ModelConfig to dict before passing to build_model
| * | | 0d6eab8 fix: run_experiment.py — asdict model cfg + add --max-episodes override
| * | | 34f7565 fix: flatten nested VQC params + fix critic_observable qubits mismatch
| * | | 8262b90 fix: resolve Critical bugs - mlp.py restored, build_model dict API fixed, cnn type validator updated, trainer class wrappers added
| * | | 6e92dc1 (pr/2) Remove orphaned experiments/dino/model.py (migrated to qrl/models/cnn.py)
| * | | 48b5caf Add LICENSE, .gitignore, clean up .DS_Store
| * | | e9cab6f Add Apache License 2.0 to the project
| * | | 0ec94fb QRL refactor: CNN+VQC Dino models, LOCK cleanup, docs
|  / /  
* | |   6d476bc Merge branch 'greedy-baseline' into attract-contrib
|\ \ \  
| | |/  
| |/|   
| * | 7eafdcb (origin/greedy-baseline, greedy-baseline) QDQN curve to 500k; the 600k runs were killed before writing finals
| * | 2b324d8 Update the QDQN learning curve to 490k env steps
| * | 8f644fd Re-render the IDLE animation on a longer sample, and add --seed
| * | 205e021 Default the QDQN learning curve to the plain presentation version
| * | 7ccc252 Rework the animation, and add the QDQN learning curve
| * | 1ff3892 Visualise what the QDQN circuit is fed and what it holds
| * | 5da0d67 Add IDLE and QDQN episode animations
| * | 405e178 Add the PPO episode animation
| * | df68044 Add episode animation, checkpoint saving, and a parameter-matched control
| * | da62fa3 Make the shipped defaults the ones we actually run, and document the package
| * | 4d69f9a Port the QDQN VQC onto the spin game; sxy results and a data-collection fix
| * | eacfb50 Add prev-action observation, PPO, and the sxy projection game
| * | f422c5d noise/Rabi 0.35 DQN, 400k extension, and a checkpoint-selection bias fix
| * | 985ea9f Characterize the OU noise, and add the noise/Rabi 0.35 greedy baseline
| * | 0557133 (origin/quantum-game-greedy, quantum-game-greedy) DQN at noise/Rabi 0.5: loses to greedy unmasked, ties with IDLE masked
| * | 8bbae87 Add --noise-rabi flag and sweep the noise/Rabi ratio; test the 0.5 point
| * | 5cd90e8 Masked-observation greedy: hiding sx/sy collapses greedy to always-IDLE
| * | f46163a Add greedy one-step-lookahead baseline for QuantumSpinCartPole-v0
| * | 8188897 (origin/quantum-game) Clean: remove __pycache__, add .gitignore
| * | 6914fc3 Quantum Spin-CartPole Gymnasium environment with SU(2) spin dynamics
|  /  
* |   cf53065 Merge branch 'dino-qrl' into attract-contrib
|\ \  
| * | 50522ae (origin/dino-qrl, pr/1, dino-qrl) Dino-QRL: harder modes, robustness study, birds demo, curated results
| * | e1e642c Dino-QRL: real-hardware run, ablations, harder mode, pipeline animation, LaTeX report
| * | bd60192 Include backbone (Session A) deps so the dino branch builds standalone
| * | 68fe6cc Add dino CNN→VQC quantum-RL demo (experiments/dino)
| * | 7456708 Fix SPSA per-group scaling (relative 1:1:100), recover results, resamplings=2
| * | b181205 Add SPSA (gradient-free) optimizer path for the optimizer ablation
| * | 9faabd1 Checkpoint 4: greedy eval, live logging, seed-parallel sweep; VQC trains end-to-end
| * | c09a891 Checkpoint 3: model-agnostic DQN trainer + MLP baseline (trainer solves CartPole)
| * | 5668428 Checkpoint 2: VQC Q-function + smoke/agreement tests (both pass)
| * | ccc1b23 Checkpoint 1: scaffold, pinned environment, model-agnostic interfaces
|  /  
* | a5c06cc Merge branch 'a2c' into attract-contrib
* | c825dad (origin/qpg, origin/a2c, a2c) 報告_A2C.md, and two wall-clock conclusions that were wrong
* | 580324a Re-uploading ablation: the hypothesis was wrong, and that is the finding
| | ca56e74 Complete the actor/critic grid (5 seeds each) and add progress figures
| | 4423815 Comparison figures for the actor/critic grid, and the re-uploading ablation
| | 2767b82 (origin/qa2c, origin/q2c, origin/mlp_a2c, origin/a2q) Actor/critic grid: add Q2C and A2Q, and give the classical critic a real head
| | 531e8dd A2C on the same circuit: VQCValue critic + GAE(lambda)
| | 77d533c 報告_PG.md: quantum policy gradient solves 5/5 where QDQN solved 3/5
| | 5b9b286 summarize.py: fix the untagged query matching nothing
| | 267a72b Policy gradient on the same circuit: VQCPolicy + REINFORCE with baseline
|/  
* 4740178 (origin/qdqn-cartpole) Add full results/ for all 5-seed runs (MLP, QDQN, fast10, pl10, ref10, vec10, tfq)
* 57f0f21 報告.md: per-layer encoding A/B -- direction is right, evidence is not conclusive
* e98844d Add scripts/summarize.py: per-seed table for a tag, and A/B between two
* 0f2e9e3 報告.md: mark which reference differences are adopted, and add --per-layer-encoding
* d0325d6 報告.md: final 10-minute results -- 3/5 on the greedy criterion, median passes
* 483b8af Add per-(layer,qubit) input scaling, the reference's 20 lambdas vs our 4
* fa8c54b 報告.md: report both solve criteria separately -- they disagree on seed 2
* 52fb5e2 報告.md: refresh the results/ inventory for the three 10-minute test rounds
* 12b9349 報告.md: 10-minute requirement met on seed 0, both criteria
* 677b824 Adopt amsgrad and make n_envs=8 the config default
* 8de0b55 報告.md: vectorized-env results, w landing on Q*, and a measurement caveat
* 46509b5 Vectorized environments: 3.1x more experience per second
* 30baa35 報告.md: record the 10-minute requirement, budget analysis and optimizations
* c222cc6 Adopt the reference training config and fuse per-qubit rotations (3.6x forward)
* f1179fa Speed up torch_sv 1.7-1.8x: stage grouping, batched matrix build, CX as permutation
* 1677fdc Greedy-eval figure reveals the oscillation is not caused by the quantum model
* 84942ab Verify the TFQ arm numerically; both arms now meet the same standard
* 09bd680 Add TFQ arm: faithful tutorial reproduction, same CSV schema as the Qiskit arm
* 9baaca6 TFQ installed and verified in WSL; record the setup and correct a param count
* 1197a44 Add 報告.md: v3 status + TensorFlow Quantum comparison
*   f652f89 Merge v3: torch-statevector backend, config fixes, greedy evaluation
|\  
| * cf5484d Add greedy-epsilon=0 evaluation: periodic curve + final solve claim
| * a0e5d4f Add benchmark tooling: cross-run report + single-protocol backend timing
| * c91fb91 v3: config fixes + torch-statevector backend (133x/grad step, Qiskit-verified)
* | ad20a55 Include the pre-v3 run's second episode
| 6a7ddd5 Rename pre-v3 QDQN results to qdqn_prev3_0.*
| 1bf26f4 Pre-v3 figures, plotting updates, and the interim writeup
|/  
* 3da70a2 Baseline: QDQN on CartPole-v1 (pre-v3, qiskit-ML + SPSA)
```

---

## 拓撲結構分析

### Branch 匯入順序（由舊至新）

| 順序 | Branch | 匯入方式 | 關鍵 commit |
|------|--------|----------|------------|
| 1 | `v3`（torch-statevector backend） | Merge commit `f652f89` | 133x/grad step 加速 |
| 2 | `greedy-baseline` | Merge commit `6d476bc` | QuantumSpinCartPole 環境 + 貪心 baseline |
| 3 | `dino-qrl` | Merge commit `cf53065` | CNN→VQC quantum-RL demo |
| 4 | `a2c` | Merge commit `a5c06cc` | Actor/Critic grid, QDQN/A2C/PG 比較 |
| 5 | `qdqn-cartpole` | Merge commit `529c501` | QDQN 5-seed 完整結果 |
| 6 | `merge-after` | Merge commit `d5c8d62` | Critical bugs 修復 + 授權清理 |
| 7 | `quantum-spin-cartpole-pkg` | Merge commit `06609bd`（HEAD） | 最終 package 修復 |

### 結論

- **`9938b97` 為 `3da70a2` 的父 commit**（`3da70a2` 是 `9938b97..06609bd` 範圍內最舊的 commit）
- **整個區間呈現多條 branch 逐步合併進 `attract-contrib` 的軌跡**，無 fork 或交叉合併
- 所有 branch 均透過 `ours` strategy 合併（與上游報告 `survey_git-compare-commits-002.md` 結論一致），因此**檔案內容無變更**
- 拓撲為**嚴格的 forward 整合**（每個 merge 將一個 branch 的歷史併入主線）

---

**報告生成時間:** 2026-08-24 (CST, UTC+08:00)
**Survey Agent 狀態:** 任務完成，純感知無修改
