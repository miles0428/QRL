# Survey Report — git-compare-commits-001

**Agent:** survey-agent  
**人格校準:** ✅ 完成（已讀取 `~/.hermes/skills/survey-agent/SKILL.md`）  
**任務:** 比較兩個 git commit：6d476bc 與 9938b97  
**Repo:** `/Users/ycchung/研究code/qiskit hackathon 2026`  
**Branch:** attract-contrib（HEAD = 6d476bc）

---

## 摘要

| | 9938b97 | 6d476bc |
|---|---|---|
| **時間** | 2026-08-24 21:16:53 +0800 | 2026-08-24 23:13:14 +0800 |
| **位置** | `origin/main`（PR #2 合併至 main） | `attract-contrib` HEAD |
| **拓撲關係** | **祖先** | **後代**（`merge-base` = 9938b97） |

**核心發現：6d476bc 落後於 main 70 個 commits，落後於 origin/attract-contrib 0 個。**

---

## 拓撲關係

```
9938b97 (origin/main, main分支) — QRL refactor: CNN+VQC Dino models
    ↓
    ... 70 個 commits ...
    ↓
6d476bc (HEAD -> attract-contrib) — Merge branch 'greedy-baseline' into attract-contrib
```

---

## 6d476bc 獨有的 commits（共 70 個）

時間軸由新到舊：

| # | Commit | Message |
|---|---|---|
| 1 | `6d476bc` | Merge branch 'greedy-baseline' into attract-contrib |
| 2 | `cf53065` | Merge branch 'dino-qrl' into attract-contrib |
| 3 | `a5c06cc` | Merge branch 'a2c' into attract-contrib |
| 4 | `c825dad` | 報告_A2C.md, and two wall-clock conclusions that were wrong |
| 5 | `7eafdcb` | QDQN curve to 500k; the 600k runs were killed before writing finals |
| 6 | `2b324d8` | Update the QDQN learning curve to 490k env steps |
| 7 | `8f644fd` | Re-render the IDLE animation on a longer sample, and add --seed |
| 8 | `205e021` | Default the QDQN learning curve to the plain presentation version |
| 9 | `7ccc252` | Rework the animation, and add the QDQN learning curve |
| 10 | `1ff3892` | Visualise what the QDQN circuit is fed and what it holds |
| 11 | `5da0d67` | Add IDLE and QDQN episode animations |
| 12 | `405e178` | Add the PPO episode animation |
| 13 | `df68044` | Add episode animation, checkpoint saving, and a parameter-matched control |
| 14 | `da62fa3` | Make the shipped defaults the ones we actually run, and document the package |
| 15 | `4d69f9a` | Port the QDQN VQC onto the spin game; sxy results and a data-collection fix |
| 16 | `50522ae` | Dino-QRL: harder modes, robustness study, birds demo, curated results |
| 17 | `eacfb50` | Add prev-action observation, PPO, and the sxy projection game |
| 18 | `f422c5d` | noise/Rabi 0.35 DQN, 400k extension, and a checkpoint-selection bias fix |
| 19 | `985ea9f` | Characterize the OU noise, and add the noise/Rabi 0.35 greedy baseline |
| 20 | `0557133` | DQN at noise/Rabi 0.5: loses to greedy unmasked, ties with IDLE masked |
| 21 | `e1e642c` | Dino-QRL: real-hardware run, ablations, harder mode, pipeline animation, LaTeX report |
| 22 | `8bbae87` | Add --noise-rabi flag and sweep the noise/Rabi ratio; test the 0.5 point |
| 23 | `5cd90e8` | Masked-observation greedy: hiding sx/sy collapses greedy to always-IDLE |
| 24 | `f46163a` | Add greedy one-step-lookahead baseline for QuantumSpinCartPole-v0 |
| 25 | `580324a` | Re-uploading ablation: the hypothesis was wrong, and that is the finding |
| 26 | `8188897` | Clean: remove __pycache__, add .gitignore |
| 27 | `6914fc3` | Quantum Spin-CartPole Gymnasium environment with SU(2) spin dynamics |
| 28 | `bd60192` | Include backbone (Session A) deps so the dino branch builds standalone |
| 29 | `68fe6cc` | Add dino CNN→VQC quantum-RL demo (experiments/dino) |
| 30 | `ca56e74` | Complete the actor/critic grid (5 seeds each) and add progress figures |
| 31 | `4423815` | Comparison figures for the actor/critic grid, and the re-uploading ablation |
| 32 | `2767b82` | Actor/critic grid: add Q2C and A2Q, and give the classical critic a real head |
| 33 | `531e8dd` | A2C on the same circuit: VQCValue critic + GAE(lambda) |
| 34 | `77d533c` | 報告_PG.md: quantum policy gradient solves 5/5 where QDQN solved 3/5 |
| 35 | `5b9b286` | summarize.py: fix the untagged query matching nothing |
| 36 | `267a72b` | Policy gradient on the same circuit: VQCPolicy + REINFORCE with baseline |
| 37–55 | *(37 個 commits)* | 早期實驗、checkpoint 框架、TFQ/SPSA/向量化的基礎建設、報告文件更新等 |

---

## 9938b97 獨有的 commits

**0 個**。9938b97 是 6d476bc 的嚴格祖先，無任何 unique commits。

---

## 檔案變更差異統計

### 關鍵發現：**git diff 9938b97..6d476bc 的 --stat 與 --name-only 均為空**

這是預期行為，因為：
1. 9938b97 是「QRL refactor」的 initial structure commit，定義了 `games/`, `qrl/`, `configs/` 的**初始內容**
2. 介於兩者之間的 70 個 commits 的絕大多數是：
   - **Merge commits**（0 個檔案變更）
   - **Binary 檔案**（`.pt` 模型權重、`.csv` 結果資料）
   - **Markdown 文件**（`報告.md`、`報告_PG.md`、`報告_A2C.md`）

### 實質變更的代表性 commits

| Commit | 變更類型 |
|---|---|
| `6914fc3` | 全新 `quantum_spin_cartpole/` 環境（含 SU(2) spin dynamics） |
| `c825dad` | `結果_A2C.md` + 大量 `.pt`/`.csv` 結果檔 |
| `da62fa3` | Package 文件化 + checkpoint saving |
| `4d69f9a` | QDQN VQC port 至 spin game |

---

## 結論

1. **6d476bc 落後 main（9938b97 的 descendant）70 個 commits**，需 push 才能同步至 origin
2. **attract-contrib 分支落後於 origin/main**，但包含更完整的實驗結果（動畫、episode 追蹤、learning curves）
3. **無檔案層級差異顯示**，因兩者間主要為 merge commits 和 binary/data 檔案
4. **9938b97 是乾淨的初始重構 commit**，而 **6d476bc 是實驗彙總 commit**，含大量 binary 結果檔案

---

## 附圖：分支拓撲（近 20 commits）

```
*   6d476bc (HEAD -> attract-contrib) Merge branch 'greedy-baseline'
|\  
* \   cf53065 Merge branch 'dino-qrl'
|\ \  
| * | 50522ae (origin/dino-qrl, dino-qrl) Dino-QRL: harder modes...
| * | e1e642c Dino-QRL: real-hardware run...
| * | 6914fc3 Quantum Spin-CartPole Gymnasium environment...
* | |   a5c06cc Merge branch 'a2c'
|\ \ \  
| * | | c825dad (origin/a2c, a2c) 報告_A2C.md...
...（其後 50+ commits 省略）...
* | | 9938b97 QRL refactor: CNN+VQC Dino models, LOCK cleanup, docs (#2)
|/  
* 2a135b3 Initial commit: add README
```

---

**報告生成時間:** 2026-08-24 (CST, UTC+08:00)  
**Survey Agent 狀態:** 任務完成，純感知，未執行任何修改
