# QPG on CartPole-v1 — 量子 policy gradient 報告

**資料截止：2026-08-13 12:40。5 seeds × 2 設定（qpg / mlp_pg）皆已完成，數字為本機實測。**
本份與 `報告.md`（QDQN，value-based）並列，不取代它。兩份共用同一顆電路、同一個後端、
同一套評估協定，所以差異可以歸因到演算法而非模型。

---

## A. 摘要

在 `報告.md` 用的同一個 10 分鐘預算下，把 value-based 換成 policy gradient：

| | QPG（量子 PG） | QDQN `vec10`（最佳 DQN 設定） |
|---|---|---|
| greedy solve | **5/5** ✅ | 3/5 |
| training solve | **5/5** ✅ | 3/5 |
| greedy 中位數 | **500.0**（範圍 499.7–500.0） | 477.5（範圍 404.6–500.0） |
| seed 內 std | **0.6** | 57.8 |
| seed 間 std | **0.1** | 35.2 |
| 訓練時鐘中位數 | 275 s | 494 s |
| 梯度步數中位數 | **60** | 12,775 |

三個要點：

1. **`報告.md` F 節記錄的策略震盪問題，在 PG 這邊沒有出現。** 那份報告說「開放問題是穩定度
   而非峰值表現」，並舉了 `404.6 ± 190.8`（在 500 和接近 0 之間擺盪）當例子。QPG 五個 seed
   的 seed 內 std 分別是 2.8、0.0、0.0、0.0、0.0。
2. **梯度步數少了 213 倍**（60 vs 12,775）。每步貴 100 倍（4.16 s vs 0.04 s），淨結果仍快 1.8 倍。
3. **QPG 的超參數沒有調過。** `configs/qpg.yaml` 裡標 GUESS 的值都是第一次猜的。QDQN 那邊
   經過 v3 參考實作對照、per-layer encoding A/B 等多輪調整後是 3/5。這個對比要小心解讀，
   見 F 節。

---

## B. 架構重用到什麼程度

`VQCPolicy`（`src/models/vqc_policy.py`）直接 import `vqc.py` 的 `build_circuit` 和
`make_backend`，不是平行實作。實際驗證（不是宣稱）：

```
QDQN  vqc backend class     : src.models.vqc._TorchStatevectorBackend
QPG   vqc backend class     : src.models.vqc._TorchStatevectorBackend
same class object           : True
compiled type (兩邊)         : src.models.torch_statevector.CompiledCircuit
  stages                    : 10 / 10   match=True
  n_qubits                  : 4 / 4     match=True
forward+backward 中 tsv.simulate 呼叫次數（7 個 states）: 1
```

也就是說同一個 `compile_circuit` 產出的閘程式、同一條 torch autograd 路徑，訓練期間沒有
任何 qiskit primitive。config 實際載入的是 `backend=torch_sv`、`gradient_method=adjoint`、
4 qubits × 5 layers、reuploading、`['ZZII','IIZZ']`——與 `configs/qdqn.yaml` 逐項相同。

### 只有兩處不同

| | QDQN (`vqc.py`) | QPG (`vqc_policy.py`) |
|---|---|---|
| head | `w * (<O>+1)/2`，每動作一個權重 | `beta * <O>` 後接 softmax，單一逆溫度 |
| 為什麼 | 必須表示 Q\* ≈ 99 | softmax 平移不變，只有 logit **差值**有意義 |
| 探索 | epsilon 排程 | 從 π 取樣；`beta` 可訓練，自己退火 |
| trainer | `trainer.py`（replay、target net、TD loss） | `pg_trainer.py`（on-policy rounds、return-to-go、baseline） |
| 參數量 | 46（40 電路 + 4 lam + 2 w） | **45**（40 電路 + 4 lam + 1 beta） |

`src/evaluate.py` 兩邊共用，一行沒改：`argmax(logits) == argmax(softmax(logits))`。

**這裡有一個會靜默失效的地方，所以做了防護。** 上面那個等式只在 `beta > 0` 時成立。若把
beta 存成原始參數而讓梯度把它走過零，greedy 評估會反過來報告一個「專挑最差動作」的
agent，而且**任何一層都不會報錯**。因此 beta 以 `softplus(beta_raw)` 參數化，結構上不可能
跨過零。

---

## C. 結果

### QPG — 5/5，兩個標準都過 ✅

`t(s)` 是**訓練時鐘**（與 `報告.md` 同一套量法：`trainer` 刻意把評估時間扣掉）。
程序總耗時另計，見下方註記。

| seed | episodes | env steps | 梯度步 | 訓練時鐘 | s/梯度步 | solve@ | greedy(100 集) | ok |
|---|---|---|---|---|---|---|---|---|
| 0 | 930 | 204,678 | 59 | 383 s | 6.50 | 930 | 499.7 ± 2.8 | ✅ |
| 1 | 1,044 | 214,909 | 66 | 275 s | 4.16 | 1,044 | 500.0 ± 0.0 | ✅ |
| 2 | 1,063 | 191,598 | 67 | 230 s | 3.43 | 1,063 | 500.0 ± 0.0 | ✅ |
| 3 | 947 | 177,975 | 60 | 290 s | 4.83 | 947 | 500.0 ± 0.0 | ✅ |
| 4 | 896 | 173,272 | 56 | 222 s | 3.96 | 896 | 500.0 ± 0.0 | ✅ |
| **中位數** | **947** | **191,598** | **60** | **275 s** | **4.16** | — | **500.0** | **5/5** |

> **註：訓練時鐘 vs 程序總時間。** 程序總耗時分別是 972 / 829 / 808 / 910 / 656 秒，
> 高於 600 秒的 cap。差額全部是評估：每 50 集一次的 5 集 greedy 曲線，加上結尾一次
> 100 集的 greedy 評估——後者是 batch-1 的逐步 forward，對電路而言很貴（約 5 分鐘）。
> 10 分鐘的宣告下在訓練時鐘上，這與 `報告.md` 的量法一致（`max_wall_clock_s` 檢查用的
> 就是扣掉評估後的時鐘）。若要含評估一起卡 10 分鐘，把 `final_eval_episodes` 調小即可。

### 古典對照 MLP-PG — 同樣 5/5

參數量 44 vs 量子的 45，REINFORCE 超參數逐項相同。

| | QPG（量子） | MLP-PG（古典） | 量子 / 古典 |
|---|---|---|---|
| greedy solve | 5/5 | 5/5 | — |
| greedy 中位數 | 500.0 | 500.0 | — |
| episodes 中位數 | **947** | 1,268 | **0.75×** |
| env steps 中位數 | **191,598** | 242,470 | **0.79×** |
| 梯度步中位數 | **60** | 80 | **0.75×** |
| 訓練時鐘中位數 | 275 s | 30 s | 9.2× |

**參數量相當的前提下，量子 policy 需要的環境互動比古典少約 21%、梯度步少 25%。** 這是
sample efficiency 上對量子模型有利的一項觀察，方向與 Jerbi et al. 的主張一致。但 wall-clock
差 9.2 倍——電路模擬的成本不會因為步數變少就消失。五個 seed 不足以把 21% 當成定論，
見 F 節。

### 與 value-based 的並排

| 設定 | 演算法 | greedy 中位數 | seed 內 std | solve | 訓練時鐘中位數 | 梯度步中位數 |
|---|---|---|---|---|---|---|
| **qpg** | REINFORCE | **500.0** | **0.6** | **5/5** | 275 s | 60 |
| mlp_pg | REINFORCE | 500.0 | 0.0 | 5/5 | 30 s | 80 |
| qdqn `vec10` | DQN | 477.5 | 57.8 | 3/5 | 494 s | 12,775 |
| qdqn `pl10` | DQN | 490.4 | 45.2 | 3/5 | 511 s | 12,428 |
| mlp_baseline | DQN | 500.0 | 0.0 | 5/5 | 222 s | 128,372 |

---

## D. 為什麼 PG 反而適合這個後端

一開始的直覺是 PG 在 wall-clock 上會吃虧：on-policy 不能重播資料，需要更多環境互動。
實測下來環境互動確實多了（191k vs QDQN 的 128k，1.5×），但總時間反而少。原因在梯度步的
經濟學：

| | QDQN `vec10` | QPG | 比值 |
|---|---|---|---|
| 梯度步數 | 12,775 | 60 | **213× 少** |
| 每步成本 | 0.04 s | 4.16 s | 100× 貴 |
| 總計 | 494 s | 275 s | **1.8× 快** |

PG 的一次梯度步吃掉整批 episode 的**每一個 state**，所以單步昂貴。但那整批是**一次**
`[T, n_qubits]` 的 batched forward——`tsv.simulate` 呼叫一次。一條 500 步的 episode 大約
只花一次電路評估的代價，而不是 500 次。DQN 那邊則是每 10 個 env step 就要對 batch 16 做
一次 forward+backward。

這也是為什麼 `pg_trainer.py` 在更新時**重算** log-probability，而不是保留取樣時的計算圖：
保留的話等於對數百個 batch-1 電路評估反向傳播；重算則把整批壓成一次 forward。

---

## E. 觀察到的訓練動態

### beta 自己退火，所以不需要 epsilon 排程

| seed | beta（首筆 → 末筆） | entropy（首筆 → 末筆） |
|---|---|---|
| 0 | 1.064 → 4.554 | 0.6845 → 0.4945 |
| 1 | 0.938 → 4.722 | 0.6887 → 0.5326 |
| 2 | 1.064 → 4.701 | 0.6854 → 0.5023 |
| 3 | 0.938 → 4.612 | 0.6861 → 0.5167 |
| 4 | 1.064 → 4.452 | 0.6845 → 0.4968 |

五個 seed 一致地把 beta 從 1 推到 4.5 附近，entropy 從接近上限（兩動作的最大熵是 0.693）
降到 0.50 左右。策略在自己收緊探索強度，這是 PG 這條路完全不需要 epsilon 排程的原因。

註：CSV 首筆是**第一次更新之後**才寫的，所以起始值不是初始化的 1.000。

### lam（輸入縮放）落點一致

| seed | lam（x, x_dot, theta, theta_dot） |
|---|---|
| 0 | 0.619, 0.799, 0.839, 1.117 |
| 1 | 0.561, 0.947, 0.729, 1.052 |
| 2 | 0.490, 0.934, 0.774, 1.084 |
| 3 | 0.595, 0.830, 0.798, 1.083 |
| 4 | 0.560, 0.852, 0.828, 1.065 |

五個 seed 收斂到相近的模式：位置分量被壓到 0.5–0.6，角速度分量升到 1.05–1.12。策略把
輸入頻寬重新分配到角速度上——對 CartPole 而言合理，而且跨 seed 一致，不是雜訊。

---

## F. 這份結果的限制（請連同 A 節一起讀）

1. **超參數不對等。** QPG 的 `lr_variational=0.01`、`episodes_per_update=16`、
   `max_grad_norm=1.0` 都是第一次猜的值，`configs/qpg.yaml` 裡標了 GUESS。QDQN 那邊是多輪
   調整後的結果。所以「PG 5/5 vs DQN 3/5」**不能**直接讀成「PG 比較強」，比較保守的讀法是
   「PG 對超參數沒那麼敏感」——但這本身也還沒有證據，因為我沒有對 PG 做敏感度掃描。
2. **5 個 seed。** `報告.md` 對自己的 A/B 也寫了同一句話：5 個 seed 不足以讓差異本身有
   統計意義。seed 內 std 0.6 vs 57.8 這種量級的差距比較難用運氣解釋，但 21% 的 sample
   efficiency 差距完全在雜訊範圍內。
3. **只有 CartPole。** 這個環境的 reward 全是 +1、回合長度上限 500，正好是 baseline 減法
   最有效的情境。換到 reward 稀疏或有負值的環境，結論不保證延續。
4. **沒有跑 finite-shot。** `evaluate_finite_shot` 只接 `VQCQFunction`，PG 這邊沒有對應的
   噪聲/取樣評估路徑。所有數字都是無限 shot 的 statevector。
5. **`make_figures.py` 沒有涵蓋 PG。** 它以 `--qdqn-config` 依名字載入並建構 `VQCQFunction`，
   維持 DQN 專用。PG 的圖還沒接。`scripts/summarize.py` 和 `src/plots.py::load_results`
   則不用改就能吃 PG 的 CSV。

---

## G. `results/` 裡哪些是這次的數據

| 檔名樣式 | 內容 |
|---|---|
| `qpg_{0..4}.csv` | 量子 PG 每集訓練紀錄，10 分鐘上限 |
| `qpg_{0..4}_eval.json` | 結尾 100 集 greedy 評估 + 兩個 solve 判定 |
| `qpg_{0..4}_final.pt` | state_dict |
| `qpg_{0..4}_weights.pt` | 後端中立權重（lam / beta_raw / 電路權重） |
| `mlp_pg_{0..4}.csv` / `_eval.json` / `_final.pt` | 古典 PG 對照 |

CSV 表頭是 DQN 那 12 欄原封不動，再往後接 `entropy`、`beta` 兩欄。`epsilon` 欄一律留白
（PG 沒有探索排程，寫 0.0 會在圖上畫出一條意義完全不同的水平線），`w` 欄也留白
（policy 沒有輸出縮放權重）。因為兩個下游工具都是**依欄名**取值，往後追加不影響它們。

---

## H. 重現方式

```bash
# 單一 run（量子 / 古典）
python scripts/train_pg.py --config configs/qpg.yaml    --seed 0 --max-wall-clock-s 600
python scripts/train_pg.py --config configs/mlp_pg.yaml --seed 0 --max-wall-clock-s 600

# 5 seeds（sweep_seeds.py 會依 config 的 model type 分派到 PG trainer）
python scripts/sweep_seeds.py --config configs/qpg.yaml --seeds 0 1 2 3 4

# 彙整（可在 sweep 進行中執行）
python scripts/summarize.py "" --config qpg
python scripts/summarize.py "" --config mlp_pg
# 註：--vs 只能比較同一個 config 底下的兩個 tag，無法跨 config 對照。
# qpg vs qdqn 的並排是分別跑兩次後人工對照的（本報告 C 節）。

# 驗證 policy 走的是同一條編譯路徑、梯度沒死
python -m src.models.vqc_policy
python -m src.pg_trainer
```

---

## I. 環境

與 `報告.md` 相同：conda env `qdqn-qtm`，Python 3.12.13、torch 2.2.2+cpu、
gymnasium 1.3.0、qiskit 1.0.2。所有計時為單機 CPU。

---

## J. 建議的下一步

1. **對 QPG 做超參數敏感度掃描**（`lr_variational`、`episodes_per_update`）。這是把
   「PG 對超參數沒那麼敏感」從猜測變成結論的唯一辦法，也是 F 節第 1 點的直接補救。
2. **把 sample efficiency 的 21% 差距做到有統計意義**——10 個以上 seed，量子 vs 古典 PG。
   這是目前最接近「量子有優勢」的一項觀察，值得認真驗。
3. **接 PG 的 finite-shot 評估路徑**，讓 `evaluate_finite_shot` 也吃 `VQCPolicy`。無限 shot
   的結果對硬體不具代表性。
4. **把 `報告.md` F 節的震盪分析用 PG 重做一次**。那節的結論是震盪不是量子模型造成的；
   PG 這邊 5/5 且 std 0.6，等於是該結論的一個獨立佐證——值得寫進去。
