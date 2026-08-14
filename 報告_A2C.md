# A2C on CartPole-v1 — actor/critic 網格報告

**資料截止：2026-08-13 15:27。四個網格 cell（qa2c / q2c / a2q / mlp_a2c）各 5 seeds，
外加 re-uploading ablation 各 5 seeds，共 30 個 run，數字全為本機實測。**

本份與 `報告.md`（QDQN，value-based）、`報告_PG.md`（REINFORCE，policy-based）並列，不取代
任何一份。三份共用同一顆電路、同一個 `torch_sv` 後端、同一套評估協定，所以差異可以歸因到
演算法而不是模型。本份要回答的是一個問題：**把 REINFORCE 的 batch-mean baseline 換成一個
學出來的 V(s)，值不值得。**

---

## A. 摘要

答案是：**在 CartPole 上不值得，而且量子 actor 這邊是明確變差。**

| | QPG（無 critic） | Q2C（最好的 critic） |
|---|---|---|
| critic 品質（explained variance 峰值中位數） | 無 critic | **0.965** |
| greedy solve | 5/5 | 5/5 |
| 環境互動中位數 | **191,598** | 428,931 |
| 每個 seed 的環境互動範圍 | **173k – 215k** | 306k – 662k |

兩個範圍**完全不重疊**。網格裡品質最好的 critic（q2c 的古典 critic，explained variance 峰值
0.965，幾乎是完美的價值估計）配上同一顆量子 actor，在**每一個 seed** 上都比沒有 critic 的
REINFORCE 需要更多環境互動。這不是雜訊能解釋的方向。

其餘三個要點：

1. **四個 cell 在 training 標準上全部 5/5 通過**，greedy 標準只有 Q2Q 掉一個 seed（4/5）。
   所以「哪一格最好」這個問題在 CartPole 上得到的是「都能解，差別在成本」。
2. **成本完全由 actor 決定，critic 幾乎免費**——因為 actor 每個環境步跑一次（約 20–66 萬次），
   critic 每個梯度步才跑一次（約 100 次）。負載歸一後的實測：換成量子 critic 每個環境步
   多 0.016 ms，換成量子 actor 多 0.351 ms——**actor 貴 22 倍**，而且兩者可加。詳見 F 節。
3. **data re-uploading 的 ablation 是一個否定結果。** 關掉之後 qpg 和 qa2c 都還是 5/5 全解，
   所以「re-uploading 是我們與 Kölle et al. 差異的原因」這個假設被自己的數據推翻。詳見 G 節。

---

## B. 網格是什麼、參數怎麼配平

沿用 Kölle et al. 2024（arXiv:2401.07043）的命名：**第一個字母是 actor，第二個是 critic**。

| cell | actor | critic | config | actor 參數 | critic 參數 | 合計 |
|---|---|---|---|---|---|---|
| **Q2Q** | 量子 | 量子 | `configs/qa2c.yaml` | 45 | 45 | 90 |
| **Q2C** | 量子 | 古典 | `configs/q2c.yaml` | 45 | 44 | 89 |
| **A2Q** | 古典 | 量子 | `configs/a2q.yaml` | 44 | 45 | 89 |
| **A2C** | 古典 | 古典 | `configs/mlp_a2c.yaml` | 44 | 44 | 88 |

四格的總參數量落在 88–90，**最大差距 2.3%**，遠在「±20% 內視為對等」的門檻裡。這是刻意配的：
`MLPPolicy(hidden=6)` = 44、`MLPValue(hidden=7)` = 44，就是為了對上 VQC 的 45/45。

除了 actor/critic 的種類，四格的其他設定逐項相同：γ=0.99、`n_envs=8`、
`episodes_per_update=16`、GAE λ=0.95、`normalize_advantages=true`、`value_coef=0.5`、
Huber value loss、`entropy_coef=0.0`、`max_grad_norm=1.0`、600 秒訓練時鐘上限。量子側一律
4 qubits × 5 layers、re-uploading、`torch_sv` + adjoint。

---

## C. 量子 actor 和量子 critic 是怎麼做的

### C.1 兩顆獨立的電路，共用 ansatz 不共用參數

`VQCPolicy` 和 `VQCValue` 都直接 import `vqc.py` 的 `build_circuit` 和 `make_backend`——不是
平行實作，是同一個建構函式。它們是**兩顆各自編譯的電路、兩組各自的權重**，只有 ansatz 的
形狀相同。差別只在讀出：

| | actor（`VQCPolicy`） | critic（`VQCValue`） |
|---|---|---|
| observables | `["ZZII", "IIZZ"]`，每個動作一個 | `"ZZZZ"`，單一個讀整個暫存器 |
| head | `beta * <O>` → softmax | `w * (<O>+1)/2` |
| 為什麼 | softmax 平移不變，只有 logit **差值**有意義，所以一個逆溫度就夠 | V\* ≈ 99，必須能表示**絕對量值**，所以需要能爬到 ~100 的 `w` |
| 參數 | 40 電路 + 4 lam + 1 beta = 45 | 40 電路 + 4 lam + 1 w = 45 |

單一觀測量選 `ZZZZ` 而不是單 qubit 的 Z，理由跟 actor 選兩體觀測量一樣：單 qubit 的 Z 只看
一個 qubit 的邊際分佈，等於把整個暫存器的相關性丟掉。

**選 separate critic 而不是在 actor 電路上多讀一個觀測量，是刻意的。** 多讀一個觀測量會讓
policy 和 value 共用同一組電路權重，兩個目標函數的梯度會互相拉扯，而且沒有辦法分開調
learning rate。代價是每個梯度步多一次電路評估——F 節會說明這個代價實際上小到量不出來。

### C.2 GAE(λ) 與一個會靜默失效的地方

```
delta_t = r_t + gamma * V(s_{t+1}) * nonterminal_t - V(s_t)
A_t     = delta_t + gamma * lambda * nonterminal_t * A_{t+1}
```

λ 在單步 TD（λ=0，低變異、被 critic 的錯誤污染）和 Monte-Carlo return（λ=1，就是 REINFORCE
的估計量）之間插值。critic 迴歸的目標是 `A_t + V(s_t)`。

**truncation 和 termination 必須分開處理，否則 critic 會被教成反的。** 一集因為桿子倒了而
結束，最後那個 state 的真實 return 就是 0；一集因為打到 500 步上限而結束，桿子還立著，那個
state 值很多。而一個訓練好的 CartPole agent **每一集都是 truncation**——所以把 truncation
當成 terminal，等於在最關鍵的地方告訴 critic「你能到達的最好狀態一文不值」。`_collect_round`
分開記錄兩種情況，`compute_gae` 在 truncation 時用 `V(final_state)` bootstrap、termination 時
用 0.0。

這一段在 REINFORCE 版本裡不存在（Monte-Carlo return 不需要 bootstrap），是 A2C 這條路獨有的
正確性細節。

### C.3 兩個 critic 的 scale 退化 bug——是量出來的，不是想出來的

**量子 critic。** 初版 `w` 從 1.0 起跳、`lr_critic_output_scaling=0.1`。實測 10 個梯度步後
`w` 只走到 2.00。A2C 全程只有約 60–190 個梯度步，所以 `w` 永遠爬不到 V\* ≈ 99。結果是 critic
輸出實質上是個常數，**GAE 的 baseline 悄悄退化回 REINFORCE 的 batch mean**——程式不會報錯，
曲線也還是會上升，只是 A2C 的機制根本沒在運作。修法是把 `w_init` 設成
`value_scale(gamma, max_steps) = (1-γ^500)/(1-γ) = 99.34`。explained variance 從 0.056 升到
0.31–0.50。

**古典 critic 有一模一樣的 bug，而且我第一次的修法是錯的。** 我先把 `MLPValue` 的輸出 bias
設成 V\*，結果沒有任何改善——因為**一個常數預測器的 explained variance 恰好是 0**，
`1 - Var(target - V)/Var(target)` 在 V 為常數時分子分母相等。真正的原因是輸出**權重**約 0.4，
要張開 [0, V\*] 的值域需要約 14。正確修法是讓 `MLPValue` 直接沿用 `VQCValue` 的
`w * sigmoid(·)` head。explained variance 從 0.004 / 0.000 升到 0.018 → 0.106，之後在完整
run 上達到 0.9 以上。

記這一段是因為它是這次工作裡唯一一個**兩種模型都中、而且只有靠測量才會發現**的錯誤。

---

## D. 結果

`t(s)` 是**訓練時鐘**（評估時間已扣除，與另外兩份報告同一套量法）。中位數取 5 個 seed。

### D.1 四個 cell

| cell | actor | critic | greedy solve | training solve | greedy 中位數 | episodes | env steps | 梯度步 | t(s) | s/梯度步 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Q2Q** `qa2c` | Q | Q | 4/5 | 5/5 | 500.0 | 1,477 | 335,443 | 93 | 399 | 4.29 |
| **Q2C** `q2c` | Q | C | **5/5** | 5/5 | 500.0 | 3,067 | 428,931 | 192 | 174 | 1.07 |
| **A2Q** `a2q` | C | Q | **5/5** | 5/5 | 500.0 | 1,531 | 340,415 | 96 | 22 | 0.22 |
| **A2C** `mlp_a2c` | C | C | **5/5** | 5/5 | 500.0 | 1,318 | 224,043 | 83 | 11 | 0.13 |

Q2Q 掉的那個 seed 是 seed 3：greedy 469.53 ± 45.33，低於 475 的門檻。它在 training 標準上是
過的（第 1224 集達標）。其餘 19 個 run 的 greedy 都是 500.0 ± 0.0，只有 Q2Q seed 4 是
496.59 ± 22.60。

> ⚠️ **`t(s)` 與 `s/梯度步` 這兩欄不可跨 config 直接比較。** Q2Q 的 399 s 是主機負載造成的
> 假象——負載歸一後它其實比 Q2C 便宜（約 139 s vs 167 s）。完整證據在 F.1，更正在 I 節第 1 點。
> `episodes` / `env steps` / `梯度步` 三欄不受影響。

### D.2 放進三個演算法一起看

| 設定 | 演算法 | actor | critic | greedy 中位數 | solve | env steps | 梯度步 | t(s) |
|---|---|---|---|---|---|---|---|---|
| `qpg` | REINFORCE | Q | – | 500.0 | 5/5 | **191,598** | 60 | 275 |
| `qa2c` (Q2Q) | A2C | Q | Q | 500.0 | 4/5 | 335,443 | 93 | 399 |
| `q2c` | A2C | Q | C | 500.0 | 5/5 | 428,931 | 192 | 174 |
| `a2q` | A2C | C | Q | 500.0 | 5/5 | 340,415 | 96 | 22 |
| `mlp_a2c` (A2C) | A2C | C | C | 500.0 | 5/5 | 224,043 | 83 | 11 |
| `mlp_pg` | REINFORCE | C | – | 500.0 | 5/5 | 242,470 | 80 | 30 |
| `qdqn` `vec10` | DQN | Q | – | 477.5 | 3/5 | 128,000 | 12,775 | 494 |
| `mlp_baseline` | DQN | C | – | 500.0 | 5/5 | 128,000 | 128,372 | 222 |

**在量子 actor 這一側，加 critic 讓 sample efficiency 變差：** 191,598 → 335,443（Q2Q，1.75×）
→ 428,931（Q2C，2.24×）。在古典 actor 那一側則大致打平：242,470 → 224,043（0.92×）。

> ⚠️ 這張表的 `t(s)` 同樣不可跨列比較（見 F.1）。特別是 `qpg` 的 275 s 和 `mlp_pg` 的 30 s
> 都跑在污染時段——後者是 `mlp_a2c`（11 s）看起來比同樣古典的 `mlp_pg` 快 3 倍這個怪現象的
> 全部原因。**可跨列比較的是 `env steps` 與 `梯度步`。**

---

## E. critic 到底有沒有做事

`explained_variance = 1 - Var(target - V) / Var(target)`。**0 代表跟預測 batch 平均一樣好**
——也就是 REINFORCE 的 baseline，所以停在 0 附近的 run 等於 A2C 的機制沒有在賺它的成本。

| cell | critic | ev 峰值中位數 | ev 末值中位數 | value loss（首 → 末，中位數） |
|---|---|---|---|---|
| Q2Q | 量子 | 0.804 | 0.144 | 38.0 → 2.3 |
| Q2C | **古典** | **0.965** | 0.306 | 25.5 → 0.2 |
| A2Q | 量子 | 0.612 | 0.190 | 42.2 → 2.0 |
| A2C | 古典 | 0.760 | 0.393 | 24.6 → 0.1 |

三件事：

1. **古典 critic 明顯比量子 critic 準**（峰值 0.965 / 0.760 vs 0.804 / 0.612）。這符合預期：
   critic 只有約 100 個梯度步可以學一個到達 99 的函數，量子電路在這個 deadline 下吃虧。
2. **每一格的 ev 都在訓練後期崩掉。** 最極端的是 Q2Q seed 0：峰值 0.816 → 末值 **−0.892**
   （比預測平均值還糟）。機制大概是：policy 收斂後幾乎每集都跑滿 500 步，return 的變異數
   趨近 0，於是 `Var(target)` 這個分母塌掉，ev 對殘差變得極度敏感。**這是一個 CartPole 特有的
   量測假象，不應該當成 critic 崩潰的證據。**
3. **critic 準不代表學得快。** Q2C 有全網格最好的 critic，卻用掉最多環境互動（429k）和最多
   梯度步（192）。這是本報告最反直覺的一點，見 A 節。

### 為什麼「更好的 baseline」反而更慢——三個候選解釋，都還沒驗證

- **GAE 早期是有偏的。** λ=0.95 的估計量會把 critic 早期的錯誤直接寫進 advantage 的方向。
  REINFORCE 的 batch mean 雖然是很爛的 baseline，但它**不會指錯方向**——它只縮放，不旋轉。
- **`normalize_advantages=true` 與 critic 交互作用。** advantage 被逐 batch 標準化之後，
  「baseline 更準」帶來的變異數降低有一部分被正規化吃掉，剩下的主要是偏差。
- **量子 actor 對梯度方向的錯誤特別敏感。** 這只是猜測，而且**目前既沒有支持也沒有反證**：
  A2Q（古典 actor + 量子 critic）的環境互動中位數 340k 也高於 MLP-PG 的 242k，但兩者的 seed
  範圍重疊（174k–477k vs 190k–269k），分不出來。

三個都需要 λ 的 sweep（λ=1.0 應該退化回 REINFORCE）和關掉 advantage 正規化的對照才能分辨。
**目前沒有跑，所以以上三點是候選假設，不是結論。**

---

## F. 成本的不對稱：actor 貴，critic 幾乎免費

這是整份報告在工程上最有用的一點，而且它是結構性的，不是這台機器的性質。

| | 一個 run 裡被呼叫幾次 | 量級 |
|---|---|---|
| **actor** | 每個環境步一次（收集時以 `n_envs=8` 批次化）＋ 每個梯度步一次 | 20–66 萬次環境步 |
| **critic** | **只有每個梯度步一次**（一次吃掉整個 round 的所有 state） | 約 80–190 次 |

### F.1 為什麼不能直接比訓練時鐘

**因為訓練時鐘被主機負載污染了，最多到 2.6×。** 證據是逐 run 的**每環境步毫秒數**（把每個
run 切成四等分，取收斂後的最後 1/4）：

| run | actor | critic | reup | 執行時段 | ms/env-step（Q4 中位數） |
|---|---|---|---|---|---|
| `mlp_a2c` | C | C | – | 13:39–13:41 | **0.039** |
| `a2q` | C | Q | on | 14:19–14:21 | **0.055** |
| `q2c` | Q | C | on | 13:50–14:17 | **0.390** |
| `qa2c` seed 3,4 | Q | Q | on | 13:29–13:37 | **0.415** |
| `qa2c` seed 0,1,2 | Q | Q | on | 12:50–13:29 | 1.035 ← 污染 |
| `qpg` | Q | – | on | 11:01–11:55 | 1.057 ← 污染 |
| `mlp_pg` | C | – | – | 12:00–12:04 | 0.113 ← 污染 |
| `qpg_noreup` | Q | – | off | 14:23–14:41 | 0.300 |
| `qa2c_noreup` | Q | Q | off | 14:52–15:26 | 0.362 |

`q2c` 和 `qpg` 用的是**完全相同的量子 actor 電路**，`qa2c` 的 seed 0–2 和 seed 3–4 是**同一個
config 的同一批 run**。前者差 2.7×、後者差 2.5×，而且差距在**四個區段上都一致**，也不能用
batch 大小解釋（每 round 的 state 數只差 1.36×）。對照時間戳，11:00–13:29 那個時段機器上有
其他負載。**13:29 之後的 run 彼此才可比。**

### F.2 在同一個時間窗內量到的不對稱

只取 13:29–14:21 這個連續時段的四個 run（`qa2c` seed 3–4、`mlp_a2c`、`q2c`、`a2q`）：

| 換掉什麼 | ms/env-step | 相對全古典的增量 |
|---|---|---|
| 全古典（A2C） | 0.039 | — |
| **critic** 古典 → 量子（A2Q） | 0.055 | **+0.016** |
| **actor** 古典 → 量子（Q2C） | 0.390 | **+0.351** |
| 兩個都換（Q2Q） | 0.415 | +0.376 |

**量子 actor 的成本是量子 critic 的 22 倍**（0.351 / 0.016）。而且兩個增量幾乎完美可加：
0.016 + 0.351 = 0.367，實測 Q2Q 是 0.376，差 2.5%——這正是「actor 每個環境步跑、critic 每個
梯度步跑」這個結構所預測的。

**在這個模擬器上，量子 critic 是便宜的、量子 actor 是貴的。** 如果要在硬體或更貴的模擬器上
做 hybrid，這個不對稱直接告訴你該把電路放在哪一邊——放在 critic 幾乎不影響成本。

反過來說，這也意味著 **A2Q（古典 actor + 量子 critic）是「最划算」的量子 cell**：它是網格裡
5/5 全解、成本只比全古典高 41% 的量子參與方案。但它同時也是量子成分貢獻最少的一格——
量子電路只負責一個在評估時**完全不參與**的函數（greedy play 只看 actor 的 logits）。

### F.3 D.1 那張表的訓練時鐘要怎麼讀

Q2Q 的 399 s 中位數**是被污染的**：它的中位數落在 seed 0–2 那三個慢的 run 上。用 seed 3–4
的速率反推，Q2Q 在乾淨的機器上約是 335,443 × 0.415 ms ≈ **139 s**，比 Q2C 的
428,931 × 0.390 ms ≈ 167 s **還便宜**——因為它需要的環境互動比較少。

**所以「Q2Q 比 Q2C 貴」這個從訓練時鐘讀出來的印象是錯的，而且方向相反。**

---

## G. re-uploading ablation：假設錯了，而這就是結果

我先前的說法是：「我們的純量子 agent 能解 CartPole 而 Kölle et al. 的不能，最可能的原因是
data re-uploading」——因為那篇論文自己的討論就點名 re-uploading 是可能的修法。這個假設**被
我們自己的 ablation 推翻**。

| 設定 | re-uploading | greedy solve | greedy 中位數 | env steps | 梯度步 | t(s) |
|---|---|---|---|---|---|---|
| `qpg` | ON | 5/5 | 500.0 | 191,598 | 60 | 275 |
| `qpg_noreup` | **OFF** | **5/5** | **500.0** | 226,227 | 68 | 95 |
| `qa2c` (Q2Q) | ON | 4/5 | 500.0 | 335,443 | 93 | 399 |
| `qa2c_noreup` | **OFF** | **5/5** | **500.0** | 511,887 | 168 | 265 |

**關掉 re-uploading，10 個 run 全部在兩個標準上通過，greedy 全部 500.0 ± 0.0。** 它不是我們
能解而他們不能的原因。剩下的候選解釋有三個——可訓練的 `lam`（輸入縮放）＋輸出縮放、兩體
`ZZ` 觀測量、以及最單純的「優化配置不同」（我們 5 層他們 2 層，learning rate 與 optimizer
不同）。**第三個目前領先，但是靠排除法領先，沒有正面證據。**

兩個附帶觀察：

- **成本：re-uploading 大約貴 15–30%，不是先前說的 1.5–2.9×。** 上表的訓練時鐘
  （qpg 275→95 s，Q2Q 399→265 s）看起來像 2.9×，但那是 F.1 的主機負載造成的：`qpg` 跑在
  11:01–11:55 的污染時段。用 F.1 的 ms/env-step 比較——`q2c`（reup ON）0.390 vs
  `qpg_noreup`（OFF）0.300，以及 `qa2c` seed 3–4（ON）0.415 vs `qa2c_noreup`（OFF）0.362——
  真正的差距是 **15–30%**。這個量級也才合理：re-uploading 在 5 層裡多加 4 組編碼閘，
  約佔全部閘數的 27%。**先前寫的 2.9× 是量測假象，在此更正。**
- **critic 反而變好：** `qa2c_noreup` 的 ev 末值中位數是 0.818，遠高於 `qa2c` 的 0.144；
  峰值 0.945 vs 0.804。這一格沒有解釋，也沒有再追。

---

## H. 與 Kölle et al. 2024 的關係

| | Kölle et al. | 本專案 |
|---|---|---|
| ansatz 層數 | 2 | 5 |
| data re-uploading | 無（討論中點名為可能的修法） | 有（**且 G 節顯示關掉也一樣解得掉**） |
| 可訓練輸入縮放 `lam` | 無 | 有 |
| 可訓練輸出縮放 | 無 | 有（critic `w`，actor `beta`） |
| 讀出 | 單 qubit 測量機率 | 兩體 `ZZ`（actor）／四體 `ZZZZ`（critic） |
| 純量子 cell 的結果 | A2Q / Q2C / Q2Q **全部學不起來**，診斷為梯度消失（平均梯度 −5.6e−5） | 四格全部 5/5 通過 training 標準 |
| 他們的修法 | 加古典後處理層才恢復效能 | 不需要 |

**這不是一次重現，是把同一個實驗跑在一個不同的 ansatz 上。** 兩邊結論相反這件事本身有價值
——它說明那篇論文報告的失敗是**該設定**的性質，不是「純量子 actor-critic 做不到 CartPole」
的性質。但 G 節同時說明：我們還**不知道**是哪個差異造成的。

---

## I. 限制（請連同 A 節一起讀）

1. **D.1 那一欄訓練時鐘帶有最多 2.7× 的主機負載污染，不能直接跨 config 比。** 完整證據與
   修正方式在 F.1；短版是：11:00–13:29 之間跑的 run（`qpg`、`mlp_pg`、`qa2c` seed 0–2）比
   13:29 之後跑的慢 2.5–2.7 倍，而 `q2c` 與 `qpg` 用的是同一顆 actor 電路。**兩個具體更正：**
   （a）Q2Q 的 399 s 是假象，負載歸一後它其實比 Q2C **便宜**（139 s vs 167 s）；
   （b）re-uploading 的成本是 15–30%，不是先前說的 1.5–2.9×（見 G 節）。
   **梯度步數與環境步數完全不受影響，是精確的**——本報告所有科學結論都只建立在這兩者上。
2. **5 個 seed。** 「Q2C 每個 seed 都比 QPG 費更多環境互動」這句因為兩個範圍完全不重疊
   （306k–662k vs 173k–215k）而還算穩；「Q2Q 也一樣」則**不成立**——它的範圍 204k–466k 與
   QPG 重疊。
3. **超參數沒有調過。** 四個 cell 的 `gae_lambda=0.95`、`value_coef=0.5`、各組 learning rate
   都是第一次猜的值，config 裡標了 GUESS。「A2C 在 CartPole 上不值得」比較保守的讀法是
   「**在這組沒調過的超參數下**不值得」。E 節提到的 λ sweep 是最直接的補救。
4. **只有 CartPole。** reward 全是 +1、回合長度上限 500、V\* 幾乎是常數——這正好是**學來的
   baseline 最沒有優勢**的情境，因為 batch mean 已經是很好的近似。換到 reward 稀疏或狀態
   之間價值差異大的環境，結論很可能反轉。**這是 A 節那個結論最重要的限制。**
5. **沒有跑 finite-shot / 含噪模擬。** 所有數字都是無限 shot 的 statevector。
   `src/models/vqc.py` 的 `evaluate_finite_shot` 只接 `VQCQFunction`，policy 和 value 這兩條
   路都還沒接上。這是相對於 challenge「Advanced goal」明列項目的一個真實缺口。
6. **explained variance 末值不可解讀。** 見 E 節第 2 點：policy 收斂後 `Var(target)` 塌掉，
   ev 會變成一個對雜訊極度敏感的量。峰值可解讀，末值不行。

---

## J. 訓練動態

### critic 讓策略收得更緊

| | beta（首 → 末，範圍） | entropy（首 → 末，範圍） |
|---|---|---|
| `qpg`（無 critic） | 0.94–1.06 → 4.45–4.72 | 0.685–0.689 → 0.49–0.53 |
| `qa2c`（Q2Q） | 0.94–1.06 → **5.32–7.09** | 0.685–0.689 → **0.33–0.44** |

五個 seed 一致：有 critic 時 `beta` 推得更高、entropy 壓得更低。兩動作的最大熵是 0.693，
所以 A2C 這邊確實把探索收得比 REINFORCE 更緊。這與 A 節「A2C 用了更多環境互動」並不矛盾——
它更快變得確定，但變確定的過程本身花了更多互動。

**兩邊都沒有 epsilon 排程。** 探索強度是策略自己透過可訓練的 `beta` 退火出來的。

---

## K. 圖

| 檔案 | 內容 |
|---|---|
| `figures/10_grid_reward.{png,pdf}` | 四格網格的 greedy 品質——一眼看出四格都解得掉 |
| `figures/11_algorithms.{png,pdf}` | 同一顆電路三種演算法（DQN / REINFORCE / A2C）＋參數配平的古典對照 |
| `figures/12_critic.{png,pdf}` | explained variance 與 value loss：critic 有沒有賺到它的成本 |
| `figures/13_cost.{png,pdf}` | 梯度步數 × 每步成本的分解，F 節那個不對稱的視覺版 |
| `figures/14_progress.{png,pdf}` | reward 進展（跨 arm 可比）與 loss 進展（**跨演算法不可比**） |
| `figures/15_entropy.{png,pdf}` | 探索自行退火，全程沒有 epsilon 排程 |

曲線一律是 5 個 seed 的中位數 + IQR，x 軸格線停在**最短**的那個 seed，避免尾端只剩一兩個
seed 撐著卻畫成實線。

重畫：`python scripts/plot_grid.py`（直接讀 `results/`，沒有中間檔）。

---

## L. 重現方式

```bash
# 單一 run
python scripts/train_a2c.py --config configs/qa2c.yaml    --seed 0 --max-wall-clock-s 600
python scripts/train_a2c.py --config configs/q2c.yaml     --seed 0 --max-wall-clock-s 600
python scripts/train_a2c.py --config configs/a2q.yaml     --seed 0 --max-wall-clock-s 600
python scripts/train_a2c.py --config configs/mlp_a2c.yaml --seed 0 --max-wall-clock-s 600

# 5 seeds（sweep_seeds.py 依 config 的 model type 分派到 A2C trainer）
python scripts/sweep_seeds.py --config configs/qa2c.yaml --seeds 0 1 2 3 4

# re-uploading ablation
python scripts/sweep_seeds.py --config configs/qpg_noreup.yaml  --seeds 0 1 2 3 4
python scripts/sweep_seeds.py --config configs/qa2c_noreup.yaml --seeds 0 1 2 3 4

# 彙整與圖
python scripts/summarize.py "" --config qa2c
python scripts/plot_grid.py

# 模組自檢（GAE 的 truncation/termination 分支、critic 梯度沒死）
python -m src.a2c_trainer
python -m src.models.vqc_value
```

**所有 sweep 都是嚴格序列執行的**，因為 wall-clock 是本專案的頭條指標之一——但 I 節第 1 點
說明，這仍然擋不掉機器上其他負載造成的污染。

---

## M. `results/` 裡哪些是這次的數據

| 檔名樣式 | 內容 |
|---|---|
| `qa2c_{0..4}.csv` / `q2c_` / `a2q_` / `mlp_a2c_` | 每集訓練紀錄，600 秒訓練時鐘上限 |
| `*_eval.json` | 結尾 100 集 greedy 評估 + 兩個 solve 判定 |
| `*_final.pt` / `*_critic.pt` | actor 與 critic 的 state_dict（分開存，因為它們是兩個模型） |
| `*_weights.pt` | 後端中立權重，有量子模型的 run 才有：`qa2c` / `q2c` 是 actor 的，`a2q` 是 critic 的，`mlp_a2c` 沒有 |
| `qpg_noreup_{0..4}` / `qa2c_noreup_{0..4}` | re-uploading ablation |

CSV 表頭是 `PG_CSV_HEADER` 再往後接 `value_loss`、`explained_variance` 兩欄。追加、不替換——
`scripts/summarize.py` 和 `src/plots.py::load_results` 都是**依欄名**取值，所以尾端加欄位對
它們是隱形的。

---

## N. 環境

與另外兩份報告相同：conda env `qdqn-qtm`，Python 3.12.13、torch 2.2.2+cpu、gymnasium 1.3.0、
qiskit 1.0.2。所有計時為單機 CPU。訓練路徑上沒有任何 qiskit primitive——actor 和 critic 都走
`compile_circuit` + `simulate` 的純 torch autograd 路徑。

---

## O. 建議的下一步

1. **λ sweep（λ ∈ {0.0, 0.5, 0.95, 1.0}）在 `q2c` 上跑。** λ=1.0 應該退化回 REINFORCE 的
   估計量；如果 sample efficiency 隨 λ 單調恢復，E 節的第一個候選解釋就從假設變成結論。
   這是本報告最值得補的一個實驗。
2. **關掉 `normalize_advantages` 再跑一次 `q2c`**，分辨 E 節的第二個候選解釋。
3. **接上 finite-shot 評估路徑**（讓 `evaluate_finite_shot` 也吃 `VQCPolicy` / `VQCValue`）。
   這是 I 節第 5 點的直接補救，也是相對於 challenge 明列項目的唯一真實缺口。
4. **換一個 baseline 不是好近似的環境**（reward 稀疏，或狀態價值差異大）重跑這個網格。
   A 節的結論在 CartPole 上成立，但 I 節第 4 點說明那多半是 CartPole 的性質。
5. **把 wall-clock 量測隔離**（固定 CPU affinity，或至少記錄同時段的系統負載），讓 I 節第 1
   點那種 3× 污染不會再發生。
