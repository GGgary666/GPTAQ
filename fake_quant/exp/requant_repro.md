# ReQuant 复现记录

论文：[ReQuant: Fixed-Grid Discrete Refinement for Post-Training Quantization](https://arxiv.org/abs/2608.07019)（arXiv 2608.07019v1）
分支：`requant_repro`　宿主：GPTAQ `fake_quant/`　官方代码：未发布

---

## 1. 结论摘要

- 算法实现与论文 Eq.3–10 及 Appendix A.1 **逐项一致**，无任何论文外的超参。
- 量化基线精确复现论文：Qwen3-14B W4A16 + QuaRot，FP **8.648**（论文 8.65），RTN **10.107**（论文 10.22）。
- 曾出现 ReQuant 让 PPL 从 11.67 恶化到 64.1。根因**不在公式**，而是工程上破坏了 GPTAQ 的分组顺序依赖（见 §4.1）。修复后 Qwen3-14B 得到 **9.179**（论文 8.85）。
- 修复前为了「救回」结果引入的交叉项缩放 `α=0.25` 是伪解，**不是论文的一部分**，修复后应弃用。

---

## 2. 论文算法与实现的对应

论文对一层线性层，把 GPTAQ 的激活感知目标放到冻结网格上做离散坐标下降：

| 论文 | 含义 | 实现位置 |
|---|---|---|
| Eq.3 | `min_{Wq∈G} ‖WX − Wq X̃‖²_F` | `requant_utils.refine_weight` |
| Eq.4 | 按输出行可分解，行间独立 | 向量化到 `[drow, dcol]`，行维并行 |
| Eq.5 | `L(e) = ‖e X̃ − w ΔX‖²`，`ΔX = X̃ − X` | `_objective` |
| Eq.6 | `g = 2(e H̃ − w B)`，`B = ΔX X̃ᵀ`，`H̃ = X̃ X̃ᵀ` | `refine_weight` 的 `G` |
| Eq.7 | `ΔL = −Δq_j g_j + (Δq_j)² H̃_jj` | 内层 `delta_L` |
| Eq.8/9 | 候选 `Δq_j = k·s_j`，`1 ≤ |k| ≤ K`，且 `z_min ≤ z_j+k ≤ z_max` | `ReQuantGrid.candidates` |
| Eq.10 | `g ← g − 2 Δq*_j H̃_{j,:}` | `G.addr_(dq, H[j], alpha=-2.0)` |
| A.1 | `L = e H̃ eᵀ − 2 e Bᵀ wᵀ + w ΔX ΔXᵀ wᵀ` | `_objective` 返回 `(total, quad, cross, const)` |

关键点：

- `X` 由**全精度模型**前向得到，`X̃` 由**已量化前缀**的模型前向得到。两者都是真实前向采集，不是近似。
- 第三项 `const = w ΔX ΔXᵀ wᵀ` 与 `e` 无关，不参与下降，但**必须算**——它是判断实现是否自洽的唯一手段（见 §4.2）。
- 论文设定：WikiText-2，512 条 × 2048 token；权重 per-channel 非对称；激活 per-tensor 非对称；默认 `T=4`、`K=2`；主表均带 QuaRot 旋转。
- **论文全文没有任何交叉项缩放系数。**

---

## 3. 工程实现

### 3.1 环境

| 项 | 值 |
|---|---|
| transformers | 5.9.0（直接适配，未降级） |
| 显存 | Qwen3-14B、512×2048 校准、`--requant_chunk 32` 下约 110 GB 峰值 |
| 模型 | Qwen3-1.7B / 8B / 14B |
| 数据 | WikiText-2（支持本地离线加载） |

### 3.2 各文件改动

**`requant_utils.py`**（新增，核心）
- Algorithm 1 的向量化实现，行维全并行。
- `ReQuantGrid` 抽象 + `AffineReQuantGrid` / `NVFP4ReQuantGrid`，统一支持仿射 INT 网格与 NVFP4 的非均匀 E2M1 网格。
- 内层循环全程在**整数码空间**操作，消除逐列的 host↔device 同步。
- `ReQuant` 统计类累积 `H̃`、`B`、`C = ΔX ΔXᵀ`，以及漂移诊断 `‖ΔX‖/‖X̃‖`。
- 校准 token 数少于列数时告警（`H̃` 秩亏，必然过拟合）。

**`hadamard_utils.py`**
- `fast_hadamard_transform` CUDA 扩展缺失时回退到纯 PyTorch。
- 修 `view` 在非连续张量上报错：改为 `contiguous().reshape()`。
- **新增 Paley I 构造 `get_hadPaley`**：Sloane 表止于 172 且跳过 68，而 Qwen3-14B 的 `intermediate_size = 17408 = 68 × 2⁸` 无法分解。67 是素数且 ≡ 3 (mod 4)，用模 67 的二次剩余可直接构造 68 阶 Hadamard 矩阵。已断言 `H Hᵀ = 68 I`，且不影响任何原有维度的分派。

**`model_utils.py`**
- 新增 Qwen3 类型与 `is_llama_like_type` / `is_llama_like_name`。
- QuaRot 需要解绑 tied embedding。
- `unpack_layer_output` / `forward_decoder_layer` 吸收 transformers 各版本 `DecoderLayer.forward` 返回值与签名差异。

**`rotation_utils.py`**
- 改为定向替换 RMSNorm，**避免误替换 Qwen3 的 `q_norm` / `k_norm`**（按类型全局 DFS 会静默破坏模型）。

**`nvfp4_utils.py`**（新增）
- NVFP4：E2M1 15 值网格 + group-16 的 fp8-e4m3 scale + 全局 fp32 scale，对齐 `compressed_tensors` 语义。
- `cast_to_fp4` 用 `torch.bucketize` 替代 `torch.where` 链，约 3× 加速，并用 `nextafter` 精确处理中点 tie-break。

**`quant_utils.py` / `utils.py` / `main.py` / `data_utils.py`**
- 激活量化粒度（per-tensor / per-token）、NVFP4 分支、`--w_format` / `--a_format` / `--seqlen` / `--eval_nsamples` 等 CLI。
- FP 权重快照在**旋转之后**采集（顺序错了交叉项会全错）。
- WikiText-2 支持本地 `save_to_disk` 或 parquet 离线加载。

### 3.3 测试

`tests/test_requant.py`（8）、`tests/test_nvfp4.py`（9）、`tests/test_hadamard.py`（4），全部 CPU 可跑。

关键几个：
- `test_matches_naive_algorithm1`：向量化结果与逐行逐列的朴素 Algorithm 1 **逐码相等**。
- `test_eq7_matches_brute_force`：Eq.7 的闭式 `ΔL` 与暴力重算的损失差一致。
- `test_objective_equals_true_layer_error`：`_objective` 必须等于暴力计算的 `‖wX − qX̃‖²`。
- `test_reported_loss_stays_nonnegative`：Eq.5 是平方和，报告的 loss 不允许为负。
- `test_generalizes_only_with_enough_calibration_tokens`：校准 token 充足时留出集上优于 RTN，不足时反而更差。
- `test_order68_is_hadamard` + `test_dispatch`：Paley-68 正确且不抢占原有维度。

```bash
for t in tests/test_requant.py tests/test_nvfp4.py tests/test_hadamard.py; do python "$t"; done
```

---

## 4. 关键 bug 与排查过程

### 4.1 主 bug：分组顺序依赖被破坏

**现象**　Qwen3-8B W4A16，RTN 11.673 → ReQuant(T=2, α=1) **64.149**。1.7B 上同样恶化。

**排查路径**

1. 先验证基线：Qwen3-14B FP 8.648 / RTN 10.107 对上论文 8.65 / 10.22 ——量化、旋转、评测链路无问题，问题在 ReQuant 本身。
2. 逐项核对代数：`ΔX` 符号、`B` 的转置、`w @ B` 的方向、`2/n` 归一化、FP 快照是否在旋转后 —— 全部正确，且单元测试对暴力枚举成立。
3. 量级不对：`gate_proj` 初始目标值比同 `dcol` 的 `q_proj` 大 100 倍，按 `e* = w B H̃⁻¹` 估算说不通。
4. **补上 `const` 项**后才看清真相（见 §4.2）：总损失恒非负且单调下降，**下降过程是对的**；但分解显示它在拿 `quad` 换 `cross`。

   | Qwen3-8B layer 1 | quad | cross | const | 总计 |
   |---|---|---|---|---|
   | `o_proj` 前 | 286.8 | −6.8 | 1247 | 1527 |
   | `o_proj` 后 | **1115** | −1985 | 1247 | 378 |

5. 定位模式：每层**第一组** `q/k/v_proj` 的 `quad` 正常下降，而紧跟在已 refine 组之后的 `o_proj`、`down_proj` 的 `quad` 上升。
6. 对照 `gptaq_utils.py`：GPTAQ 是 `for names in sequential:` —— **每组单独跑一次量化前向，组内量化完再跑下一组**。

**根因**　此前为提速，把一层内 4 组（`q/k/v` → `o` → `up/gate` → `down`）的统计收集合并成**一次**前向。于是 `o_proj` 的 `X̃` 是在 `q/k/v` **尚未 refine** 时测的。交叉项要抵消的漂移在真正推理时已不存在，坐标下降便忠实地牺牲可泛化的权重保真项，去追一个错误方向：校准损失下降，PPL 爆炸。

**修复**　恢复分组顺序：每组收统计 → refine 该组 → 下一组的前向已能看到上一组的新权重。组内共享输入的模块仍共享 `H̃/B/C`。FP 分支与 refine 无关，可按组重跑，仅最后一组推进 `fp_inps`。代价是每层前向次数增加，约 3×。

### 4.2 次生问题：目标函数缺常数项，导致主 bug 无法被发现

`_objective` 原本省略了与 `e` 无关的 `w ΔX ΔXᵀ wᵀ`。省略不影响 argmin，但报告出的 "loss" **可以为负**（实测 `gate_proj` 得到 −1031），于是：

- 无法判断损失下降是否合理；
- 无法发现 `quad` 正在被卖掉。

现在 `ReQuant` 额外累积 `C = ΔX ΔXᵀ`，`_objective` 返回 `(total, quad, cross, const)`，日志逐模块打印分解，并在 `total < 0` 时告警。这是定位主 bug 的决定性工具，代价是每个采集点多一个 `dcol²` 矩阵。

### 4.3 `α = 0.25`：曾判为伪解，现被数据推翻

主 bug 期间，参考 GPTAQ `fasterquant(alpha=0.25)` 给交叉项加了缩放，`α=0.25` 时结果「变好」且随 `T` 单调。当时的结论是：它只是把坏交易砍掉 3/4，**不是论文内容**（论文全文无此系数），根因修复后应回到 `α=1`。

**这个结论是错的。** Qwen3-8B 上，修复前 `α=0.25` 得到 10.096（回收 RTN→FP 缺口的 81%，与论文量级相当），而修复后 `α=1` 只有 11.549（回收 6%）。`q8_rq_a1` / `q8_rq_a025` 同脚本同代码只差 α（64.149 vs 10.096），修复前的 A/B 是干净的。

需要注意 GPTAQ 里 alpha 的真实语义：`gptaq_utils.add_batch` 对 `H` 和 `dXXT` 施加**完全相同**的 `sqrt(2/n)` 归一化，所以 `fasterquant` 中「scale it by alpha due to collection of dXXT and H」的注释具有误导性——0.25 是阻尼超参，不是归一化修正。而且本文所有实验的 initializer 是 RTN，GPTAQ 的 alpha 不进入该路径；`--requant_cross_alpha` 缩放的是 ReQuant 自己的 `B`。

当前无法归因：10.096 与 11.549 之间同时差了「分组修复」和「α」两个变量。**缺口实验是修复后 `α=0.25`**。若它仍显著优于 `α=1`，则说明忠实的 Eq.5 在本设定下确实不如带阻尼的版本，那不是工程 bug，而是复现的实质性发现。

### 4.4 其他已修

| 问题 | 处理 |
|---|---|
| `ModuleNotFoundError: fast_hadamard_transform` | 纯 PyTorch 回退 |
| `HfUriError: Invalid HF URI 'hf://datasets/wikitext@...'` | 改用 `Salesforce/wikitext` |
| `view size is not compatible with stride` | `contiguous().reshape()` |
| `get_hadK` 对 17408 断言失败 | Paley-68 构造 |
| `add_batch` 靠形状推断 FP 激活方向 | 改为显式约定 `[dcol, tokens]` + 断言 |

最后一项是加固而非修 bug：`_cache_fp_input` 一直存的是 `x.t()`，与 `add_batch` 内部的转置按构造恒等，那段推断分支从未进入过，`dcol == tokens` 的方形情形（Qwen3-1.7B 在 seqlen 2048 下 `hidden_size` 恰为 2048）也是走「形状相等、不变换」这条正确路径。但方形下任何基于形状的猜测都无法区分 `X` 与 `X̃ᵀ`，一旦将来有调用方传入未转置的张量就会静默算错，因此改为约定方向并对不符者报错。`tests/test_requant.py` 加了两个用例：方形输入下 `B` 必须等于直接计算值、且必须不等于用转置 FP 激活算出的值；以及传入 `[tokens, dcol]` 时必须抛错。

---

## 5. 实验结果

统一设定：QuaRot 旋转，权重 INT4 per-channel 非对称 + `--w_clip`，WikiText-2 校准 512×2048，评测为 WikiText-2 全测试集、seqlen 2048。

### 5.1 基线复现（可信）

| 模型 | FP16 | RTN W4A16 | 论文 FP | 论文 RTN |
|---|---|---|---|---|
| Qwen3-14B | **8.648** | **10.107** | 8.65 | 10.22 |
| Qwen3-8B | 9.723 | 11.673 | — | — |

Qwen3-14B 两项均与论文吻合，说明旋转、量化、评测链路正确。

### 5.2 ReQuant

「缺口回收」指相对 FP16 的 PPL 缺口被收回的比例，论文在 Llama-3 8B / Qwen3-14B 上约 75–87%。

| 模型 | 设定 | RTN | + ReQuant | 缺口回收 |
|---|---|---|---|---|
| Qwen3-8B | T=2，**修复前**，α=1 | 11.673 | 64.149 | 崩 |
| Qwen3-8B | T=2，**修复前**，α=0.25 | 11.673 | **10.096** | 81% |
| Qwen3-8B | T=2，修复后，α=1 | 11.673 | 11.549 | 6% |
| Qwen3-14B | T=4，修复后，α=1 | 10.107 | **9.179** | 64% |

论文 Qwen3-14B 对应值为 8.85。当前差 ≈ 0.33 PPL，方向已正确（RTN 相对 FP 的 +1.46 缺口收回到 +0.53）。

但 α=1 在 8B 上只收回 0.12，远弱于 14B 的 0.93，也远弱于同模型上 α=0.25 的 1.58。逐层分解显示它在网络后段大量出现 `quad` 上升的坏交易，见 §6.1；α 的归因见 §4.3。

修复生效的直接证据（Qwen3-14B layer 0，`quad` 恢复下降）：

```
o_proj    : quad 386.4 -> 102.3
down_proj : quad 2005  -> 687.9
```

### 5.3 Qwen3-1.7B 的历史数据（**不可信，仅存档**）

以下全部在主 bug（§4.1）存在时产出，**不能作为 ReQuant 的结论**：

| 设定 | T=0 | α=1 | α=0.25 |
|---|---|---|---|
| NVFP4 W4A16 | 19.400 | 24.217 (T=2) | 17.624 / 17.426 / 17.031 (T=1/2/4) |
| NVFP4 W4A4 | 25.003 | 28.744 (T=2)，29.058 (T=4) | 18.811 (T=2) |
| INT4 W4A16 | 26.828 | 24.481 (T=2) | — |

FP16 = 16.768。注意 INT4 下 RTN 相对 FP 的缺口高达 +10.06，是 8B/14B 的 5 倍以上，本就在论文设定的适用范围之外。

### 5.4 耗时（Qwen3-1.7B NVFP4，内层优化后、分组修复前，单卡）

| 档位 | 耗时 |
|---|---|
| T=1 | 405 s |
| T=2 | 535 s |
| T=4 | 772 s |
| W4A4 T=2 | 553 s |

内层优化（整数码空间 + 消除 host 同步 + `bucketize` 版 `cast_to_fp4`）相对初版 ≥ 8×，且 T=2 复现了优化前的 17.426，逐位一致。分组修复会使前向次数约 3×，需重新计时。

论文报告 Llama-3 8B RTN+QuaRot W4A16 端到端：T=1 50.82 min，T=2 60.00 min，T=4 81.80 min。

---

## 6. 已知问题与待办

### 6.1 中后层的 `quad` 上升坏交易（已判定：漂移是固有的，不改算法）

Qwen3-8B W4A16 上最明显。`o_proj` 随深度的分解：

| `‖ΔX‖/‖X̃‖` | `quad` | `cross` | 总损失 |
|---|---|---|---|
| 0.056 | 180.8 → 57.2 | 0.4 → −9 | 232 → 99 |
| 0.195 | 1.07e4 → **1.65e4** | −444 → −2.2e4 | 4.2e4 → 2.6e4 |
| 0.503 | 6.5e4 → **7.8e5** | −7.7e3 → −1.5e6 | 1.2e6 → 4.1e5 |
| 0.684 | 7.9e5 → **2.4e7** | −1.8e5 → −4.6e7 | 2.9e7 → 6.3e6 |

Eq.5 允许这种交易，总损失确实单调下降，但泛化到推理时是亏的：8B 全程只从 11.673 降到 11.549（0.12），而同样修复后的 14B 降了 0.93。

关键疑点是 0.68 这个漂移量级——一个 PPL 11.5 的 W4A16 模型不该在残差流上带 68% 的相对误差。两种可能：

- **固有**：RTN 自身就漂移到这个程度，交叉项响应的是真实错配，坏交易属于忠实行为，不该改算法；
- **自激**：ReQuant 砸坏后段权重 → 下一层漂移更大 → 交叉项更强势 → 交易更差，形成正反馈。

判别手段：`--requant_sweeps 0` 只收集 `H/B/C` 并打印同样的诊断、不写回任何权重，得到的漂移曲线即初始化器自身的（`scripts/remote_drift.sh`）。该配置跑出的 PPL 为 11.673，与 RTN 基线逐位相同，确认它确实没有改动任何权重。

Qwen3-8B `o_proj` 每 5 层采样的 `‖ΔX‖/‖X̃‖`：

| layer | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 |
|---|---|---|---|---|---|---|---|---|
| T=0（仅 RTN） | 0.080 | 0.324 | 0.511 | 0.451 | 0.536 | 0.592 | 0.649 | 0.695 |
| T=2（ReQuant） | 0.056 | 0.093 | 0.195 | 0.192 | 0.503 | 0.678 | 0.684 | 0.400 |

**结论：漂移以固有为主，但深层 `o_proj` 存在局部倒挂。** 三点：

1. 漂移是 RTN 固有的。不做任何 refine，深层就已经漂到 0.55–0.72（layer 35 T=0 为 0.695）。所以 0.68 不是 ReQuant 造成的，交叉项响应的是真实存在的错配。
2. ReQuant 总体在**降低**漂移。前中段 `o_proj` 压到四分之一（layer 10 从 0.511 到 0.195），`down_proj` 则全深度都降（layer 34 从 0.513 到 0.406）。交叉项和分组修复都在干实事。
3. 但 layer 24–30 的 `o_proj` 出现倒挂：T=2 的漂移**高于** T=0（layer 25 为 0.678 vs 0.592，layer 28 为 0.632 vs 0.552）。这正是 `quad` 被放大 30 倍的位置。深层 `o_proj` 的坏交易确实会把误差原样还回去，只是范围局限于这一带，不是全局正反馈。

因此**不动算法**。给 `quad` 加信任域或对 `cross` 设上限确实能立刻改善 PPL，但既然漂移是固有的，那种改法只是用超参掩盖模型特性，而且属于改论文。

遗留的真问题变成：ReQuant 在 8B 前中段明显压低了漂移，PPL 却几乎没动（11.673 → 11.549）。说明主导输出质量的是后段，而后段恰好是漂移压不下去、`quad` 又被牺牲最多的区域。这是模型特性还是网格选择的后果，取决于下一项。

### 6.2 `--w_clip` 是否解释剩余差距（已否定）

当前 RTN 用了 `--w_clip`（MSE 搜索裁剪），论文未提及 RTN 做裁剪。我们的 RTN 基线 10.107 优于论文 10.22，而 ReQuant 后 9.179 差于论文 8.85——起点更好、终点更差。由于 ReQuant **冻结网格**，网格不同则终点不可比，所以补了一组不带 `--w_clip` 的对照。

| Qwen3-14B RTN W4A16 | PPL |
|---|---|
| 带 `--w_clip`（当前设定） | **10.107** |
| 论文 | 10.22 |
| 不带 `--w_clip` | 11.858 |

去掉裁剪后偏离 1.64，远差于论文；带裁剪只差 0.11。**论文的 RTN 显然是带裁剪的，`--w_clip` 应当保留，它不是 0.33 差距的来源。**

不过这组对照还有第二重用途，正在跑：论文从 10.22 收回到 8.85（1.37），我们从更好的 10.107 只收回到 9.179（0.93）——起点越差反而收得越多。用 11.858 这个更差的网格跑 ReQuant T=4，可以直接检验「起点差留给 ReQuant 的空间就大」这一解释是否成立。

### 6.3 待跑

1. Qwen3-8B 修复后 T=2：已完成，11.549（见 §5.2）。
2. `T ∈ {0,1,2,4,8}` 单调性，对齐论文 Table 4。
3. NVFP4（W4A16 / W4A4）在分组修复后全部重跑；5.3 的数据作废。
4. GPTQ / GPTAQ initializer + ReQuant 的完整矩阵。
5. **Qwen3-8B 修复后 `α=0.25`，T=2**——当前最高优先级。修复前 `α=0.25` 的 10.096 远好于修复后 `α=1` 的 11.549，但两者差了两个变量，必须解耦。见 §4.3。
6. 不带 `--w_clip` 的对照（见 §6.2）：RTN 基线已出 11.858，ReQuant T=4 运行中。

### 6.4 未实现

1. AWQ initializer（论文四个 initializer 之一）。
2. lm-eval 十项 zero-shot、UltraChat-2k / NuminaMath PPL、对 FP 的 KL。
3. `fast_hadamard_transform` CUDA 扩展未编译成功，当前走纯 PyTorch 回退，有性能损失但不影响数值。

---

## 7. 复现命令

```bash
# 基线 + ReQuant（论文默认 T=4 / K=2）
python main.py --save_name requant_t4 \
  --model "$MODEL" \
  --dataset_dir "$DATA_DIR" \
  --cal_dataset wikitext2 --eval_dataset wikitext2 \
  --seqlen 2048 --nsamples 512 --bsz 4 \
  --k_bits 16 --v_bits 16 --a_bits 16 \
  --rotate --w_bits 4 --w_groupsize -1 --w_asym --w_clip --w_rtn \
  --requant --requant_sweeps 4 --requant_neighborhood 2 \
  --requant_chunk 32 --requant_cross_alpha 1.0
```

- `--w_rtn` 换成 GPTQ / GPTAQ 的开关即可切换 initializer。
- `--w_format nvfp4 --a_format nvfp4 --a_bits 4` 切到 W4A4 NVFP4。
- `--requant_chunk` 控制 FP 激活缓存的显存占用，调小可在小显存卡上跑。
- `--requant_objective simplified` 丢弃交叉项，退化为纯权重重建目标，可用于消融。
