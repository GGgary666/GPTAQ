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

### 4.3 伪解：`α = 0.25`

主 bug 期间，参考 GPTAQ `fasterquant(alpha=0.25)` 给交叉项加了缩放，`α=0.25` 时结果「变好」且随 `T` 单调。

这只是把坏交易砍掉 3/4，**不是论文内容**（论文全文无此系数）。根因修复后应回到 `α=1`。`--requant_cross_alpha` 保留仅作消融用途。

### 4.4 其他已修

| 问题 | 处理 |
|---|---|
| `ModuleNotFoundError: fast_hadamard_transform` | 纯 PyTorch 回退 |
| `HfUriError: Invalid HF URI 'hf://datasets/wikitext@...'` | 改用 `Salesforce/wikitext` |
| `view size is not compatible with stride` | `contiguous().reshape()` |
| `get_hadK` 对 17408 断言失败 | Paley-68 构造 |

---

## 5. 实验结果

统一设定：QuaRot 旋转，权重 INT4 per-channel 非对称 + `--w_clip`，WikiText-2 校准 512×2048，评测为 WikiText-2 全测试集、seqlen 2048。

### 5.1 基线复现（可信）

| 模型 | FP16 | RTN W4A16 | 论文 FP | 论文 RTN |
|---|---|---|---|---|
| Qwen3-14B | **8.648** | **10.107** | 8.65 | 10.22 |
| Qwen3-8B | 9.723 | 11.673 | — | — |

Qwen3-14B 两项均与论文吻合，说明旋转、量化、评测链路正确。

### 5.2 ReQuant（修复前 vs 修复后，均 α=1）

| 模型 | 设定 | RTN | + ReQuant |
|---|---|---|---|
| Qwen3-8B | T=2，**修复前** | 11.673 | 64.149 |
| Qwen3-14B | T=4，**修复后** | 10.107 | **9.179** |
| Qwen3-8B | T=2，修复后 | 11.673 | 待补 |

论文 Qwen3-14B 对应值为 8.85。当前差 ≈ 0.33 PPL，方向已正确（RTN 相对 FP 的 +1.46 缺口收回到 +0.53）。

修复生效的直接证据（Qwen3-14B layer 0，`quad` 恢复下降）：

```
o_proj    : quad 386.4 -> 102.3
down_proj : quad 2005  -> 687.9
```

### 5.3 Qwen3-1.7B 的历史数据（**不可信，仅存档**）

以下全部在主 bug（§4.1）存在时产出，且该模型还命中 §6 的方向歧义问题，**不能作为 ReQuant 的结论**：

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

**待修**

1. `ReQuant.add_batch` 的方向判定在 `dcol == tokens` 时有歧义：转置后形状相同，`x_fp.shape != inp.shape` 判为相等，会静默地把 `X̃ᵀ` 当作 `X`。**Qwen3-1.7B 恰好 `hidden_size = 2048 = seqlen`，全部命中。** 8B/14B 因 4096/5120 ≠ 2048 走正确分支。应改为显式约定方向而非靠形状推断。
2. 中后层仍存在 `quad` 上升的坏交易，例如 Qwen3-14B layer 20 `o_proj`（`quad 8.3e4 → 1.4e6`，此时 `‖ΔX‖/‖X̃‖ = 0.471`）。漂移累积到一定程度后交叉项仍会过度牺牲权重保真，可能是与论文剩余 0.33 PPL 差距的来源。

**待跑**

3. Qwen3-8B 修复后的 T=2 结果。
4. `T ∈ {0,1,2,4,8}` 单调性，对齐论文 Table 4。
5. NVFP4（W4A16 / W4A4）在分组修复后全部重跑；5.3 的数据作废。
6. GPTQ / GPTAQ initializer + ReQuant 的完整矩阵。
7. `α=1` 与 `α=0.25` 在修复后的对比，确认 `α` 可以彻底移除。

**未实现**

8. AWQ initializer（论文四个 initializer 之一）。
9. lm-eval 十项 zero-shot、UltraChat-2k / NuminaMath PPL、对 FP 的 KL。
10. `fast_hadamard_transform` CUDA 扩展未编译成功，当前走纯 PyTorch 回退，有性能损失但不影响数值。

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
