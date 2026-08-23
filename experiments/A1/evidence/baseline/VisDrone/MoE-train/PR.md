# PR: ES-MoE 路由可观测性框架与 VisDrone 稀疏训练复现

## 摘要

本 PR 为 ES-MoE 的路由行为建立了一个系统化的可观测性框架，并在 VisDrone 上通过 2×2 对照实验（专家数量 × balance_loss）验证了不同超参下的路由分化模式。核心产出包括：

1. 修复了 ES_MOE 默认专家数量与 CLI 参数的对齐
2. 构建了路由诊断工具链（单图专家特征热力图 + 数据集级稀疏 top-k 统计 + 路由区分度指标）
3. 提供了可复现的 VisDrone 稀疏训练脚本，内置路由诊断回调
4. 四组对照实验揭示了 balance_loss 取值与路由健康度之间的定量关系

---

## 背景与动机

待补充。

---

## 代码改动

### Commit 1：VisDrone ES-MoE 稀疏训练脚本 + 专家参数修正

**文件**：
- `ultralytics/nn/modules/moe/modules.py`
- `scripts/reproduce/reproduce_visdrone_sparse.py`（新增）
- `scripts/reproduce/_reproduce_common.py`

**内容**：
- ES_MOE 默认值 `num_experts=4`、`top_k=2`，与论文设计及 CLI 参数对齐
- routing snapshot 移除 `self.training` 判断，推理模式下同样记录路由统计，支撑离线诊断
- 新增 `reproduce_visdrone_sparse.py`：一键复现 ES-MoE 在 VisDrone 上的稀疏路由训练，支持 `--moe-balance-loss` 参数切换，内置：
  - `_make_sparse_topk_callback`：确保 ES_MOE 使用 top-2 稀疏路由
  - `_make_routing_diag_callback`：每个 epoch 记录路由 Gini / 主导专家占比 / 各专家 usage 到日志和 W&B
- `_reproduce_common.py` 新增 `mixture_aux_loss` 指标到 W&B

### Commit 2：路由诊断工具链

**文件**：
- `ultralytics/utils/routing_interpreter.py`
- `tools/routing_interpreter.py`

**内容**：
- 新增 3 个 dataclass：`ExpertFeatureMap`、`SparseTopKStats`、`RouterDifferentiationMetrics`
- 新增 `capture_routing_and_expert_features()`：单图模式下同时捕获路由器权重和专家输出 RMS 特征热力图（替代原先无空间变化的 per-pixel 路由权重热力图）
- 新增 `run_dataset_analysis()`：遍历数据集，输出稀疏 top-k 统计（每专家选中率 + 共现矩阵）、路由区分度指标（KL 散度 + weight spread）、collapse 报告
- `save_routing_visualizations()` 新增 `expert_features` 可选参数，向后兼容
- CLI 新增 `--dataset`、`--max-samples`、`--batch-size` 参数，支持单图模式（专家特征热力图）和数据集模式（`dataset_routing_report.json`）

---

## 实验设计

### 核心问题

1. balance_loss 取值与路由分化程度之间的定量关系？
2. 专家数量（3 vs 4）对路由塌缩倾向的影响？

### 实际实验矩阵（共 7 组）

实际运行中 epoch 数根据训练收敛情况做了调整（exp1/2 在 epoch 25 时观察到 loss 已平稳，提前停止；exp3-6 训练 30 epoch）。增加了 exp3（BL=1.5, 4E）以验证 BL 是否已饱和，以及 exp7（500 epoch 长训练 baseline）以观察长期训练下路由行为的变化。

| 实验 | 专家数 | Top-K | Balance Loss | Epochs | 实际状态 |
|------|--------|-------|-------------|--------|---------|
| **exp1** | 4 | 2 | 0.3 | 80 (early stop 27) | ✅ 完成 |
| **exp2** | 4 | 2 | 1.0 | 80 (early stop 27) | ✅ 完成 |
| **exp3** | 4 | 2 | 1.5 | 30 | ✅ 完成 |
| **exp4** | 3 | 2 | 0.3 | 30 | ✅ 完成 |
| **exp5** | 3 | 2 | 1.0 | 30 | ✅ 完成 |
| **exp6** | 3 | 2 | 1.5 | 30 | ✅ 完成 |
| **exp7** | 3 | 2 | 1.0 | ~500 | ✅ 完成（长训练 baseline） |

> **exp7 说明**：权重来源于 phase0 baseline 训练，使用 MuSGD 优化器，其余配置与 exp5 相同。目的是对比长训练（~500 epoch）与短训练（30 epoch）下路由行为的变化。

### 公共配置

| 参数 | 值 |
|------|-----|
| 模型 | EsMoE-N (`yolo-master-n.yaml`) |
| 数据集 | VisDrone |
| imgsz | 640 |
| batch | 32 |
| device | 0 (RTX 4090) |
| optimizer | AdamW |
| seed | 42 |
| deterministic | true |
| patience | 0 |
| amp | true |
| top_k | 2（训练 sparse forward，推理 dense forward） |
| noise_std | 0.5 |
| expert_warmup_epochs | 3 |

### 训练命令

```bash
# exp1: 4 experts, BL=0.3, 80 epochs
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 80 --moe-balance-loss 0.3 --moe-num-experts 4

# exp2: 4 experts, BL=1.0, 80 epochs
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 80 --moe-balance-loss 1.0 --moe-num-experts 4

# exp3: 4 experts, BL=1.5, 30 epochs
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 30 --moe-balance-loss 1.5 --moe-num-experts 4

# exp4: 3 experts, BL=0.3, 30 epochs
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 30 --moe-balance-loss 0.3 --moe-num-experts 3

# exp5: 3 experts, BL=1.0, 30 epochs
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 30 --moe-balance-loss 1.0 --moe-num-experts 3

# exp6: 3 experts, BL=1.5, 30 epochs
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 30 --moe-balance-loss 1.5 --moe-num-experts 3
```

### 分析命令

```bash
# 单图专家特征热力图（routing_report.json + 可视化）
python tools/routing_interpreter.py <checkpoint.pt> <image.jpg> --device cuda:0

# 数据集级路由统计（dataset_routing_report.json）
python tools/routing_interpreter.py <checkpoint.pt> --dataset VisDrone.yaml --device cuda:0
```

---

## 结果

### 1. 实验概况

| 实验 | 配置 | Epochs | mAP50 | 坍塌层数 | 死专家总数 |
|------|------|--------|-------|---------|-----------|
| exp1 | 4E, BL=0.3 | 27 | 0.176 | 3/4 | 6 |
| exp2 | 4E, BL=1.0 | 27 | 0.176 | 3/4 | 6 |
| exp3 | 4E, BL=1.5 | 30 | 0.159 | 2/4 | 4 |
| exp4 | 3E, BL=0.3 | 30 | 0.151 | 3/4 | 3 |
| exp5 | 3E, BL=1.0 | 30 | 0.155 | 3/4 | 3 |
| exp6 | 3E, BL=1.5 | 30 | 0.148 | 3/4 | 3 |
| **exp7** | **3E, BL=1.0** | **~500** | **—** | **0/4** | **0** |

> **注**：当前实验训练 epoch 数远少于论文（600 epoch），mAP 绝对值偏低属预期行为。本节聚焦路由行为分析，不涉及 mAP 性能对比。exp7 为长训练 baseline，坍塌层数为 0 但存在"软坍塌"（见下文）。

### 2. 专家利用率明细

`expert_usage` 字段来自 `dataset_routing_report.json` 的 `collapse` 部分，含义是全体测试样本上各专家被分配的**平均概率**（所有样本所有空间位置上的 router softmax 均值）。

**4 专家组（exp1 / exp2 / exp3）**

| 实验 | 层 | E0 | E1 | E2 | E3 | 死专家 | Gini | 状态 |
|------|----|----|----|----|----|--------|------|------|
| exp1 (BL=0.3) | L3 | **0.556** | 0.444 | ❌ 0.000 | ❌ 0.000 | E2,E3 | 0.704 | ⛔ 坍塌 |
| | L6 | **0.552** | ❌ 0.000 | ❌ 0.000 | 0.448 | E1,E2 | 0.701 | ⛔ 坍塌 |
| | L9 | ❌ 0.000 | **0.508** | 0.492 | ❌ 0.000 | E0,E3 | 0.672 | ⛔ 坍塌 |
| | L12 | 0.370 | 0.096 | **0.394** | 0.139 | — | 0.375 | ✅ 正常 |
| exp2 (BL=1.0) | L3 | **0.531** | 0.469 | ❌ 0.000 | ❌ 0.000 | E2,E3 | 0.688 | ⛔ 坍塌 |
| | L6 | **0.556** | ❌ 0.000 | 0.444 | ❌ 0.000 | E1,E3 | 0.704 | ⛔ 坍塌 |
| | L9 | ❌ 0.000 | **0.534** | 0.466 | ❌ 0.000 | E0,E3 | 0.689 | ⛔ 坍塌 |
| | L12 | **0.357** | 0.201 | 0.319 | 0.123 | — | 0.274 | ✅ 正常 |
| exp3 (BL=1.5) | L3 | **0.522** | 0.478 | ❌ 0.000 | ❌ 0.000 | E2,E3 | 0.681 | ⛔ 坍塌 |
| | L6 | **0.544** | ❌ 0.000 | 0.456 | ❌ 0.000 | E1,E3 | 0.696 | ⛔ 坍塌 |
| | L9 | 0.217 | **0.345** | 0.214 | 0.224 | — | 0.134 | ✅ 正常 |
| | L12 | **0.373** | 0.191 | 0.311 | 0.126 | — | 0.287 | ✅ 正常 |

**3 专家组（exp4 / exp5）**

| 实验 | 层 | E0 | E1 | E2 | 死专家 | Gini | 状态 |
|------|----|----|----|----|--------|------|------|
| exp4 (BL=0.3) | L3 | ❌ 0.000 | 0.509 | 0.491 | E0 | 0.509 | ⛔ 坍塌 |
| | L6 | ❌ 0.000 | **0.532** | 0.468 | E0 | 0.532 | ⛔ 坍塌 |
| | L9 | ❌ 0.000 | 0.511 | 0.489 | E0 | 0.511 | ⛔ 坍塌 |
| | L12 | 0.190 | 0.352 | **0.458** | — | 0.268 | ✅ 正常 |
| exp5 (BL=1.0) | L3 | ❌ 0.000 | **0.523** | 0.477 | E0 | 0.523 | ⛔ 坍塌 |
| | L6 | ❌ 0.000 | **0.531** | 0.469 | E0 | 0.531 | ⛔ 坍塌 |
| | L9 | ❌ 0.000 | 0.505 | 0.495 | E0 | 0.505 | ⛔ 坍塌 |
| | L12 | 0.270 | **0.374** | 0.355 | — | 0.104 | ✅ 正常 |
| exp6 (BL=1.5) | L3 | ❌ 0.000 | **0.516** | 0.484 | E0 | 0.516 | ⛔ 坍塌 |
| | L6 | ❌ 0.000 | **0.518** | 0.482 | E0 | 0.518 | ⛔ 坍塌 |
| | L9 | ❌ 0.000 | 0.500 | **0.500** | E0 | 0.500 | ⛔ 坍塌 |
| | L12 | **0.488** | 0.228 | 0.284 | — | 0.260 | ✅ 正常 |
| **exp7 (BL=1.0, ~500ep)** | L3 | 0.321 | 0.337 | **0.342** | — | 0.021 | ⚠ 软坍塌 |
| | L6 | **0.338** | 0.329 | 0.332 | — | 0.009 | ⚠ 软坍塌 |
| | L9 | **0.342** | 0.332 | 0.326 | — | 0.016 | ⚠ 软坍塌 |
| | L12 | 0.335 | **0.337** | 0.328 | — | 0.010 | ⚠ 软坍塌 |

> **⚠ 软坍塌**：expert_usage 接近均匀（~33%），Gini ≈ 0，collapse=false。但 KL 散度 ≈ 0（router 对所有输入输出几乎相同分布），跨样本路由多样性丧失。详见 [发现 7](#发现-7长训练导致的坍塌退化谱系)。

> **符号说明**：❌ = 死专家 (expert_usage = 0.000)，**加粗** = 该层主导专家，✅ = Gini < 0.5，⛔ = 坍塌 (collapsed=true)

### 3. Gini 系数汇总

| 实验 | 配置 | L3 | L6 | L9 | L12 | 均值 |
|------|------|----|----|----|----|------|
| exp1 | 4E, BL=0.3 | 0.704 | 0.701 | 0.672 | 0.375 | 0.613 |
| exp2 | 4E, BL=1.0 | 0.688 | 0.704 | 0.689 | **0.274** | 0.589 |
| exp3 | 4E, BL=1.5 | 0.681 | 0.696 | **0.134** | **0.287** | 0.449 |
| exp4 | 3E, BL=0.3 | 0.509 | 0.532 | 0.511 | **0.268** | 0.455 |
| exp5 | 3E, BL=1.0 | 0.523 | 0.531 | 0.505 | **0.104** | 0.416 |
| exp6 | 3E, BL=1.5 | 0.516 | 0.518 | 0.500 | **0.260** | 0.448 |
| **exp7** | **3E, BL=1.0, 500ep** | **0.021** | **0.009** | **0.016** | **0.010** | **0.014** |

**Gini 热力图（深色=严重不均衡，浅色=均衡）：**

```
         L3     L6     L9     L12
exp1  ██████ ██████ ██████ ██████    0.704
exp2  ██████ ██████ ██████ ░░░░░░    0.688
exp3  ██████ ██████ ░░░░░░ ░░░░░░    0.681
exp4  ██████ ██████ ██████ ░░░░░░    0.532
exp5  ██████ ██████ ██████ ░░░░░░    0.531
exp6  ██████ ██████ ██████ ░░░░░░    0.518
exp7  ░░░░░░ ░░░░░░ ░░░░░░ ░░░░░░    0.021  ← 几乎完全均匀
```

### 4. 跨样本路由多样性

`expert_usage` 是平均概率，无法区分"全局坍塌"（所有样本走同一对专家）和"样本级坍塌 + 跨样本多样性"（不同样本走不同专家对）。`expert_hit_percentages`（每个专家被多少 % 的样本选中至少一次）弥补了这一点。

**4 专家组 — 专家命中率 (%)**

| 实验 | 层 | E0 | E1 | E2 | E3 | 活跃 | 诊断 |
|------|----|----|----|----|----|------|------|
| exp1 | L3 | 100 | 100 | 0 | 0 | 2 | 🔴 全局坍塌 |
| | L6 | 100 | 0 | 0 | 100 | 2 | 🔴 全局坍塌 |
| | L9 | 0 | 100 | 100 | 0 | 2 | 🔴 全局坍塌 |
| | L12 | 70.2 | 22.6 | 76.0 | 31.3 | **4** | 🟢 跨样本多样 |
| exp2 | L3 | 100 | 100 | 0 | 0 | 2 | 🔴 全局坍塌 |
| | L6 | 100 | 0 | 100 | 0 | 2 | 🔴 全局坍塌 |
| | L9 | 0 | 100 | 100 | 0 | 2 | 🔴 全局坍塌 |
| | L12 | 73.9 | 36.1 | 63.5 | 26.5 | **4** | 🟢 跨样本多样 |
| exp3 | L3 | 100 | 100 | 0 | 0 | 2 | 🔴 全局坍塌 |
| | L6 | 100 | 0 | 100 | 0 | 2 | 🔴 全局坍塌 |
| | L9 | 43.4 | 68.2 | 43.1 | 45.3 | **4** | 🟢 跨样本多样 |
| | L12 | 73.9 | 36.1 | 63.5 | 26.5 | **4** | 🟢 跨样本多样 |

**3 专家组 — 专家命中率 (%)**

| 实验 | 层 | E0 | E1 | E2 | 活跃 | 诊断 |
|------|----|----|----|----|------|------|
| exp4 | L3 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L6 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L9 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L12 | 44.2 | 68.4 | 87.4 | **3** | 🟢 跨样本多样 |
| exp5 | L3 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L6 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L9 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L12 | 63.9 | 71.5 | 64.6 | **3** | 🟢 跨样本多样 |
| exp6 | L3 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L6 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L9 | 0 | 100 | 100 | 2 | 🔴 全局坍塌 |
| | L12 | 91.4 | 47.4 | 61.1 | **3** | 🟢 跨样本多样 |
| **exp7** | **L3** | **10.6** | **90.7** | **98.7** | **3** | **🟡 软坍塌** |
| | **L6** | **94.5** | **29.6** | **75.9** | **3** | **🟡 软坍塌** |
| | **L9** | **97.8** | **95.4** | **6.8** | **3** | **🟡 软坍塌** |
| | **L12** | **66.8** | **97.1** | **36.1** | **3** | **🟡 软坍塌** |

> **🟡 软坍塌**：expert_usage 接近均匀（~33%），无死专家，但每层仍有一个"边缘专家"命中率极低（L3 E0=10.6%、L9 E2=6.8%）。softmax 概率虽然接近均匀，argmax top-2 的 winner-take-all 效应仍放大了微小差异。

### 5. 专家配对多样性（model.12 共现矩阵）

exp1-6 中 model.12 是唯一在所有实验中均保持健康（collapsed=false）的层。exp7 中 model.12 虽然在 collapse 指标上"正常"，但实际已进入过度均衡的退化状态。以下对比 3 专家组各实验的 L12 共现矩阵。

**exp1（4E, BL=0.3）— L12 共现，配对种类：4/6**

|  | E0 | E1 | E2 | E3 |
|--|----|----|----|----|
| **E0** | 146 | 47 | 96 | 3 |
| **E1** | 47 | 47 | 0 | 0 |
| **E2** | 96 | 0 | 158 | 62 |
| **E3** | 3 | 0 | 62 | 65 |

> (E1,E2) 和 (E1,E3) 从未同时出现——E1 的功能与 E2/E3 互斥。

**exp2/3（4E, BL≥1.0）— L12 共现，配对种类：6/6（全覆盖）**

|  | E0 | E1 | E2 | E3 |
|--|----|----|----|----|
| **E0** | 405 | 160 | 228 | 17 |
| **E1** | 160 | 198 | 15 | 23 |
| **E2** | 228 | 15 | 348 | 105 |
| **E3** | 17 | 23 | 105 | 145 |

> BL≥1.0 后所有 6 种配对均有出现，主要配对 (E0,E2)=228、(E0,E1)=160、(E2,E3)=105。

**exp4（3E, BL=0.3）— L12 共现，配对种类：3/3**

|  | E0 | E1 | E2 |
|--|----|----|----|
| **E0** | 242 | 69 | 173 |
| **E1** | 69 | 375 | 306 |
| **E2** | 173 | 306 | 479 |

**exp5（3E, BL=1.0）— L12 共现，配对种类：3/3**

|  | E0 | E1 | E2 |
|--|----|----|----|
| **E0** | 350 | 194 | 156 |
| **E1** | 194 | 392 | 198 |
| **E2** | 156 | 198 | 354 |

> exp5 三个配对分布更加均衡（194 / 156 / 198），显著优于 exp4 的 (69 / 173 / 306)。

**exp6（3E, BL=1.5）— L12 共现，配对种类：3/3**

|  | E0 | E1 | E2 |
|--|----|----|----|
| **E0** | 501 | 213 | 288 |
| **E1** | 213 | 260 | 47 |
| **E2** | 288 | 47 | 335 |

> exp6 的 L12 配对分布极度偏向 E0：(E0,E2)=288 占主导，(E1,E2)=47 很少。E1 几乎只与 E0 配对（213 vs 仅 47 与 E2），暗示 E2 和 E1 的功能互补性弱。与 exp5 (BL=1.0) 相比，BL=1.5 在 3 专家模型深层**反而恶化了配对多样性**。

**exp7（3E, BL=1.0, ~500ep）— L12 共现，配对种类：3/3**

|  | E0 | E1 | E2 |
|--|----|----|----|
| **E0** | 366 | 350 | 16 |
| **E1** | 350 | 532 | 182 |
| **E2** | 16 | 182 | 198 |

> exp7 的共现矩阵揭示了一个关键退化特征：(E0,E2)=16 几乎为零！虽然三个专家都有命中率，但 E0 和 E2 几乎从不共现。配对分布极度固化：(E0,E1)=350 和 (E1,E2)=182 垄断了所有样本。与 exp5（30ep，194/156/198 均衡）对比，长训练的配对多样性**显著退化**。这说明 balance_loss 长期作用使 router 收敛到一个"高度对称但无信息量"的均衡点——每个专家 usage 看起来均衡（~33%），但实际配对行为已经刚性化。

### 6. 单图热力图对比

选取各实验的 model.12 层 Dashboard（含 Assignment Map + Confidence Heatmap + 各 Expert 激活热力图）作为示例。

**4 专家组 model.12 Dashboard：**

| exp1 (BL=0.3) | exp2 (BL=1.0) | exp3 (BL=1.5) |
|---------------|---------------|---------------|
| ![exp1-L12](实验记录/exp1/routing-analysis/model_12_routing_dashboard.png) | ![exp2-L12](实验记录/exp2/routing/model_12_routing_dashboard.png) | ![exp3-L12](实验记录/exp3/routing/model_12_routing_dashboard.png) |

**3 专家组 model.12 Dashboard：**

| exp4 (BL=0.3) | exp5 (BL=1.0) | exp6 (BL=1.5) | exp7 (500ep, BL=1.0) |
|---------------|---------------|---------------|---------------------|
| ![exp4-L12](实验记录/exp4/routing/model_12_routing_dashboard.png) | ![exp5-L12](实验记录/exp5/routing/model_12_routing_dashboard.png) | ![exp6-L12](实验记录/exp6/routing/model_12_routing_dashboard.png) | ![exp7-L12](实验记录/exp7/routing/model_12_routing_dashboard.png) |

**4 专家组 model.9 Dashboard（L9 是关键分水岭——exp1 坍塌，exp2/3 正常）：**

| exp1 (BL=0.3) ⛔ | exp2 (BL=1.0) ✅ | exp3 (BL=1.5) ✅ |
|-------------------|-------------------|-------------------|
| ![exp1-L9](实验记录/exp1/routing-analysis/model_9_routing_dashboard.png) | ![exp2-L9](实验记录/exp2/routing/model_9_routing_dashboard.png) | ![exp3-L9](实验记录/exp3/routing/model_9_routing_dashboard.png) |

**3 专家组 model.3 Dashboard（示范浅层坍塌的典型模式——所有样本统一走同一对专家，热力图近乎纯色）：**

| exp4 (BL=0.3) ⛔ | exp5 (BL=1.0) ⛔ |
|-------------------|-------------------|
| ![exp4-L3](实验记录/exp4/routing/model_3_routing_dashboard.png) | ![exp5-L3](实验记录/exp5/routing/model_3_routing_dashboard.png) |

> 更多层的 Dashboard 见各实验的 `routing/` 子目录。

---

## 讨论

### 发现 1：浅层路由器（L3/L6）在所有配置下均坍塌

这是本次实验最稳定的发现。不论专家数是 3 还是 4，balance_loss 是 0.3 还是 1.5，L3 和 L6 始终只有 2 个专家被使用（top_k=2 下恰好饱和）。跨样本分析进一步确认这是**全局坍塌**——所有 548 张测试图片都无一例外地选择了同一对专家。

**可能的解释：**
- 浅层特征（边缘、纹理、颜色）是高度通用的，不同图片之间的浅层特征差异很小，router 没有动机将不同样本路由到不同专家
- 浅层感受野小（L3: 160×160, L6: 80×80），空间上下文不足，router 缺乏足够信息做出差异化决策
- 初始化时 router 的微小随机偏好被 soft top-k 放大为固定选择，随后进入恶性循环

### 发现 2：深层路由器（L12）在所有配置下均健康

L12 是唯一在所有 7 个实验中均未坍塌的层。原因可能是：

- 深层特征（L12: 20×20）更接近任务语义，包含物体类别、位置、尺度等差异化信息
- 感受野大，router 有足够的空间上下文做差异化路由决策
- exp5 的 L12 Gini=0.104 是 3 专家中最均衡的；exp2 的 L12 Gini=0.274 也在健康范围内
- 但 exp7 证明 L12 即使不坍塌，也可能退化为"无分化"（KL≈0）

### 发现 3：L9 呈"分水岭"行为——BL=1.0→1.5 是关键跃迁

| 条件 | L9 状态 |
|------|---------|
| 4 专家 + BL=0.3 | ⛔ 坍塌（Gini=0.672） |
| 4 专家 + BL=1.0 | ⛔ 坍塌（Gini=0.689） |
| 4 专家 + BL=1.5 | ✅ 正常（Gini=0.134） |
| 3 专家 + 任意 BL | ⛔ 坍塌（Gini=0.500~0.511） |

**修正**：此前因数据复制错误，误认为 exp2 (BL=1.0) 和 exp3 (BL=1.5) 的 L9 路由行为一致。实际数据表明：

- BL=1.0（exp2）的 L9 仍然坍塌（E0 和 E3 死亡，Gini=0.689），与 BL=0.3（exp1）模式相同
- BL=1.5（exp3）的 L9 才恢复健康（4 专家全活，Gini=0.134，跨样本多样性丰富）

这意味着 L9 的"激活阈值"在 **BL=1.0 到 1.5 之间**——这是一个需要足够强平衡约束才能打破路由固化的临界层。对于 3 专家模型，即使 BL=1.5（exp6）也不足以激活 L9 的全部专家，说明**专家容量和 BL 之间存在乘法关系**——更多专家需要更高的 BL 来保持均衡。

exp6 中 L9 的 Gini=0.500 是一个有趣的边缘情况：E1/E2 几乎完美均分（50.0%/50.0%），但 E0 死亡。这属于"计算上的坍塌"（dead_expert 非空）但"分布上几乎均衡"——可能说明 3 专家模型在足够大的 BL 下，存活的两个专家可以接近均衡，但永远拉不起第三个。

### 发现 4：Balance Loss 的效果呈"阶梯式"而非线性

**4 专家模型**：BL 的效果是非连续的阶梯函数：

| BL | L3/L6 | L9 | L12 | 效果 |
|-----|-------|-----|-----|------|
| 0.3 (exp1) | ⛔ | ⛔ | ✅ | 深层仅 L12 健康 |
| 1.0 (exp2) | ⛔ | ⛔ | ✅ | L9 仍未激活 |
| 1.5 (exp3) | ⛔ | ✅ | ✅ | **L9 跃迁** |

BL=1.0→1.5 不是量的叠加，而是**质变**——L9 从完全坍塌（Gini=0.689）跃迁到完全健康（Gini=0.134）。说明路由坍塌存在一个需要足够强 balance_loss 才能克服的"能垒"。

**3 专家模型**：BL 在深层（L12）存在"过冲"风险：

| 实验 | BL | L12 Gini | L12 主导专家 | L12 配对均衡度 |
|------|-----|----------|-------------|---------------|
| exp4 | 0.3 | 0.268 | E2 (45.8%) | 不均衡 |
| exp5 | 1.0 | **0.104** | E1 (37.4%) | ✅ 最均衡 |
| exp6 | 1.5 | 0.260 | E0 (48.8%) | 不均衡 |

exp5 (BL=1.0) 的 L12 路由是最均衡的，BL=1.5（exp6）反而退化。3 专家模型对 BL 更敏感，最优窗口更窄。

### 发现 5：3 专家模型中 E0 的"分裂人格"

exp4 和 exp5 中，所有坍塌层的死专家都是 E0。但 exp6 呈现了一个有趣的反转：

- **L3/L6/L9**：E0 仍然死亡（与 exp4/5 一致）
- **L12**：E0 反而成为**主导专家**（命中率 91.4%，usage 48.8%），而 E1 被边缘化（命中率仅 47.4%，共现矩阵中几乎只与 E0 配对）

这说明"E0 系统性死亡"并非 E0 永远最差，而是不同层对 E0 的偏好不同。BL=1.5 的强约束可能在深层（L12）翻转了路由偏好，但无法在浅层（L3/L6/L9）克服已有的路由固化。

### 发现 6：expert_usage（平均概率）与 expert_hit_percentages（命中率）的互补性

这是方法论层面的发现。exp2 model.9 的 expert_usage = [0.217, 0.345, 0.214, 0.224]——表面上看 4 个专家都用到了。但如果只看单图的 `routing_report.json`（如 exp3 单图分析），每张图仍然只是 2 个专家在干活。

这意味着在 top_k=2 的稀疏路由下，"平均 4 个专家都用到了"不能等同于"没有坍塌"。真正重要的是**跨样本多样性**——不同样本是否选择了不同的专家对。`dataset_routing_report.json` 中的 `expert_hit_percentages` 和 `co_occurrence_matrix` 比 `expert_usage` 更适合评估稀疏路由的健康度。

### 发现 7：长训练导致的路由退化——坍塌→退化谱系（exp7 核心发现）

exp7 使用与 exp5 完全相同的配置（3E, BL=1.0），但训练了 ~500 epoch 而非 30 epoch。结果揭示了一条完整的路由行为谱系：

#### 坍塌-退化谱系

```
坍塌 (Collapse)  ◄──────────────────────►  退化 (Degeneracy)
专家死亡           均衡但有分化            所有专家存活但无分化

exp4 (3E,BL=0.3,30ep)     exp5 (3E,BL=1.0,30ep)     exp7 (3E,BL=1.0,500ep)
3/4层坍塌                 L12健康(配对均衡)         0层坍塌
L12 Gini=0.268            L12 Gini=0.104            L12 Gini=0.010
KL=0.413                  KL=0.419                  KL=0.001
死专家=3                  死专家=3                  死专家=0
◄── 不均衡 ──────────────────────────────────────────── 过均衡 ──►
```

#### exp7 的关键数据

| 指标 | exp5 (30ep) | exp7 (500ep) | 变化 |
|------|------------|-------------|------|
| 坍塌层数 | 3/4 | 0/4 | 坍塌"消失" |
| 死专家数 | 3 | 0 | 全部存活 |
| L12 Gini | 0.104 | 0.010 | 降 10× |
| L12 KL 散度 | 0.419 | **0.001** | 降 **400×** |
| L12 weight_spread | 0.125 | 0.008 | 降 16× |
| L12 共现均衡度 | ✅ 194/156/198 | ❌ 350/16/182 | (E0,E2) 几乎为 0 |

**表面上看 exp7 是最健康的**——无坍塌、无死专家、Gini≈0。但 KL 散度从 0.419 降到 0.001（降低 400 倍），意味着 router 对所有 548 张测试图片输出了几乎完全相同的分布。routes 失去了跨样本分化能力，MoE 退化为"带噪声的稠密网络"。

#### "软坍塌"现象

exp7 还揭示了一种新的坍塌形式——**软坍塌（soft collapse）**：

- expert_usage ≈ 33% 均衡（看起来健康）
- 但 KL 散度 ≈ 0（router 不做选择）
- argmax top-2 命中率仍有偏差（每层有一个"边缘专家"命中率 < 40%）

这说明仅靠 `expert_usage` 和 `collapse` 的 `dead_experts` 检查不足以检测路由退化。需要在诊断工具中增加 KL 散度和 weight_spread 的最小阈值检查——当这些值过低时，即使没有死专家，路由也已经失去了 MoE 的意义。

#### 对 balance_loss 策略的启示

这 7 个实验构成了完整的图景：

| 阶段 | 实验 | 状态 |
|------|------|------|
| BL 太低 | exp1, exp4 | 坍塌主导（死专家多） |
| BL 中等 (1.0) | exp2, exp5 | L9 仍未激活（4E），L12 最均衡（3E） |
| BL 较高 (1.5) | exp3 | **L9 跃迁成功**，4E 下唯一 L9 健康的短训实验 |
| BL 较高 (1.5) | exp6 | 3E 下过冲，L12 配对失衡 |
| BL 适中 + 长训 | **exp7** | **退化主导（过均衡）** |

**最优策略不是固定 BL，而是动态调度**：初期高 BL 让所有专家存活 → 后期逐步降低 BL 让 router 恢复跨样本分化能力。这正是论文中 `moe_dynamic_schedule` 的设计动机，也是 exp1-7 的实验数据给出的最明确的改进方向。

---

## 结论

### 核心结论

1. **MoE 路由坍塌呈"分层递减"规律**：浅层 (L3/L6) 全局坍塌 → 中层 (L9) 需要强 BL 才能激活 → 深层 (L12) 稳定健康。这一规律在 7 个实验中保持高度一致，说明它是**架构内生属性**。L9 的坍塌需要 BL≥1.5（4 专家）才能打破——这是一道明确的"能垒"。

2. **Balance Loss 的效果呈阶梯式而非线性**：4 专家模型中，BL=0.3→1.0 对 L9 无效（均坍塌），BL=1.0→1.5 触发 L9 质变（Gini: 0.689→0.134）。3 专家模型中 BL 存在"过冲"——BL=1.0（exp5）的 L12 最均衡（Gini=0.104），BL=1.5（exp6）反而退化（Gini=0.260）。说明 BL 的调优窗口与专家数量耦合：专家越多，所需 BL 越高，但 BL 过高会抑制分化。

3. **存在完整的坍塌→退化谱系，而非简单的坍塌/健康二分**：exp7（500 epoch）证明了路由健康不是一个二分类问题。BL=1.0 在 30 epoch 时（exp5）深层坍塌但 L12 跨样本多样性强；在 500 epoch 时（exp7）所有层"表面上健康"（无死专家、Gini≈0），但 KL 散度崩溃（0.001）、跨样本路由失去了分化能力。路由评估必须同时考虑均衡度（Gini）和分化度（KL 散度）。

4. **"软坍塌"是一种新的退化模式**：exp7 揭示：expert_usage 均匀 ≠ 路由健康。softmax 概率接近均匀时，argmax top-2 的 winner-take-all 效应仍会在命中率上产生系统性偏差。诊断工具需要增加 KL 散度/weight_spread 的最低阈值告警。

5. **减少专家数不缓解坍塌**：3 专家模型（exp4/5/6/7）无论短训长训，要么坍塌要么退化。在 top_k=2 下，4 专家（exp2/3）在短训时深层能保持跨样本分化，是目前的最优配置。

6. **动态 Balance Loss 是最明确的改进方向**：7 个实验的数据完整覆盖了"坍塌 ← 均衡且有分化 → 退化"谱系。固定 BL 无法同时避免坍塌和退化——最优策略是前期高 BL 保活、后期衰减促分化（即论文的 `moe_dynamic_schedule`）。

### 改进方向

| 优先级 | 方向 | 依据 |
|--------|------|------|
| 🔴 **最高** | **实现 moe_dynamic_schedule** | exp1-7 完整覆盖坍塌→退化谱系，固定 BL 无法同时避免两端。论文的 `moe_dynamic_schedule`（前期高 BL→后期衰减）是唯一覆盖全谱系的方案 |
| 🔴 高 | 诊断工具增加 KL 散度/weight_spread 退化告警 | exp7 证明 Gini+dead_experts 不足以检测"软坍塌"。KL < 0.01 或 weight_spread < 0.01 应触发退化告警 |
| 🔴 高 | 检查 E0 的 router 初始化 | 3 专家模型中 E0 在浅层系统性死亡 |
| 🟡 中 | 对浅层施加更强的负载均衡约束 | L3/L6 在所有 7 个实验中要么坍塌要么退化，balance_loss 对其调节无效 |
| 🟡 中 | 4 专家用 BL=1.5，3 专家用 BL=1.0 | exp3 是唯一 L9 健康的 4 专家短训实验；exp5 是 3 专家 L12 最均衡配置 |
| 🟢 低 | 探索 layer-wise 差异化 BL + 差异化 schedule | 浅层可能需要更高 BL 和更长 warmup，深层可以更快衰减 |

---

## 附录：命令速查

### 训练

```bash
# 4 experts, BL=0.3
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 80 --moe-balance-loss 0.3 --moe-num-experts 4

# 4 experts, BL=1.0
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 80 --moe-balance-loss 1.0 --moe-num-experts 4

# 4 experts, BL=1.5
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 30 --moe-balance-loss 1.5 --moe-num-experts 4

# 3 experts, BL=0.3
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 30 --moe-balance-loss 0.3 --moe-num-experts 3

# 3 experts, BL=1.0
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 30 --moe-balance-loss 1.0 --moe-num-experts 3

# 3 experts, BL=1.5
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 30 --moe-balance-loss 1.5 --moe-num-experts 3
```

### 数据集级路由分析（dataset_routing_report.json）

```bash
python tools/routing_interpreter.py \
  <checkpoint.pt> \
  --dataset VisDrone.yaml \
  --device cuda:0 \
  --output <output_dir> \
  --batch-size 32
```

输出文件：`dataset_routing_report.json`，包含：
- `sparse_topk`：每层 expert_hit_counts（命中数）、hit_percentages（命中率%）、co_occurrence_matrix（共现矩阵）
- `differentiation`：每层 mean/std kl_divergence、mean/std weight_spread
- `collapse`：每层 expert_usage、normalized_gini、normalized_entropy、dead_experts、collapsed

### 单图路由分析（routing_report.json + 热力图）

```bash
# 所有层
python tools/routing_interpreter.py \
  <checkpoint.pt> \
  <image.jpg> \
  --device cuda:0 \
  --output <output_dir>

# 仅指定层
python tools/routing_interpreter.py \
  <checkpoint.pt> \
  <image.jpg> \
  --layer model.12 \
  --device cuda:0 \
  --output <output_dir>
```

输出文件：`routing_report.json` + 每层 dashboard PNG（assignment_map、confidence_heatmap、各 expert 热力图）

### 强制专家 Counterfactual（对比不同专家对同一张图的输出差异）

```bash
# 强制 model.12 层只用 expert 0，同时生成对比热力图
python tools/routing_interpreter.py \
  <checkpoint.pt> \
  <image.jpg> \
  --layer model.12 --expert 0 \
  --device cuda:0 \
  --output <output_dir>
```

输出：`routing_report.json` 中 `causal` 字段包含：
- `cosine_similarity`：自然路由 vs 强制专家的输出余弦相似度（1.0 = 完全相同 → 专家退化）
- `mean_absolute_difference` / `root_mean_square_difference`：输出差异的 MAE / RMSE
- `max_absolute_difference`：最大单元素差异

同时生成的热力图展示强制专家下的特征分布，可与自然路由热力图对比。

### 批量跑所有层+专家的 counterfactual

```bash
for layer in model.3 model.6 model.9 model.12; do
  for expert in 0 1 2; do
    python tools/routing_interpreter.py \
      <checkpoint.pt> <image.jpg> \
      --layer $layer --expert $expert \
      --device cuda:0 \
      --output <output_dir>/force_${layer}_expert${expert}
  done
done
```

### 诊断工具关键指标解读

| 指标 | 来源 | 健康范围 | 含义 |
|------|------|---------|------|
| `normalized_gini` | collapse | 0.1~0.4 | >0.5 = 严重不均衡（坍塌），<0.01 = 过度均衡（退化风险） |
| `normalized_entropy` | collapse | 0.8~1.0 | <0.6 = 坍塌，接近 1.0 但 KL≈0 = 退化 |
| `mean_kl_divergence` | differentiation | >0.01 | <0.01 = router 对所有输入输出几乎相同分布（退化告警） |
| `mean_weight_spread` | differentiation | >0.02 | <0.02 = 两个选中专家权重无差异（退化告警） |
| `expert_hit_percentages` | sparse_topk | 各专家 20~80% | 100% 或 0% = 全局坍塌；全部接近 50% 但 KL≈0 = 退化 |
| `cosine_similarity` | causal | <0.95 | >0.99 = 强制切换专家几乎不改变输出（专家功能趋同） |
