# Issue #52 分阶段规划 — moe-pruning-signals

> 分支: `dev/moe-pruning-signals`

## 背景

YOLO-Master 使用 `balance_loss` 强制 MoE 路由器均衡使用 expert。经过 200 epoch 训练后，4 个 expert 的使用率高度均衡，导致剪枝时 hit-rate 信号缺乏区分度。

**核心洞察**: 剪枝和动态调度解决的是同一个矛盾——expert 应该均衡还是分化。固定 balance_loss 迫使均衡，expert 同等重要，剪枝效果差。动态调度让 expert 自然分化，分化后剪枝才可能有效。

## 四条链路，六个实验点

| 实验点 | 简称 | 说明 |
|--------|------|------|
| ① | baseline | 基线模型（固定 balance_loss=1.0, 200 epoch） |
| ② | baseline-pruned | 对 ① 做专家剪枝 |
| ③ | baseline-pruned-lora | 对 ② 做 LoRA 微调恢复 |
| ④ | dynamic-trained | 动态调度重新训练（Gini 自适应 balance_loss） |
| ⑤ | dynamic-pruned | 对 ④ 做专家剪枝 |
| ⑥ | dynamic-pruned-lora | 对 ⑤ 做 LoRA 微调恢复 |

实验点构成两条对比链路：

```
链路 A（固定调度）： ① → ② → ③
链路 B（动态调度）： ④ → ⑤ → ⑥
```

核心对比：**链路 A vs 链路 B 的剪枝效果**，回答"动态调度是否让 expert 分化，从而使剪枝更有效"。

---

## 第一阶段：基线评估

**目标**: 搞清楚 ① 的专家实际行为。

- [ ] 4 层 MoE 各自的 expert 命中率分布
- [ ] 每个 expert 被选中时的平均权重
- [ ] 每层 Gini 系数
- [ ] 是否存在某 expert 几乎不用（collapse）或垄断（dominant）

**输出**: 基线路由诊断报告

---

## 第二阶段：固定调度下的剪枝与恢复（链路 A）

**目标**: 在 ① 上做剪枝和恢复，建立 baseline 链条。

### 2.1 专家剪枝: ② baseline-pruned

- [ ] usage 信号 × 5 阈值 {0.05, 0.10, 0.15, 0.20, 0.30}
- [ ] usage_weight 信号 × 5 阈值
- [ ] 每轮记录: 剪枝后专家结构、mAP50-95、mAP50、GFLOPs、Latency、Params

**评估**: 记录有效剪枝点（no-op 点也如实记录）。

### 2.2 LoRA 恢复: ③ baseline-pruned-lora

- [ ] 对每个有效剪枝点做 LoRA 微调
- [ ] 与 ① baseline 和 ② 直接推理对比

**评估**: LoRA 恢复后精度能回到 baseline 的多少。

---

## 第三阶段：动态超参数调度训练

**目标**: 用动态调度重新训练模型，得到 ④。

使用已有 `GiniBalanceScheduler`，公式:
```
ema_t = beta × ema_{t-1} + (1-beta) × gini_t
coeff_{t+1} = clip(base × exp(alpha × (ema_t - target_gini)), min, max)
```

三组对照:

- [ ] Fixed Baseline（balance_loss=1.0）
- [ ] Gini Dynamic（moe_dynamic_schedule=gini）
- [ ] Fixed-Low Ablation（balance_loss=0.3）

**输出**: 三个训练 trace（mAP 曲线、Gini 变化曲线、balance_loss 变化曲线）

**产出**: ④ dynamic-trained（取 Gini Dynamic 组的 best checkpoint）

---

## 第四阶段：动态调度下的剪枝与恢复（链路 B）

**目标**: 在 ④ 上重复第二阶段流程，形成对比。

### 4.1 专家剪枝: ⑤ dynamic-pruned

- [ ] 与 2.1 相同的剪枝流程（两种信号 × 5 阈值）
- [ ] 与 ② 对比剪枝效果差异

### 4.2 LoRA 恢复: ⑥ dynamic-pruned-lora

- [ ] 与 2.2 相同的恢复流程
- [ ] 与 ③ 对比恢复效果差异

**核心评估**: 链路 B 的有效剪枝点数量和质量是否优于链路 A。

---

## 产出清单

| 序号 | 产出 | 阶段 |
|------|------|------|
| 1 | 基线路由诊断报告 | 一 |
| 2 | ② 剪枝结果表（usage + usage_weight × 5 阈值） | 二 |
| 3 | ③ LoRA 恢复对比表 | 二 |
| 4 | ④ 动态调度三组对照 trace | 三 |
| 5 | ⑤ 动态模型剪枝结果表 | 四 |
| 6 | ⑥ 动态模型 LoRA 恢复对比表 | 四 |
| 7 | 链路 A vs 链路 B 综合对比 | 四 |
| 8 | Pareto 前沿图 + Sweet Spot 推荐 | 四 |
| 9 | 技术报告 | 收尾 |
| 10 | GitHub Discussion 文章 | 收尾 |
