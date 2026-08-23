# Exp2: balance_loss=1.0, soft Top-2, 4 experts

## 实验目的
https://wandb.ai/delei-kong-szu/yolo-master-reproduce/workspace?nw=nwuserdeleikong
对比 exp1 (balance_loss=0.3)，验证 balance_loss=1.0 能否打破专家塌缩循环。

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | EsMoE-N (yolo-master-n.yaml) |
| 专家数 | 4 per layer |
| top_k | 2 |
| balance_loss | **1.0** |
| warmup | expert_warmup_epochs=3 |
| noise_std | 0.5 |
| epochs | 30（训练完成）|
| batch | 32 |
| imgsz | 640 |
| `use_dense` 逻辑 | `self.training` 保留（正确） |

## 训练结果（epoch 27）

| 指标 | 值 |
|------|-----|
| mAP50 | 0.176 |
| mAP50-95 | 0.076 |

## 路由分化（epoch 25）

| Layer | 活跃 | 死 | Gini |
|:-----:|------|-----|:---:|
| L3 | E0(52%) E1(48%) | E2(0%) E3(0%) | 0.51 |
| L6 | E0(56%) E2(44%) | E1(0%) E3(0%) | 0.53 |
| L9 | E1(49%) E2(51%) | E0(0%) E3(0%) | 0.50 |
| L12 | E0(54%) E1(29%) E2(15%) E3(1%) | — | 0.43 |

## exp1 vs exp2 对比

| 指标 | exp1 (BL=0.3) | exp2 (BL=1.0) |
|------|:-----:|:-----:|
| mAP50 | 0.176 | 0.176 |
| 死专家/层 | 2 | 2 |
| L12 Gini | 0.49 | 0.43 |

## 结论

balance_loss 从 0.3 提到 1.0，**路由僵化模式完全相同**——每层仍有 2 个死专家。

### 分析

1. balance_loss 是全局约束，但路由器已经形成了稳定偏好，30 epoch 内不足以翻转
2. 初始化 + 前几个 epoch 的路由选择锁定了"谁活谁死"
3. 论文消融实验中 λ=1.5 配合 DFL 移除才能达到最佳效果
4. 可能需要更长的训练（论文 600 epoch）让 balance_loss 逐步起作用
5. 也可能需要调低 noise_std 或延长 warmup

### 下一步

- 考虑 **warmup 策略**：expert_warmup_epochs 从 3 提到 10+
- 或直接采用动态调度：初期高 balance → 后期衰减
