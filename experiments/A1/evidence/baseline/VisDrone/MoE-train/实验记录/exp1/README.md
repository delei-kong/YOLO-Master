# Exp1: balance_loss=0.3, soft Top-2, 4 experts

## 实验目的

https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/8di125l1?nw=nwuserdeleikong
验证论文 ES-MoE 设计的正确训练方式：训练时 `_dense_forward`（所有 expert 计算，soft Top-K 保梯度），但 balance_loss=0.3 是否足以防止专家塌缩。

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | EsMoE-N (yolo-master-n.yaml) |
| 专家数 | 4 per layer |
| top_k | 2 |
| balance_loss | **0.3** |
| warmup | 无 (固定 balance_loss) |
| noise_std | 0.5 |
| epochs | 80（25 时提前停止）|
| batch | 32 |
| imgsz | 640 |
| optimizer | MuSGD (lr=0.01) |
| `use_dense` 逻辑 | `self.training` 保留（正确） |

## 训练结果（epoch 27）

| 指标 | 值 |
|------|-----|
| mAP50 | 0.176 |
| mAP50-95 | 0.089 |

## 路由分化

| Layer | 活跃 | 死 | Gini | 持续 |
|:-----:|------|-----|:---:|:---:|
| L3 | E0(57%) E1(43%) | E2(0%) E3(0%) | 0.54 | 25/25 epoch |
| L6 | E0(52%) E2(48%) | E1(0%) E3(0%) | 0.51 | 25/25 epoch |
| L9 | E1(56%) E2(44%) | E0(0%) E3(0%) | 0.53 | 25/25 epoch |
| L12 | E0(44%) E1(56%) | E2(0%) E3(0%) | 0.49 | 25/25 epoch |

## 结论

balance_loss=0.3 **不足以防止专家塌缩**。

### 原因分析

1. 初始化时 Router 有微小随机偏好 → soft Top-K 放大为固定选择
2. 被冷落的 expert 在 `_dense_forward` 中权重=0 → 检测 loss 梯度=0
3. 被冷落的 expert 在 Router softmax 中 mask=0 → Router 梯度也断
4. 唯一梯度来源是 balance_loss，但 0.3 力度不够拉起 dead expert
5. 恶性循环：不被选 → 没梯度 → 不变好 → 永远不被选

### 论文依据

论文 3.5 节明确指出 load balancing loss 的设计目的就是防止 "better-initialized experts" 导致的 collapse。消融实验中 λ=0.5~1.5，我们的 0.3 低于最低值。

### 下一步

提高 balance_loss 至 1.0，或采用 warmup+衰减策略（初期 1.0 强制均衡 → 后期降低促分化）。
