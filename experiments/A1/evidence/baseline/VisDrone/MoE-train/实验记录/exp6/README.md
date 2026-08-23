# exp6: balance_loss=1.5, soft Top-2, 3 experts

## 实验目的

验证 3 experts + balance_loss=1.5 下的路由分化行为。

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | EsMoE-N (yolo-master-n.yaml) |
| 专家数 | 3 per layer |
| top_k | 2 |
| balance_loss | 1.5 |
| epochs | 30 |
| batch | 32 |
| imgsz | 640 |

## 训练结果 (epoch 30)

| 指标 | 值 |
|------|-----|
| mAP50 | 0.14839 |

## 路由分析

见 `routing/dataset_routing_report.json` 和热力图。
