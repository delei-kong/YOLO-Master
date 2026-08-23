# Baseline: 4 experts, balance_loss=1.0, 300 epochs

## 训练目的

按论文 ES-MoE 设计训练完整的 sparse baseline，用于后续剪枝实验。

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | EsMoE-N (yolo-master-n.yaml) |
| MoE 层数 | 4 (backbone L3/L6/L9/L12) |
| 每层专家数 | 4 |
| top_k | 2 (50% 稀疏) |
| balance_loss | **1.0** |
| epochs | **300** |
| imgsz | 640 |
| batch | 32 |
| device | 0 (RTX 4090) |
| optimizer | auto (MuSGD lr=0.01) |
| workers | 0 |
| warmup | expert_warmup_epochs=3 |
| noise_std | 0.5 |
| 路由诊断 | 每 5 epoch + W&B |

https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/0hvmw80p

## 代码关键配置

- `use_dense`: `self.training` 保留, 训练走 `_dense_forward` (all experts compute, soft Top-K)
- `_record_moe_snapshot`: 训练和验证都记录 (去掉 `if self.training`)
- W&B: `wandb.log()` 发送 routing 指标

## 训练命令

```bash
cd /root/workspace/YOLO-Master
rm -rf VisDrone_EsMoE-N  # 在 project 目录下
bash /root/workspace/YOLO-Master-docs/issue2/VisDrone/MoE-train/train.sh
```
