# 实验记录：construction-ppe — YOLO-Master-v0.1-N

## 基本信息

| 字段 | 值 |
|------|-----|
| **实验 ID** | 6-ppe-1 |
| **Issue** | [#49](https://github.com/Tencent/YOLO-Master/issues/49) |
| **数据集** | [construction-ppe](https://docs.ultralytics.com/datasets/detect/construction-ppe/) |
| **模型** | YOLO-Master-v0.1-N |
| **配置文件** | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` |
| **训练日期** | 2026-07-21 ~ 2026-07-22 |
| **状态** | ✅ 完成 |

## 数据集概况

| 属性 | 值 |
|------|-----|
| 类别数 | 11（helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots） |
| 训练图像 | 1,132 |
| 验证图像 | 143 |
| 测试图像 | 141 |
| 场景 | 工业安全 — 工地个人防护装备检测 |

## 训练配置

| 参数 | 值 |
|------|-----|
| `epochs` | 200 |
| `imgsz` | 640 |
| `batch` | 32 |
| `optimizer` | SGD (auto) |
| `lr0` | 0.01 |
| `lrf` | 0.01 |
| `momentum` | 0.9 |
| `weight_decay` | 0.0005 |
| `warmup_epochs` | 3 |
| `cos_lr` | True |
| `mosaic` | 1.0 |
| `auto_augment` | randaugment |
| `erasing` | 0.4 |
| `pretrained` | False（从零训练） |
| `lora_r` | 0（全量训练，非 LoRA） |
| `seed` | 42 |
| `deterministic` | True |
| `AMP` | True |
| `device` | NVIDIA GeForce RTX 4090 (24GB) |
| `workers` | 16 |
| `训练耗时` | ~41 分钟（200 epochs） |

完整参数见：[args.yaml](./v0.1-N/args.yaml)

## 训练命令

```bash
conda activate yolo_master
cd /root/workspace/YOLO-Master

# 训练
python scripts/reproduce/reproduce_ppe.py \
  --model v0.1-N \
  --epochs 200 \
  --batch 32 \
  --device 0
```

## 最佳结果

| 指标 | 值 | epoch |
|------|-----|-------|
| **mAP50** | **0.5269** | 142 |
| **mAP50-95** | **0.2529** | 142 |
| Precision | 0.7820 | 142 |
| Recall | 0.4637 | 142 |

### 最终 epoch (200) 结果

| 指标 | 值 |
|------|-----|
| mAP50 | 0.5193 |
| mAP50-95 | 0.2521 |
| Precision | 0.7753 |
| Recall | 0.4644 |

## Loss 收敛趋势

| Loss | epoch 1 | best epoch (142) | epoch 200 | 下降幅度 |
|------|---------|-----------------|-----------|---------|
| train/box_loss | 3.580 | 1.597 | 1.503 | -58.0% |
| train/cls_loss | 4.631 | 1.197 | 1.048 | -77.4% |
| train/dfl_loss | 4.249 | 1.701 | 1.730 | -59.3% |
| train/moe_loss | 3.021 | 0.999 | 1.001 | -66.9% |

moe_loss 从 3.02 收敛到 ~1.0，MoE 模块正常学习到路由策略。

## 产物清单

```
construction-ppe-v0.1N/
├── v0.1-N/
│   ├── results.csv      # per-epoch 完整训练指标（200 行）
│   ├── results.png      # 训练曲线图（loss + mAP + precision/recall + LR）
│   ├── args.yaml        # 完整训练参数（可复现）
│   └── labels.jpg       # 数据集标签分布
└── summary.csv          # 汇总表
```

## W&B 训练曲线

🔗 https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/3z8vcfuy

## 已知问题

### 1. final_eval 阶段 Router NaN

**现象**：训练 200 epochs 完成后，`final_eval()` 在加载 `best.pt` 时报错：
```
RuntimeError: Router input contains NaN/Inf values [EfficientSpatialRouter]
```

**影响**：per-epoch validation 指标全部正常（results.csv 中有 200 个 epoch 的完整 mAP、loss 数据），不影响训练过程本身的指标收集。

**可能原因**：construction-ppe 数据量较小（1132 张训练图，11 类），MoE router 在 final_eval 时出现数值不稳定。EfficientSpatialRouter 的输入包含 NaN/Inf。

**解决方案**：待排查。后续可尝试：
- 减少 epochs 或增加 warmup_epochs
- 降低学习率
- 在 final_eval 前对 checkpoint 做数值稳定性检查

## 结论

v0.1-N 在 construction-ppe（工业安全 PPE 检测）上从零训练 200 epochs，达到了 **mAP50=0.527, mAP50-95=0.253**。MoE router 正常收敛，moe_loss 从 3.02 降至 ~1.0。该结果可作为 construction-ppe 数据集上的 baseline。

与 VisDrone/SKU-110K 报告结果的对比：construction-ppe 作为 11 类小样本工业场景，mAP 值低于 VisDrone（10 类/6481 训练图）是合理的。11 类中有多个相似类别（如 helmet vs no_helmet），增加了分类难度。
