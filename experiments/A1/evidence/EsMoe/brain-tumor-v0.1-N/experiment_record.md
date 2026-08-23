# 实验记录：brain-tumor — YOLO-Master-v0.1-N

## 基本信息

| 字段 | 值 |
|------|-----|
| **实验 ID** | 6-bt-1 |
| **Issue** | [#49](https://github.com/Tencent/YOLO-Master/issues/49) |
| **数据集** | [brain-tumor](https://docs.ultralytics.com/datasets/detect/brain-tumor/) |
| **模型** | YOLO-Master-v0.1-N |
| **配置文件** | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` |
| **训练日期** | 2026-07-23 |
| **状态** | ✅ 完成 |

## 数据集概况

| 属性 | 值 |
|------|-----|
| 类别数 | 2（negative, positive） |
| 训练图像 | 893 |
| 验证图像 | 223 |
| 场景 | 医学影像 — 脑肿瘤检测 |

## 训练配置

| 参数 | 值 |
|------|-----|
| `epochs` | 200 |
| `imgsz` | 640 |
| `batch` | 32 |
| `optimizer` | auto → AdamW (lr=0.000526) |
| `pretrained` | False（从零训练） |
| `seed` | 42 |
| `device` | NVIDIA GeForce RTX 4090 (24GB) |
| `训练耗时` | 2819s = **47.0 min**（200 epochs） |
| `平均每 epoch` | **14.1s** |

## 训练命令

```bash
python scripts/reproduce/reproduce_brain_tumor.py \
  --model v0.1-N \
  --epochs 200 \
  --batch 32 \
  --device 0
```

## 最佳结果

| 指标 | 值 | epoch |
|------|-----|-------|
| **mAP50** | **0.5305** | 40 |
| **mAP50-95** | **0.3639** | 40 |

### 最终 epoch (200)

| 指标 | 值 |
|------|-----|
| mAP50 | 0.4597 |
| mAP50-95 | 0.3316 |

> ⚠️ 严重过拟合：epoch 40 达到最佳后持续下降至 0.46。893 张训练图对 200 epochs 过多，建议早停。

## Loss 收敛

| Loss | epoch 1 | best (40) | epoch 200 |
|------|---------|-----------|-----------|
| train/box_loss | 3.643 | 1.128 | 0.773 |
| train/cls_loss | 6.272 | 1.184 | 0.669 |
| train/dfl_loss | 4.430 | 1.287 | 1.042 |

## W&B 训练曲线

🔗 https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/n1qozc3y

## 产物清单

```
brain-tumor-v0.1-N/
└── v0.1-N/
    ├── results.csv
    ├── results.png
    ├── args.yaml
    ├── labels.jpg
    ├── output.log
    ├── wandb_history.csv
    ├── wandb_meta.json
    ├── wandb-summary.json
    ├── config.yaml
    └── requirements.txt
```
