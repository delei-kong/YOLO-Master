# 实验记录：construction-ppe — YOLO-Master-EsMoE-N

## 基本信息

| 字段 | 值 |
|------|-----|
| **实验 ID** | 6-ppe-2 |
| **Issue** | [#49](https://github.com/Tencent/YOLO-Master/issues/49) |
| **数据集** | [construction-ppe](https://docs.ultralytics.com/datasets/detect/construction-ppe/) |
| **模型** | YOLO-Master-EsMoE-N |
| **配置文件** | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` |
| **训练日期** | 2026-07-22 ~ 2026-07-23 |
| **状态** | ✅ 完成 |

## 数据集概况

| 属性 | 值 |
|------|-----|
| 类别数 | 11 |
| 训练图像 | 1,132 |
| 验证图像 | 143 |
| 场景 | 工业安全 — 工地个人防护装备检测 |

## 训练配置

| 参数 | 值 |
|------|-----|
| `epochs` | 200 |
| `imgsz` | 640 |
| `batch` | 32 |
| `optimizer` | SGD (auto, lr0=0.01, momentum=0.9) |
| `pretrained` | False（从零训练） |
| `seed` | 42 |
| `deterministic` | True |
| `AMP` | True |
| **`--no-sparse-eval`** | **True**（ES_MOE dense 评估） |
| `device` | NVIDIA GeForce RTX 4090 (24GB) |
| `训练耗时` | 2506s = **41.8 min**（200 epochs） |
| `平均每 epoch` | **12.5s** |

完整参数见：[args.yaml](./EsMoE-N/args.yaml)

## 训练命令

```bash
python scripts/reproduce/reproduce_construction_ppe_v01n.py \
  --model EsMoE-N \
  --epochs 200 \
  --batch 32 \
  --device 0 \
  --no-sparse-eval
```

## 最佳结果

| 指标 | 值 | epoch |
|------|-----|-------|
| **mAP50** | **0.5348** | 171 |
| **mAP50-95** | **0.2669** | 171 |

### 最终 epoch (200)

| 指标 | 值 |
|------|-----|
| mAP50 | 0.5281 |
| mAP50-95 | 0.2660 |
| Precision | 0.6851 |
| Recall | 0.4938 |

## Loss 收敛

| Loss | epoch 1 | best epoch (171) | epoch 200 |
|------|---------|-----------------|-----------|
| train/box_loss | 3.636 | 1.492 | 1.442 |
| train/cls_loss | 4.633 | 1.078 | 0.981 |
| train/dfl_loss | 4.263 | 1.632 | 1.689 |
| train/moe_loss | 2.803 | 1.000 | 1.000 |

## v0.1-N vs EsMoE-N 对比

| 指标 | v0.1-N | EsMoE-N | 差异 |
|------|--------|---------|------|
| **mAP50** | 0.5269 | **0.5348** | +0.008 (+1.5%) |
| **mAP50-95** | 0.2529 | **0.2669** | +0.014 (+5.5%) |
| **参数** | 7.55M | **2.69M** | -64% |
| **参数量** | 7.55M | **2.69M** | -64% |
| **每 epoch 耗时** | 12.3s | 12.5s | +1.6% |
| **总训练时间** | 41.0 min | 41.8 min | +2.0% |
| **最佳 epoch** | 142 | 171 | |
| **MoE 模块** | ModularRouterExpertMoE | **ES_MOE** | |
| `--no-sparse-eval` | 不需要 | **必须** | |

> 结论：EsMoE-N 以仅 1/3 的参数量，在 mAP50 和 mAP50-95 上均略优于 v0.1-N。训练速度几乎相同。

## W&B 训练曲线

🔗 https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/cvsslgdu

## 产物清单

```
construction-ppe-EsMoE-N/
├── summary.csv
└── EsMoE-N/
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

## 已知问题

### 1. final_eval 阶段 Router NaN（与 v0.1-N 相同）

**现象**：训练 200 epochs 完成后，final_eval() 阶段报错：
```
RuntimeError: Router input contains NaN/Inf values [EfficientSpatialRouter]
```

**影响**：per-epoch 指标完整保存（200 epoch），不影响结果收集。

## 结论

EsMoE-N 在 construction-ppe 上使用 dense 评估（`--no-sparse-eval`）达到 **mAP50=0.535, mAP50-95=0.267**，以 **2.69M 参数**（v0.1-N 的 36%）**略优于 v0.1-N**。在工业安全 PPE 检测场景下，EsMoE-N 是更优选择：更小、更准。
