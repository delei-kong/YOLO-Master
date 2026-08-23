# 实验记录：brain-tumor — YOLO-Master-EsMoE-N

## 基本信息

| 字段 | 值 |
|------|-----|
| **实验 ID** | 6-bt-2 |
| **Issue** | [#49](https://github.com/Tencent/YOLO-Master/issues/49) |
| **数据集** | brain-tumor |
| **模型** | YOLO-Master-EsMoE-N |
| **配置文件** | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` |
| **训练日期** | 2026-07-23 |
| **状态** | ✅ 完成 |

## 训练配置

| 参数 | 值 |
|------|-----|
| `epochs` | 200 |
| `imgsz` | 640 |
| `batch` | 32 |
| `--no-sparse-eval` | **True** |
| `optimizer` | auto |
| `训练耗时` | 1906s = **31.8 min**（200 epochs） |
| `平均每 epoch` | **9.5s** |

## 训练命令

```bash
python scripts/reproduce/reproduce_brain_tumor.py \
  --model EsMoE-N \
  --epochs 200 \
  --batch 32 \
  --device 0 \
  --no-sparse-eval
```

## 最佳结果

| 指标 | 值 | epoch |
|------|-----|-------|
| **mAP50** | **0.5515** | 38 |
| **mAP50-95** | **0.3809** | 38 |

> 与 v0.1-N 相似，epoch 38 后过拟合。

## v0.1-N vs EsMoE-N

| 指标 | v0.1-N | EsMoE-N | 差异 |
|------|--------|---------|------|
| mAP50 | 0.531 | **0.551** | +3.9% |
| mAP50-95 | 0.364 | **0.381** | +4.7% |
| 参数量 | 7.55M | **2.69M** | -64% |
| 每 epoch | 14.1s | **9.5s** | -33% |
| 总耗时 | 47.0min | **31.8min** | -32% |

> EsMoE-N 以 1/3 参数、2/3 时间，mAP 优于 v0.1-N。

## W&B 训练曲线

🔗 https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/hpi4dvhj

## 产物清单

```
brain-tumor-EsMoE-N/
└── EsMoE-N/
    ├── results.csv, results.png
    ├── args.yaml, config.yaml
    ├── output.log
    ├── wandb_history.csv, wandb_meta.json, wandb-summary.json
    └── requirements.txt
```
