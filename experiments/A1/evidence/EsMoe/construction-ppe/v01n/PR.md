# PR: YOLO-Master 垂类数据集复现 — construction-ppe

## 概述

本 PR 在 **construction-ppe**（工业安全 PPE 检测）数据集上复现了 YOLO-Master 的 nano 基线模型训练。

相关 Issue: [#49](https://github.com/Tencent/YOLO-Master/issues/49)

## 环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| CUDA | 12.4 |
| PyTorch | 2.5.1 |
| ultralytics | 8.3.240 |
| Python | 3.11.15 |
| OS | Ubuntu 22.04 |

## 数据集

[construction-ppe](https://docs.ultralytics.com/datasets/detect/construction-ppe/) — 工地个人防护装备检测

| 属性 | 值 |
|------|-----|
| 类别数 | 11 |
| 训练集 | 1,132 张 |
| 验证集 | 143 张 |
| 场景 | 工业安全 |

## 训练配置

```
imgsz: 640
epochs: 200
batch: 32
optimizer: SGD (auto, lr0=0.01, momentum=0.9)
pretrained: False（从零训练）
LoRA: 关闭（全量训练）
seed: 42
```

## 结果

### YOLO-Master-v0.1-N

| 指标 | 值 | epoch |
|------|-----|-------|
| **mAP50** | **0.527** | 142 |
| **mAP50-95** | **0.253** | 142 |
| Precision | 0.782 | 142 |
| Recall | 0.464 | 142 |

### Loss 收敛

| Loss | 初始值 | 最终值 | 下降 |
|------|--------|--------|------|
| box_loss | 3.58 | 1.50 | -58% |
| cls_loss | 4.63 | 1.05 | -77% |
| dfl_loss | 4.25 | 1.73 | -59% |
| moe_loss | 3.02 | 1.00 | -67% |

## W&B 训练曲线

🔗 https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/3z8vcfuy

![训练曲线](./v0.1-N/results.png)

## 复现命令

### 1. 下载数据集

```bash
# construction-ppe 数据集从 GitHub Release 下载（~170MB）
wget -P <datasets_dir> \
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"

# 自动解压 + 标注转换
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('construction-ppe.yaml', autodownload=True)"
```

### 2. 训练

```bash
# v0.1-N (200 epochs)
python scripts/reproduce/reproduce_ppe.py \
  --model v0.1-N \
  --epochs 200 \
  --batch 32 \
  --device 0

# EsMoE-N (必须加 --no-sparse-eval)
python scripts/reproduce/reproduce_ppe.py \
  --model EsMoE-N \
  --epochs 200 \
  --batch 32 \
  --device 0 \
  --no-sparse-eval
```

## 新增文件

```
scripts/reproduce/reproduce_ppe.py    # construction-ppe 复现脚本（~30 行）
```

## 已知问题

1. **final_eval Router NaN**：v0.1-N 在 200 epoch 训练后 final_eval 阶段 EfficientSpatialRouter 出现 NaN，best.pt 未能成功保存。但 200 epoch 的 per-epoch validation 指标全部正常。可能原因：数据量小（1132 张图）导致 Router 数值不稳定。
2. **EsMoE-N 需 `--no-sparse-eval`**：与 VisDrone/SKU-110K 相同，EsMoE-N 的 sparse inference 与 dense training 不一致会导致 val mAP 崩塌。

## 产物

完整训练日志和曲线见 `docs/issue6/construction-ppe-v0.1N/`：
- `v0.1-N/results.csv` — per-epoch 指标
- `v0.1-N/results.png` — 训练曲线
- `v0.1-N/args.yaml` — 完整参数

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
