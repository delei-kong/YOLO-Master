# PR: brain-tumor — YOLO-Master-v0.1-N 复现

## 概述

在 brain-tumor 数据集上完成 v0.1-N 基线训练。

Related: [#49](https://github.com/Tencent/YOLO-Master/issues/49)

## 结果

| 指标 | 值 | epoch |
|------|-----|-------|
| **mAP50** | **0.531** | 40 |
| **mAP50-95** | **0.364** | 40 |

> ⚠️ 小数据集过拟合：最佳 epoch 40，之后 mAP 持续下降至 0.46。

## W&B

🔗 https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/n1qozc3y

## 复现命令

```bash
# 下载（~4MB）
wget -P <datasets_dir> "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('brain-tumor.yaml', autodownload=True)"

# 训练
python scripts/reproduce/reproduce_brain_tumor.py --model v0.1-N --epochs 200 --batch 32 --device 0
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
