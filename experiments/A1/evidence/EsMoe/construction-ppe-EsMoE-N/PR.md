# PR: construction-ppe — YOLO-Master-EsMoE-N 复现

## 概述

在 construction-ppe 数据集上完成 **EsMoE-N** 基线训练，与已提交的 v0.1-N 结果形成完整对比。

Related: [#49](https://github.com/Tencent/YOLO-Master/issues/49)

## 环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 (24GB) |
| CUDA | 12.4 |
| PyTorch | 2.5.1 |
| ultralytics | 8.3.240 |

## 结果

### EsMoE-N

| 指标 | 值 | epoch |
|------|-----|-------|
| **mAP50** | **0.535** | 171 |
| **mAP50-95** | **0.267** | 171 |

### v0.1-N vs EsMoE-N

| 指标 | v0.1-N | EsMoE-N |
|------|--------|---------|
| mAP50 | 0.527 | **0.535** |
| mAP50-95 | 0.253 | **0.267** |
| 参数量 | 7.55M | **2.69M** |
| 每 epoch | 12.3s | 12.5s |

> EsMoE-N 以 1/3 参数略优于 v0.1-N，证明 ES_MOE 在工业安全场景有效。

## W&B 训练曲线

🔗 https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/cvsslgdu

## 复现命令

```bash
# 数据集下载（~170MB）
wget -P <datasets_dir> \
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('construction-ppe.yaml', autodownload=True)"

# EsMoE-N（⚠️ 必须加 --no-sparse-eval）
python scripts/reproduce/reproduce_construction_ppe_v01n.py \
  --model EsMoE-N \
  --epochs 200 \
  --batch 32 \
  --device 0 \
  --no-sparse-eval
```

## 产物

完整日志见 `docs/issue6/construction-ppe-EsMoE-N/`

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
