# PR: brain-tumor — YOLO-Master-EsMoE-N 复现

## 结果

| 指标 | 值 | epoch |
|------|-----|-------|
| **mAP50** | **0.551** | 38 |
| **mAP50-95** | **0.381** | 38 |

### v0.1-N vs EsMoE-N

| 指标 | v0.1-N | EsMoE-N |
|------|--------|---------|
| mAP50 | 0.531 | **0.551** |
| mAP50-95 | 0.364 | **0.381** |
| 参数量 | 7.55M | **2.69M** |
| 每 epoch | 14.1s | **9.5s** |

> EsMoE-N 更小、更快、更准。

## W&B

🔗 https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/hpi4dvhj

## 复现命令

```bash
python scripts/reproduce/reproduce_brain_tumor.py \
  --model EsMoE-N --epochs 200 --batch 32 --device 0 --no-sparse-eval
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
