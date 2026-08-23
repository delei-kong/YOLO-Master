# 8.24 准入检查日志：detect 基线复验

> 目的：在锁定 commit `2c8253f` 工作区上复验 detect 基线可运行（准入检查 7.1 的"锁定后复验"证据）。
> 历史训练/验证证据见 `experiments/A1/evidence/baseline/VisDrone/` 与 `experiments/A1/evidence/EsMoe/`。

| 项 | 值 |
|----|----|
| 日期 | 2026-08-24 |
| 基线 commit | `2c8253fda85c6cc24354f24f569ede86cf18b1ff` |
| 环境 | Python 3.11.15 / torch 2.5.1 / ultralytics 8.4.101（本地 repo 生效）/ CUDA RTX 4090 24G |
| 入口 | dispatcher：`python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.predict","inputs":{"model":"yolo11n.pt","source":"ultralytics/assets/bus.jpg"},"params":{"device":"0"}}'` |
| 实际 CLI | `yolo predict model=/root/workspace/YOLO-Master/yolo11n.pt source=/root/workspace/YOLO-Master/ultralytics/assets/bus.jpg device=0 project=runs/agent name=yolo-predict-8cca069e` |

## 结果

```
Ultralytics 8.4.101 🚀 Python-3.11.15 torch-2.5.1 CUDA:0 (NVIDIA GeForce RTX 4090, 24081MiB)
YOLO11n summary (fused): 100 layers, 2,616,248 parameters, 0 gradients, 6.5 GFLOPs

image 1/1 /root/workspace/YOLO-Master/ultralytics/assets/bus.jpg: 640x480 4 persons, 1 bus, 87.5ms
Speed: 5.1ms preprocess, 87.5ms inference, 19.8ms postprocess per image at shape (1, 3, 640, 480)
Results saved to /root/workspace/YOLO-Master/runs/agent/yolo-predict-8cca069e
```

- 检出：4 persons, 1 bus（640x480）
- 延迟（batch=1, RTX 4090）：preprocess 5.1ms / inference 87.5ms / postprocess 19.8ms
- 产物目录：`runs/agent/yolo-predict-8cca069e/`（含 `skill_manifest.json` 完整运行记录）
