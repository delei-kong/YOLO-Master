#!/usr/bin/env python3
"""快速检查 checkpoint 的路由行为（只跑 100 张图，不占训练 GPU 时间）。
用法: python check_routing.py [last.pt 路径，默认从 weights/last.pt]
"""

import sys
sys.path.insert(0, "/root/workspace/YOLO-Master")

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train/VisDrone_EsMoE-N/weights/last.pt"

print(f"[check] 加载: {CHECKPOINT}")
model = YOLO(CHECKPOINT)

# 确认 sparse 设置
for m in model.model.modules():
    if isinstance(m, ES_MOE):
        print(f"  {type(m).__name__}: sparse={m.use_sparse_inference}, top_k={m.top_k}, experts={m.num_experts}")

# 快速路由诊断（仅跑 100 张）
with ExpertUsageTracker(model.model) as tracker:
    model.val(data="VisDrone.yaml", split="test", imgsz=640, batch=16, device=0)
    tracker.print_report()
