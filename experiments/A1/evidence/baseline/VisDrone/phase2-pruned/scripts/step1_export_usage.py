#!/usr/bin/env python3
"""Step 1: 导出 ExpertUsageTracker 累积的 usage_stats 到 JSON，供 MoEPruner 注入使用。"""

import sys, json

sys.path.insert(0, "/root/workspace/YOLO-Master")

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

MODEL = "/root/workspace/YOLO-Master-docs/issue2/VisDrone/phase1-baseline/baseline/best.pt"
DATA = "/root/workspace/YOLO-Master/ultralytics/cfg/datasets/VisDrone.yaml"
OUT = "/root/workspace/YOLO-Master-docs/issue2/VisDrone/phase2-pruned/usage_stats.json"

# 1. 加载模型 + 设置 sparse 模式
print("[1/3] 加载 baseline 模型...")
m = YOLO(MODEL)
for mod in m.model.modules():
    if isinstance(mod, ES_MOE):
        mod.use_sparse_inference = True
        mod.use_top_k = True
        mod.top_k = 2

# 2. 运行 ExpertUsageTracker
print("[2/3] 运行 test 集诊断（1610 张，约 2 分钟）...")
with ExpertUsageTracker(m.model) as tracker:
    m.val(data=DATA, split="test", imgsz=640, batch=16, device=0)

# 3. 导出 usage_stats
print("[3/3] 导出 usage_stats...")
stats_out = {}
for name, experts in tracker.usage_stats.items():
    short = name.replace("model.", "L").replace(".routing", "")
    stats_out[short] = {}
    total = sum(s.hits for s in experts.values())
    for eid, s in experts.items():
        stats_out[short][str(eid)] = {
            "hits": float(s.hits),
            "weighted_sum": float(s.weighted_sum),
            "usage_pct": float(s.hits / total * 100) if total > 0 else 0.0,
            "avg_weight": float(s.avg_weight),
        }

with open(OUT, "w") as f:
    json.dump(stats_out, f, indent=2, ensure_ascii=False)

# 打印摘要
print("\n每层 expert 利用率 (usage_pct):")
for name, experts in stats_out.items():
    usage_str = "  ".join(f"E{e}: {v['usage_pct']:.1f}%" for e, v in experts.items())
    weight_str = "  ".join(f"E{e}: {v['avg_weight']:.3f}" for e, v in experts.items())
    print(f"  {name}: {usage_str}")
    print(f"  {' ' * len(name)}  {weight_str}")

print(f"\n已导出到 {OUT}")
