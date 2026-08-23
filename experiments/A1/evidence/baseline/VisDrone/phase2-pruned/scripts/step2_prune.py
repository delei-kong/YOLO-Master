#!/usr/bin/env python3
"""Step 2: 对 baseline 模型执行专家剪枝。

用法:
    python step2_prune.py        # 循环 2 mode × 5 threshold = 10 组
    python step2_prune.py --mode usage --threshold 0.15  # 单组

产出: phase2-pruned/results/
"""

import sys, json, csv
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/root/workspace/YOLO-Master")

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.pruning import MoEPruner

MODEL_PATH = "/root/workspace/YOLO-Master-docs/issue2/VisDrone/phase1-baseline/baseline/best.pt"
DATA_YAML  = "/root/workspace/YOLO-Master/ultralytics/cfg/datasets/VisDrone.yaml"
USAGE_JSON = "/root/workspace/YOLO-Master-docs/issue2/VisDrone/phase2-pruned/usage_stats.json"
OUT_DIR    = Path("/root/workspace/YOLO-Master-docs/issue2/VisDrone/phase2-pruned/results")

# ============================================================
# 1. 加载预计算的 usage_stats，重建 ExpertStats 对象
# ============================================================
with open(USAGE_JSON) as f:
    raw = json.load(f)

# 短名称 -> 完整路由器模块名 的映射
NAME_MAP = {
    "L3":  "model.3.routing",
    "L6":  "model.6.routing",
    "L9":  "model.9.routing",
    "L12": "model.12.routing",
}

usage_stats = {}
for short, experts in raw.items():
    layer_key = NAME_MAP[short]
    usage_stats[layer_key] = {}
    for eid_str, info in experts.items():
        usage_stats[layer_key][int(eid_str)] = SimpleNamespace(
            hits=info["hits"],
            avg_weight=info["avg_weight"],
        )

# ============================================================
# 2. 评估剪枝后模型的函数
# ============================================================
def eval_pruned(pt_path: str) -> dict:
    """在 test 集上评估剪枝模型（sparse top-2）。返回指标字典。"""
    m = YOLO(pt_path)
    for mod in m.model.modules():
        if isinstance(mod, ES_MOE):
            mod.use_sparse_inference = True
            mod.use_top_k = True
            mod.top_k = 2
    results = m.val(data=DATA_YAML, split="test", imgsz=640, batch=16, device=0, verbose=False)
    # 从 model 获取结构信息
    expert_counts = {}
    for name, mod in m.model.named_modules():
        if isinstance(mod, ES_MOE):
            short = name.replace("model.", "").replace(".routing", "").replace(".", "_")
            expert_counts[short] = mod.num_experts
    return {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map50_95),
        "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 3),
        "experts_per_layer": expert_counts,
    }


# ============================================================
# 3. 主函数：跑一组 (mode, threshold) 的剪枝
# ============================================================
def run_one(mode: str, threshold: float) -> dict:
    label = f"{mode}_t{threshold:.2f}"
    out_pt = str(OUT_DIR / f"pruned_{label}.pt")

    print(f"\n{'='*60}")
    print(f" 剪枝: {label}")
    print(f"{'='*60}")

    pruner = MoEPruner(
        MODEL_PATH,
        threshold=threshold,
        dataset=DATA_YAML,
        device="0",
        importance_mode=mode,
        usage_stats=usage_stats,
    )
    success = pruner.prune(out_pt)
    if not success:
        print(f"  {label}: 剪枝失败")
        return {"mode": mode, "threshold": threshold, "status": "failed"}

    # 评估
    print(f"\n 评估 {label}...")
    metrics = eval_pruned(out_pt)
    row = {
        "mode": mode, "threshold": threshold,
        "status": "ok",
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "params_M": metrics["params_M"],
        "experts": str(metrics["experts_per_layer"]),
    }
    print(f"  mAP50={row['map50']:.4f}  mAP50-95={row['map50_95']:.4f}  params={row['params_M']}M  experts={row['experts']}")
    return row


# ============================================================
# 4. 入口
# ============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["usage", "usage_weight"], default=None)
    p.add_argument("--threshold", type=float, default=None)

    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    MODES = ["usage", "usage_weight"]
    THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]

    if args.mode and args.threshold is not None:
        # 单组模式
        result = run_one(args.mode, args.threshold)
        print(json.dumps(result, indent=2))
    else:
        # 全量循环
        rows = []
        for mode in MODES:
            for t in THRESHOLDS:
                row = run_one(mode, t)
                rows.append(row)

        # 写入 CSV
        csv_path = OUT_DIR / "pruning_results.csv"
        keys = ["mode", "threshold", "status", "map50", "map50_95", "params_M", "experts"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"\n结果已保存到 {csv_path}")
