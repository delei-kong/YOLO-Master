#!/usr/bin/env python3
"""步骤3：baseline 模型路由专家行为评估。

评估指标：
- 每层 expert 命中率分布（利用 ExpertUsageTracker 的 forward hooks 累积全量数据）
- 每个 expert 被选中时的平均权重
- 每层 Gini 系数（基于 ExpertUsageTracker 累积数据计算，覆盖全部 1610 张图）
- 每层 expert 权重分布（从 last_routing_snapshot 获取详细分布）

运行方式：
    bash step3_route_analysis.sh
"""

import sys

sys.path.insert(0, "/root/workspace/YOLO-Master")

import numpy as np
from ultralytics import YOLO
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker
from ultralytics.nn.modules.moe.diagnostics import _gini
from ultralytics.nn.modules.moe.modules import ES_MOE

MODEL_PATH = "/root/workspace/YOLO-Master-docs/issue2/VisDrone/phase1-baseline/baseline/best.pt"
DATA_YAML = "/root/workspace/YOLO-Master/ultralytics/cfg/datasets/VisDrone.yaml"


def compute_gini_from_usage(usage_dict: dict) -> float:
    """从 ExpertUsageTracker 累积的 usage_stats 计算 Gini 系数。"""
    values = [stats.hits for stats in usage_dict.values()]
    if not values or sum(values) == 0:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    arr = arr / arr.sum()
    return float(_gini(list(arr)))


print("=" * 60)
print("步骤3：路由专家行为评估")
print("=" * 60)

# 1. 加载模型
print("[1/4] 加载模型...")
model = YOLO(MODEL_PATH)

# 2. 开启 ES_MOE sparse inference (top_k=2)，模型本身即为 sparse 训练
print("[2/4] 设置 ES_MOE 为 sparse inference (top_k=2)...")
count = 0
for module in model.model.modules():
    if isinstance(module, ES_MOE):
        module.use_sparse_inference = True
        module.use_top_k = True
        module.top_k = 2  # 4 选 2
        count += 1
print(f"      已将 {count} 个 ES_MOE 模块设为 sparse (top_k=2)")

# 3. 使用 ExpertUsageTracker 进行推理，收集路由统计
print("[3/4] 使用 ExpertUsageTracker 进行 test 集推理（1610 张）...")
with ExpertUsageTracker(model.model) as tracker:
    model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=16,
        device=0,
    )
    tracker.print_report()

# 4. 基于 ExpertUsageTracker 累积数据计算 Gini 系数（覆盖全量数据）
print("\n" + "=" * 60)
print("[4/4] 全量路由统计汇总（基于 ExpertUsageTracker 累积数据）")
print("=" * 60)

print(f"\n总 token 数: {tracker.total_tokens:,}\n")

for layer_name in sorted(tracker.usage_stats.keys()):
    expert_stats = tracker.usage_stats[layer_name]
    total_hits = sum(s.hits for s in expert_stats.values())
    num_experts = len(expert_stats)

    usages = []
    weights = []
    for eid in sorted(expert_stats.keys()):
        s = expert_stats[eid]
        share = s.hits / total_hits * 100 if total_hits > 0 else 0
        usages.append(share)
        weights.append(s.avg_weight)

    gini = compute_gini_from_usage(expert_stats)
    dominant = max(usages) / 100 if usages else 0

    print(f"  [{layer_name}]  experts={num_experts}")
    print(f"    利用率:  {', '.join(f'E{e}: {u:.2f}%' for e, u in enumerate(usages))}")
    print(f"    平均权重: {', '.join(f'E{e}: {w:.4f}' for e, w in enumerate(weights))}")
    print(f"    Gini: {gini:.4f}  |  主导占比: {dominant:.3f}  |  "
          f"Collapse: {dominant > 0.8}")
    print()

# 5. 补充：从 last_routing_snapshot 读取最后 batch 的快照细节
#    （仅作抽样参考，完整统计以 ExpertUsageTracker 累积数据为准）
print("=" * 60)
print("[补充] 最后 batch 的 routing snapshot 快照...")
print("=" * 60)

from ultralytics.nn.modules.moe.diagnostics import routing_runtime_metrics as _rtm, collect_moe_diagnostics as _cmd

rt_metrics = _rtm(model.model)
if rt_metrics["routed_layers"] > 0:
    for name, layer in rt_metrics["layers"].items():
        print(f"  [{layer['module_type']}] {name}")
        print(f"    Gini: {layer['gini']:.4f},  Entropy: {layer['entropy']:.4f},  "
              f"Collapse: {layer['collapse_flag']}")
else:
    print("  (无数据)")

diags = _cmd(model.model)
if diags:
    for diag in diags:
        weight_str = ", ".join(f"w{i}:{w:.4f}" for i, w in enumerate(diag.mean_topk_weight or []))
        print(f"  [{diag.module_type}] {diag.name}")
        print(f"    aux_loss={diag.aux_loss:.6f}, top_k={diag.top_k}/{diag.num_experts}")
        print(f"    avg top-k weights: {weight_str}")
else:
    print("  (无数据)")

# ================================================================
# 汇总：所有评估指标一览
# ================================================================
print("\n" + "=" * 60)
print("                    📊 BASELINE 评估指标汇总")
print("=" * 60)

# 从模型获取 FLOPs / Params
model_info = model.info(imgsz=640, detailed=False)
num_params = sum(p.numel() for p in model.model.parameters()) / 1e6
# model.info() 返回 (layers, params, gradients, gflops)
gflops = model_info[3] if len(model_info) >= 4 else "N/A"

# 收集路由统计
num_moe_layers = len(tracker.usage_stats)
experts_per_layer = {}
gini_per_layer = {}
for layer_name, stats in tracker.usage_stats.items():
    experts_per_layer[layer_name] = len(stats)
    gini_per_layer[layer_name] = compute_gini_from_usage(stats)

print(f"\n  {'指标':<20} {'值':>25}")
print(f"  {'-' * 45}")
print(f"  {'mAP50':<20} {'见 step2_val_test 输出':>25}")
print(f"  {'mAP50-95':<20} {'见 step2_val_test 输出':>25}")
print("  {:<20} {:>25}".format("FLOPs", "{:.1f} GFLOPs".format(gflops)))
print("  {:<20} {:>25}".format("Params", "{:.2f} M".format(num_params)))
print("  {:<20} {:>25}".format("Latency (inference)", "见 step2_val_test 输出"))
print("  {:<20} {:>25}".format("MoE 层数", str(num_moe_layers)))
expert_count = list(experts_per_layer.values())[0] if experts_per_layer else "N/A"
expert_detail = str(expert_count) + " (x" + str(num_moe_layers) + "层, sparse top-2)"
print("  {:<20} {:>25}".format("每层保留专家数", expert_detail))
for name, g in gini_per_layer.items():
    short = name.replace("model.", "L")
    print("  {:<20} {:>25}".format("  Gini (" + short + ")", "{:.4f}".format(g)))
avg_gini = sum(gini_per_layer.values()) / len(gini_per_layer) if gini_per_layer else 0.0
print("  {:<20} {:>25}".format("Gini (平均)", "{:.4f}".format(avg_gini)))
print()

print("\n路由专家行为评估完成!")
