#!/usr/bin/env python3
"""
路由器退化深度分析 — 从 sparse top-k 选择角度验证。
在 sparse inference 模式下统计:
  1) 每个专家的实际被选中率（top-2 离散选择）
  2) 专家共现矩阵（哪两个专家经常一起被选）
  3) 跨图片的专家偏好方差（不同图片是否选不同专家）
  4) 单张图片的空间选择可视化（有没有空间结构）
"""

import sys
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, "/root/workspace/YOLO-Master")

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE

CHECKPOINT = "/root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train/balance-loss-1.0/VisDrone_EsMoE-N/weights/last.pt"
DATASET = "VisDrone.yaml"
DEVICE = 0

print("=" * 80)
print(" 🔬 路由器退化深度分析 (Sparse Top-K 选择)")
print("=" * 80)

model = YOLO(CHECKPOINT)
es_moe_layers = {name: m for name, m in model.model.named_modules() if isinstance(m, ES_MOE)}

print(f"检测到 {len(es_moe_layers)} 个 ES_MOE 层\n")

# 先检查 sparse inference 是否启用
for name, m in es_moe_layers.items():
    print(f"  {name}: sparse={m.use_sparse_inference}, top_k={m.top_k}, "
          f"num_experts={m.num_experts}, eager_enabled={m._eager_sparse_enabled()}")

# ═══════════════════════════════════════════════════════════════
# 方法: 注册 hook 拦截 _sparse_forward 中的 topk_indices
# ═══════════════════════════════════════════════════════════════

# 收集器: layer_name -> { 'selections': [list-of-expert-ids per token], 'images': N }
records = defaultdict(lambda: {
    'per_image_selections': [],     # 每张图 [H*W, top_k] expert indices
    'per_image_weights': [],        # 对应的 weights
    'per_image_names': [],
})

# Hook 到 routing 输出
hooks = []
for name, m in es_moe_layers.items():
    def make_hook(layer_name):
        def hook(module, input, output):
            # output 是 routing_weights [B, E, H, W]
            if isinstance(output, torch.Tensor):
                weights = output.detach().cpu()
            elif isinstance(output, tuple):
                weights = output[0].detach().cpu()
            else:
                return
            B, E, H, W = weights.shape
            # 模拟 top-k 选择
            flat = weights.permute(0, 2, 3, 1).reshape(-1, E)  # [B*H*W, E]
            # ES_MOE 实际用的是 mean over spatial → topk over experts
            # 但更准确的做法是: 对每个空间位置做 top-k
            # ES_MOE._sparse_forward 里是先 mean over spatial 再 topk
            spatial_weights = weights  # [B, E, H, W]
            per_sample_importance = spatial_weights.mean(dim=(2, 3))  # [B, E]
            # 对每张图做 top-k
            topk_vals, topk_idx = torch.topk(per_sample_importance, k=min(2, E), dim=1)  # [B, topk]
            # 记录
            for b in range(B):
                records[layer_name]['per_image_selections'].append(topk_idx[b].tolist())
                records[layer_name]['per_image_weights'].append(topk_vals[b].tolist())
        return hook
    hooks.append(m.routing.register_forward_hook(make_hook(name)))

# 跑验证
print("\n[1] 在测试集上以 sparse 模式运行验证...")
try:
    model.val(data=DATASET, split="test", imgsz=640, batch=16, device=DEVICE)
except Exception as e:
    print(f"  ⚠️ 验证出错 (可能正常): {e}")

for h in hooks:
    h.remove()

# ═══════════════════════════════════════════════════════════════
# 分析
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 📊 分析结果")
print("=" * 80)

for layer_name in sorted(records.keys()):
    data = records[layer_name]
    selections = data['per_image_selections']
    weights = data['per_image_weights']

    if not selections:
        print(f"\n  {layer_name}: 无数据")
        continue

    # 找到 expert 数量
    num_e = len(es_moe_layers[layer_name].experts)
    n_images = len(selections)

    print(f"\n{'─' * 60}")
    print(f" 📍 {layer_name}  ({n_images} 张图片)")
    print(f"{'─' * 60}")

    # ── 指标 1: 每个专家被选中的频率 ──
    expert_hit_count = [0] * num_e
    for sel in selections:
        for eid in sel:
            expert_hit_count[eid] += 1

    print(f"\n  ┌ 1) 专家被选频率 (top-2 离散选择):")
    total_hits = sum(expert_hit_count)
    ideal_rate = (2.0 / num_e) * 100  # top-2 out of 3 = 66.67%
    print(f"     {'Expert':<10} {'选中次数':<12} {'选中率':<10} {'与理想偏差':<10}")
    print(f"     {'─'*10} {'─'*12} {'─'*10} {'─'*10}")
    for ei in range(num_e):
        rate = expert_hit_count[ei] / max(total_hits, 1) * 100
        bias = rate - ideal_rate
        flag = "✅" if abs(bias) < 5 else ("⚠️" if abs(bias) < 15 else "🔴")
        print(f"     Expert {ei}   {expert_hit_count[ei]:>10,}    {rate:>6.2f}%    {bias:>+7.2f}%  {flag}")

    # ── 指标 2: 专家共现矩阵 ──
    cooccur = np.zeros((num_e, num_e))
    for sel in selections:
        for i in range(len(sel)):
            for j in range(len(sel)):
                cooccur[sel[i]][sel[j]] += 1
    # 归一化
    cooccur = cooccur / cooccur.sum(axis=1, keepdims=True)

    print(f"\n  ┌ 2) 专家共现矩阵 P(专家j | 专家i 被选中):")
    header = " " * 10 + "".join(f"  Expert{j} " for j in range(num_e))
    print(f"     {header}")
    for i in range(num_e):
        row = f"     Expert {i}  " + "".join(f"  {cooccur[i,j]:.4f} " for j in range(num_e))
        print(row)

    # ── 指标 3: 跨图片的选择方差 ──
    # 每张图片每个专家的选中次数
    per_img_expert_counts = []
    for sel in selections:
        counts = [0] * num_e
        for eid in sel:
            counts[eid] += 1
        per_img_expert_counts.append(counts)

    per_img_counts_arr = np.array(per_img_expert_counts)  # [N_images, num_e]
    img_std = per_img_counts_arr.std(axis=0)

    # 图片间的偏好是否一致
    # 计算每张图片的"专家偏好向量"，然后看图片间的方差
    per_img_pref = np.array([
        [c / max(sum(cnts), 1) for c in cnts]
        for cnts in per_img_expert_counts
    ])  # [N_images, num_e]

    img_pref_std = per_img_pref.std(axis=0)

    print(f"\n  ┌ 3) 跨图片偏好方差:")
    for ei in range(num_e):
        print(f"     Expert {ei}: per_img_pref_std={img_pref_std[ei]:.4f}"
              f"  ({'图片间有差异 ✅' if img_pref_std[ei] > 0.1 else '图片间无差异 🔴'})")

    # ── 指标 4: 熵分析 ──
    # 每张图片的 top-2 选择的"独特性"
    # 如果是随机的，每张图的熵 = 理想 (top-2 均匀)
    diversity_scores = []
    for pref in per_img_pref:
        eps = 1e-9
        h = -np.sum(pref * np.log(np.clip(pref, eps, 1)))
        diversity_scores.append(h)
    avg_diversity = np.mean(diversity_scores)
    max_diversity = np.log(num_e)
    print(f"\n  ┌ 4) 图片级选择熵:")
    print(f"     平均熵: {avg_diversity:.4f} / 最大熵: {max_diversity:.4f} "
          f"= {avg_diversity/max_diversity*100:.1f}%")
    print(f"     {'→ 每张图的选择偏好几乎一样 🔴' if avg_diversity/max_diversity > 0.95 else '→ 有一定差异 ✅'}")

    # ── 指标 5: 路由权重的区分度 ──
    all_weights = np.array([np.array(w) for w in weights])  # [N_images, top_k]
    weight_spread = all_weights.max(axis=1) - all_weights.min(axis=1)
    avg_spread = weight_spread.mean()

    print(f"\n  ┌ 5) top-k 权重分散度:")
    print(f"     top-2 权重的平均 max-min 差: {avg_spread:.6f}")
    print(f"     {'→ 权重无从区分 (退化) 🔴' if avg_spread < 0.01 else '→ 有一定区分度 ✅'}")

    # ── 综合判断 ──
    print(f"\n  ┌ 🧾 综合判断:")
    issues = []
    if abs(expert_hit_count[0]/max(total_hits,1)*100 - ideal_rate) < 5:
        issues.append("hit_rate: 每个专家选中率完全等于理想值(66.7%)")
    if avg_diversity / max_diversity > 0.95:
        issues.append("diversity: 图片间无差异")
    if avg_spread < 0.01:
        issues.append("weight_spread: 权重极窄")
    if all(s < 0.05 for s in img_pref_std):
        issues.append("cross_img: 跨图片方差几乎为0")

    if len(issues) >= 3:
        print(f"     🔴 路由器退化确认 ({len(issues)}/4 项命中)")
    elif len(issues) >= 1:
        print(f"     🟡 路由器部分退化 ({len(issues)}/4 项命中)")
    else:
        print(f"     🟢 路由器工作正常")
    for iss in issues:
        print(f"        • {iss}")

print("\n" + "=" * 80)
print(" ✅ 完成")
print("=" * 80)
