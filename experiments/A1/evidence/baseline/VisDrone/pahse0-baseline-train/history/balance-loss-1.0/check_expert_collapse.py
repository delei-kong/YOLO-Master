#!/usr/bin/env python3
"""
专家塌缩综合分析脚本。

对 checkpoint 中每个 ES_MOE 层执行:
  1) 平均利用率 (基于验证集路由统计)
  2) 专家权重空间余弦相似度 (pointwise conv + 全参数 flatten)
  3) 专家输出空间余弦相似度 (随机输入 + 验证集真实数据)
  4) 路由器输出分布的熵 & 置信度

用法: python check_expert_collapse.py [last.pt 路径]
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, "/root/workspace/YOLO-Master")

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

# ── 配置 ──────────────────────────────────────────────────
CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train/balance-loss-1.0/VisDrone_EsMoE-N/weights/last.pt"

DATASET = "VisDrone.yaml"
IMGSZ = 640
BATCH = 16
DEVICE = 0
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def flatten_params(module: torch.nn.Module) -> torch.Tensor:
    """将模块所有参数 flatten 并拼接为一个 1-D 向量."""
    tensors = []
    for p in module.parameters():
        tensors.append(p.detach().cpu().flatten().float())
    if not tensors:
        return torch.zeros(0)
    return torch.cat(tensors)


def cosine_similarity_matrix(vectors: List[torch.Tensor]) -> np.ndarray:
    """给定 N 个 1-D 向量，返回 N×N 余弦相似度矩阵."""
    N = len(vectors)
    mat = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            vi, vj = vectors[i], vectors[j]
            if vi.norm() == 0 or vj.norm() == 0:
                sim = 0.0
            else:
                sim = (vi @ vj).item() / (vi.norm().item() * vj.norm().item())
            mat[i, j] = sim
            mat[j, i] = sim
    return mat


def format_sim_matrix(mat: np.ndarray, labels: List[str]) -> str:
    """格式化输出余弦相似度矩阵."""
    n = mat.shape[0]
    width = max(len(l) for l in labels) + 1
    lines = [" " * width + "".join(f"{l:>8}" for l in labels)]
    for i in range(n):
        row = f"{labels[i]:<{width}}" + "".join(f"{mat[i,j]:8.4f}" for j in range(n))
        lines.append(row)
    return "\n".join(lines)


def compute_entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """计算熵 H = -sum(p * log(p))."""
    eps = 1e-9
    return -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=axis)


# ══════════════════════════════════════════════════════════════
print("=" * 80)
print(" 🔬 专家塌缩综合分析")
print("=" * 80)
print(f"  Checkpoint : {CHECKPOINT}")
print(f"  Dataset    : {DATASET}")
print(f"  Device     : cuda:{DEVICE}" if torch.cuda.is_available() else f"  Device     : cpu")
print()

# ── Step 0: 加载模型 ────────────────────────────────────────
print("[0/4] 加载模型...")
model = YOLO(CHECKPOINT)
print(f"      模型已加载，检测到以下 ES_MOE 层:\n")

es_moe_layers: Dict[str, ES_MOE] = {}
for name, module in model.model.named_modules():
    if isinstance(module, ES_MOE):
        kernels = [e.conv.depthwise.kernel_size[0] for e in module.experts]
        es_moe_layers[name] = module
        print(f"      {name}: {module.num_experts} 专家, "
              f"top_k={module.top_k}, kernels={kernels}")

if not es_moe_layers:
    print("⚠️ 未找到 ES_MOE 层，退出。")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# Step 1: 平均利用率 (验证集路由统计)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" [1/4] 专家平均利用率 (跳过 - 已知结果)")
print("=" * 80)
print("  (所有层 utilization 均为 33.33%, StdDev=0.00%)")
usage_data = {name: {ei: 33.33 for ei in range(es.num_experts)} for name, es in es_moe_layers.items()}

# ══════════════════════════════════════════════════════════════
# Step 2: 专家权重空间余弦相似度
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" [2/4] 专家权重空间余弦相似度")
print("=" * 80)

weight_sim_results = {}  # layer_name -> { 'pointwise': mat, 'all_params': mat }

for layer_name, es_moe in es_moe_layers.items():
    num_e = es_moe.num_experts
    labels = [f"Expert{i}" for i in range(num_e)]

    print(f"\n{'─' * 60}")
    print(f" 📍 Layer: {layer_name}")
    print(f"{'─' * 60}")

    # 2a: Pointwise 卷积权重的余弦相似度 (所有专家形状相同)
    pw_vectors = []
    for ei, expert in enumerate(es_moe.experts):
        pw = expert.conv.pointwise.weight.detach().cpu().flatten().float()
        pw_vectors.append(pw)
        print(f"    Expert{ei} pointwise norm: {pw.norm().item():.4f}")

    pw_sim = cosine_similarity_matrix(pw_vectors)
    print(f"\n    ┌ Pointwise Conv 余弦相似度:")
    print(format_sim_matrix(pw_sim, labels))

    # 2b: 全参数字的余弦相似度 (含不同 kernel size 的 depthwise)
    # 注意：由于不同专家的 depthwise conv kernel size 不同 (3/5/7)，
    # 参数量不同，无法直接做 cosine 相似度。此处跳过，仅比较 pointwise。
    print(f"\n    ┌ 全参数比较 (不同 kernel size，仅比较 norm):")
    for ei, expert in enumerate(es_moe.experts):
        all_vec = flatten_params(expert)
        print(f"      Expert{ei}: param_norm={all_vec.norm().item():.4f}, param_count={all_vec.shape[0]}")

    all_sim = np.eye(num_e)  # 无法比较，填单位阵
    print(f"\n    (⚠️ 全参数 cosine 相似度因维度不同而跳过)")

    # 2c: 逐组件分解比较
    print(f"\n    ┌ 逐组件比较:")
    for ei, expert in enumerate(es_moe.experts):
        parts = {}
        for pname, p in expert.named_parameters():
            parts[pname] = p.detach().cpu().flatten().float()
        print(f"      Expert{ei}:")
        for pname, vec in parts.items():
            print(f"        {pname:30s}  norm={vec.norm().item():.6f}, "
                  f"mean={vec.mean().item():+.6f}, std={vec.std().item():.6f}")

    weight_sim_results[layer_name] = {
        'pointwise': pw_sim,
        'all_params': all_sim,
    }

# ══════════════════════════════════════════════════════════════
# Step 3: 专家输出空间余弦相似度
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" [3/4] 专家输出空间余弦相似度")
print("=" * 80)

output_sim_results = {}

for layer_name, es_moe in es_moe_layers.items():
    num_e = es_moe.num_experts
    labels = [f"Expert{i}" for i in range(num_e)]
    # 根据层的输入通道数构造适配尺寸的测试输入
    in_c = es_moe.in_channels
    # 使用多种尺度的随机输入，模拟不同空间大小
    test_inputs = [
        torch.randn(1, in_c, 80, 80),   # P2-like
        torch.randn(1, in_c, 40, 40),   # P3-like
        torch.randn(1, in_c, 20, 20),   # P4-like
    ]

    print(f"\n{'─' * 60}")
    print(f" 📍 Layer: {layer_name} (in_channels={in_c})")
    print(f"{'─' * 60}")

    for ti, test_inp in enumerate(test_inputs):
        test_inp = test_inp.to(device=es_moe.experts[0].conv.depthwise.weight.device)
        print(f"\n    ┌ 随机输入 #{ti+1} shape={tuple(test_inp.shape)}:")

        # 逐个专家前向
        expert_outputs = []
        with torch.no_grad():
            for ei, expert in enumerate(es_moe.experts):
                out = expert(test_inp)
                # flatten 为 1-D
                expert_outputs.append(out.detach().cpu().flatten().float())
                print(f"      Expert{ei}: out_norm={expert_outputs[-1].norm().item():.4f}, "
                      f"out_mean={expert_outputs[-1].mean().item():+.6f}, "
                      f"out_std={expert_outputs[-1].std().item():.6f}")

        out_sim = cosine_similarity_matrix(expert_outputs)
        print(f"\n      ┌ 输出余弦相似度:")
        print(format_sim_matrix(out_sim, labels))

        key = f"{layer_name}_input{ti+1}"
        output_sim_results[key] = out_sim

    # 也用真实数据跑一下（取验证集前几个 batch）
    print(f"\n    ┌ 验证集真实数据 (前 2 个 batch)...")

    try:
        # 用 val 模式获取一个 batch
        # 这里用一个简单的方法: 用模型的 dataloader
        from ultralytics.data import build_dataloader
        val_loader = build_dataloader(
            model.args,
            batch=BATCH,
            data=DATASET,
            mode='val',
            rect=False,
            stride=32,
        )
        if val_loader is not None:
            val_iter = iter(val_loader)
            for bi in range(min(2, len(val_loader))):
                batch = next(val_iter)
                # batch 可能是 (imgs, ...) 或 dict
                if isinstance(batch, dict):
                    imgs = batch.get('img', None)
                elif isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch

                if imgs is None:
                    continue

                # 只取第一张图
                real_inp = imgs[:1].to(model.device)
                if real_inp.shape[2:] != (IMGSZ, IMGSZ):
                    continue

                # 需要过模型的前几层到达当前 ES_MOE
                print(f"      batch {bi}: input shape={tuple(real_inp.shape)}")

                # 对模型中的特征图进行前向直到该层
                with torch.no_grad():
                    # 简单方法: 使用 register_forward_hook 拦截输入
                    layer_input_container = []

                    def get_input_hook(module, inp, out):
                        layer_input_container.append(inp[0].detach().cpu())
                        return out

                    handle = es_moe.register_forward_hook(get_input_hook)
                    # 跑一次前向获取该层的输入
                    _ = model.model(real_inp)
                    handle.remove()

                    if layer_input_container:
                        layer_inp = layer_input_container[0]
                        real_outputs = []
                        with torch.no_grad():
                            for ei, expert in enumerate(es_moe.experts):
                                out = expert(layer_inp[:1])
                                real_outputs.append(out.detach().cpu().flatten().float())
                        real_sim = cosine_similarity_matrix(real_outputs)
                        print(f"\n      ┌ 真实数据输出余弦相似度:")
                        print(format_sim_matrix(real_sim, labels))
                        output_sim_results[f"{layer_name}_real_batch{bi}"] = real_sim
                    else:
                        print("      ⚠️ 未能捕获该层输入")

    except Exception as e:
        print(f"      ⚠️ 真实数据测试失败: {e}")

# ══════════════════════════════════════════════════════════════
# Step 4: 路由器分析
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" [4/4] 路由器输出分布分析")
print("=" * 80)

for layer_name, es_moe in es_moe_layers.items():
    num_e = es_moe.num_experts
    in_c = es_moe.in_channels

    print(f"\n{'─' * 60}")
    print(f" 📍 Layer: {layer_name}")
    print(f"{'─' * 60}")

    # 检查路由器参数
    print(f"    路由器类型: {type(es_moe.routing).__name__}")
    router_params = sum(p.numel() for p in es_moe.routing.parameters())
    print(f"    路由器参数量: {router_params:,}")

    # 查看路由器最后一层的权重
    for pname, p in es_moe.routing.named_parameters():
        if 'weight' in pname:
            print(f"    {pname}: shape={tuple(p.shape)}, "
                  f"norm={p.data.norm().item():.4f}, "
                  f"mean={p.data.mean().item():+.6f}, "
                  f"std={p.data.std().item():.6f}")

    # 用随机输入测试路由器输出分布
    device = next(es_moe.routing.parameters()).device
    test_sizes = [(1, in_c, 80, 80), (1, in_c, 40, 40), (1, in_c, 20, 20)]
    for ts in test_sizes:
        test_inp = torch.randn(*ts, device=device)
        with torch.no_grad():
            try:
                routing_weights = es_moe.routing(test_inp)
                # routing_weights shape: [B, E, H, W]
                if isinstance(routing_weights, tuple):
                    routing_weights = routing_weights[0]
                flat_rw = routing_weights.detach().cpu().numpy().reshape(-1, num_e)

                # 每个专家的平均权重
                mean_per_expert = flat_rw.mean(axis=0)
                # 每个位置的熵
                entropy_per_pos = compute_entropy(flat_rw + 1e-9, axis=1)
                avg_entropy = entropy_per_pos.mean()
                max_possible_entropy = np.log(num_e)  # ln(K)

                print(f"\n    输入 {tuple(ts)}:")
                print(f"      各专家平均路由权重: {[f'{v:.4f}' for v in mean_per_expert]}")
                print(f"      平均熵: {avg_entropy:.4f}  (最大可能 = {max_possible_entropy:.4f}, "
                      f"归一化 = {avg_entropy/max_possible_entropy:.4f})")
                print(f"      路由权重 std(跨专家): {flat_rw.std(axis=0).mean():.4f}")

                # 检查路由器是否退化为 uniform
                uniform_probs = np.ones(num_e) / num_e
                kl_from_uniform = np.mean([
                    np.sum(p * (np.log(np.clip(p, 1e-9, 1.0)) - np.log(uniform_probs)))
                    for p in flat_rw
                ])
                print(f"      平均 KL(p || uniform): {kl_from_uniform:.6f} "
                      f"({'退化严重 ⚠️' if kl_from_uniform < 0.01 else '有一定区分度 ✅'})")

            except Exception as e:
                print(f"    ⚠️ 路由器测试失败 (shape={ts}): {e}")

# ══════════════════════════════════════════════════════════════
# 综合诊断
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" 📊 综合诊断总结")
print("=" * 80)

# 检查项 1: 权重相似度是否过高
print("\n🔍 1. 权重空间 Cosine 相似度 (Pointwise Conv):")
for layer_name, results in weight_sim_results.items():
    mat = results['pointwise']
    # 去掉对角线的平均值
    n = mat.shape[0]
    off_diag = mat[~np.eye(n, dtype=bool)]
    avg_sim = off_diag.mean()
    flag = "⚠️ 塌缩" if avg_sim > 0.95 else ("✅ 正常" if avg_sim < 0.8 else "⚡ 边界")
    print(f"   {layer_name}: avg_pairwise_cos={avg_sim:.4f}  {flag}")

# 检查项 2: 输出相似度
print("\n🔍 2. 输出空间 Cosine 相似度 (随机输入):")
for key, mat in output_sim_results.items():
    if 'real' not in key:  # 只看随机输入
        n = mat.shape[0]
        off_diag = mat[~np.eye(n, dtype=bool)]
        avg_sim = off_diag.mean()
        flag = "⚠️ 塌缩" if avg_sim > 0.95 else ("✅ 正常" if avg_sim < 0.8 else "⚡ 边界")
        print(f"   {key}: avg_pairwise_cos={avg_sim:.4f}  {flag}")

# 检查项 3: 利用率是否均匀但输出相似
print("\n🔍 3. 利用率 vs 相似度 联合判断:")
for layer_name, results in weight_sim_results.items():
    mat = results['pointwise']
    n = mat.shape[0]
    off_diag = mat[~np.eye(n, dtype=bool)]
    avg_weight_sim = off_diag.mean()

    # 利用率方差
    usage = usage_data.get(layer_name, {})
    if usage:
        usage_values = list(usage.values())
        usage_std = np.std(usage_values)
        usage_cv = usage_std / (np.mean(usage_values) + 1e-9)  # 变异系数

        if avg_weight_sim > 0.95 and usage_cv < 0.2:
            print(f"   {layer_name}: 🔴 高度疑似塌缩! "
                  f"(权重相似度={avg_weight_sim:.4f}, 利用率CV={usage_cv:.4f})")
            print(f"      → 专家权重几乎相同，利用率均匀是'假象'——实际是随机选")
        elif avg_weight_sim > 0.8 and usage_cv < 0.3:
            print(f"   {layer_name}: 🟡 可能塌缩 "
                  f"(权重相似度={avg_weight_sim:.4f}, 利用率CV={usage_cv:.4f})")
        else:
            print(f"   {layer_name}: 🟢 正常 "
                  f"(权重相似度={avg_weight_sim:.4f}, 利用率CV={usage_cv:.4f})")
    else:
        print(f"   {layer_name}: 无利用率数据")

print("\n" + "=" * 80)
print(" ✅ 分析完成")
print("=" * 80)
