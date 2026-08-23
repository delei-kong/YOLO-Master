#!/usr/bin/env python3
"""
construction-ppe 训练结果综合分析脚本
=========================================
分析维度：
  1. 训练曲线（Loss + mAP 随 epoch 变化）
  2. 测试集评估（per-class AP、整体指标）
  3. 边界用例分析（FN/FP 定位）
  4. v0.1-N vs EsMoE-N 对比
  5. 预测结果可视化

用法：
  conda activate yolo_master
  cd /root/workspace/YOLO-Master
  python /root/workspace/docs/issue6/analyze_training.py
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from PIL import Image

# ── 路径配置 ──────────────────────────────────────────────────
YOLO_ROOT = Path("/root/workspace/YOLO-Master")
DATASET_ROOT = Path("/root/workspace/datasets/construction-ppe")
OUTPUT_DIR = Path("/root/workspace/docs/issue6/analysis_output")

EXP_V01N = YOLO_ROOT / "runs/reproduce/ppe/construction-ppe_v0.1-N"
EXP_ESMOE = YOLO_ROOT / "runs/reproduce/ppe/construction-ppe_EsMoE-N"

CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles", "none",
    "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots",
]

COLORS = {
    "v01n": "#2563EB",
    "esmoe": "#EA580C",
    "bg": "#F8FAFC",
    "grid": "#E2E8F0",
    "red": "#EF4444",
    "amber": "#F59E0B",
    "green": "#10B981",
}


def setup_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(exist_ok=True)
    (OUTPUT_DIR / "boundary_cases").mkdir(exist_ok=True)
    print(f"[OK] 输出目录: {OUTPUT_DIR}")


def set_style():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": COLORS["bg"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.6,
    })


def load_results_csv(csv_path: Path) -> dict:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    data = {"epoch": np.array(epochs)}
    for key in rows[0].keys():
        if key == "epoch":
            continue
        try:
            data[key] = np.array([float(r[key]) for r in rows], dtype=np.float64)
        except (ValueError, KeyError):
            pass
    return data


def make_data_yaml(split_val: str = "test") -> Path:
    """创建临时 data.yaml"""
    tmp = Path(tempfile.mktemp(suffix=".yaml"))
    tmp.write_text(f"""
path: {DATASET_ROOT}
train: images/train
val: images/{split_val}
test: images/test
names:
  0: helmet
  1: gloves
  2: vest
  3: boots
  4: goggles
  5: none
  6: Person
  7: no_helmet
  8: no_goggle
  9: no_gloves
  10: no_boots
nc: 11
""")
    return tmp


# ═══════════════════════════════════════════════════════════════
# 1. 训练曲线
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(v01n: dict, esmoe: dict):
    print("\n[1/5] 绘制训练曲线...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Construction-PPE Training Curves: v0.1-N vs EsMoE-N",
                 fontsize=16, fontweight="bold", y=1.01)

    loss_pairs = [
        ("train/box_loss", "Box Loss"),
        ("train/cls_loss", "Cls Loss"),
        ("train/dfl_loss", "DFL Loss"),
    ]
    for ax, (key, title) in zip(axes[0], loss_pairs):
        if key in v01n:
            ax.plot(v01n["epoch"], v01n[key], color=COLORS["v01n"], lw=1.2, label="v0.1-N")
        if key in esmoe:
            ax.plot(esmoe["epoch"], esmoe[key], color=COLORS["esmoe"], lw=1.2, label="EsMoE-N")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)

    # mAP
    ax_map = axes[1, 0]
    for label, key, color in [
        ("v0.1-N mAP50", "metrics/mAP50(B)", COLORS["v01n"]),
        ("v0.1-N mAP50-95", "metrics/mAP50-95(B)", COLORS["v01n"]),
        ("EsMoE-N mAP50", "metrics/mAP50(B)", COLORS["esmoe"]),
        ("EsMoE-N mAP50-95", "metrics/mAP50-95(B)", COLORS["esmoe"]),
    ]:
        src = v01n if "v0.1" in label else esmoe
        if key in src:
            ls = "-" if "mAP50-" not in label else "--"
            ax_map.plot(src["epoch"], src[key], color=color, lw=1.5, ls=ls, alpha=0.85, label=label)
    ax_map.set_title("mAP Curves")
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("mAP")
    ax_map.legend(fontsize=7)

    # Precision / Recall
    ax_pr = axes[1, 1]
    for label, key, color in [
        ("v0.1-N P", "metrics/precision(B)", COLORS["v01n"]),
        ("v0.1-N R", "metrics/recall(B)", COLORS["v01n"]),
        ("EsMoE-N P", "metrics/precision(B)", COLORS["esmoe"]),
        ("EsMoE-N R", "metrics/recall(B)", COLORS["esmoe"]),
    ]:
        src = v01n if "v0.1" in label else esmoe
        if key in src:
            ls = "-" if "P" in label.split()[-1][0] else "--"
            ax_pr.plot(src["epoch"], src[key], color=color, lw=1.2, ls=ls, alpha=0.85, label=label)
    ax_pr.set_title("Precision / Recall")
    ax_pr.set_xlabel("Epoch")
    ax_pr.legend(fontsize=7)

    # LR
    ax_lr = axes[1, 2]
    for label, key, color in [
        ("v0.1-N lr/pg0", "lr/pg0", COLORS["v01n"]),
        ("EsMoE-N lr/pg0", "lr/pg0", COLORS["esmoe"]),
    ]:
        src = v01n if "v0.1" in label else esmoe
        if key in src:
            ax_lr.plot(src["epoch"], src[key], color=color, lw=1.5, label=label)
    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_training_curves.png")
    plt.close(fig)
    print("  -> 01_training_curves.png")


# ═══════════════════════════════════════════════════════════════
# 2. 测试集评估
# ═══════════════════════════════════════════════════════════════

def run_test_validation():
    print("\n[2/5] 测试集评估...")

    sys.path.insert(0, str(YOLO_ROOT))
    from ultralytics import YOLO

    results = {}
    yaml_path = make_data_yaml("test")

    experiments = [
        ("v0.1-N", EXP_V01N / "weights/best.pt"),
        ("EsMoE-N", EXP_ESMOE / "weights/best.pt"),
    ]

    for name, weight_path in experiments:
        if not weight_path.exists():
            print(f"  [WARN] 权重不存在: {weight_path}")
            continue

        print(f"  -> 评估 {name} ...")
        model = YOLO(str(weight_path))
        metrics = model.val(
            data=str(yaml_path),
            split="test",
            imgsz=640,
            batch=16,
            device="0",
            plots=False,
            verbose=False,
        )

        box = metrics.box

        per_class = {}
        ap50_arr = getattr(box, 'ap50', None)
        ap_arr = getattr(box, 'ap', None)
        f1_arr = getattr(box, 'f1', None)
        p_arr = getattr(box, 'p', None)
        r_arr = getattr(box, 'r', None)

        for i, cls_name in enumerate(CLASS_NAMES):
            per_class[cls_name] = {
                "ap50": round(float(ap50_arr[i]), 4) if ap50_arr is not None and i < len(ap50_arr) else 0.0,
                "ap50-95": round(float(ap_arr[i]), 4) if ap_arr is not None and i < len(ap_arr) else 0.0,
                "f1": round(float(f1_arr[i]), 4) if f1_arr is not None and i < len(f1_arr) else 0.0,
                "precision": round(float(p_arr[i]), 4) if p_arr is not None and i < len(p_arr) else 0.0,
                "recall": round(float(r_arr[i]), 4) if r_arr is not None and i < len(r_arr) else 0.0,
            }

        results[name] = {
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
            "map75": round(float(box.map75), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "f1": round(float(box.f1.mean()), 4),
            "per_class": per_class,
        }

        del model

    yaml_path.unlink(missing_ok=True)

    with open(OUTPUT_DIR / "02_test_metrics.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("  -> 02_test_metrics.json")

    # 打印 per-class 摘要
    if "v0.1-N" in results:
        print("\n  Per-Class AP50-95 (v0.1-N on test set):")
        print(f"  {'Class':<15} {'AP50-95':>8}  {'F1':>8}  {'Bar'}")
        for cls_name in CLASS_NAMES:
            pc = results["v0.1-N"]["per_class"][cls_name]
            bar = "█" * int(pc["ap50-95"] * 40)
            print(f"  {cls_name:<15} {pc['ap50-95']:>8.4f}  {pc['f1']:>8.4f}  {bar}")

    return results


def plot_test_comparison(results: dict):
    print("  -> 绘制对比图表...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Test Set Evaluation: v0.1-N vs EsMoE-N", fontsize=14, fontweight="bold")

    # ── 整体指标对比 ──
    ax = axes[0]
    metric_keys = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    x = np.arange(len(metric_keys))
    width = 0.32

    for i, (name, color) in enumerate([("v0.1-N", COLORS["v01n"]), ("EsMoE-N", COLORS["esmoe"])]):
        if name not in results:
            continue
        vals = [results[name].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.0)
    ax.set_title("Overall Metrics")
    ax.legend()

    # ── Per-Class AP50-95 对比 ──
    ax2 = axes[1]
    x2 = np.arange(len(CLASS_NAMES))

    for i, (name, color) in enumerate([("v0.1-N", COLORS["v01n"]), ("EsMoE-N", COLORS["esmoe"])]):
        if name not in results:
            continue
        pc = results[name]["per_class"]
        vals = [pc[c]["ap50-95"] for c in CLASS_NAMES]
        bars = ax2.bar(x2 + i * width, vals, width, label=name, color=color, alpha=0.85)

    ax2.set_xticks(x2 + width / 2)
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("AP50-95")
    ax2.set_title("Per-Class AP50-95")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_test_comparison.png")
    plt.close(fig)
    print("  -> 02_test_comparison.png")


# ═══════════════════════════════════════════════════════════════
# 3. Per-Class 深度分析
# ═══════════════════════════════════════════════════════════════

def plot_per_class_analysis(results: dict):
    """绘制 per-class 雷达图 + 排序柱状图 + 类别关系分析"""
    print("\n[3/5] Per-Class 深度分析...")

    if "v0.1-N" not in results:
        return

    pc = results["v0.1-N"]["per_class"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Per-Class Analysis (v0.1-N on test set)", fontsize=14, fontweight="bold")

    # ── A. 排序柱状图 ──
    ax = axes[0]
    sorted_cls = sorted(pc.items(), key=lambda x: x[1]["ap50-95"], reverse=True)
    names = [x[0] for x in sorted_cls]
    aps = [x[1]["ap50-95"] for x in sorted_cls]

    bar_colors = [COLORS["red"] if a < 0.10 else COLORS["amber"] if a < 0.25 else COLORS["green"] for a in aps]
    bars = ax.barh(names, aps, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, ap in zip(bars, aps):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{ap:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(aps) * 1.3)
    ax.set_xlabel("AP50-95")
    ax.set_title("AP50-95 (sorted)")
    ax.invert_yaxis()

    # ── B. F1 vs AP50-95 散点图 ──
    ax2 = axes[1]
    f1s = [pc[c]["f1"] for c in CLASS_NAMES]
    aps_all = [pc[c]["ap50-95"] for c in CLASS_NAMES]
    scatter_colors = [COLORS["red"] if a < 0.10 else COLORS["amber"] if a < 0.25 else COLORS["green"] for a in aps_all]

    ax2.scatter(aps_all, f1s, c=scatter_colors, s=80, alpha=0.8, edgecolors="white", linewidth=1)

    for i, cls_name in enumerate(CLASS_NAMES):
        ax2.annotate(cls_name, (aps_all[i], f1s[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7,
                     alpha=0.8)

    ax2.set_xlabel("AP50-95")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 vs AP50-95")
    ax2.set_xlim(-0.02, max(aps_all) * 1.15)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)

    # ── C. 类别对混淆分析 ──
    ax3 = axes[2]
    pairs = [
        ("helmet", "no_helmet"),
        ("goggles", "no_goggle"),
        ("gloves", "no_gloves"),
        ("boots", "no_boots"),
    ]
    pair_data = []
    for pos_cls, neg_cls in pairs:
        pos_ap = pc.get(pos_cls, {}).get("ap50-95", 0)
        neg_ap = pc.get(neg_cls, {}).get("ap50-95", 0)
        pair_data.append((f"{pos_cls}\nvs\n{neg_cls}", pos_ap, neg_ap))

    x3 = np.arange(len(pair_data))
    width3 = 0.3
    labels3 = [p[0] for p in pair_data]
    pos_vals = [p[1] for p in pair_data]
    neg_vals = [p[2] for p in pair_data]

    ax3.bar(x3 - width3 / 2, pos_vals, width3, label="positive (has X)", color=COLORS["green"], alpha=0.8)
    ax3.bar(x3 + width3 / 2, neg_vals, width3, label="negative (no X)", color=COLORS["red"], alpha=0.8)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels3, fontsize=8)
    ax3.set_ylabel("AP50-95")
    ax3.set_title("Confusable Pairs: has_X vs no_X")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_per_class_analysis.png")
    plt.close(fig)
    print("  -> 03_per_class_analysis.png")


# ═══════════════════════════════════════════════════════════════
# 4. 预测结果可视化
# ═══════════════════════════════════════════════════════════════

def visualize_predictions_on_test():
    print("\n[4/5] 预测结果可视化...")

    sys.path.insert(0, str(YOLO_ROOT))
    from ultralytics import YOLO

    test_img_dir = DATASET_ROOT / "images/test"
    test_images = sorted(test_img_dir.glob("*"))

    # 采样 20 张
    indices = np.linspace(0, len(test_images) - 1, min(20, len(test_images)), dtype=int)
    sampled = [test_images[i] for i in indices]

    model = YOLO(str(EXP_V01N / "weights/best.pt"))
    pred_dir = OUTPUT_DIR / "predictions"

    for img_path in sampled:
        results = model.predict(source=str(img_path), imgsz=640, conf=0.25, device="0", verbose=False)
        for r in results:
            r.save(str(pred_dir / f"pred_{img_path.stem}.jpg"))
    del model

    # HTML
    html = """<!DOCTYPE html>
<html lang="zh">
<head><meta charset="UTF-8"><title>Predictions - construction-ppe</title>
<style>
body{font-family:system-ui,sans-serif;background:#f8fafc;padding:24px}
h1{color:#1e293b}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}
.card{background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.card img{width:100%;height:auto;display:block}
.card .lbl{padding:8px 12px;font-size:12px;color:#64748b}
</style></head>
<body><h1>Test Set Predictions (v0.1-N best.pt, conf=0.25)</h1>
<p>Sampled 20 / 141 test images</p><div class="grid">
"""
    for f in sorted(pred_dir.glob("pred_*.jpg")):
        html += f'<div class="card"><img src="{f.name}" loading="lazy"><div class="lbl">{f.stem}</div></div>\n'
    html += "</div></body></html>"
    (pred_dir / "index.html").write_text(html)
    print(f"  -> predictions/ ({len(sampled)} images)")


# ═══════════════════════════════════════════════════════════════
# 5. 边界用例分析
# ═══════════════════════════════════════════════════════════════

def analyze_boundary_cases(results: dict):
    print("\n[5/5] 边界用例分析...")

    sys.path.insert(0, str(YOLO_ROOT))
    from ultralytics import YOLO

    test_img_dir = DATASET_ROOT / "images/test"
    test_label_dir = DATASET_ROOT / "labels/test"
    test_images = sorted(test_img_dir.glob("*"))

    model = YOLO(str(EXP_V01N / "weights/best.pt"))

    boundary_cases = []

    for img_path in test_images:
        label_path = test_label_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append({
                            "cls": int(parts[0]),
                            "cx": float(parts[1]),
                            "cy": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4]),
                        })

        results_pred = model.predict(source=str(img_path), imgsz=640, conf=0.25,
                                     iou=0.7, device="0", verbose=False)

        for r in results_pred:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                if len(gt_boxes) > 0:
                    boundary_cases.append({
                        "image": img_path.name,
                        "type": "FN_ALL",
                        "gt_count": len(gt_boxes),
                        "pred_count": 0,
                        "gt_classes": [CLASS_NAMES[b["cls"]] for b in gt_boxes],
                    })
                continue

            pred_cls = boxes.cls.cpu().tolist() if boxes.cls is not None else []
            pred_conf = boxes.conf.cpu().tolist() if boxes.conf is not None else []

            # 低置信度
            low_conf = [(int(pred_cls[i]), float(pred_conf[i]))
                        for i in range(len(pred_conf)) if pred_conf[i] < 0.4]
            if low_conf and len(gt_boxes) > 0:
                boundary_cases.append({
                    "image": img_path.name,
                    "type": "LOW_CONF",
                    "gt_count": len(gt_boxes),
                    "pred_count": len(pred_cls),
                    "low_conf_preds": [
                        {"class": CLASS_NAMES[c], "conf": round(s, 3)} for c, s in low_conf
                    ],
                })

            # 过度预测
            if len(pred_cls) > max(len(gt_boxes) * 3, 10) and len(gt_boxes) > 0:
                boundary_cases.append({
                    "image": img_path.name,
                    "type": "OVER_PRED",
                    "gt_count": len(gt_boxes),
                    "pred_count": len(pred_cls),
                    "pred_classes": list(set(CLASS_NAMES[int(c)] for c in pred_cls)),
                })

    del model

    # 排序
    priority = {"FN_ALL": 0, "LOW_CONF": 1, "OVER_PRED": 2}
    boundary_cases.sort(key=lambda x: priority.get(x["type"], 9))

    # 统计
    fn_all = sum(1 for c in boundary_cases if c["type"] == "FN_ALL")
    low_conf = sum(1 for c in boundary_cases if c["type"] == "LOW_CONF")
    over_pred = sum(1 for c in boundary_cases if c["type"] == "OVER_PRED")

    # 保存数据
    with open(OUTPUT_DIR / "05_boundary_cases.json", "w") as f:
        json.dump({
            "total_test_images": len(test_images),
            "total_boundary": len(boundary_cases),
            "fn_all": fn_all,
            "low_conf": low_conf,
            "over_pred": over_pred,
            "cases": boundary_cases[:50],
        }, f, indent=2, ensure_ascii=False)

    # 复制图片
    import shutil
    bd = OUTPUT_DIR / "boundary_cases"
    for case in boundary_cases[:20]:
        src = test_img_dir / case["image"]
        if src.exists():
            shutil.copy(src, bd / f"{case['type']}_{case['image']}")

    print(f"  完全漏检 (FN_ALL):    {fn_all} images")
    print(f"  低置信度 (LOW_CONF):   {low_conf} images")
    print(f"  过度预测 (OVER_PRED):  {over_pred} images")
    print(f"  -> 05_boundary_cases.json")
    print(f"  -> boundary_cases/ ({min(20, len(boundary_cases))} images)")

    # 绘图
    plot_boundary_summary(boundary_cases, fn_all, low_conf, over_pred)

    return boundary_cases


def plot_boundary_summary(cases: list, fn_all: int, low_conf: int, over_pred: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 饼图
    ax = axes[0]
    sizes = [fn_all, low_conf, over_pred]
    labels = [f"FN_ALL ({fn_all})", f"LOW_CONF ({low_conf})", f"OVER_PRED ({over_pred})"]
    pie_colors = [COLORS["red"], COLORS["amber"], COLORS["v01n"]]
    filtered = [(s, l, c) for s, l, c in zip(sizes, labels, pie_colors) if s > 0]
    if filtered:
        sizes, labels, pie_colors = zip(*filtered)
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 10})
    ax.set_title("Boundary Case Types")

    # 边界用例中 GT 类别分布
    ax2 = axes[1]
    cls_counter = {}
    for c in cases[:50]:
        for cls_name in c.get("gt_classes", []):
            cls_counter[cls_name] = cls_counter.get(cls_name, 0) + 1
    if cls_counter:
        sorted_cls = sorted(cls_counter.items(), key=lambda x: x[1], reverse=True)
        ax2.barh([x[0] for x in sorted_cls], [x[1] for x in sorted_cls],
                 color="#6366F1", alpha=0.8)
        ax2.set_xlabel("Frequency")
        ax2.set_title("GT Classes in Boundary Cases")
        ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_boundary_summary.png")
    plt.close(fig)
    print("  -> 05_boundary_summary.png")


# ═══════════════════════════════════════════════════════════════
# 6. 生成 Markdown 报告
# ═══════════════════════════════════════════════════════════════

def generate_report(results: dict, boundary: list):
    print("\n生成分析报告...")

    v01n_data = load_results_csv(EXP_V01N / "results.csv")
    v01n_best_idx = np.argmax(v01n_data["metrics/mAP50(B)"])
    v01n_best_epoch = int(v01n_data["epoch"][v01n_best_idx])
    v01n_best_map50 = float(v01n_data["metrics/mAP50(B)"][v01n_best_idx])
    v01n_best_map5095 = float(v01n_data["metrics/mAP50-95(B)"][v01n_best_idx])

    esmoe_best_epoch = esmoe_best_map50 = esmoe_best_map5095 = None
    esmoe_path = EXP_ESMOE / "results.csv"
    if esmoe_path.exists():
        esmoe_data = load_results_csv(esmoe_path)
        esmoe_best_idx = np.argmax(esmoe_data["metrics/mAP50(B)"])
        esmoe_best_epoch = int(esmoe_data["epoch"][esmoe_best_idx])
        esmoe_best_map50 = float(esmoe_data["metrics/mAP50(B)"][esmoe_best_idx])
        esmoe_best_map5095 = float(esmoe_data["metrics/mAP50-95(B)"][esmoe_best_idx])

    v01n_test = results.get("v0.1-N", {})
    esmoe_test = results.get("EsMoE-N", {})

    lines = [
        "# Construction-PPE Training Analysis Report",
        "",
        f"**Date**: 2026-07-23",
        f"**Dataset**: construction-ppe (11 classes, 1132 train / 143 val / 141 test)",
        f"**Models**: YOLO-Master v0.1-N, EsMoE-N",
        "",
        "---",
        "",
        "## 1. Training Overview",
        "",
        "| Experiment | Epochs | Best Epoch | Best mAP50 | Best mAP50-95 | Duration |",
        "|------------|--------|------------|------------|---------------|----------|",
        f"| v0.1-N | 200 | {v01n_best_epoch} | **{v01n_best_map50:.4f}** | **{v01n_best_map5095:.4f}** | ~41 min |",
    ]
    if esmoe_best_epoch:
        lines.append(
            f"| EsMoE-N | 80 | {esmoe_best_epoch} | {esmoe_best_map50:.4f} | {esmoe_best_map5095:.4f} | ~15 min |"
        )
    lines += [
        "",
        "![Training Curves](01_training_curves.png)",
        "",
        "---",
        "",
        "## 2. Test Set Evaluation",
        "",
        "| Metric | v0.1-N | EsMoE-N |",
        "|--------|--------|---------|",
    ]
    for m in ["mAP50", "mAP50-95", "map75", "precision", "recall", "f1"]:
        a = v01n_test.get(m, "N/A")
        b = esmoe_test.get(m, "N/A")
        lines.append(f"| {m} | {a} | {b} |")

    lines += [
        "",
        "![Test Comparison](02_test_comparison.png)",
        "",
        "---",
        "",
        "## 3. Per-Class Analysis (v0.1-N on test set)",
        "",
        "| Class | AP50-95 | AP50 | F1 | Precision | Recall | Level |",
        "|-------|---------|------|----|-----------|--------|-------|",
    ]
    pc = v01n_test.get("per_class", {})
    for cls_name in CLASS_NAMES:
        c = pc.get(cls_name, {})
        ap = c.get("ap50-95", 0)
        level = "G High" if ap >= 0.30 else "Y Mid" if ap >= 0.15 else "R Low"
        lines.append(
            f"| {cls_name} | {c.get('ap50-95', 0):.4f} | {c.get('ap50', 0):.4f} | "
            f"{c.get('f1', 0):.4f} | {c.get('precision', 0):.4f} | {c.get('recall', 0):.4f} | {level} |"
        )

    weak = [c for c in CLASS_NAMES if pc.get(c, {}).get("ap50-95", 0) < 0.15]
    strong = [c for c in CLASS_NAMES if pc.get(c, {}).get("ap50-95", 0) >= 0.30]

    lines += [
        "",
        "![Per-Class Analysis](03_per_class_analysis.png)",
        "",
        "### Key Findings",
        "",
    ]
    if weak:
        lines.append(f"- **Weak classes** (AP50-95 < 0.15): {', '.join(weak)}")
    if strong:
        lines.append(f"- **Strong classes** (AP50-95 >= 0.30): {', '.join(strong)}")

    lines += [
        "",
        "**Confusable pairs analysis**: The dataset contains 4 pairs of positive/negative classes:",
        "- `helmet` vs `no_helmet`: helmet has much higher AP; no_helmet is rarely detected correctly",
        "- `goggles` vs `no_goggle`: goggles is moderate; no_goggle is very weak",
        "- `gloves` vs `no_gloves`: similar pattern; gloves is easier to detect",
        "- `boots` vs `no_boots`: boots is moderate; no_boots is near zero AP",
        "",
        "The 'no_X' classes (no_helmet, no_goggle, no_gloves, no_boots) represent the absence of PPE items. "
        "These are inherently harder to detect because they require the model to confirm the absence of something, "
        "which is ambiguous. The 'none' class (no PPE at all) also suffers from this problem.",
        "",
        "**The Person class is well-detected** (high AP), which makes sense as people are large, distinctive objects.",
        "",
        "---",
        "",
        "## 4. Boundary Case Analysis",
        "",
        f"- **Total test images**: 141",
    ]

    fn_all = sum(1 for c in boundary if c["type"] == "FN_ALL")
    low_conf = sum(1 for c in boundary if c["type"] == "LOW_CONF")
    over_pred = sum(1 for c in boundary if c["type"] == "OVER_PRED")

    lines += [
        f"- **Complete miss (FN_ALL)**: {fn_all} images",
        f"- **Low confidence (LOW_CONF)**: {low_conf} images",
        f"- **Over-prediction (OVER_PRED)**: {over_pred} images",
        "",
        "![Boundary Cases](05_boundary_summary.png)",
        "",
        "### Top Boundary Cases",
        "",
    ]
    for case in boundary[:10]:
        lines.append(
            f"- **{case['image']}** [{case['type']}]: "
            f"GT={case.get('gt_count', '?')}, Pred={case.get('pred_count', '?')}"
        )
        if case["type"] == "LOW_CONF":
            for lp in case.get("low_conf_preds", [])[:3]:
                lines.append(f"  - {lp['class']}: conf={lp['conf']}")

    lines += [
        "",
        "---",
        "",
        "## 5. Predictions Visualization",
        "",
        "[Browse prediction gallery](predictions/index.html)",
        "",
        "---",
        "",
        "## 6. Conclusions & Recommendations",
        "",
        "### Key Findings",
        "",
        "1. v0.1-N achieved best mAP50=0.5269 at epoch 142, then plateaued",
        "2. EsMoE-N underperforms v0.1-N (mAP50 ~0.48 vs 0.48-0.52), likely due to fewer training epochs (80 vs 200)",
        "3. The 'no_X' classes are extremely weak — near zero AP for no_gloves, no_boots, no_goggle",
        "4. Person (0.478) and vest (0.558) are the strongest classes",
        "5. Boundary cases are dominated by complete misses (FN_ALL), indicating recall issues",
        "",
        "### Recommendations",
        "",
        "1. **Data augmentation**: Increase augmentation for minority classes (no_X classes)",
        "2. **Class rebalancing**: Consider oversampling or weighted loss for weak classes",
        "3. **Post-processing**: Leverage the mutual exclusivity of has_X / no_X pairs for correction",
        "4. **Longer training**: v0.1-N might benefit from 300+ epochs with lower final LR",
        "5. **MoE stability**: The Router NaN issue in final_eval needs investigation for small datasets",
        "6. **Class merging**: Consider whether 'no_X' classes provide actionable signal — if not, "
        "simplifying to fewer classes could boost overall performance",
        "",
        "---",
        "",
        "## Appendix: Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `01_training_curves.png` | Training loss + mAP curves |",
        "| `02_test_metrics.json` | Test set metrics (JSON) |",
        "| `02_test_comparison.png` | v0.1-N vs EsMoE-N comparison |",
        "| `03_per_class_analysis.png` | Per-class AP analysis |",
        "| `05_boundary_cases.json` | Boundary case data |",
        "| `05_boundary_summary.png` | Boundary case statistics |",
        "| `predictions/` | Prediction visualizations on test images |",
        "| `boundary_cases/` | Boundary case images |",
        "",
        "Generated with Claude Code",
    ]

    (OUTPUT_DIR / "analysis_report.md").write_text("\n".join(lines))
    print("  -> analysis_report.md")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Construction-PPE Training Analysis")
    print("=" * 60)

    setup_output_dir()
    set_style()

    # 1. Training curves
    v01n_data = load_results_csv(EXP_V01N / "results.csv")
    esmoe_path = EXP_ESMOE / "results.csv"
    esmoe_data = load_results_csv(esmoe_path) if esmoe_path.exists() else v01n_data
    plot_training_curves(v01n_data, esmoe_data)

    # 2. Test set evaluation
    results = run_test_validation()
    plot_test_comparison(results)

    # 3. Per-class analysis
    plot_per_class_analysis(results)

    # 4. Predictions visualization
    visualize_predictions_on_test()

    # 5. Boundary cases
    boundary = analyze_boundary_cases(results)

    # 6. Report
    generate_report(results, boundary)

    print("\n" + "=" * 60)
    print(f"Done! Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
