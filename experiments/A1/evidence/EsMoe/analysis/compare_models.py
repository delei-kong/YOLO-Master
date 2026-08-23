#!/usr/bin/env python3
"""Compare YOLO-Master v0.1-N vs EsMoE-N across datasets.

Experiments:
  1. Our trained weights: v0.1-N vs EsMoE-N on VisDrone / construction-ppe / brain-tumor
  2. Author pretrained vs ours on VisDrone
  3. Per-class AP dump and boundary-case detection

Output: docs/issue6/exp/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "docs/issue6/exp"
RUNS = ROOT / "runs/reproduce"

# ── Experiment config ────────────────────────────────────────────

OUR_WEIGHTS = [
    # (label, checkpoint_path, data_yaml)
    ("VisDrone-v0.1-N",   RUNS / "visdrone/VisDrone_v0.1-N/weights/best.pt",    "VisDrone.yaml"),
    ("VisDrone-EsMoE-N",  RUNS / "visdrone/VisDrone_EsMoE-N/weights/best.pt",   "VisDrone.yaml"),
    ("ppe-v0.1-N",        RUNS / "ppe/construction-ppe_v0.1-N/weights/best.pt",  "construction-ppe.yaml"),
    ("ppe-EsMoE-N",       RUNS / "ppe/construction-ppe_EsMoE-N/weights/best.pt", "construction-ppe.yaml"),
    ("bt-v0.1-N",         RUNS / "brain-tumor/brain-tumor_v0.1-N/weights/best.pt", "brain-tumor.yaml"),
    ("bt-EsMoE-N",        RUNS / "brain-tumor/brain-tumor_EsMoE-N/weights/best.pt", "brain-tumor.yaml"),
]

AUTHOR_WEIGHTS = [
    ("VisDrone-author-v0.1-N",  RUNS / "visdrone/author_v0.1-N.pt",  "VisDrone.yaml"),
    ("VisDrone-author-EsMoE-N", RUNS / "visdrone/author_EsMoE-N.pt", "VisDrone.yaml"),
]

ALL_RUNS = OUR_WEIGHTS + AUTHOR_WEIGHTS

BATCH = 32
DEVICE = 0
IMGSZ = 640


def run_val(label: str, ckpt: Path, data: str) -> dict:
    """Run validation and return metrics dict."""
    print(f"\n{'='*60}")
    print(f"[{label}] Loading {ckpt.name} on {data} ...")
    t0 = time.perf_counter()
    model = YOLO(str(ckpt))
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = model.val(data=data, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                        verbose=False, plots=False, save_json=False)
    val_time = time.perf_counter() - t0

    # Collect metrics
    out = {
        "label": label,
        "ckpt": str(ckpt),
        "data": data,
        "load_time_s": round(load_time, 2),
        "val_time_s": round(val_time, 2),
        "mAP50": round(float(results.box.map50), 6),
        "mAP50-95": round(float(results.box.map), 6),
        "mAP75": round(float(results.box.map75), 6),
    }

    # Per-class AP50
    try:
        ap50_per_class = results.box.ap_class_index
        if ap50_per_class is not None and len(ap50_per_class) > 0:
            out["per_class_ap50"] = [round(float(x), 6) for x in results.box.maps50]
            out["class_indices"] = [int(x) for x in ap50_per_class]
    except Exception:
        out["per_class_ap50"] = []

    # Inference speed
    try:
        speed = results.speed  # (preprocess, inference, postprocess, total) ms
        out["speed_preprocess_ms"] = round(speed["preprocess"], 2) if "preprocess" in speed else None
        out["speed_inference_ms"] = round(speed["inference"], 2) if "inference" in speed else None
        out["speed_postprocess_ms"] = round(speed["postprocess"], 2) if "postprocess" in speed else None
        out["speed_total_ms"] = round(speed.get("total", sum(speed.values())), 2) if speed else None
    except Exception:
        pass

    # Pretty print
    print(f"  mAP50={out['mAP50']:.4f}  mAP50-95={out['mAP50-95']:.4f}  "
          f"mAP75={out['mAP75']:.4f}  val_time={val_time:.1f}s")
    if "speed_total_ms" in out and out["speed_total_ms"]:
        print(f"  speed: pre={out['speed_preprocess_ms']}ms  "
              f"inf={out['speed_inference_ms']}ms  total={out['speed_total_ms']}ms")

    return out


def compute_per_class_table(runs: list[dict], class_names: dict[str, list[str]]) -> pd.DataFrame | None:
    """Build per-class AP50 comparison table for a dataset."""
    rows = []
    for r in runs:
        if not r.get("per_class_ap50"):
            continue
        data_name = r["data"].replace(".yaml", "")
        names = class_names.get(data_name, [f"cls{i}" for i in r.get("class_indices", [])])
        for i, ap in zip(r.get("class_indices", []), r["per_class_ap50"]):
            cls_name = names[i] if i < len(names) else f"cls{i}"
            rows.append({"model": r["label"], "class_id": i, "class_name": cls_name, "ap50": ap})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["class_id", "class_name"], columns="model", values="ap50")
    return pivot


def find_boundary_cases(runs_by_dataset: dict[str, list[dict]], threshold_diff: float = 0.15):
    """Flag classes where models disagree (>threshold_diff AP gap) or both poor (<0.3 AP)."""
    findings = {}
    for data_name, runs in runs_by_dataset.items():
        pairs = {}
        for r in runs:
            # Group by model family
            family = None
            if "v0.1-N" in r["label"]:
                family = "v0.1-N"
            elif "EsMoE-N" in r["label"]:
                family = "EsMoE-N"
            if family:
                pairs.setdefault(family, []).append(r)

        # Find v0.1-N vs EsMoE-N gap per class
        if "v0.1-N" in pairs and "EsMoE-N" in pairs:
            v01 = {r["data"]: r for r in pairs["v0.1-N"]}.get(data_name + ".yaml") or pairs["v0.1-N"][0]
            esm = {r["data"]: r for r in pairs["EsMoE-N"]}.get(data_name + ".yaml") or pairs["EsMoE-N"][0]
            if v01.get("per_class_ap50") and esm.get("per_class_ap50"):
                big_diff = []
                both_poor = []
                for i, (a1, a2) in enumerate(zip(v01["per_class_ap50"], esm["per_class_ap50"])):
                    diff = abs(a1 - a2)
                    cls = i  # class index
                    if diff > threshold_diff:
                        big_diff.append({"class": cls, "v0.1-N": round(a1, 4), "EsMoE-N": round(a2, 4), "diff": round(diff, 4)})
                    if a1 < 0.3 and a2 < 0.3:
                        both_poor.append({"class": cls, "v0.1-N": round(a1, 4), "EsMoE-N": round(a2, 4)})
                findings[data_name] = {"big_diff": big_diff, "both_poor": both_poor}

    return findings


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run all validations ──────────────────────────────────────
    all_results = []
    for label, ckpt, data in ALL_RUNS:
        if not ckpt.exists():
            print(f"[SKIP] {label}: checkpoint not found at {ckpt}")
            continue
        try:
            r = run_val(label, ckpt, data)
            all_results.append(r)
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            all_results.append({"label": label, "error": str(e)})

    # ── Save raw results ─────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = EXP_DIR / f"val_results_{timestamp}.json"
    raw_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))

    # ── Summary table (Markdown) ─────────────────────────────────
    lines = [
        "# 模型验证对比结果\n",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 整体指标\n",
        "| 模型 | 数据集 | mAP50 | mAP50-95 | mAP75 | 推理耗时(s) | 推理速度(ms) |",
        "|------|--------|-------|----------|-------|------------|-------------|",
    ]
    for r in all_results:
        if "error" in r:
            continue
        speed = r.get("speed_total_ms", "")
        speed_str = f"{speed:.1f}" if speed else "-"
        lines.append(
            f"| {r['label']} | {r['data']} | {r['mAP50']:.4f} | {r['mAP50-95']:.4f} "
            f"| {r['mAP75']:.4f} | {r['val_time_s']:.1f} | {speed_str} |"
        )

    # ── Per-class AP tables ──────────────────────────────────────
    lines.append("\n## Per-Class AP50\n")

    # Try to load class names from dataset YAMLs
    class_names = {}
    for dataset_name in ["VisDrone", "construction-ppe", "brain-tumor"]:
        yaml_path = ROOT / "ultralytics/cfg/datasets" / f"{dataset_name}.yaml"
        if yaml_path.exists():
            import yaml as _yaml
            with open(yaml_path) as fh:
                cfg = _yaml.safe_load(fh)
                class_names[dataset_name] = cfg.get("names", [])

    for data_name in ["VisDrone.yaml", "construction-ppe.yaml", "brain-tumor.yaml"]:
        dataset_runs = [r for r in all_results if r.get("data") == data_name and "error" not in r]
        if not dataset_runs:
            continue
        dataset_key = data_name.replace(".yaml", "")
        lines.append(f"### {dataset_key}\n")

        # Build table manually
        # Collect all unique class names across runs
        all_classes = {}
        for r in dataset_runs:
            for i, ap in zip(r.get("class_indices", []), r.get("per_class_ap50", [])):
                names = class_names.get(dataset_key, [])
                name = names[i] if i < len(names) else f"cls{i}"
                key = (i, name)
                if key not in all_classes:
                    all_classes[key] = {}
                all_classes[key][r["label"]] = ap

        if all_classes:
            models_in_dataset = sorted(set(r["label"] for r in dataset_runs))
            header = "| class | " + " | ".join(models_in_dataset) + " |"
            sep = "|-------|" + "|".join(["-" * max(len(m), 4) for m in models_in_dataset]) + "|"
            lines.append(header)
            lines.append(sep)
            for (cid, cname) in sorted(all_classes.keys()):
                vals = []
                for m in models_in_dataset:
                    v = all_classes[(cid, cname)].get(m)
                    vals.append(f"{v:.4f}" if v is not None else "-")
                lines.append(f"| {cname} | " + " | ".join(vals) + " |")
            lines.append("")

    # ── Speed comparison ─────────────────────────────────────────
    lines.append("## 推理速度对比\n")
    lines.append("| 模型 | 数据集 | 预处理(ms) | 推理(ms) | 后处理(ms) | 总耗时(ms) |")
    lines.append("|------|--------|-----------|----------|-----------|-----------|")
    for r in all_results:
        if "error" in r:
            continue
        pre = r.get("speed_preprocess_ms", "-")
        inf = r.get("speed_inference_ms", "-")
        post = r.get("speed_postprocess_ms", "-")
        tot = r.get("speed_total_ms", "-")
        lines.append(f"| {r['label']} | {r['data']} | {pre} | {inf} | {post} | {tot} |")

    summary_path = EXP_DIR / f"comparison_report_{timestamp}.md"
    summary_path.write_text("\n".join(lines))

    # ── Boundary case detection ──────────────────────────────────
    runs_by_ds = {}
    for r in all_results:
        if "error" in r:
            continue
        runs_by_ds.setdefault(r["data"], []).append(r)
    boundary = find_boundary_cases(runs_by_ds)
    boundary_path = EXP_DIR / f"boundary_cases_{timestamp}.json"
    boundary_path.write_text(json.dumps(boundary, indent=2, ensure_ascii=False))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results saved:")
    print(f"  Raw:      {raw_path}")
    print(f"  Report:   {summary_path}")
    print(f"  Boundary: {boundary_path}")

    for ds, info in boundary.items():
        print(f"\n[{ds}]")
        print(f"  Big model gap (>0.15 AP50): {len(info['big_diff'])} classes")
        for c in info["big_diff"]:
            print(f"    class {c['class']}: v0.1-N={c['v0.1-N']:.4f}  EsMoE-N={c['EsMoE-N']:.4f}  gap={c['diff']:.4f}")
        print(f"  Both poor (<0.3 AP50): {len(info['both_poor'])} classes")
        for c in info["both_poor"]:
            print(f"    class {c['class']}: v0.1-N={c['v0.1-N']:.4f}  EsMoE-N={c['EsMoE-N']:.4f}")


if __name__ == "__main__":
    main()
