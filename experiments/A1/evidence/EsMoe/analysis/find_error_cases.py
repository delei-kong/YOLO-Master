#!/usr/bin/env python3
"""Find and visualize error cases: v0.1-N vs EsMoE-N vs GT."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT.parent / "docs/issue6/exp"  # /root/workspace/docs/issue6/exp

CKPTS = {
    "v0.1-N": {
        "VisDrone": ROOT / "runs/reproduce/visdrone/VisDrone_v0.1-N/weights/best.pt",
        "construction-ppe": ROOT / "runs/reproduce/ppe/construction-ppe_v0.1-N/weights/best.pt",
        "brain-tumor": ROOT / "runs/reproduce/brain-tumor/brain-tumor_v0.1-N/weights/best.pt",
    },
    "EsMoE-N": {
        "VisDrone": ROOT / "runs/reproduce/visdrone/VisDrone_EsMoE-N/weights/best.pt",
        "construction-ppe": ROOT / "runs/reproduce/ppe/construction-ppe_EsMoE-N/weights/best.pt",
        "brain-tumor": ROOT / "runs/reproduce/brain-tumor/brain-tumor_EsMoE-N/weights/best.pt",
    },
}

DATASETS = {
    "VisDrone": "VisDrone.yaml",
    "construction-ppe": "construction-ppe.yaml",
    "brain-tumor": "brain-tumor.yaml",
}

# ── per-data conf maps & class filter hints ──
WEAK_CLASSES = {
    "VisDrone": ["bicycle", "awning-tricycle", "people"],
    "construction-ppe": ["no_boots", "no_goggle", "no_gloves", "Person", "helmet"],
    "brain-tumor": ["positive"],
}
BIG_GAP_CLASSES = {
    "VisDrone": ["bus", "tricycle"],
    "construction-ppe": ["goggles", "gloves", "vest", "boots", "Person", "helmet"],
    "brain-tumor": ["negative"],
}

COLORS = [(0, 255, 0), (255, 0, 0)]  # green: GT / blue: pred


def draw_boxes(img, boxes, color, label_prefix="", class_names=None):
    """Draw bounding boxes on image, return annotated copy."""
    out = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4] if len(box) > 4 else None
        cls_id = int(box[5]) if len(box) > 5 else None
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Use class name if available
        if class_names and cls_id is not None and cls_id < len(class_names):
            cls_str = class_names[cls_id]
        else:
            cls_str = str(cls_id) if cls_id is not None else ""
        text = f"{label_prefix}{cls_str}"
        if conf is not None:
            text += f" {conf:.2f}"
        cv2.putText(out, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def run_predict(model: YOLO, img_path: str, conf: float = 0.25):
    """Run predict on single image, return list of [x1,y1,x2,y2,conf,cls]."""
    results = model(img_path, conf=conf, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is not None:
            for b in r.boxes.data.cpu().numpy():
                boxes.append(b.tolist())  # [x1, y1, x2, y2, conf, cls]
    return boxes


def load_gt(label_path: str, img_w: int, img_h: int):
    """Load YOLO-format GT labels, return [x1, y1, x2, y2, cls] (absolute coords)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2, cls_id])
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def score_case(pred_boxes, gt_boxes, cls_id_target, iou_thresh=0.5):
    """Score how interesting a case is:
    - For 'both_poor': both models miss many GT boxes of cls_id_target
    - For 'big_gap': models disagree significantly on cls_id_target
    Returns (miss_rate_v01, miss_rate_esm) or None if no GT of target class.
    """
    gt_target = [b for b in gt_boxes if int(b[4]) == cls_id_target]
    if not gt_target:
        return None

    def miss_rate(preds):
        matched = 0
        pred_target = [b for b in preds if int(b[5]) == cls_id_target]
        for gt in gt_target:
            for p in pred_target:
                if compute_iou(gt[:4], p[:4]) >= iou_thresh:
                    matched += 1
                    break
        return 1.0 - matched / len(gt_target)

    return miss_rate([b + [0] for b in []]), miss_rate([b + [0] for b in []])  # placeholder


def main():
    os.makedirs(EXP_DIR, exist_ok=True)

    # Load class name maps
    import yaml
    class_names = {}
    for ds, yaml_file in DATASETS.items():
        with open(ROOT / "ultralytics/cfg/datasets" / yaml_file) as f:
            raw_names = yaml.safe_load(f).get("names", [])
        if isinstance(raw_names, dict):
            class_names[ds] = [raw_names[i] for i in range(len(raw_names))]
        else:
            class_names[ds] = list(raw_names)

    # Load models
    models = {}
    for model_name, ckpt_map in CKPTS.items():
        models[model_name] = {}
        for ds, ckpt_path in ckpt_map.items():
            models[model_name][ds] = YOLO(str(ckpt_path))

    # Process each dataset
    for ds, yaml_file in DATASETS.items():
        print(f"\n{'='*60}\n{ds}\n{'='*60}")

        # Get class ids for target classes
        names = class_names[ds]
        name_to_id = {n: i for i, n in enumerate(names)}

        weak_ids = [name_to_id[n] for n in WEAK_CLASSES.get(ds, []) if n in name_to_id]
        gap_ids = [name_to_id[n] for n in BIG_GAP_CLASSES.get(ds, []) if n in name_to_id]

        # Find val images
        data_dir = ROOT.parent.parent / "gpufree-data/datasets" / ds
        val_img_dir = data_dir / "images" / "val"
        val_lbl_dir = data_dir / "labels" / "val"
        img_files = sorted(val_img_dir.glob("*.[jp][pn][g]"))

        print(f"  Images: {len(img_files)}, weak_ids={weak_ids}, gap_ids={gap_ids}")

        # Collect cases
        cases = []  # [(img_path, v01_preds, esm_preds, gt_boxes, case_type, cid, score, desc)]

        for img_path in img_files:  # scan ALL images
            label_path = val_lbl_dir / (img_path.stem + ".txt")
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            gt_boxes = load_gt(str(label_path), w, h)
            v01_preds = run_predict(models["v0.1-N"][ds], str(img_path))
            esm_preds = run_predict(models["EsMoE-N"][ds], str(img_path))

            # Helper: match pred boxes to GT boxes
            def match_preds(preds, gt_boxes, cid_filter=None):
                """Return (matched_count, false_positives).
                matched: pred box has IoU>=0.5 with some GT of same class
                fp: pred box has IoU<0.3 with ALL GT boxes (hallucination)"""
                matched = 0
                fps = 0
                for p in preds:
                    p_cls = int(p[5])
                    if cid_filter is not None and p_cls != cid_filter:
                        continue
                    best_iou = max([compute_iou(p[:4], g[:4]) for g in gt_boxes]) if gt_boxes else 0
                    if best_iou >= 0.5:
                        # Check if class matches the best GT
                        best_g_idx = max(range(len(gt_boxes)), key=lambda i: compute_iou(p[:4], gt_boxes[i][:4]))
                        if int(gt_boxes[best_g_idx][4]) == p_cls:
                            matched += 1
                    elif best_iou < 0.3:
                        fps += 1
                return matched, fps

            # Score for weak classes (both models poor)
            for cid in weak_ids:
                gt_target = [b for b in gt_boxes if int(b[4]) == cid]
                if not gt_target:
                    continue
                v01_target = [b for b in v01_preds if int(b[5]) == cid]
                esm_target = [b for b in esm_preds if int(b[5]) == cid]
                v01_match = sum(1 for gt in gt_target for p in v01_target if compute_iou(gt[:4], p[:4]) >= 0.5)
                esm_match = sum(1 for gt in gt_target for p in esm_target if compute_iou(gt[:4], p[:4]) >= 0.5)
                v01_miss = len(gt_target) - v01_match
                esm_miss = len(gt_target) - esm_match
                if v01_miss >= 2 and esm_miss >= 2:
                    score = v01_miss + esm_miss
                    desc = f"v0.1-N miss {v01_miss}/{len(gt_target)}, EsMoE-N miss {esm_miss}/{len(gt_target)}"
                    cases.append((img_path, v01_preds, esm_preds, gt_boxes, "both_poor", cid, score, desc))
                    if len(cases) < 5:  # only print first few
                        print(f"    both_poor: {img_path.name} cls={names[cid]} {desc}")

            # Score for big gap classes
            for cid in gap_ids:
                gt_target = [b for b in gt_boxes if int(b[4]) == cid]
                if not gt_target:
                    continue
                v01_target = [b for b in v01_preds if int(b[5]) == cid]
                esm_target = [b for b in esm_preds if int(b[5]) == cid]
                v01_hit = sum(1 for gt in gt_target for p in v01_target if compute_iou(gt[:4], p[:4]) >= 0.5)
                esm_hit = sum(1 for gt in gt_target for p in esm_target if compute_iou(gt[:4], p[:4]) >= 0.5)
                gap = abs(v01_hit - esm_hit)
                if gap >= 1 and len(gt_target) >= 1:
                    desc = f"v0.1-N hit {v01_hit}, EsMoE-N hit {esm_hit} (gap={gap})"
                    cases.append((img_path, v01_preds, esm_preds, gt_boxes, "big_gap", cid, gap, desc))
                    print(f"    big_gap: {img_path.name} cls={names[cid]} {desc}")

            # Detect false positives (hallucinations)
            for model_name, preds in [("v0.1-N", v01_preds), ("EsMoE-N", esm_preds)]:
                fps = 0
                for p in preds:
                    best_iou = max([compute_iou(p[:4], g[:4]) for g in gt_boxes]) if gt_boxes else 0
                    if best_iou < 0.3:
                        fps += 1
                if fps >= 2:
                    cid = -1  # generic fp
                    desc = f"{model_name} has {fps} false positives (no matching GT)"
                    cases.append((img_path, v01_preds, esm_preds, gt_boxes, "false_positive", cid, fps, desc))
                    print(f"    false_positive: {img_path.name} {desc}")

        # Pick top 3 cases per dataset with diversity
        both_poor_cases = sorted([c for c in cases if c[4] == "both_poor"], key=lambda x: -x[6])
        big_gap_cases = sorted([c for c in cases if c[4] == "big_gap"], key=lambda x: -x[6])
        fp_cases = sorted([c for c in cases if c[4] == "false_positive"], key=lambda x: -x[6])

        # Pick diverse: 1 both_poor + 1 big_gap + 1 fp if available. Otherwise fill with both_poor.
        selected = []
        # Deduplicate by class for diversity
        seen_cls = set()
        for pool in [big_gap_cases, fp_cases, both_poor_cases]:
            for c in pool:
                cid = c[5]
                if cid not in seen_cls and c not in [s for s in selected]:
                    selected.append(c)
                    seen_cls.add(cid)
                    break
            if len(selected) >= 3:
                break
        # Fill remaining
        for c in both_poor_cases:
            if len(selected) >= 3:
                break
            if c not in selected:
                selected.append(c)

        print(f"  Selected {len(selected)} cases")

        # Draw and save
        for idx, (img_path, v01_preds, esm_preds, gt_boxes, case_type, cid, score, desc) in enumerate(selected):
            img = cv2.imread(str(img_path))
            cname = names[cid] if cid >= 0 and cid < len(names) else "fp"

            # GT
            gt_img = draw_boxes(img, [[b[0], b[1], b[2], b[3], 1.0, b[4]] for b in gt_boxes], (0, 255, 0), "GT:", class_names[ds])
            # v0.1-N
            v01_img = draw_boxes(img.copy(), v01_preds, (255, 0, 0), "v0.1-N:", class_names[ds])
            # EsMoE-N
            esm_img = draw_boxes(img.copy(), esm_preds, (0, 0, 255), "EsMoE-N:", class_names[ds])

            # Stack 3 images horizontally
            combined = np.hstack([v01_img, esm_img, gt_img])
            short_desc = desc[:40].replace(" ", "_").replace("/", "_").replace(",", "").replace(".", "")
            out_name = f"{ds}_{case_type}_{cname}_case{idx+1}.jpg"
            out_path = EXP_DIR / out_name
            cv2.imwrite(str(out_path), combined)
            print(f"    Saved: {out_name}  [{desc}]")

    print(f"\nDone! Images saved to {EXP_DIR}/")


if __name__ == "__main__":
    main()
