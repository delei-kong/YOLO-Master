#!/usr/bin/env python3
"""
为 boundary_cases 中的图片分别绘制 GT 标注和预测框，保存为独立图片。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

YOLO_ROOT = Path("/root/workspace/YOLO-Master")
DATASET_ROOT = Path("/root/workspace/datasets/construction-ppe")
BOUNDARY_DIR = Path("/root/workspace/docs/issue6/analysis_output/boundary_cases")

CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles", "none",
    "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots",
]

# 每类用不同颜色
CLASS_COLORS = [
    "#00BFFF", "#FFD700", "#32CD32", "#FF6347", "#9370DB", "#808080",
    "#FF1493", "#FF4500", "#00CED1", "#FF8C00", "#8B0000",
]


def draw_boxes(img: Image.Image, boxes: list, out_path: Path, title: str):
    """在图片上绘制边界框并保存"""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        cls_id = box["cls"]
        cx, cy, bw, bh = box["cx"], box["cy"], box["w"], box["h"]
        # YOLO normalized -> pixel
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        name = CLASS_NAMES[cls_id]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # label background
        draw.rectangle([x1, y1 - 18, x1 + len(name) * 8 + 10, y1], fill=color)
        draw.text((x1 + 3, y1 - 16), name, fill="white", font=font)

    # 标题
    draw.text((8, 8), title, fill="white", font=font)
    # shadow
    draw.text((7, 7), title, fill="black", font=font)
    draw.text((8, 8), title, fill="white", font=font)

    img.save(out_path)
    print(f"  -> {out_path.name}")


def draw_pred_boxes(img: Image.Image, boxes: list, out_path: Path, title: str):
    """在图片上绘制预测框（带置信度）"""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        cls_id = box["cls"]
        conf = box.get("conf", 0.0)
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        name = CLASS_NAMES[cls_id]
        label = f"{name} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 8 + 10, y1], fill=color)
        draw.text((x1 + 3, y1 - 16), label, fill="white", font=font)

    # 标题
    draw.text((7, 7), title, fill="black", font=font)
    draw.text((8, 8), title, fill="white", font=font)

    img.save(out_path)
    print(f"  -> {out_path.name}")


def main():
    sys.path.insert(0, str(YOLO_ROOT))
    from ultralytics import YOLO

    model = YOLO(str(YOLO_ROOT / "runs/reproduce/ppe/construction-ppe_v0.1-N/weights/best.pt"))

    test_img_dir = DATASET_ROOT / "images/test"
    test_label_dir = DATASET_ROOT / "labels/test"

    # 获取已保存的边界用例列表
    existing = sorted(BOUNDARY_DIR.glob("LOW_CONF_*"))

    # 删掉旧的纯拷贝图片，换成 GT/Pred 分开的
    for f in existing:
        f.unlink()

    # 读取边界用例 JSON 获取用例列表
    import json
    with open(Path("/root/workspace/docs/issue6/analysis_output/05_boundary_cases.json")) as f:
        bc_data = json.load(f)

    print(f"处理 {min(20, len(bc_data['cases']))} 个边界用例...")

    for case in bc_data["cases"][:20]:
        img_name = case["image"]
        stem = Path(img_name).stem  # 用 stem 避免 .jpeg vs .jpg 不一致

        img_path = test_img_dir / img_name
        label_path = test_label_dir / f"{stem}.txt"

        if not img_path.exists():
            # 尝试 jpeg 扩展名
            img_path_jpeg = test_img_dir / f"{stem}.jpeg"
            if img_path_jpeg.exists():
                img_path = img_path_jpeg
            else:
                print(f"  [SKIP] 找不到图片: {img_name}")
                continue

        # ── 1. 画 GT ──
        img_gt = Image.open(img_path).convert("RGB")
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
        draw_boxes(img_gt, gt_boxes, BOUNDARY_DIR / f"{stem}_gt.jpg",
                   f"GT: {img_name} ({len(gt_boxes)} objects)")

        # ── 2. 画 Prediction ──
        img_pred = Image.open(img_path).convert("RGB")
        results = model.predict(source=str(img_path), imgsz=640, conf=0.25,
                                iou=0.7, device="0", verbose=False)
        pred_boxes = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                cls_arr = boxes.cls.cpu().tolist()
                conf_arr = boxes.conf.cpu().tolist()
                xyxy_arr = boxes.xyxy.cpu().tolist()
                for i in range(len(cls_arr)):
                    pred_boxes.append({
                        "cls": int(cls_arr[i]),
                        "conf": float(conf_arr[i]),
                        "x1": int(xyxy_arr[i][0]),
                        "y1": int(xyxy_arr[i][1]),
                        "x2": int(xyxy_arr[i][2]),
                        "y2": int(xyxy_arr[i][3]),
                    })
        draw_pred_boxes(img_pred, pred_boxes, BOUNDARY_DIR / f"{stem}_pred.jpg",
                        f"Pred: {img_name} ({len(pred_boxes)} objects, conf>=0.25)")

    del model
    print(f"\n完成！共 {len(bc_data['cases'][:20])} 个边界用例，每个生成 _gt.jpg 和 _pred.jpg")


if __name__ == "__main__":
    main()
