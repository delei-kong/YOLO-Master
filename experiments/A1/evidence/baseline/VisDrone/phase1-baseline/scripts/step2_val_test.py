#!/usr/bin/env python3
"""步骤2：baseline 模型在 VisDrone test 集上推理评估。

ES-MoE 论文设计：训练 soft Top-K (dense) → 推理 hard Top-K (sparse)。
正确推理模式：use_sparse_inference=True, top_k=2。

运行方式：
    bash step2_val_test.sh
"""

import sys
sys.path.insert(0, "/root/workspace/YOLO-Master")

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE

MODEL_PATH = "/root/workspace/YOLO-Master-docs/issue2/VisDrone/phase1-baseline/baseline/best.pt"
DATA_YAML = "/root/workspace/YOLO-Master/ultralytics/cfg/datasets/VisDrone.yaml"

# 1. 加载模型
print("=" * 60)
print("[1/3] 加载模型...")
model = YOLO(MODEL_PATH)

# 2. 开启 ES_MOE sparse inference (hard Top-K)，与论文部署模式一致
print("[2/3] 设置 ES_MOE 为 sparse inference (top_k=2, 4 选 2)...")
count = 0
for module in model.model.modules():
    if isinstance(module, ES_MOE):
        module.use_sparse_inference = True
        module.use_top_k = True
        module.top_k = 2
        count += 1
print(f"      已将 {count} 个 ES_MOE 模块设为 sparse (top_k=2)")

# 3. 在 test 集上评估
print("[3/3] 在 VisDrone test 集上评估...")
print("=" * 60)

results = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=640,
    batch=16,
    device=0,
)

print("\n评估完成!")
