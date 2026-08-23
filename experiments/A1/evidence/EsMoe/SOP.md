# 垂类数据集复现 SOP（标准作业流程）

> 适用范围：对 `ultralytics/cfg/datasets/` 下的任意检测数据集，用 YOLO-Master 两个 nano 模型做 baseline 复现。
> 无论换哪个数据集，流程不变，只在 Step 0 换一个数据集名。
>
> **实战验证**：本 SOP 已通过 construction-ppe 数据集完整验证。参考产物：`docs/issue6/construction-ppe-v0.1N/`

---

## 速览：我需要完成什么？

| # | 步骤 | 产出物 | 预计耗时 |
|---|------|--------|---------|
| 0 | 选数据集 | 确定 `DATASET.yaml`，确认下载源 | 5 min |
| 1 | 写复现脚本 | `scripts/reproduce/reproduce_xxx.py`（~30 行） | 10 min |
| 2 | 下载数据集 | 数据就位，`check_det_dataset` 通过 | 10 min ~ 1h |
| 3 | 环境自检 | `--check-build` + `--dry-run` 通过 | 2 min |
| 4 | 冒烟测试 | 2 epoch 不报错，loss 正常下降 | 5 min |
| 5 | 完整训练 | v0.1-N + EsMoE-N 各一轮，W&B 日志 | 数小时 |
| **6** | **收集产物** | **13+ 个文件，含 W&B 导出 + 本地日志** | **10 min** |
| 7 | 写实验记录 + PR 文档 | experiment_record.md + PR.md | 15 min |
| 8 | 提交 PR | 脚本 + 日志 + 对比表 | 30 min |

---

## Step 0：选数据集

### 0.1 选数据集的铁律

**优先选 GitHub 源的数据集**（国内通过 `gh-proxy.org` 加速下载 ~2.8MB/s）。避免 Zenodo / S3 / cocodataset.org 等源（~13KB/s，无法加速）。

扫描数据集的下载源：
```bash
grep -Eo 'https?://[^"() ]+' /root/workspace/YOLO-Master/ultralytics/cfg/datasets/<DatasetName>.yaml
```

> GitHub 源的特征：`https://github.com/ultralytics/assets/releases/download/v0.0.0/<name>.zip`

### 0.2 确认 YAML 存在并了解特征

```bash
cat /root/workspace/YOLO-Master/ultralytics/cfg/datasets/<DatasetName>.yaml
```

关注：`nc`（类别数）、`names`（类别名）、`path`（数据目录）、`download`（下载源）。

---

## Step 1：写复现脚本（~30 行模板）

在 `scripts/reproduce/` 下新建 `reproduce_<dataset_lower>.py`：

```python
#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N and YOLO-Master-EsMoE-N baselines on <DatasetName>.

<DatasetName> (<场景描述>), built-in config <DatasetName>.yaml.
By default the models are reproduced as-is (EsMoE-N keeps its sparse eval, which
collapses mAP). Add --no-sparse-eval to opt into the corrected dense evaluation
for EsMoE-N (train==eval); v0.1-N is unaffected.

Examples:
    python scripts/reproduce/reproduce_<dataset>.py --check-build
    python scripts/reproduce/reproduce_<dataset>.py --epochs 100 --batch 32
    python scripts/reproduce/reproduce_<dataset>.py --model EsMoE-N --no-sparse-eval
    python scripts/reproduce/reproduce_<dataset>.py --model v0.1-N --no-wandb
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reproduce_common import DatasetSpec, run_dataset  # noqa: E402

DATASET = DatasetSpec(
    name="<DatasetName>",                          # 和 YAML 文件名一致，不含 .yaml
    data="<DatasetName>.yaml",
    project="runs/reproduce/<dataset_lower>",      # 训练产物输出目录
)


if __name__ == "__main__":
    raise SystemExit(run_dataset(DATASET))
```

> 核心设计：**新数据集只需要写一个 30 行的 `DatasetSpec` 声明文件**，所有训练逻辑复用 `_reproduce_common.py`。

---

## Step 2：下载数据集

### 2.1 GitHub 源数据集（推荐，快）

```bash
# 先确认 GitHub Release URL，然后手动下载
wget -P /root/gpufree-data/datasets/ \
  "https://gh-proxy.org/https://github.com/ultralytics/assets/releases/download/v0.0.0/<dataset>.zip"

# zip 已存在，ultralytics 会跳过下载，只做解压和标注转换
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('<DatasetName>.yaml', autodownload=True)"
```

### 2.2 非 GitHub 源（Zenodo/S3 等）

如果数据集不在 GitHub 上，提前评估下载时间（~13KB/s = 每 GB 约 22 小时）。必要时换用其他 GitHub 源的数据集。

### 2.3 验证

```bash
# 确认数据目录已就位，images/labels 子目录存在
ls /root/gpufree-data/datasets/<dataset_name>/images/ | head -5
ls /root/gpufree-data/datasets/<dataset_name>/labels/ | head -5
```

---

## Step 3：环境自检

```bash
conda activate yolo_master
cd /root/workspace/YOLO-Master

# 3.1 确认 GPU 和显存
nvidia-smi

# 3.2 验证模型能正常构建
python scripts/reproduce/reproduce_<dataset>.py --check-build

# 3.3 dry-run 打印完整参数
python scripts/reproduce/reproduce_<dataset>.py --dry-run --epochs 100 --batch 32
```

`--check-build` 成功输出：
```
[build-ok] v0.1-N: 7.547M
[build-ok] EsMoE-N: 2.694M
```

---

## Step 4：冒烟测试（epochs=2）

```bash
# 删掉之前的测试 run，从零开始
rm -rf /root/workspace/YOLO-Master/runs/reproduce/<project>/<Dataset>_v0.1-N

python scripts/reproduce/reproduce_<dataset>.py \
  --model v0.1-N --epochs 2 --batch 32 --device 0
```

### 冒烟通过标准

- [ ] 数据加载无报错
- [ ] train/box_loss 从 ~3.5 下降到 ~3.0 左右（2 个 epoch 后）
- [ ] val 正常完成（输出 mAP50 等指标）
- [ ] results.csv 生成且 mAP 开始从 0 上涨
- [ ] W&B 收到 2 个 epoch 的日志点

---

## Step 5：完整训练

### 5.1 配置 W&B

```bash
pip install wandb
wandb login   # 获取 API key: https://wandb.ai/authorize
```

### 5.2 训练命令（⚠️ 必须重定向日志！）

**关键教训**：训练命令必须显式保存完整日志到文件。不要裸跑，不要只靠 W&B——W&B 的 `output.log` 不含进度条且可能丢失部分输出。

```bash
# 清掉冒烟测试 run，从零开始
rm -rf /root/workspace/YOLO-Master/runs/reproduce/<project>/<Dataset>_v0.1-N

# v0.1-N（nohup + 日志重定向）
nohup python scripts/reproduce/reproduce_<dataset>.py \
  --model v0.1-N \
  --epochs 200 \
  --batch 32 \
  --device 0 \
  > /tmp/<dataset>_v01n.log 2>&1 &

# EsMoE-N（⚠️ 必须加 --no-sparse-eval）
nohup python scripts/reproduce/reproduce_<dataset>.py \
  --model EsMoE-N \
  --epochs 200 \
  --batch 32 \
  --device 0 \
  --no-sparse-eval \
  > /tmp/<dataset>_esmoe.log 2>&1 &
```

> **为什么用 nohup**：防止 SSH 断连/定时关机导致进程被杀。`nohup` 忽略 SIGHUP。
> **为什么重定向到 /tmp**：确保完整的进度条、WARNING、ERROR 都被保存。W&B `output.log` 不含进度条。
> **为什么不用 `&& echo PID`**：nohup 下 `$!` 不可靠，用 `ps aux | grep reproduce` 查 PID。
> 推荐 epochs=200（平衡效果和耗时）。RTX 4090 24GB 上 batch=32 安全。
> 训练过程中记录 W&B run URL（控制台会打印），后续导出需要用到。

### 5.3 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 300 | 推荐 200 |
| `--batch` | 64 | RTX 4090 24G 建议 32 |
| `--no-sparse-eval` | off | **EsMoE-N 必须加**，v0.1-N 不需要 |
| `--wandb-mode offline` | online | 没网络时用 offline，之后 `wandb sync` 上传 |

---

## Step 6：收集训练产物（关键！按此清单逐项执行）

训练完成后，所有产物统一保存到 `docs/issue6/<dataset>-<model>/`。

### 6.1 本地训练产物

```bash
DATASET="<dataset-name>"      # 如 construction-ppe
MODEL="v0.1-N"                 # 模型名
RUN_DIR="<Dataset>_<model>"    # 如 construction-ppe_v0.1-N
PROJECT="<project>"            # 如 ppe
OUT="docs/issue6/${DATASET}-${MODEL}/${MODEL}"

mkdir -p "$OUT"

# 从 ultralytics run 目录拷贝
cp runs/reproduce/${PROJECT}/${RUN_DIR}/results.csv   "$OUT/"
cp runs/reproduce/${PROJECT}/${RUN_DIR}/args.yaml      "$OUT/"
cp runs/reproduce/${PROJECT}/${RUN_DIR}/labels.jpg     "$OUT/" 2>/dev/null
cp runs/reproduce/${PROJECT}/${RUN_DIR}/weights/best.pt "$OUT/" 2>/dev/null
```

### 6.2 生成 summary.csv

```bash
python scripts/reproduce/reproduce_<dataset>.py --summary-only
cp runs/reproduce/${PROJECT}/summary.csv "docs/issue6/${DATASET}-${MODEL}/"
```

### 6.3 从 W&B 云端导出（必须逐项执行）

记录训练时控制台打印的 W&B URL，或通过 `wandb` 命令查看。

```bash
# 查看 W&B run URL（如果忘了）
# 训练控制台输出中有: wandb: 🚀 View run ... at: https://wandb.ai/...

WANDB_RUN="<entity>/<project>/<run_id>"   # 如 delei-kong-szu/yolo-master-reproduce/3z8vcfuy
```

然后运行导出脚本：

```python
import wandb

api = wandb.Api()
run = api.run("<WANDB_RUN>")
out_dir = "docs/issue6/<dataset>-<model>/<model>/"

# 1) output.log — 控制台完整日志（最重要！）
run.file("output.log").download(root=out_dir, replace=True)

# 2) requirements.txt — 环境依赖
run.file("requirements.txt").download(root=out_dir, replace=True)

# 3) config.yaml — 训练配置
run.file("config.yaml").download(root=out_dir, replace=True)

# 4) wandb-summary.json — 最终汇总指标
run.file("wandb-summary.json").download(root=out_dir, replace=True)

# 5) wandb_history.csv — per-epoch 完整指标
import pandas as pd
history = run.history()
history.to_csv(f"{out_dir}/wandb_history.csv", index=False)

# 6) wandb_meta.json — run 元信息
import json
meta = {
    "name": run.name, "id": run.id, "url": run.url,
    "state": run.state, "created_at": str(run.created_at),
    "config": dict(run.config) if run.config else {},
    "tags": list(run.tags) if run.tags else [],
}
with open(f"{out_dir}/wandb_meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

print(f"All W&B files exported to {out_dir}")
```

### 6.4 保存完整训练日志（⚠️ 训练完成后立刻做！）

**training_full.log 是比 W&B output.log 更完整的日志**（含进度条、所有 WARNING/ERROR）。**训练完成后必须立刻保存，不能等**——/tmp 可能被清理，定时关机可能生效。

**标准流程：直接从 nohup 日志拷贝**（Step 5.2 中重定向到 /tmp 的日志文件）：

```bash
LOG_SRC="/tmp/<dataset>_v01n.log"   # 或 <dataset>_esmoe.log
LOG_DST="docs/issue6/<dataset>-<model>/<model>/training_full.log"
cp "$LOG_SRC" "$LOG_DST"
echo "training_full.log saved: $(wc -l < "$LOG_DST") lines"
```

如果是从 Claude Code 后台任务启动的，额外取任务输出：
```bash
# 路径类似: /tmp/claude-0/-root-workspace/.../tasks/<task_id>.output
cp <task_output_file> "docs/issue6/<dataset>-<model>/<model>/training_full.log"
```

如果以上都没有（裸跑命令/只用了 `&`），**用 W&B output.log 替代**（不含进度条，但包含所有 WARNING/ERROR）：
```bash
cp "docs/issue6/<dataset>-<model>/<model>/output.log" "docs/issue6/<dataset>-<model>/<model>/training_full.log"
```

> **教训**：本次 brain-tumor EsMoE-N 和 construction-ppe EsMoE-N 因未用 nohup 重定向，丢失了含进度条的训练日志。**以后一律用 Step 5.2 的 nohup 方式启动。**

### 6.5 手动生成 results.png（如果未自动生成）

ultralytics 的 results.png 在 final_eval 失败时不会生成。手动画图：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("docs/issue6/<dataset>-<model>/<model>/results.csv")
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("<Dataset> — <Model> (<epochs> epochs)", fontsize=14)

# Loss curves
for ax, key, label in [
    (axes[0,0], "train/box_loss", "Box"),
    (axes[0,1], "train/cls_loss", "Cls"),
    (axes[0,2], "train/dfl_loss", "DFL"),
]:
    ax.plot(df["epoch"], df[key], label=f"train", color="blue")
    vkey = f"val/{key.split('/')[-1]}"
    if vkey in df.columns:
        ax.plot(df["epoch"], df[vkey], label=f"val", color="orange")
    ax.set_title(f"{label} Loss"); ax.legend(); ax.grid(True)

# mAP
axes[1,0].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="green")
axes[1,0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color="red")
axes[1,0].set_title("mAP"); axes[1,0].legend(); axes[1,0].grid(True)

# Precision/Recall
axes[1,1].plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="purple")
axes[1,1].plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="brown")
axes[1,1].set_title("Precision & Recall"); axes[1,1].legend(); axes[1,1].grid(True)

# LR
for col in ["lr/pg0", "lr/pg1", "lr/pg2"]:
    if col in df.columns:
        axes[1,2].plot(df["epoch"], df[col], label=col)
axes[1,2].set_title("LR Schedule"); axes[1,2].legend(); axes[1,2].grid(True)

plt.tight_layout()
plt.savefig("docs/issue6/<dataset>-<model>/<model>/results.png", dpi=150)
```

### 6.6 产物完整性检查清单

```bash
DATASET="<dataset-name>"
MODEL="<model>"
DIR="docs/issue6/${DATASET}-${MODEL}"

echo "=== 检查 ${DATASET}-${MODEL} ==="
for f in \
    "${DIR}/${MODEL}/results.csv" \
    "${DIR}/${MODEL}/results.png" \
    "${DIR}/${MODEL}/args.yaml" \
    "${DIR}/${MODEL}/labels.jpg" \
    "${DIR}/${MODEL}/output.log" \
    "${DIR}/${MODEL}/training_full.log" \
    "${DIR}/${MODEL}/wandb_history.csv" \
    "${DIR}/${MODEL}/wandb_meta.json" \
    "${DIR}/${MODEL}/wandb-summary.json" \
    "${DIR}/${MODEL}/config.yaml" \
    "${DIR}/${MODEL}/requirements.txt" \
    "${DIR}/summary.csv" \
    "${DIR}/experiment_record.md" \
    "${DIR}/PR.md"; do
  if [ -f "$f" ]; then echo "  ✅ $f"; else echo "  ❌ MISSING: $f"; fi
done
```

### 6.7 全部产物清单（13+ 文件）

```
docs/issue6/<dataset>-<model>/
├── PR.md                         # PR 提交文档
├── experiment_record.md           # 实验记录表
├── summary.csv                    # 汇总表
└── <model>/
    ├── results.csv                # ultralytics 原生 per-epoch 指标
    ├── results.png                # 训练曲线图（6 子图）
    ├── args.yaml                  # ultralytics 完整训练参数
    ├── labels.jpg                 # 标签分布图
    ├── output.log                 # W&B 云端控制台日志
    ├── training_full.log          # 最完整的训练日志（含进度条）
    ├── wandb_history.csv          # W&B per-epoch 指标
    ├── wandb_meta.json            # W&B run 元信息
    ├── wandb-summary.json         # W&B 最终汇总
    ├── config.yaml                # W&B 训练配置
    ├── requirements.txt           # Python 依赖列表
    └── best.pt                    # 最优模型权重（可选）
```

---

## Step 7：写实验记录 + PR 文档

### 7.1 设计实验
设计几个简单的实验和可视化效果，例如：
1. 用预训练的权重去跑一下

### 7.1 experiment_record.md

记录以下内容（参考 `docs/issue6/construction-ppe-v0.1N/experiment_record.md`）：
- 基本信息（ID、数据集、模型、日期、状态）
- 数据集概况（类别、训练/验证/测试数量）
- 训练配置表格
- 训练命令
- 最佳结果 + 最终结果
- Loss 收敛趋势表
- 产物清单
- W&B URL
- 已知问题
- 结论

### 7.2 PR.md

记录以下内容（参考 `docs/issue6/construction-ppe-v0.1N/PR.md`）：
- 环境表格
- 数据集概况
- 训练配置
- 结果表格（mAP50 / mAP50-95）
- W&B 链接
- 复现命令
- 新增文件列表
- 已知问题

---

## Step 8：提交 PR

### 8.1 PR 应包含的文件

- [ ] `scripts/reproduce/reproduce_<dataset>.py` — 新增复现脚本
- [ ] `docs/issue6/<dataset>-<model>/` — 完整训练日志和产物
- [ ] 更新 `scripts/reproduce/README.md`（可选）

### 8.2 PR Description

直接用 `docs/issue6/<dataset>-<model>/PR.md` 的内容贴到 PR body。

---

## 已知问题速查

| # | 问题 | 现象 | 解决方案 |
|---|------|------|---------|
| 1 | **EsMoE-N val mAP 崩塌** | val mAP ≈ 0.01，train loss 正常 | 必须加 `--no-sparse-eval` |
| 2 | **final_eval Router NaN** | 训练完成但 `RuntimeError: Router input contains NaN/Inf` | 不影响 per-epoch 指标收集；results.png 需手动画；待进一步排查 AMP 和 Router 交互 |
| 3 | **SKU-110K tar 权限** | `Cannot change ownership` | `tar --no-same-owner --no-same-permissions` |
| 4 | **续训 GradScaler 报错** | `RuntimeError: The source state dict is empty` | 删除 checkpoint 中的 `scaler` key |
| 5 | **国内下载慢** | Zenodo/S3 源 ~13KB/s | 优先选 GitHub 源数据集 + gh-proxy |
| 6 | **EsMoE-N 极小目标路由崩塌** | mAP ≈ 0 | 换用 v0.1-N |

---

## 附录 A：GitHub 源候选数据集

以下数据集托管在 GitHub 上，可通过 `gh-proxy.org` 加速（~2.8MB/s）：

| 数据集 | 场景 | 类别数 | 训练图数 | 大小 |
|--------|------|--------|---------|------|
| 🦺 **construction-ppe** | 工业安全 | 11 | 1,132 | ~170MB |
| 🧠 **brain-tumor** | 医疗·脑肿瘤 | 1 | — | 小 |
| 💊 **medical-pills** | 医疗·药丸 | 1 | — | 小 |
| 🛰️ **DOTAv1** | 航拍·15类 | 15 | — | 中 |
| 🦁 **african-wildlife** | 野生动物 | 4 | — | 小 |
| 🏠 **HomeObjects-3K** | 家居物品 | 100+ | — | 中 |
| 🚗 **kitti** | 自动驾驶 | 8 | — | 中 |

---

## 附录 B：核心脚本架构

```
scripts/reproduce/
├── _reproduce_common.py       # 共享训练引擎（不需要改）
│   ├── DatasetSpec / ModelSpec
│   ├── train_one()             # 训练单个模型（断点续训内置）
│   ├── write_summary()         # 生成 summary.csv
│   ├── build_parser()          # 命令行参数
│   ├── run_dataset()           # 入口函数
│   ├── _make_dense_inference_callback()  # EsMoE-N dense eval
│   └── _make_wandb_callbacks()          # W&B 日志上报
│
├── reproduce_visdrone.py       # VisDrone 复现入口
├── reproduce_sku110k.py        # SKU-110K 复现入口
├── reproduce_ppe.py            # construction-ppe 复现入口
└── reproduce_<your>.py         # ← 你新建的入口（~30 行）
```
