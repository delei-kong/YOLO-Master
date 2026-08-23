# Issue #49 PR 提交检查清单

> 目标：在 VisDrone + SKU-110K 上完成 4 轮训练，提交可复现 PR。
> 我们先用 VisDrone 跑通 pipeline，验证一切正常，再换另一个数据集。

---

## 一、PR 需要包含什么（按 Issue 要求）

### 1. 可复现脚本 ✅ 已完成
- [x] `scripts/reproduce/reproduce_visdrone.py` — 已有（34 行，调用 `_reproduce_common.py`）
- [x] `scripts/reproduce/reproduce_sku110k.py` — 已有（33 行，调用 `_reproduce_common.py`）
- [x] `scripts/reproduce/_reproduce_common.py` — 共享训练逻辑（425 行）

> 这些脚本已存在，不需要新写。但如果你的目标数据集不是 VisDrone/SKU-110K，比如你想换用 **AI-TOD-v2** 或其他数据集，你只需要照猫画虎写一个 ~30 行的新脚本即可（参考 `reproduce_aitodv2.py`）。

### 2. 训练日志 — 需要收集
- [ ] **W&B 项目**：包含 4 轮训练的完整 per-epoch 曲线
  - 每轮记录：`mAP50`, `mAP50-95`, `box_loss`, `cls_loss`, `moe_loss`（train + val）
  - 设置 W&B project 为公开，PR 中贴出 URL
  - 参考：已有作者提供的 [W&B 面板](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce)
- [ ] **results.csv**：每轮的本地 per-epoch 数据（`runs/reproduce/<dataset>/<Dataset>_<model>/results.csv`）
- [ ] **results.png**：训练曲线图（自动生成）

### 3. 结果对比表 — 需要填写
| 数据集 | 模型 | mAP50 | mAP50-95 | 参数量 | 备注 |
|--------|------|-------|----------|--------|------|
| VisDrone | v0.1-N | ? | ? | 7.55M | |
| VisDrone | EsMoE-N | ? | ? | 2.69M | 须加 `--no-sparse-eval` |
| SKU-110K | v0.1-N | ? | ? | 7.55M | |
| SKU-110K | EsMoE-N | ? | ? | 2.69M | 须加 `--no-sparse-eval` |

> 已有作者的 baseline 可以参考（见 `scripts/reproduce/README.md` 第 16-22 行）：
> - VisDrone v0.1-N: mAP50=0.344, mAP50-95=0.201
> - VisDrone EsMoE-N: mAP50=0.350, mAP50-95=0.203
> - SKU-110K v0.1-N: mAP50=0.906, mAP50-95=0.582
> - SKU-110K EsMoE-N: mAP50=0.904, mAP50-95=0.583

### 4. README.md — 需要确认/补充
- [ ] 数据集下载命令
- [ ] 训练命令
- [ ] 预期结果
- [ ] 已知问题与解决方案
- [ ] 环境要求（GPU、CUDA、PyTorch 版本）

> `scripts/reproduce/README.md` 已有完整文档。你可能只需要补充你的实际运行结果。

### 5. 模型权重（可选但建议）
- [ ] 将 `best.pt` 上传到 GitHub Release 或 HuggingFace
- [ ] 在 README 中提供下载链接

---

## 二、具体操作步骤

### Step 1：配置 W&B（一次性）
```bash
conda activate yolo_master
pip install wandb
wandb login           # 输入你的 API key（https://wandb.ai/authorize）
```

### Step 2：下载数据集
```bash
# VisDrone（2.3GB）— 国内用 gh-proxy 加速
wget -P /root/workspace/datasets/VisDrone \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip" \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip" \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip"

# 解压 + 标注转换
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"

# SKU-110K（13.6GB）
wget -P /root/workspace/datasets \
  "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
tar -xzf /root/workspace/datasets/SKU110K_fixed.tar.gz \
  --no-same-owner --no-same-permissions -C /root/workspace/datasets/
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```

### Step 3：训练前检查
```bash
# 确认模型能正常构建
python scripts/reproduce/reproduce_visdrone.py --check-build
python scripts/reproduce/reproduce_sku110k.py --check-build

# dry-run 打印完整参数
python scripts/reproduce/reproduce_visdrone.py --dry-run --epochs 100 --batch 32
```

### Step 4：启动训练（4 轮）

**推荐用 tmux 持久化，每轮一个 session：**
```bash
# 终端 1: VisDrone v0.1-N
tmux new -s visdrone_v01
python scripts/reproduce/reproduce_visdrone.py \
  --model v0.1-N --epochs 100 --batch 32 --device 0 --cache ram

# 终端 2: VisDrone EsMoE-N（⚠️ 必须加 --no-sparse-eval）
tmux new -s visdrone_esmoe
python scripts/reproduce/reproduce_visdrone.py \
  --model EsMoE-N --epochs 100 --batch 32 --device 0 --cache ram --no-sparse-eval

# 终端 3: SKU-110K v0.1-N
tmux new -s sku_v01
python scripts/reproduce/reproduce_sku110k.py \
  --model v0.1-N --epochs 100 --batch 32 --device 0 --cache ram

# 终端 4: SKU-110K EsMoE-N（⚠️ 必须加 --no-sparse-eval）
tmux new -s sku_esmoe
python scripts/reproduce/reproduce_sku110k.py \
  --model EsMoE-N --epochs 100 --batch 32 --device 0 --cache ram --no-sparse-eval
```

### Step 5：训练完成后收集产物
```bash
# 生成 summary
python scripts/reproduce/reproduce_visdrone.py --summary-only
python scripts/reproduce/reproduce_sku110k.py --summary-only

# 产出的文件位置：
# runs/reproduce/visdrone/
#   ├── VisDrone_v0.1-N/results.csv, results.png, weights/best.pt
#   ├── VisDrone_EsMoE-N/results.csv, results.png, weights/best.pt
#   └── summary.csv
# runs/reproduce/sku110k/
#   ├── SKU-110K_v0.1-N/results.csv, results.png, weights/best.pt
#   ├── SKU-110K_EsMoE-N/results.csv, results.png, weights/best.pt
#   └── summary.csv
```

### Step 6：填写实验注册表
更新 `/root/workspace/docs/issue6/experiment_registry.md`，记录：
- W&B URL
- 最终 mAP50 / mAP50-95
- 遇到的问题和解决方案

---

## 三、已知坑位（按严重程度排序）

| # | 问题 | 现象 | 解决方案 |
|---|------|------|---------|
| 1 | **EsMoE-N sparse eval 导致 mAP 崩塌** | val mAP ≈ 0.01，训练 loss 正常 | 必须加 `--no-sparse-eval` |
| 2 | **SKU-110K 解压权限错误** | `Cannot change ownership` | `tar --no-same-owner --no-same-permissions` |
| 3 | **续训时 GradScaler 空 dict 报错** | `RuntimeError: The source state dict is empty` | 删除 checkpoint 中的 scaler key，或从头训练 |
| 4 | **国内下载慢** | ~14KB/s | 用 `gh-proxy.org` 代理 GitHub release 源 |
| 5 | **EsMoE-N 在 AI-TOD-v2 上路由崩塌** | mAP ≈ 0，单个专家使用率 >0.8 | 换用 v0.1-N（有 shared expert 保底） |
| 6 | **cache=ram 在 val loader 构建时可能 hang** | 网络挂载盘上卡住 | 换 `--cache disk` |

---

## 四、你的实际情况 & 建议

### 关于数据集选择

Issue #49 要求 VisDrone + SKU-110K，但如果你计划换用其他数据集：

| 数据集 | 大小 | 类别数 | 特点 | 已有脚本 |
|--------|------|--------|------|---------|
| VisDrone | 2.3GB | 10 类 | 航拍密集小目标 | ✅ `reproduce_visdrone.py` |
| SKU-110K | 13.6GB | 1 类 | 零售密集商品 | ✅ `reproduce_sku110k.py` |
| AI-TOD-v2 | ~27GB | 8 类 | 极小目标（~12px）| ✅ `reproduce_aitodv2.py` |

如果换数据集，需要做的事：
1. 在 `ultralytics/cfg/datasets/` 下新增 `YourDataset.yaml`
2. 新建 `scripts/reproduce/reproduce_yourdataset.py`（照 `reproduce_visdrone.py` 抄，改 `DatasetSpec` 即可）
3. 下载数据集并验证 `check_det_dataset` 能通过
4. 正常训练

### 建议执行顺序
1. **先跑 VisDrone v0.1-N 冒烟测试**：`--epochs 2 --batch 32`，确认不报错
2. **VisDrone 跑通 4 轮完整训练**（W&B 日志 + results.csv + summary）
3. **commit VisDrone 结果**
4. **再搞第二个数据集**（SKU-110K 或其他）
