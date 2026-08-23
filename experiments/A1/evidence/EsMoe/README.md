# Issue 6：模型训练专项 — 垂类数据集基线训练

> Issue 链接：https://github.com/Tencent/YOLO-Master/issues/49
>
> 仓库路径：`/root/workspace/YOLO-Master`

## 1. 磁盘架构

```
/root/workspace/datasets  →  (软链接)  →  /root/gpufree-data/datasets  (49GB 数据盘)
```

所有代码和 ultralytics 只认 `/root/workspace/datasets`，实际数据自动落到 49GB 数据盘，不占 30GB 系统盘。

```bash
mkdir -p /root/gpufree-data/datasets
ln -sfn /root/gpufree-data/datasets /root/workspace/datasets
```

## 2. 下载数据集

ultralytics 内置的自动下载走 `ultralytics.com/assets`，国内极慢（~14KB/s）。
实际底层是 GitHub release asset，通过 `gh-proxy.org` 代理后速度 ~2.8MB/s（约 600 倍提升）。

### VisDrone（2.3GB）

```bash
conda activate yolo_master

# 通过 gh-proxy 下载 3 个 zip 文件
wget -P /root/workspace/datasets/VisDrone \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip" \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip" \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip"

# 运行 auto-download 完成解压 + VisDrone→YOLO 标注转换（zip 已存在，跳过下载阶段）
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
```

### SKU-110K（13.6GB）

```bash
# 下载 SKU-110K 原始 tar.gz（S3 源，可能需要代理）
wget -P /root/workspace/datasets \
  "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"

# 手动解压（避免 tar 权限报错：Cannot change ownership ... Operation not permitted）
tar -xzf /root/workspace/datasets/SKU110K_fixed.tar.gz \
  --no-same-owner --no-same-permissions -C /root/workspace/datasets/

# 运行 auto-download 完成标注转换（CSV→YOLO txt）
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```

## 3. 训练前自检

```bash
conda activate yolo_master

# 验证模型 config 可 build
python scripts/reproduce/reproduce_visdrone.py --check-build
python scripts/reproduce/reproduce_sku110k.py --check-build

# dry-run 打印完整计划
python scripts/reproduce/reproduce_visdrone.py --dry-run
```

## 4. 训练

```bash
# VisDrone — v0.1-N
python scripts/reproduce/reproduce_visdrone.py \
  --model v0.1-N --epochs 100 --batch 24 --device 0

# VisDrone — EsMoE-N（注意：必须加 --no-sparse-eval，否则 val mAP 崩塌到 0.01）
python scripts/reproduce/reproduce_visdrone.py \
  --model EsMoE-N --epochs 100 --batch 24 --device 0 --no-sparse-eval

# SKU-110K — v0.1-N
python scripts/reproduce/reproduce_sku110k.py \
  --model v0.1-N --epochs 100 --batch 24 --device 0

# SKU-110K — EsMoE-N
python scripts/reproduce/reproduce_sku110k.py \
  --model EsMoE-N --epochs 100 --batch 24 --device 0 --no-sparse-eval
```

## 5. 模型配置参考

| 模型 | 配置文件 | 参数量 | MoE 模块 | 关键注意 |
|------|---------|--------|---------|---------|
| v0.1-N | `v0_1/det/yolo-master-n.yaml` | 7.55M | ModularRouterExpertMoE | 含 shared expert，无稀疏问题 |
| EsMoE-N | `v0/det/yolo-master-n.yaml` | 2.69M | ES_MOE | 默认 sparse eval 导致 mAP 崩塌，须加 `--no-sparse-eval` |

## 6. 基础设施

### 断点续训

现有脚本已内置：自动检测 `runs/reproduce/<dataset>/<Dataset>_<model>/weights/last.pt`，存在则自动恢复。

### SwanLab 远程监控（待实现）

替代 W&B，适配现有 `_reproduce_common.py` 中的 callback 模式。

### 实验注册表

见同目录 `experiment_registry.md`。

## 7. 已知问题

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| EsMoE-N sparse eval 导致 mAP 崩塌 | val mAP ≈ 0.01（VisDrone） | `--no-sparse-eval` |
| SKU-110K tar 权限报错 | `Cannot change ownership` | `--no-same-owner --no-same-permissions` |
| 国内下载慢 | ultralytics.com ~14KB/s | 通过 gh-proxy.org 代理 GitHub release 源，2.8MB/s |

## 8. 产出物清单

- [ ] 4 轮训练完整日志（wandb/SwanLab 链接）
- [ ] `runs/reproduce/visdrone/summary.csv` + `sku110k/summary.csv`
- [ ] 对比表（v0.1-N vs EsMoE-N，两个数据集）
- [ ] scripts/reproduce/ 下的可复现脚本（已有，验证可运行）
- [ ] Pull Request
