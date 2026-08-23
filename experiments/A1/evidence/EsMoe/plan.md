# Issue 6：模型训练专项 — 垂类数据集基线训练

> Issue 链接：https://github.com/Tencent/YOLO-Master/issues/49
>
> 仓库路径：`/root/workspace/YOLO-Master`

## 目标

在 VisDrone（航拍密集小目标）和 SKU-110K（零售密集商品）数据集上，
分别训练 YOLO-Master-v0.1-N 和 YOLO-Master-EsMoE-N，共 4 轮。
产出完整的训练日志、可复现脚本和结果对比表，提交 PR。

## 快速命令

### 数据集下载

```bash
conda activate yolo_master

# 修改 ultralytics settings 将 datasets_dir 指向数据盘
python -c "
from ultralytics.utils import SETTINGS
SETTINGS['datasets_dir'] = '/root/gpufree-data/datasets'
SETTINGS.save()
print('datasets_dir ->', SETTINGS['datasets_dir'])
"

# VisDrone（2.3GB）
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"

# SKU-110K（13.6GB）— 如遇 tar 权限错误，手动解压：
# tar -xzf SKU110K_fixed.tar.gz --no-same-owner --no-same-permissions -C /root/gpufree-data/datasets
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```

### 训练前自检

```bash
# 模型构建检查
python scripts/reproduce/reproduce_visdrone.py --check-build
python scripts/reproduce/reproduce_sku110k.py --check-build

# dry-run 打印完整参数
python scripts/reproduce/reproduce_visdrone.py --dry-run
```

### 训练

```bash
# VisDrone — v0.1-N
python scripts/reproduce/reproduce_visdrone.py \
  --model v0.1-N --epochs 100 --batch 24 --device 0

# VisDrone — EsMoE-N（注意：必须加 --no-sparse-eval）
python scripts/reproduce/reproduce_visdrone.py \
  --model EsMoE-N --epochs 100 --batch 24 --device 0 --no-sparse-eval

# SKU-110K — v0.1-N
python scripts/reproduce/reproduce_sku110k.py \
  --model v0.1-N --epochs 100 --batch 24 --device 0

# SKU-110K — EsMoE-N
python scripts/reproduce/reproduce_sku110k.py \
  --model EsMoE-N --epochs 100 --batch 24 --device 0 --no-sparse-eval
```

## 基础设施 SOP

### 1. 断点续训

现有脚本已内置：自动检测 `runs/reproduce/<dataset>/<Dataset>_<model>/weights/last.pt`，
存在则从该 epoch 恢复继续训练。

```bash
# 直接重复执行相同命令即可自动续训
python scripts/reproduce/reproduce_visdrone.py --model v0.1-N --epochs 100 --batch 24
```

### 2. SwanLab 远程监控

替代 W&B，国内网络友好。基于现有 `_reproduce_common.py` 中的 W&B callback 模式进行适配：

```bash
pip install swanlab
swanlab login   # 首次使用，获取 API key
```

用法：
```bash
# 替换 --wandb-project 为 --swanlab
# 在 _reproduce_common.py 中新增 SwanLab callback，API 与 W&B 几乎一致
```

### 3. 训练前环境自检

```bash
# 一键检查脚本（待实现：scripts/check_env.py）
python scripts/check_env.py
```

检查项：
- CUDA 驱动 vs PyTorch CUDA 版本对齐
- 数据集路径存在且包含 images/labels
- 模型 config 可成功 build（DetectionModel 前向通过）
- 磁盘剩余空间 >= 10GB

### 4. 实验注册表

每次训练前/后各填一行，追踪全部 Issue 的实验历程。

路径：`/root/workspace/docs/experiment_registry.md`

---

## 已知问题

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| EsMoE-N mAP 崩塌 | val mAP ≈ 0.01（VisDrone）或远低于 v0.1-N | 训练时加 `--no-sparse-eval` |
| SKU-110K tar 权限错误 | `Cannot change ownership ... Operation not permitted` | 手动加 `--no-same-owner --no-same-permissions` |
| Python 3.14 standalone val 崩溃 | `ConnectionResetError` | 设置 `workers=0` |

## 模型配置参考

| 模型 | Config | 参数量 | MoE 模块 |
|------|--------|--------|---------|
| v0.1-N | `v0_1/det/yolo-master-n.yaml` | 7.55M | ModularRouterExpertMoE |
| EsMoE-N | `v0/det/yolo-master-n.yaml` | 2.69M | ES_MOE |

## 现有脚本路径

```
scripts/reproduce/
├── _reproduce_common.py      # 共享训练逻辑（425 行）
├── reproduce_visdrone.py     # VisDrone 复现入口
├── reproduce_sku110k.py      # SKU-110K 复现入口
└── README.md                # 详细文档
```
