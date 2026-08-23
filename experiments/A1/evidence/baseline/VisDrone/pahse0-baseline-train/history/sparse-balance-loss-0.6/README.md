# Sparse Baseline — balance_loss=0.6, 4 experts, top-2

> 按论文 ES-MoE 设计训练：soft Top-2 训练 + hard Top-2 验证，真正稀疏路由

## 模型架构

| 项 | 值 |
|----|-----|
| 模型 | YOLO-Master EsMoE-N (yolo-master-n.yaml) |
| MoE 层数 | 4（backbone L3/L6/L9/L12） |
| 每层专家数 | **4** |
| top_k | **2**（每 token 激活 2 个 expert） |
| top_k / num_experts | 50% 稀疏率 |
| 参数量 | ~3.0M（估算，比 3 expert 的 2.65M 多 ~13%） |
| GFLOPs (dense) | ~9.0（估算） |
| GFLOPs (sparse) | ~7.5（估算，仅 2 expert 计算） |

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 500 | 论文训练 600 epoch |
| imgsz | 640 | |
| batch | 32 | RTX 4090 24GB |
| optimizer | auto → MuSGD (lr=0.01) | |
| cos_lr | True | 余弦退火 |
| close_mosaic | 10 | 最后 10 epoch 关闭 |
| seed | 42 | |
| AMP | True | |
| pretrained | False | 从零训练 |
| workers | 0 | 避免 dataloader worker 崩溃 |
| **moe_balance_loss** | **0.6** | 比原始 dense 训练的 1.0 更低 |
| moe_top_k | 2 | |
| moe_num_experts | 4 | ES_MOE 默认已改为 4 |
| moe_dynamic_schedule | none | 固定 balance_loss |
| DFL | 1.5 | 保留（对照论文 Config 3） |

## 关键代码修复

### ES_MOE 训练时走 sparse forward

**文件**: `ultralytics/nn/modules/moe/modules.py:522`

```python
# 修复前：训练时强制 dense
use_dense = (
    self.training                     # ← 删掉这行
    or torch.onnx.is_in_onnx_export()
    or torch.jit.is_tracing()
    or not self._eager_sparse_enabled()
)

# 修复后：训练和推理一致走 sparse
use_dense = (
    torch.onnx.is_in_onnx_export()
    or torch.jit.is_tracing()
    or not self._eager_sparse_enabled()
)
```

### ES_MOE 默认专家数 3→4

**文件**: `ultralytics/nn/modules/moe/modules.py:399`

```python
def __init__(self, in_channels, out_channels=None, num_experts=4, ...):
    # num_experts: 3 → 4
```

### 路由诊断回调

**文件**: `scripts/reproduce/reproduce_visdrone_sparse.py`

每 5 个 epoch 在验证后收集路由指标：
- 每层 expert 利用率分布
- Gini 系数
- 主导占比

指标记录到 W&B `routing/*` 命名空间。

## 训练命令

```bash
cd /root/workspace/YOLO-Master

# 新训练（不续训之前的 checkpoint）
rm -rf /root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train/sparse-balance-loss-0.6/VisDrone_EsMoE-N

python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 500 \
  --batch 32 \
  --device 0 \
  --workers 0 \
  --moe-balance-loss 0.6 \
  --routing-diag-interval 5 \
  --project /root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train/sparse-balance-loss-0.6 \
  2>&1 | tee /root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train/sparse-balance-loss-0.6/logs/train_$(date +%Y%m%d_%H%M%S).log
```

## 历史实验对照

| 实验 | 专家数 | balance_loss | 训练模式 | epoch 100 mAP50 |
|------|:-----:|:----------:|------|:--------------:|
| dense baseline | 3 | 1.0 | dense train + dense val | 0.326 |
| sparse broken v1 | 3 | 1.0 | dense train + sparse val | 0.03 |
| sparse broken v2 | 3 | 0.6 | dense train + sparse val | 0.04 |
| sparse fixed (当前) | 3 | 0.6 | sparse train + sparse val | 0.31 |
| **本次实验** | **4** | **0.6** | **sparse train + sparse val** | **0.342 (epoch 223)** |
| dense baseline | 3 | 1.0 | dense | 0.339 (epoch 200) |

## 4 expert 训练结果

| 指标 | 值 |
|------|-----|
| 训练 epoch | 223/500（提前停止，已收敛）|
| 最终 mAP50 | **0.342** |
| Dense baseline mAP50 | 0.339 |
| 优势 | **+0.003 (超越 dense)** |
| 每 token 激活专家 | 2/4 (50% 稀疏) |
| 路由分化 Gini | 0.47~0.52 |
| 每层死专家数 | 2（连续 220+ epoch 稳定） |
| W&B | https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/i27tkil4 |

### 最终路由分化（epoch 215）

| Layer | 活跃专家 | 死专家 | Gini |
|:-----:|---------|--------|:---:|
| L3 | E3(52%) E0(48%) | E1,E2 | 0.51 |
| L6 | E0(51%) E2(49%) | E1,E3 | 0.50 |
| L9 | E1(50%) E2(50%) | E0,E3 | 0.50 |
| L12 | E1(62%) E3(24%) E0(15%) | E2 | 0.49 |

## 前置实验：3 experts 版本

先跑了 3 expert + balance_loss=0.6 + 代码修复的版本，验证 sparse 训练可行性：

| 指标 | 值 |
|------|-----|
| 训练 epoch | 111/500 |
| 最终 mAP50 | **0.314** |
| 最终 mAP50-95 | 0.180（估算） |
| Dense 同期 mAP50 | 0.331 |
| 性能差距 | 5.1% |
| 路由均匀度 | Gini=0.000, 所有 expert avg_weight≈0.333 |
| W&B | https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/c4mtb7bz |

产物见 `3experts/` 目录。
