# A1｜MoE × End-to-End：现有 one-to-one 能否真正免 NMS

> 本文档 = **个人 fork 与历史成果一次性登记** + **2026-08-24 准入检查记录**。
> 登记作用：划清既有成果边界（不为既有成果追加本轮分数），并固定个人基线。
> 截止后不得通过更换基线 SHA、改写历史或补报旧报告来扩大"历史成果"范围。

| 项目 | 内容 |
|------|------|
| 登记日期 | 2026-08-24 |
| 登记截止 | 2026-08-25 23:59:59（UTC+8） |
| 基线 SHA | `2c8253fda85c6cc24354f24f569ede86cf18b1ff` |

---

## 1. 课题与成员

| 字段 | 内容 |
|------|------|
| 课题编号 | A1 |
| 题目 | MoE × End-to-End：现有 one-to-one 能否真正免 NMS |
| 难度/性质 | 研究型｜高风险 |
| 算力 | 中高（单卡 24G） |
| 团队 | 2–3 人 |
| Owner | delei-kong |
| 成员及职责 | 现阶段与涂嘉隽同学沟通方案与分工事项 |

## 2. 仓库

| 字段 | 内容 |
|------|------|
| Repository URL | https://github.com/delei-kong/YOLO-Master |
| 默认分支（GitHub 实际） | `main` |
| 登记/工作分支（Owner 指定） | `dev/delei-kong/A1`（当前 HEAD 所在；⚠️ 尚未推送远端，GitHub 默认分支仍为 `main`） |
| 上游公共仓库 | https://github.com/Tencent/YOLO-Master |

## 3. 个人基线（锁定前 HEAD）

| 字段 | 内容 |
|------|------|
| Commit SHA | `2c8253fda85c6cc24354f24f569ede86cf18b1ff` |
| 提交信息 | `docs: 历史成果冻结与准入检查日志证据` |
| 提交时间 | 2026-08-24 00:16:45 +0800 |
| 固定 commit URL | https://github.com/delei-kong/YOLO-Master/commit/2c8253fda85c6cc24354f24f569ede86cf18b1ff |


## 4. 既有成果范围

按文件、模块、后端、数据集、结论逐项列出。以下均为锁定前既有成果，不追加本轮分数。

### 4.1 数据集与后端

| 项 | 内容 |
|----|------|
| 数据集 | VisDrone（Issue #52）；brain-tumor、construction-ppe（Issue #6 垂类专项） |
| 后端 | NVIDIA RTX 4090 24G，CUDA，torch 2.5.1，Python 3.11.15，ultralytics 8.4.101（本地 repo 生效） |
| 数据盘 | `/root/workspace/datasets → /root/gpufree-data/datasets`（软链，49GB 数据盘） |

### 4.2 模块与实验（文件级清单）

- **Issue #52（moe-pruning-signals）规划**：`experiments/A1/evidence/baseline/plan.md`
  - 四阶段设计：固定 balance_loss 基线 → 专家剪枝 → LoRA 恢复；动态调度对照链路（链路 A vs 链路 B）
- **VisDrone baseline（300 epochs 训练 + 验证 + 路由记录）**：
  - `experiments/A1/evidence/baseline/VisDrone/baseline/`（args.yaml / best.pt / results.csv / routing/ / logs）
  - 配置：EsMoE-N（yolo-master-n.yaml），4 MoE 层 × 4 experts，top_k=2（50% 稀疏），balance_loss=1.0，imgsz 640，batch 32，RTX 4090
  - 终值（epoch 268）：P 0.4518 / R 0.3677 / mAP50 0.3402 / mAP50-95 0.1945
- **VisDrone MoE-train 再训练**：
  - `experiments/A1/evidence/baseline/VisDrone/MoE-train/`（train.sh / run_experiments.sh / PR.md / 实验记录/）
  - 终值（epoch 182）：P 0.4538 / R 0.3671 / mAP50 0.3412 / mAP50-95 0.1949
- **VisDrone 初版训练**：`pahse0-baseline-train/`（train.sh、check_routing.py、logs）
- **VisDrone 推理 + 路由分析**：`phase1-baseline/`（step2_val_test.py / step3_route_analysis.py 及对应 logs）
- **VisDrone 专家剪枝**：`phase2-pruned/`（usage_stats.json、剪枝 scripts）
- **Issue #6 垂类基线对照（v0.1-N 无 MoE vs EsMoE-N 有 MoE）**：
  - `experiments/A1/evidence/EsMoe/` 下 brain-tumor / construction-ppe / visdrone 三组，各自 `v0.1-N`、`EsMoE-N` 子目录含 PR.md、experiment_record.md、summary.csv
  - 配套：README.md、experiment_registry.md、dataset.md、SOP.md、checklist.md、笔记.md、提交PR.md、plan.md、draw_boundary.py、analyze_training.py、analysis/

### 4.3 既有结论

1. MoE 训练管线（EsMoE-N）在 VisDrone 与三个垂类数据集上可收敛、可验证、可出 PR 记录。
2. Issue #52：固定 balance_loss=1.0 训练 200+ epoch 后，4 个 expert 使用率高度均衡，导致剪枝 hit-rate 信号缺乏区分度；动态调度让 expert 分化后剪枝才可能有效。


## 5. 8.24 准入检查

### 5.1 ✅ 能运行 detect 基线

- **历史证据**：§4.2 中 VisDrone 300-epoch 训练与 val 记录（best.pt + results.csv + W&B）。
- **锁定后复验**（2026-08-24，commit `2c8253f` 工作区）：
  - 命令（dispatcher）：`yolo.predict model=yolo11n.pt source=ultralytics/assets/bus.jpg device=0`
  - 结果：`640x480 4 persons, 1 bus`；Speed：preprocess 5.1ms / inference 87.5ms / postprocess 19.8ms（RTX 4090，batch=1）
  - 产物：`runs/agent/yolo-predict-8cca069e/`（含 skill_manifest.json）；日志副本 `evidence/admission/2026-08-24_detect_smoke.md`

### 5.2 ✅ 能定位 end-to-end / assigner / postprocess

| 环节 | 位置 | 说明 |
|------|------|------|
| 标签分配器 | `ultralytics/utils/tal.py:14` | `TaskAlignedAssigner`（tal_topk 决定 one-to-many 或 one-to-one 匹配）；`:356` RotatedTaskAlignedAssigner |
| one-to-many / one-to-one 损失 | `ultralytics/utils/loss.py:1186-1196` | `v8DetectionLoss`：one2many 用 tal_topk=10，one2one 用 tal_topk=1，两路损失相加 |
| o2m/o2o 权重调度 | `ultralytics/utils/loss.py:1210-1231` | 训练后期 o2m 权重衰减至 0，one-to-one 监督占主导 |
| MultiTaskLoss | `ultralytics/utils/loss.py:1418` | 多任务损失入口 |
| 检测头 one-to-one 分支 | `ultralytics/nn/modules/head.py:122-124, 137-144, 157-171` | `end2end` 时深拷贝 `one2one_cv2/cv3`；训练输出 `{"one2many","one2one"}`；推理走 one2one 分支 |
| postprocess（免 NMS 分支） | `ultralytics/models/yolo/detect/predict.py:33-63` → `ultralytics/utils/nms.py:66` | `shape[-1]==6 or end2end` 时仅按 conf 过滤 top-k，**不做 NMS** |
| 训练/验证 o2o 接入 | `ultralytics/engine/trainer.py:502, 1550` | 恢复/初始化 o2o、o2m 损失与参数 |
| MultiTaskHead | `ultralytics/nn/modules/multitask/head.py:30` | `MultiTaskHead(Detect)` |
| 导出 end2end 支持 | `ultralytics/engine/exporter.py:663-672` | 部分格式不支持 end2end 自动禁用；静态量化会坍缩 end2end 类别索引输出 |
| MoE 路由 | `ultralytics/nn/modules/moe/routers.py`（UltraEfficientRouter :58、BaseRouter :168、EfficientSpatialRouter :268）、`ultralytics/nn/modules/dynamic_moe.py` | 稀疏路由核心（4-D NCHW 约束） |
| 模型 YAML end2end 开关 | `ultralytics/cfg/models/26/yolo26.yaml:9`、`yolo26-master-n.yaml:7` | **默认 `end2end: True`**——仓库标配即免 NMS 检测头 |
| MoE YAML | `ultralytics/cfg/models/26/yolo26-master-n.yaml:20-24` | A2C2fMoE 于 P3/P4/P5，experts 4/8/16，top_k=2 |

### 5.3 ✅ 一页 2×2 实验设计

**核心问题**：MoE 稀疏路由与一对一（one-to-one）监督是否冲突？仓库现有 one-to-one 路径能否形成训练→推理→导出→评测的完整免 NMS 闭环？

**2×2 因素**：MoE 开/关（骨干）× NMS 开/关（推理路径，由 YAML `end2end` 开关控制）。

| | MoE off（yolo26.yaml） | MoE on（yolo26-master-n.yaml，A2C2fMoE） |
|---|---|---|
| **NMS on**（end2end=False，one-to-many + NMS） | **A 基线**：现有 one-to-many 检测闭环 | B：MoE 主干 + 传统 NMS 闭环 |
| **NMS off**（end2end=True，one-to-one 免 NMS） | C：one-to-one 免 NMS 闭环（无 MoE） | **D 核心格**：MoE + one-to-one 免 NMS 全闭环 |

说明：`end2end: False` 时头部不构建 one2one 分支（`head.py:122,162`），推理走 one-to-many + NMS；`end2end: True` 时同时训 one2many + one2one，推理走 one2one 免 NMS 分支（`nms.py:66`）。

**每组四环节闭环检查**：训练（o2m/o2o 损失正确性）→ 推理（是否真正跳过 NMS）→ 导出（ONNX/静态量化对 end2end 输出的支持）→ 评测（mAP 与延迟口径一致）。

**指标与成功判据**（P0/P1）：

- mAP50-95：**D vs A 掉点 ≤ 2**（COCO-mini 起步 → COCO val）
- 延迟：GPU/CPU batch=1 延迟及标准差（≥3 次重复），NMS off 相对 NMS on 需有可测改善
- 路由诊断（P2）：expert 使用率/Gini、梯度稀疏性、匹配冲突、路由坍塌——允许有证据的负结果

**数据与环境**：COCO mini 起步，资源允许跑 COCO val；单卡 RTX 4090 24G。

**风险与降级**：one-to-one 不收敛 → 降级为 one-to-many 主分支 + one-to-one 辅助分支；算力不足缩数据不缩变量。

---

*登记人：delei-kong（Owner）｜2026-08-24*
