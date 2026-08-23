#!/bin/bash
# 自动运行 exp3-6 并收集产出
# exp3: 4 experts BL=1.5 (当前训练中)
# exp4: 3 experts BL=0.3
# exp5: 3 experts BL=1.0
# exp6: 3 experts BL=1.5

set -e
cd /root/workspace/YOLO-Master

TRAIN_DIR="/root/workspace/YOLO-Master-docs/issue2/VisDrone/MoE-train"
RECORDS="${TRAIN_DIR}/实验记录"
YOLO_MODEL_YAML="ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml"
ROUTING_IMAGE="/root/gpufree-data/datasets/VisDrone/images/test/0000006_00159_d_0000001.jpg"

# ── 实验配置 ──
# 格式: "exp_id|num_experts|balance_loss"
EXPERIMENTS=(
  "exp3|4|1.5"
  "exp4|3|0.3"
  "exp5|3|1.0"
  "exp6|3|1.5"
)

run_single_exp() {
  local EXP_ID="$1"
  local NUM_EXPERTS="$2"
  local BL="$3"

  echo ""
  echo "============================================================"
  echo " 开始 ${EXP_ID}: experts=${NUM_EXPERTS} BL=${BL}"
  echo "============================================================"

  # 1. 修改 ES_MOE 默认专家数
  sed -i "s/num_experts=[0-9]*,/num_experts=${NUM_EXPERTS},/" \
    ultralytics/nn/modules/moe/modules.py

  # 2. 清理旧产物
  rm -rf "${TRAIN_DIR}/VisDrone_EsMoE-N" "${TRAIN_DIR}/weights"/*

  # 3. 训练
  python scripts/reproduce/reproduce_visdrone_sparse.py \
    --epochs 30 \
    --batch 32 \
    --device 0 \
    --workers 0 \
    --moe-balance-loss "${BL}" \
    --routing-diag-interval 5 \
    --project "${TRAIN_DIR}" \
    2>&1 | tee "${TRAIN_DIR}/logs/train_${EXP_ID}_$(date +%Y%m%d_%H%M%S).log"

  # 4. 收集产物
  local EXP_DIR="${RECORDS}/${EXP_ID}"
  mkdir -p "${EXP_DIR}/weights" "${EXP_DIR}/logs" "${EXP_DIR}/routing"

  local RUN_DIR=$(ls -dt "${TRAIN_DIR}"/VisDrone_EsMoE-N* 2>/dev/null | head -1)
  if [ -n "${RUN_DIR}" ]; then
    cp "${RUN_DIR}/weights/best.pt" "${EXP_DIR}/weights/" 2>/dev/null
    cp "${RUN_DIR}/weights/last.pt" "${EXP_DIR}/weights/" 2>/dev/null
    cp "${RUN_DIR}/results.csv" "${EXP_DIR}/" 2>/dev/null
    cp "${RUN_DIR}/args.yaml" "${EXP_DIR}/" 2>/dev/null
  fi

  # 5. 跑路由诊断（dataset 模式）
  python tools/routing_interpreter.py \
    "${EXP_DIR}/weights/last.pt" \
    --dataset VisDrone.yaml \
    --output "${EXP_DIR}/routing" \
    --device cuda:0 --imgsz 640 --batch-size 16 \
    2>&1 | tail -3

  # 6. 跑路由诊断（单图热力图）
  python tools/routing_interpreter.py \
    "${EXP_DIR}/weights/last.pt" \
    "${ROUTING_IMAGE}" \
    --output "${EXP_DIR}/routing" \
    --device cuda:0 --imgsz 640 \
    2>&1 | tail -3

  # 7. 写 README
  local LAST_EPOCH=$(tail -1 "${EXP_DIR}/results.csv" | cut -d',' -f1)
  local MAP50=$(tail -1 "${EXP_DIR}/results.csv" | cut -d',' -f9)
  cat > "${EXP_DIR}/README.md" << EOF
# ${EXP_ID}: balance_loss=${BL}, soft Top-2, ${NUM_EXPERTS} experts

## 实验目的

验证 ${NUM_EXPERTS} experts + balance_loss=${BL} 下的路由分化行为。

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | EsMoE-N (yolo-master-n.yaml) |
| 专家数 | ${NUM_EXPERTS} per layer |
| top_k | 2 |
| balance_loss | ${BL} |
| epochs | 30 |
| batch | 32 |
| imgsz | 640 |

## 训练结果 (epoch ${LAST_EPOCH})

| 指标 | 值 |
|------|-----|
| mAP50 | ${MAP50} |

## 路由分析

见 \`routing/dataset_routing_report.json\` 和热力图。
EOF

  echo "[${EXP_ID}] 完成!"
}

# ── 主循环 ──
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r EXP_ID NUM_EXPERTS BL <<< "$exp"

  # exp3 检查是否已经在跑（当前训练）
  if [ "$EXP_ID" = "exp3" ]; then
    if pgrep -f "reproduce_visdrone_sparse" > /dev/null 2>&1; then
      echo "[exp3] 训练进行中，等待完成..."
      while pgrep -f "reproduce_visdrone_sparse" > /dev/null 2>&1; do
        sleep 60
      done
      echo "[exp3] 训练结束，收集产物..."
      # 收集
      EXP_DIR="${RECORDS}/exp3"
      mkdir -p "${EXP_DIR}/weights" "${EXP_DIR}/logs" "${EXP_DIR}/routing"
      RUN_DIR=$(ls -dt "${TRAIN_DIR}"/VisDrone_EsMoE-N* 2>/dev/null | head -1)
      if [ -n "${RUN_DIR}" ]; then
        cp "${RUN_DIR}/weights/best.pt" "${EXP_DIR}/weights/" 2>/dev/null
        cp "${RUN_DIR}/weights/last.pt" "${EXP_DIR}/weights/" 2>/dev/null
        cp "${RUN_DIR}/results.csv" "${EXP_DIR}/" 2>/dev/null
        cp "${RUN_DIR}/args.yaml" "${EXP_DIR}/" 2>/dev/null
      fi
      cp "${TRAIN_DIR}/logs/"train_*.log "${EXP_DIR}/logs/" 2>/dev/null
      # 路由诊断
      python tools/routing_interpreter.py \
        "${EXP_DIR}/weights/last.pt" \
        --dataset VisDrone.yaml \
        --output "${EXP_DIR}/routing" \
        --device cuda:0 --imgsz 640 --batch-size 16 \
        2>&1 | tail -3
      python tools/routing_interpreter.py \
        "${EXP_DIR}/weights/last.pt" \
        "${ROUTING_IMAGE}" \
        --output "${EXP_DIR}/routing" \
        --device cuda:0 --imgsz 640 \
        2>&1 | tail -3
      LE=$(tail -1 "${EXP_DIR}/results.csv" | cut -d',' -f1)
      MP=$(tail -1 "${EXP_DIR}/results.csv" | cut -d',' -f9)
      cat > "${EXP_DIR}/README.md" << EOFME
# exp3: balance_loss=1.5, soft Top-2, 4 experts

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | EsMoE-N |
| 专家数 | 4 per layer |
| top_k | 2 |
| balance_loss | 1.5 |
| epochs | 30 |
| batch | 32 |
| imgsz | 640 |

## 训练结果 (epoch ${LE})
| 指标 | 值 |
|------|-----|
| mAP50 | ${MP} |

## 路由分析
见 routing/ 目录。
EOFME
    fi
  else
    run_single_exp "$EXP_ID" "$NUM_EXPERTS" "$BL"
  fi
done

echo ""
echo "============================================================"
echo " 全部实验完成!"
echo "============================================================"
echo "exp3: 4 experts, BL=1.5 (已收集)"
echo "exp4: 3 experts, BL=0.3"
echo "exp5: 3 experts, BL=1.0"
echo "exp6: 3 experts, BL=1.5"
