#!/bin/bash
# Phase 0: 按论文 ES-MoE 设计训练 VisDrone baseline
# soft Top-2 训练 + hard Top-2 验证，epochs=200
#
# 产出:
#   logs/      — 完整训练日志
#   weights/   — best.pt, last.pt
#   args.yaml, results.csv, results.png 等

set -e
cd /root/workspace/YOLO-Master

SCRIPT_DIR="/root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train"
LOG_DIR="${SCRIPT_DIR}/logs"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
RUN_DIR="/root/workspace/YOLO-Master-docs/issue2/VisDrone/pahse0-baseline-train/ultralytics-run"

mkdir -p "${LOG_DIR}" "${WEIGHTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "============================================================"
echo " Phase 0: VisDrone EsMoE-N Sparse Baseline 训练"
echo " 配置: soft Top-2 训练 + hard Top-2 验证, epochs=500"
echo " 日志: ${LOG_FILE}"
echo " 权重: ${WEIGHTS_DIR}"
echo "============================================================"
echo ""

# 训练（tee 保存完整日志）
python scripts/reproduce/reproduce_visdrone_sparse.py \
  --epochs 500 \
  --batch 32 \
  --device 0 \
  --workers 0 \
  --moe-balance-loss 0.6 \
  --project "${SCRIPT_DIR}" \
  2>&1 | tee "${LOG_FILE}"

# 训练完成后收集产物
echo ""
echo "============================================================"
echo " 收集训练产物..."
echo "============================================================"

# 查找 ultralytics 实际 run 目录
UL_RUN_DIR=$(ls -dt "${SCRIPT_DIR}"/VisDrone_EsMoE-N* 2>/dev/null | head -1)

if [ -n "${UL_RUN_DIR}" ]; then
    echo "Run dir: ${UL_RUN_DIR}"

    # 权重
    if [ -f "${UL_RUN_DIR}/weights/best.pt" ]; then
        cp -v "${UL_RUN_DIR}/weights/best.pt" "${WEIGHTS_DIR}/"
    fi
    if [ -f "${UL_RUN_DIR}/weights/last.pt" ]; then
        cp -v "${UL_RUN_DIR}/weights/last.pt" "${WEIGHTS_DIR}/"
    fi

    # 训练元数据
    for f in args.yaml results.csv results.png labels.jpg; do
        if [ -f "${UL_RUN_DIR}/${f}" ]; then
            cp -v "${UL_RUN_DIR}/${f}" "${SCRIPT_DIR}/"
        fi
    done
else
    echo "WARNING: 未找到 ultralytics run 目录"
fi

echo ""
echo "============================================================"
echo " 训练完成!"
echo " 日志: ${LOG_FILE}"
echo " 权重: ${WEIGHTS_DIR}"
echo "============================================================"
