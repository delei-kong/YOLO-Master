#!/bin/bash
# 步骤2：baseline 模型在 VisDrone test 集上推理评估
# 同时保存完整终端日志到 logs/ 目录

set -e
cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/val_test_${TIMESTAMP}.log"

echo "日志将保存至: ${LOG_FILE}"
echo ""

cd /root/workspace/YOLO-Master
python "${SCRIPT_DIR}/step2_val_test.py" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "日志已保存至: ${LOG_FILE}"
