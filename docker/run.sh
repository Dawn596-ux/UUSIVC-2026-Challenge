#!/usr/bin/env bash
set -euo pipefail

# UUSIVC 2026 推理入口
# 环境变量：INPUT_DIR(默认 /app/input, 只读) / OUTPUT_DIR(默认 /app/output) / DEVICE(默认 cuda)

INPUT_DIR="${INPUT_DIR:-/app/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/output}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUTPUT_DIR"

# 双 checkpoint：主 aug2 rank1（task1/2/5）+ 辅 stage2_cls rank2（task3/4 混合修复）
python -B /app/predict.py \
  --data-root "$INPUT_DIR" \
  --phase test \
  --checkpoint /app/checkpoint.pth \
  --secondary-checkpoint /app/secondary_checkpoint.pth \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --no-zip
