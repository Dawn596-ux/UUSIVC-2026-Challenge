#!/usr/bin/env bash
set -euo pipefail

# UUSIVC 2026 推理入口
# 环境变量：INPUT_DIR(默认 /app/input, 只读) / OUTPUT_DIR(默认 /app/output) / DEVICE(默认 cuda)

INPUT_DIR="${INPUT_DIR:-/app/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/app/output}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUTPUT_DIR"

# 显式 --checkpoint 指向 rank2（勿用 --which best 默认，会解析到坏 rank1）
python -B /app/predict.py \
  --data-root "$INPUT_DIR" \
  --phase test \
  --checkpoint /app/checkpoint.pth \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --no-zip
