#!/usr/bin/env bash
set -euo pipefail

# UUSIVC 2026 final inference entry — three-slot recipe (gate 0.8449631, validated 2026-09-05)
# Routing per-task, tertiary > secondary > primary:
#   primary   MEM : configs/stage2_cls_mem.yaml -> video_seg, ceus_seg, ceus_cls (memory video head)
#   secondary MAIN: configs/stage2_cls.yaml     -> image_seg
#   tertiary  OLD : configs/stage2_cls.yaml     -> image_cls
# All three task lists EXPLICIT (--secondary-tasks legacy default "video_seg,image_cls" would mis-route).

INPUT_DIR="${INPUT_DIR:-/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
DEVICE="${DEVICE:-cuda}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "[run.sh] ERROR: input dir '$INPUT_DIR' not found." >&2
  echo "[run.sh] Mount the official test package: -v /path/to/input:/input:ro" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "[run.sh] INPUT_DIR=$INPUT_DIR OUTPUT_DIR=$OUTPUT_DIR DEVICE=$DEVICE"

python -B /app/predict.py \
  --data-root "$INPUT_DIR" \
  --phase test \
  --config /app/configs/stage2_cls_mem.yaml \
  --checkpoint /app/checkpoints/primary_mem_stage2_cls_rank1.pth \
  --secondary-config /app/configs/stage2_cls.yaml \
  --secondary-checkpoint /app/checkpoints/secondary_main_stage2_cls_aug2_rank1.pth \
  --secondary-tasks image_seg \
  --tertiary-config /app/configs/stage2_cls.yaml \
  --tertiary-checkpoint /app/checkpoints/tertiary_old_stage2_cls_rank2.pth \
  --tertiary-tasks image_cls \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --no-zip
