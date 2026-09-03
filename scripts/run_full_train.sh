#!/usr/bin/env bash
# Full-train (100% train data) driver for deployment weights — single split
# (the default 10% local-val manifest, same split convention as the aug2
# 0.6505-era full-train), NOT the K3 fold manifests.
#
# Mirrors scripts/run_cv.sh conventions: OMP cap, resume-aware stages with
# TRAIN_DONE markers, tee'd per-stage logs, predict_val scoring at the end.
#
# Usage (server, from the repo root):
#   TAG=mem_full bash scripts/run_full_train.sh              # memhead recipe (default)
#   TAG=aug2_full STAGE1_CFG=configs/stage1_seg.yaml \
#     STAGE2_CFG=configs/stage2_cls.yaml bash scripts/run_full_train.sh
#
# Outputs:
#   outputs/stage1_seg_$TAG/      stage1 checkpoints (best_checkpoints/rank* + latest)
#   outputs/stage2_cls_$TAG/      stage2 checkpoints
#   outputs/predict_val_$TAG/     offline-scored single-model predictions
# Logs: logs/full_${TAG}_stage{1,2}.log, logs/full_${TAG}_predict.log
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data}
TAG=${TAG:-mem_full}
STAGE1_CFG=${STAGE1_CFG:-configs/stage1_seg_mem.yaml}
STAGE2_CFG=${STAGE2_CFG:-configs/stage2_cls_mem.yaml}
S1DIR=outputs/stage1_seg_${TAG}
S2DIR=outputs/stage2_cls_${TAG}
mkdir -p "$S1DIR" "$S2DIR" logs

echo "[FULL] tag=$TAG stage1=$STAGE1_CFG stage2=$STAGE2_CFG data=$DATA_ROOT"

# stage 1: segmentation on the full train split (no --cv-fold => default manifest)
if [ ! -f "$S1DIR/TRAIN_DONE" ]; then
  RESUME_ARG=""
  if [ -f "$S1DIR/latest_stage1_seg.pth" ]; then
    RESUME_ARG="--resume-checkpoint $S1DIR/latest_stage1_seg.pth"
    echo "[FULL] [resume] stage1 from latest_stage1_seg.pth"
  fi
  $PY -B train.py --stage stage1_seg --config "$STAGE1_CFG" \
    --data-root "$DATA_ROOT" \
    --save-dir "$S1DIR" \
    $RESUME_ARG \
    2>&1 | tee "logs/full_${TAG}_stage1.log"
  touch "$S1DIR/TRAIN_DONE"
fi

# stage 2: classification, chained from this run's stage1 rank1
if [ ! -f "$S2DIR/TRAIN_DONE" ]; then
  RESUME_ARG=""
  if [ -f "$S2DIR/latest_stage2_cls.pth" ]; then
    RESUME_ARG="--resume-checkpoint $S2DIR/latest_stage2_cls.pth"
    echo "[FULL] [resume] stage2 from latest_stage2_cls.pth"
  fi
  $PY -B train.py --stage stage2_cls --config "$STAGE2_CFG" \
    --data-root "$DATA_ROOT" \
    --init-checkpoint "$S1DIR/best_checkpoints/best_stage1_seg_rank1.pth" \
    --save-dir "$S2DIR" \
    $RESUME_ARG \
    2>&1 | tee "logs/full_${TAG}_stage2.log"
  touch "$S2DIR/TRAIN_DONE"
fi

# single-model offline scoring on the full-split val (slot-decision input)
$PY -B predict_val.py --config "$STAGE2_CFG" --data-root "$DATA_ROOT" --device cuda \
  --val-manifest "outputs/uusivc2026_fixed/manifests/val_entries.json" \
  --checkpoint "$S2DIR/best_checkpoints/best_stage2_cls_rank1.pth" \
  --output-dir "outputs/predict_val_${TAG}" --no-zip \
  > "logs/full_${TAG}_predict.log" 2>&1
echo "[FULL] scored: $(grep overall_score "outputs/predict_val_${TAG}/predict_val_summary.json" | head -1)"
echo "[FULL] done: $S2DIR + outputs/predict_val_${TAG}"
