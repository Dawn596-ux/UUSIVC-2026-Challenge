#!/usr/bin/env bash
# Disjoint patient-level K-fold CV (default K=3) for training-side changes.
#
# Per fold: stage1 seg (aug2 recipe) -> stage2 cls (chained from the same fold's
# stage1 rank1) -> predict_val on that fold's val split -> metrics.json with
# per-sample records. A single checkpoint per arm is used (no mixed dual
# checkpoint): the CV table compares training recipes on equal footing; the
# frozen mixed-checkpoint deployment trick is deliberately not reproduced here
# (it would double the cost per fold without changing paired deltas).
#
# Resumable per fold: a fold is skipped when its predict/metrics.json already
# exists (AutoDL bad-restart safety). Stages carry their own TRAIN_DONE markers
# so a crash mid-fold resumes at the right stage.
#
# Usage (server):
#   bash scripts/run_cv.sh                                  # baseline run
#   TAG=aug3 bash scripts/run_cv.sh                         # candidate recipe
#   TAG=aug3 BASELINE=outputs/cv/K3_baseline bash scripts/run_cv.sh
#   K=5 TAG=x bash scripts/run_cv.sh                        # more folds
#
# Aggregate afterwards (also run automatically at the end):
#   python -B aggregate_cv.py outputs/cv/K3_<TAG> --baseline outputs/cv/K3_baseline
set -euo pipefail
cd /root/autodl-tmp/project/code
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
DATA_ROOT=/root/autodl-tmp/data
K=${K:-3}
SEED=${SEED:-2024}
TAG=${TAG:-baseline}
CVROOT=outputs/cv/K${K}_${TAG}
mkdir -p "$CVROOT" logs

echo "[CV] root=$CVROOT K=$K seed=$SEED tag=$TAG"

for FOLD in $(seq 0 $((K - 1))); do
  FOLDDIR=$CVROOT/fold$FOLD
  if [ -f "$FOLDDIR/predict/metrics.json" ]; then
    echo "[CV] [skip] fold$FOLD already scored"
    continue
  fi
  mkdir -p "$FOLDDIR"
  echo "[CV] === fold$FOLD ==="

  # stage 1: segmentation (aug2 recipe) on this fold's train split
  if [ ! -f "$FOLDDIR/stage1_seg_aug2/TRAIN_DONE" ]; then
    $PY -B train.py --stage stage1_seg --config configs/stage1_seg.yaml \
      --data-root "$DATA_ROOT" \
      --split-seed "$SEED" --cv-num-folds "$K" --cv-fold "$FOLD" \
      --save-dir "$FOLDDIR/stage1_seg_aug2" \
      2>&1 | tee "logs/cv_${TAG}_fold${FOLD}_stage1.log"
    touch "$FOLDDIR/stage1_seg_aug2/TRAIN_DONE"
  fi

  # stage 2: classification, chained from this fold's stage1 best
  if [ ! -f "$FOLDDIR/stage2_cls_aug2/TRAIN_DONE" ]; then
    $PY -B train.py --stage stage2_cls --config configs/stage2_cls.yaml \
      --data-root "$DATA_ROOT" \
      --init-checkpoint "$FOLDDIR/stage1_seg_aug2/best_checkpoints/best_stage1_seg_rank1.pth" \
      --split-seed "$SEED" --cv-num-folds "$K" --cv-fold "$FOLD" \
      --save-dir "$FOLDDIR/stage2_cls_aug2" \
      2>&1 | tee "logs/cv_${TAG}_fold${FOLD}_stage2.log"
    touch "$FOLDDIR/stage2_cls_aug2/TRAIN_DONE"
  fi

  # inference + file-based scoring on this fold's val split
  $PY -B predict_val.py --data-root "$DATA_ROOT" --device cuda \
    --val-manifest "outputs/uusivc2026_fixed/manifests_cv${K}/fold${FOLD}/val_entries.json" \
    --checkpoint "$FOLDDIR/stage2_cls_aug2/best_checkpoints/best_stage2_cls_rank1.pth" \
    --output-dir "$FOLDDIR/predict" --no-zip \
    > "logs/cv_${TAG}_fold${FOLD}_predict.log" 2>&1
  echo "[CV] fold$FOLD scored: $(grep overall_score "$FOLDDIR/predict/predict_val_summary.json" | head -1)"
done

echo "[CV] aggregating..."
$PY -B aggregate_cv.py "$CVROOT" \
  ${BASELINE:+--baseline "$BASELINE"} \
  --output-json "$CVROOT/summary.json"
echo "[CV] done: $CVROOT/summary.json"
