#!/usr/bin/env bash
# Regression gate: reproduce the 0.6505-equivalent baseline metrics on the local
# labeled holdout (patient-level 10% split, seed 2024) with the mixed dual
# checkpoint. Run this after ANY code change — metrics.json must match the
# known baseline (overall=0.8450, image_seg task=0.7430) exactly before any
# A/B comparison of a candidate change is meaningful.
#
# To evaluate a candidate change, add its flags/edits as a B-arm below
# (copy the block, change --output-dir), then diff the two metrics.json.
set -euo pipefail
cd /root/autodl-tmp/project/code
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
MAIN=outputs/stage2_cls_aug2/best_checkpoints/best_stage2_cls_rank1.pth
SEC=outputs/stage2_cls/best_checkpoints/best_stage2_cls_rank2.pth

# A-arm: baseline recipe (0.6505 inference, no extras)
$PY -B predict_val.py --data-root /root/autodl-tmp/data --device cuda \
  --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
  --output-dir outputs/pv_baseline --no-zip > logs/pv_baseline.log 2>&1

# B-arm template — append a candidate change here, e.g.:
# $PY -B predict_val.py --data-root /root/autodl-tmp/data --device cuda \
#   --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
#   --output-dir outputs/pv_candidate --no-zip > logs/pv_candidate.log 2>&1

echo "=== baseline (gate) ==="
tail -n 25 logs/pv_baseline.log
