#!/usr/bin/env bash
# Master driver for the image_seg augmentation-ablation experiment.
#
# Serial stages (each resumable via run_cv.sh's per-fold skip markers):
#   A) baseline K=3 CV          (TAG=baseline, aug_extra=none)
#   B) fold0 screens            (ONLY_FOLD=0 for each candidate operator)
#   C) auto-select winners      (fold0 overall > baseline fold0 overall)
#      -> combined full K=3     (AUG_EXTRA=comma-joined winners)
#      -> adoption report       (aggregate_cv.py vs baseline)
#
# PREREQUISITE: scripts/run_cv.sh must forward AUG_EXTRA to train.py --aug-extra.
#   (Applied only AFTER the baseline run finishes, to avoid editing a running script.)
#
# Usage (server):
#   bash scripts/run_aug_experiment.sh
#   BASELINE_TAG=baseline CANDIDATES="clahe elastic gaussnoise" K=3 bash scripts/run_aug_experiment.sh
set -euo pipefail
cd /root/autodl-tmp/project/code
PY=/root/miniconda3/bin/python

BASELINE_TAG=${BASELINE_TAG:-baseline}
CANDIDATES=${CANDIDATES:-"clahe elastic gaussnoise"}
K=${K:-3}

echo "[AUG-EXP] =============================================="
echo "[AUG-EXP] K=$K baseline_tag=$BASELINE_TAG candidates=[$CANDIDATES]"
echo "[AUG-EXP] =============================================="

echo "[AUG-EXP] Stage A: baseline K=$K (resumable)"
TAG="$BASELINE_TAG" bash scripts/run_cv.sh

echo "[AUG-EXP] Stage B: fold0 screens"
for cand in $CANDIDATES; do
  echo "[AUG-EXP] --- fold0 screen: $cand ---"
  TAG="$cand" AUG_EXTRA="$cand" ONLY_FOLD=0 bash scripts/run_cv.sh
done

echo "[AUG-EXP] Stage C: select winners (fold0 overall vs baseline fold0)"
COMBO=$($PY -B scripts/select_winners.py \
  --cv-root outputs/cv --k "$K" --baseline-tag "$BASELINE_TAG" --candidates "$CANDIDATES")

if [ -z "$COMBO" ]; then
  echo "[AUG-EXP] No operator improved fold0 over baseline -> keep baseline recipe. DONE."
  exit 0
fi

echo "[AUG-EXP] Winners: $COMBO -> combined full K=$K"
TAG="combo" AUG_EXTRA="$COMBO" bash scripts/run_cv.sh

echo "[AUG-EXP] Adoption report: combo vs baseline"
$PY -B aggregate_cv.py "outputs/cv/K${K}_combo" \
  --baseline "outputs/cv/K${K}_${BASELINE_TAG}" \
  --output-json "outputs/cv/K${K}_combo/summary.json"

echo "[AUG-EXP] ALL DONE. See outputs/cv/K${K}_combo/summary.json for ADOPT/REJECT."
