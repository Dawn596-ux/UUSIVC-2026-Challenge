#!/usr/bin/env bash
# Phase 4: gate + mixed-recipe predict_val (video worktree, GPU)
# Arm A: old-recipe gate — video-branch code (dbe7240+3b3b10c wiring), old
#        checkpoints, default routing. Must reproduce pv_baseline 0.844963.
# Arm B: three-slot deployment — mem_full(video,ceus) + aug2 r1(image_seg)
#        + old r2(image_cls). image_seg/image_cls must equal Arm A exactly.
set -euo pipefail
cd /root/autodl-tmp/project/code-video-temporal
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/data
MAN=outputs/uusivc2026_fixed/manifests/val_entries.json
MAIN=/root/autodl-tmp/project/code/outputs/stage2_cls_aug2/best_checkpoints/best_stage2_cls_rank1.pth
OLD=/root/autodl-tmp/project/code/outputs/stage2_cls/best_checkpoints/best_stage2_cls_rank2.pth
MEM=outputs/stage2_cls_mem_full/best_checkpoints/best_stage2_cls_rank1.pth

echo "[P4] Arm A: old-recipe gate"
$PY -B predict_val.py --data-root $DATA --device cuda \
  --val-manifest $MAN \
  --checkpoint $MAIN --secondary-checkpoint $OLD \
  --output-dir outputs/pv_gate_wired --no-zip > logs/pv_gate_wired.log 2>&1
echo "[P4] Arm A done"

echo "[P4] Arm B: three-slot deployment recipe"
$PY -B predict_val.py --data-root $DATA --device cuda \
  --val-manifest $MAN \
  --config configs/stage2_cls_mem.yaml \
  --checkpoint $MEM \
  --secondary-config configs/stage2_cls.yaml --secondary-checkpoint $MAIN --secondary-tasks image_seg \
  --tertiary-config configs/stage2_cls.yaml --tertiary-checkpoint $OLD --tertiary-tasks image_cls \
  --output-dir outputs/pv_mixed_v1 --no-zip > logs/pv_mixed_v1.log 2>&1
echo "[P4] Arm B done"
echo "PHASE4_RUNS_DONE"
