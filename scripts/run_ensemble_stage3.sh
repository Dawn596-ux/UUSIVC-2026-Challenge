#!/usr/bin/env bash
# 260903 双 checkpoint 概率级集成 — 阶段 3：E 合并臂 + 独立判读 + 600s 推理计时复测。
# E 成分（阶段 1/2 判读锁定）：image_seg = B2+third（无 secondary-tta）；ceus_cls = D 基础形态。
# E 判据：vs A overall Δ>0 且 p_pos>=0.95；计时 >600s = REJECT 硬线，目标 <300s。
set -uo pipefail
cd /root/autodl-tmp/project/code
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/data
MAIN=outputs/stage2_cls_aug2/best_checkpoints/best_stage2_cls_rank1.pth
SEC=outputs/stage2_cls/best_checkpoints/best_stage2_cls_rank2.pth
THIRD=outputs/stage2_cls_aug2/best_checkpoints/best_stage2_cls_rank2.pth
mkdir -p logs

df_guard () {
  local avail; avail=$(df --output=avail -B1G /root/autodl-tmp | tail -1 | tr -dc '0-9')
  if [ -z "$avail" ] || [ "$avail" -lt 5 ]; then
    echo "[ABORT] disk free ${avail}G < 5G before arm $1"; exit 1
  fi
}

echo "########## ARM E (merged survivors) ##########"
df_guard e
echo "=== [$(date '+%m-%d %H:%M:%S')] arm e START ==="
$PY -B predict_val.py --data-root "$DATA" --device cuda \
  --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
  --output-dir outputs/pv_ensemble_e --no-zip \
  --ensemble-tasks image_seg,ceus_cls --ensemble-weight 0.5 \
  --ensemble-third-checkpoint "$THIRD" --ensemble-third-weight 0.5 \
  > logs/pv_ensemble_e.log 2>&1
echo "=== arm e rc=$? ==="

echo "--- isolation check e (vs A, expected: image_seg,ceus_cls) ---"
if ! $PY -B scripts/check_isolation.py \
    outputs/pv_ensemble_a/metrics_records.json \
    outputs/pv_ensemble_e/metrics_records.json \
    --expected-changed image_seg,ceus_cls; then
  echo "[ABORT] isolation broken at arm e"; exit 1
fi

echo "--- evaluate_ab e vs A ---"
$PY -B evaluate_ab.py \
  outputs/pv_ensemble_a/metrics_records.json \
  outputs/pv_ensemble_e/metrics_records.json \
  --bootstrap 1000 --output-json logs/ab_e.json > logs/ab_e.log 2>&1
$PY -c "
import json
r = json.load(open('logs/ab_e.json'))['rows']
o = r['overall']
verdict = 'ADOPT' if (o['delta'] > 0 and o['p_pos'] >= 0.95) else 'REJECT'
print(f'[E-judge] overall delta={o[\"delta\"]:+.6f} ci=[{o[\"ci_low\"]:+.6f},{o[\"ci_high\"]:+.6f}] p_pos={o[\"p_pos\"]} worst={o[\"worst\"]:+.6f} -> {verdict}')
for k in ('image_seg/overall', 'ceus_cls/overall'):
    if k in r:
        v = r[k]
        print(f'[E-judge] {k}: delta={v[\"delta\"]:+.6f} p_pos={v[\"p_pos\"]}')
m = json.load(open('outputs/pv_ensemble_e/metrics.json'))
print(f'[E-judge] e overall={m[\"overall_score\"]!r}')
"

echo "########## TIMING (600s hard limit, <300s target) ##########"
df_guard timing_base
T0=$(date +%s)
$PY -B predict.py --phase val --data-root "$DATA" --device cuda \
  --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
  --output-dir outputs/timing_base --no-zip > logs/timing_base.log 2>&1
T1=$(date +%s)
echo "TIMING_BASELINE_ELAPSED_SECONDS=$((T1-T0))"

df_guard timing_e
T0=$(date +%s)
$PY -B predict.py --phase val --data-root "$DATA" --device cuda \
  --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
  --ensemble-tasks image_seg,ceus_cls --ensemble-weight 0.5 \
  --ensemble-third-checkpoint "$THIRD" --ensemble-third-weight 0.5 \
  --output-dir outputs/timing_e --no-zip > logs/timing_e.log 2>&1
T1=$(date +%s)
echo "TIMING_E_ELAPSED_SECONDS=$((T1-T0))"

echo "########## STAGE 3 DONE [$(date '+%m-%d %H:%M:%S')] ##########"
