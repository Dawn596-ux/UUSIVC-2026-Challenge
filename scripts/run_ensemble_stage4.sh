#!/usr/bin/env bash
# 260903 双 checkpoint 概率级集成 — 阶段 4：E2 修正合并臂 + 判读 + 计时。
# E2 = image_seg(B2+third) + ceus_cls(D 无 third)；--ensemble-third-tasks image_seg 作用域。
# 端点恒等断言：E2 image_seg 必须精确复现 f_seg、ceus_cls 必须精确复现 d。
set -uo pipefail
cd /root/autodl-tmp/project/code
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/data
MAIN=outputs/stage2_cls_aug2/best_checkpoints/best_stage2_cls_rank1.pth
SEC=outputs/stage2_cls/best_checkpoints/best_stage2_cls_rank2.pth
THIRD=outputs/stage2_cls_aug2/best_checkpoints/best_stage2_cls_rank2.pth
mkdir -p logs

avail=$(df --output=avail -B1G /root/autodl-tmp | tail -1 | tr -dc '0-9')
[ -z "$avail" ] || [ "$avail" -lt 5 ] && { echo "[ABORT] disk ${avail}G"; exit 1; }

echo "########## ARM E2 (scoped third: seg only) ##########"
echo "=== [$(date '+%m-%d %H:%M:%S')] arm e2 START ==="
$PY -B predict_val.py --data-root "$DATA" --device cuda \
  --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
  --output-dir outputs/pv_ensemble_e2 --no-zip \
  --ensemble-tasks image_seg,ceus_cls --ensemble-weight 0.5 \
  --ensemble-third-checkpoint "$THIRD" --ensemble-third-weight 0.5 \
  --ensemble-third-tasks image_seg \
  > logs/pv_ensemble_e2.log 2>&1
echo "=== arm e2 rc=$? ==="

echo "--- isolation check e2 (vs A, expected: image_seg,ceus_cls) ---"
if ! $PY -B scripts/check_isolation.py \
    outputs/pv_ensemble_a/metrics_records.json \
    outputs/pv_ensemble_e2/metrics_records.json \
    --expected-changed image_seg,ceus_cls; then
  echo "[ABORT] isolation broken at arm e2"; exit 1
fi

echo "--- endpoint identity assertion ---"
$PY -c "
import json
e2 = json.load(open('outputs/pv_ensemble_e2/metrics.json'))['per_task']
f_seg = json.load(open('outputs/pv_ensemble_f_seg/metrics.json'))['per_task']
d = json.load(open('outputs/pv_ensemble_d/metrics.json'))['per_task']
ok_seg = e2['image_seg']['score'] == f_seg['image_seg']['score']
ok_cls = e2['ceus_cls']['score'] == d['ceus_cls']['score']
print(f'IDENTITY image_seg: e2={e2[\"image_seg\"][\"score\"]!r} f_seg={f_seg[\"image_seg\"][\"score\"]!r} -> {\"OK\" if ok_seg else \"MISMATCH\"}')
print(f'IDENTITY ceus_cls:  e2={e2[\"ceus_cls\"][\"score\"]!r} d={d[\"ceus_cls\"][\"score\"]!r} -> {\"OK\" if ok_cls else \"MISMATCH\"}')
if not (ok_seg and ok_cls):
    print('[FAIL] endpoint identity broken — scoping flag did not route as intended'); exit(1)
m = json.load(open('outputs/pv_ensemble_e2/metrics.json'))
print(f'[E2] overall={m[\"overall_score\"]!r}')
"

echo "--- evaluate_ab e2 vs A ---"
$PY -B evaluate_ab.py \
  outputs/pv_ensemble_a/metrics_records.json \
  outputs/pv_ensemble_e2/metrics_records.json \
  --bootstrap 1000 --output-json logs/ab_e2.json > logs/ab_e2.log 2>&1
$PY -c "
import json
r = json.load(open('logs/ab_e2.json'))['rows']
o = r['overall']
verdict = 'ADOPT' if (o['delta'] > 0 and o['p_pos'] >= 0.95) else 'REJECT'
print(f'[E2-judge] overall delta={o[\"delta\"]:+.6f} ci=[{o[\"ci_low\"]:+.6f},{o[\"ci_high\"]:+.6f}] p_pos={o[\"p_pos\"]} worst={o[\"worst\"]:+.6f} -> {verdict}')
for k in ('image_seg', 'ceus_cls'):
    if k in r:
        v = r[k]
        print(f'[E2-judge] {k}: delta={v[\"delta\"]:+.6f} p_pos={v[\"p_pos\"]}')
"

echo "########## TIMING (600s hard, <300s target) ##########"
T0=$(date +%s)
$PY -B predict.py --phase val --data-root "$DATA" --device cuda \
  --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
  --output-dir outputs/timing_base2 --no-zip > logs/timing_base2.log 2>&1
T1=$(date +%s)
echo "TIMING_BASELINE_ELAPSED_SECONDS=$((T1-T0))"

T0=$(date +%s)
$PY -B predict.py --phase val --data-root "$DATA" --device cuda \
  --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
  --ensemble-tasks image_seg,ceus_cls --ensemble-weight 0.5 \
  --ensemble-third-checkpoint "$THIRD" --ensemble-third-weight 0.5 \
  --ensemble-third-tasks image_seg \
  --output-dir outputs/timing_e2 --no-zip > logs/timing_e2.log 2>&1
T1=$(date +%s)
echo "TIMING_E2_ELAPSED_SECONDS=$((T1-T0))"

rm -rf outputs/timing_base2/image_seg outputs/timing_base2/video_seg outputs/timing_base2/ceus_seg \
       outputs/timing_e2/image_seg outputs/timing_e2/video_seg outputs/timing_e2/ceus_seg \
       outputs/pv_ensemble_e2/image_seg outputs/pv_ensemble_e2/video_seg outputs/pv_ensemble_e2/ceus_seg
echo "########## STAGE 4 DONE [$(date '+%m-%d %H:%M:%S')] ##########"
