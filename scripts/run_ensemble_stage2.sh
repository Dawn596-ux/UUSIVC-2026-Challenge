#!/usr/bin/env bash
# 260903 双 checkpoint 概率级集成 — 阶段 2：F 臂（第三成员），逐任务 vs 各自基础臂判读。
# 阶段 1 结论：B2（image_seg, 无 secondary-tta）与 D（ceus_cls）幸存；C 否决。
# F 判据：vs 基础臂 delta>0 且 p_pos>=0.95 才进 E，否则该任务用基础臂形态进 E。
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

run_arm () {
  local tag="$1"; shift
  df_guard "$tag"
  echo "=== [$(date '+%m-%d %H:%M:%S')] arm $tag START: $* ==="
  $PY -B predict_val.py --data-root "$DATA" --device cuda \
    --checkpoint "$MAIN" --secondary-checkpoint "$SEC" \
    --output-dir "outputs/pv_ensemble_${tag}" --no-zip "$@" \
    > "logs/pv_ensemble_${tag}.log" 2>&1
  echo "=== arm $tag rc=$? ==="
}

# F vs 基础臂的配对 bootstrap；隔离自检仍以 A 为基准（expected-changed 限目标任务）。
judge_f () {
  local tag="$1" task="$2" base="$3"
  echo "--- isolation check $tag (vs A, expected: $task) ---"
  if ! $PY -B scripts/check_isolation.py \
      outputs/pv_ensemble_a/metrics_records.json \
      "outputs/pv_ensemble_${tag}/metrics_records.json" \
      --expected-changed "$task"; then
    echo "[ABORT] isolation broken at arm $tag"; exit 1
  fi
  echo "--- evaluate_ab $tag vs base $base ---"
  $PY -B evaluate_ab.py \
    "outputs/pv_ensemble_${base}/metrics_records.json" \
    "outputs/pv_ensemble_${tag}/metrics_records.json" \
    --bootstrap 1000 --output-json "logs/ab_${tag}.json" > "logs/ab_${tag}.log" 2>&1
  $PY -c "
import json
o = json.load(open('logs/ab_${tag}.json'))['rows']['overall']
verdict = 'ADOPT' if (o['delta'] > 0 and o['p_pos'] >= 0.95) else 'DROP'
print(f'[F-judge ${tag}] vs ${base}: delta={o[\"delta\"]:+.6f} ci=[{o[\"ci_low\"]:+.6f},{o[\"ci_high\"]:+.6f}] p_pos={o[\"p_pos\"]} -> {verdict}')
"
}

echo "########## ARM F_SEG (B2 form + third member) ##########"
run_arm f_seg --ensemble-tasks image_seg --ensemble-weight 0.5 \
  --ensemble-third-checkpoint "$THIRD" --ensemble-third-weight 0.5
judge_f f_seg image_seg b2

echo "########## ARM F_CEUS (D form + third member) ##########"
run_arm f_ceus --ensemble-tasks ceus_cls --ensemble-weight 0.5 \
  --ensemble-third-checkpoint "$THIRD" --ensemble-third-weight 0.5
judge_f f_ceus ceus_cls d

echo "########## STAGE 2 DONE [$(date '+%m-%d %H:%M:%S')] — F verdicts above decide E composition ##########"
