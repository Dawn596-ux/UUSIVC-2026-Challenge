#!/usr/bin/env bash
# 260903 双 checkpoint 概率级集成 — 实验矩阵驱动（阶段 1：A/B1/B2/C/D）
# A 臂为硬停 gate：overall_score 必须 repr 级复现 0.8449631268228422。
# 每臂：结构隔离自检（FAIL=接线 bug，中止）+ evaluate_ab 配对 bootstrap。
# F/E 臂由阶段 2 依判读结果另行执行。用法：nohup bash scripts/run_ensemble_matrix.sh > logs/ensemble_matrix.log 2>&1 &
set -uo pipefail
cd /root/autodl-tmp/project/code
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/data
MAIN=outputs/stage2_cls_aug2/best_checkpoints/best_stage2_cls_rank1.pth
SEC=outputs/stage2_cls/best_checkpoints/best_stage2_cls_rank2.pth
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
  local rc=$?
  echo "=== arm $tag rc=$rc ==="
  return $rc
}

summarize () {
  $PY - "$1" <<'EOF'
import json, sys
tag = sys.argv[1]
m = json.load(open(f"outputs/pv_ensemble_{tag}/metrics.json"))
print(f"[score] {tag} overall={m['overall_score']!r}")
for t in ("image_seg", "image_cls", "ceus_cls"):
    pt = m["per_task"].get(t)
    print(f"[score] {tag} {t} score={pt['score']!r}" if pt else f"[score] {tag} {t} MISSING")
EOF
}

judge () {
  local tag="$1" expected="$2"
  echo "--- isolation check $tag (expected changed: $expected) ---"
  if ! $PY -B scripts/check_isolation.py \
      outputs/pv_ensemble_a/metrics_records.json \
      "outputs/pv_ensemble_${tag}/metrics_records.json" \
      --expected-changed "$expected"; then
    echo "[ABORT] isolation broken at arm $tag (= wiring bug, fix before judging)"; exit 1
  fi
  echo "--- evaluate_ab $tag vs A ---"
  $PY -B evaluate_ab.py \
    outputs/pv_ensemble_a/metrics_records.json \
    "outputs/pv_ensemble_${tag}/metrics_records.json" \
    --bootstrap 1000 --output-json "logs/ab_${tag}.json" > "logs/ab_${tag}.log" 2>&1
  tail -n 45 "logs/ab_${tag}.log"
}

echo "########## ARM A (gate, hard stop) ##########"
run_arm a
A_OVERALL=$($PY -c "import json; print(repr(json.load(open('outputs/pv_ensemble_a/metrics.json'))['overall_score']))")
echo "arm A overall_score = $A_OVERALL"
if [ "$A_OVERALL" != "0.8449631268228422" ]; then
  echo "[GATE FAIL] arm A overall $A_OVERALL != 0.8449631268228422 — default path broken, no further arms."; exit 1
fi
if ! diff -q outputs/pv_baseline/metrics.json outputs/pv_ensemble_a/metrics.json > /dev/null; then
  echo "[GATE FAIL] metrics.json differs from outputs/pv_baseline/metrics.json"; exit 1
fi
echo "[GATE PASS] arm A reproduces anchor exactly."
summarize a

echo "########## ARM B1 (image_seg, dual TTA) ##########"
run_arm b1 --ensemble-tasks image_seg --ensemble-weight 0.5 --ensemble-secondary-tta \
  && judge b1 image_seg && summarize b1

echo "########## ARM B2 (image_seg, secondary no TTA) ##########"
run_arm b2 --ensemble-tasks image_seg --ensemble-weight 0.5 \
  && judge b2 image_seg && summarize b2

echo "########## ARM C (image_cls) ##########"
run_arm c --ensemble-tasks image_cls --ensemble-weight 0.5 \
  && judge c image_cls && summarize c

echo "########## ARM D (ceus_cls) ##########"
run_arm d --ensemble-tasks ceus_cls --ensemble-weight 0.5 \
  && judge d ceus_cls && summarize d

echo "########## STAGE 1 DONE [$(date '+%m-%d %H:%M:%S')] — decide F/E from logs/ab_*.log ##########"
