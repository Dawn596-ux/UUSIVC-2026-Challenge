"""结构隔离自检：两个 predict_val 臂的 records 级逐位对比。

每臂判读前必跑（最便宜的 bug 探测器）：臂 X 只允许 --expected-changed 里的任务
相对基准臂 A 发生变化，其余任务必须逐位一致；允许集任务 0 变更 = 接线 bug（WARN）。

Usage:
  python -B scripts/check_isolation.py outputs/pv_ensemble_a/metrics_records.json \
      outputs/pv_ensemble_b1/metrics_records.json --expected-changed image_seg

PASS 条件：key 集合一致 + 所有变更记录的 task ∈ expected-changed。
另打印各任务变更计数——目标任务的计数应等于其样本数（偏小 = extra 成员接错 checkpoint）。
"""
import argparse
import json
from pathlib import Path


def load_records(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {f"{r['task']}/{r['key']}": r for r in data}


def main() -> None:
    ap = argparse.ArgumentParser(description="Arm isolation check over metrics_records.json files.")
    ap.add_argument("records_a", help="Baseline arm records json (metrics_records.json).")
    ap.add_argument("records_b", help="Candidate arm records json (metrics_records.json).")
    ap.add_argument("--expected-changed", type=str, default="",
                    help="Comma-separated tasks allowed to change between the two arms.")
    args = ap.parse_args()

    a, b = load_records(args.records_a), load_records(args.records_b)
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))[:5]
        only_b = sorted(set(b) - set(a))[:5]
        raise SystemExit(f"FAIL: key sets differ (+{only_b} -{only_a}); both runs must use the same val split.")

    allowed = {t.strip() for t in args.expected_changed.split(",") if t.strip()}
    changed: dict = {}
    for key, rec_a in a.items():
        if json.dumps(rec_a, sort_keys=True) != json.dumps(b[key], sort_keys=True):
            task = rec_a["task"]
            changed[task] = changed.get(task, 0) + 1
    print(f"changed per task: {dict(sorted(changed.items())) or '{}'}")

    unexpected = {t: n for t, n in changed.items() if t not in allowed}
    if unexpected:
        raise SystemExit(f"FAIL: isolation broken, unexpected changed tasks: {unexpected}")
    silent = sorted(allowed - set(changed))
    if silent:
        print(f"WARN: expected-changed tasks with 0 changed records (wiring bug?): {silent}")
    print("PASS: isolation ok")


if __name__ == "__main__":
    main()
