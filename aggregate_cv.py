"""Aggregate K-fold CV runs into the mean+-std / worst-fold decision table.

Consumes the per-fold ``metrics.json`` files produced by ``scripts/run_cv.sh``
(each fold directory holds one ``predict_val.py`` output with ``metrics.json``)
and reports, per task and per dataset, the per-fold scores plus mean / std /
worst fold — the pessimistic alignment with the official 3-seed-worst rule.

With ``--baseline <other-cv-root>`` the folds are compared pairwise (same fold
index = same patient split, so deltas are paired) and the handover adoption
rules are evaluated automatically:

1. overall improves in >= ceil(K/2) folds;
2. mean overall delta > std of the fold deltas;
3. no single task regresses in every fold (unanimous regression).

Dataset-level unanimous regressions are reported as warnings. Per the domain
gap, only deltas are meaningful — absolute values are not comparable to the
official leaderboard.

Run: python -B aggregate_cv.py outputs/cv/K3_candidate --baseline outputs/cv/K3_baseline
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from score_submission import load_json, write_json


def discover_folds(cv_root: Path) -> List[Path]:
    if not cv_root.is_dir():
        raise FileNotFoundError(f"CV root not found: {cv_root}")
    folds = sorted(p for p in cv_root.iterdir() if p.is_dir() and (p / "metrics.json").exists())
    if not folds:
        raise FileNotFoundError(f"No fold*/metrics.json under {cv_root}")
    return folds


def load_run(cv_root: Path) -> Dict[str, Any]:
    folds = discover_folds(cv_root)
    run: Dict[str, Any] = {
        "root": str(cv_root),
        "fold_names": [p.name for p in folds],
        "overall": [],
        "per_task": {},
        "per_dataset": {},
        "n_dataset": {},
    }
    for p in folds:
        m = load_json(p / "metrics.json")
        run["overall"].append(float(m["overall_score"]))
        for task, tm in m["per_task"].items():
            run["per_task"].setdefault(task, []).append(float(tm["score"]))
        for ds, dm in m["per_dataset"].items():
            run["per_dataset"].setdefault(ds, []).append(float(dm["score"]))
            run["n_dataset"].setdefault(ds, []).append(int(dm["n"]))
    return run


def fmt_series(values: List[float]) -> str:
    return " ".join(f"{v:.4f}" for v in values)


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "folds": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "worst": float(np.min(values)),
    }


def render_single_run(run: Dict[str, Any]) -> str:
    k = len(run["fold_names"])
    lines = [f"\nCV run: {run['root']} (K={k}: {', '.join(run['fold_names'])})",
             f"{'segment':<42} {'n/fold':>12} {'mean':>7} {'std':>7} {'worst':>7}  folds"]
    rows = [("overall", None, run["overall"])]
    rows += [(task, None, vals) for task, vals in sorted(run["per_task"].items())]
    rows += [(ds, run["n_dataset"][ds], vals) for ds, vals in sorted(run["per_dataset"].items())]
    for name, ns, vals in rows:
        s = summarize(vals)
        n_str = "/".join(str(n) for n in ns) if ns else "-"
        flag = " *" if (ns and max(ns) < 100) else ""
        lines.append(
            f"{name:<42} {n_str:>12} {s['mean']:>7.4f} {s['std']:>7.4f} {s['worst']:>7.4f}  "
            f"[{fmt_series(vals)}]{flag}"
        )
    lines.append("\n(* = every fold n<100 — small-sample dataset, trust the worst fold, expect a large std)")
    return "\n".join(lines)


def evaluate_rules(run: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    if run["fold_names"] != base["fold_names"]:
        raise ValueError(
            f"Fold sets differ: {run['root']}={run['fold_names']} vs {base['root']}={base['fold_names']} — "
            "runs must use the same K and fold indices."
        )
    k = len(run["fold_names"])
    deltas = {
        "overall": [r - b for r, b in zip(run["overall"], base["overall"])],
    }
    for task in sorted(run["per_task"]):
        deltas[task] = [r - b for r, b in zip(run["per_task"][task], base["per_task"][task])]

    overall = summarize(deltas["overall"])
    rule1 = sum(d > 0 for d in deltas["overall"]) >= math.ceil(k / 2)
    rule2 = overall["mean"] > overall["std"]
    unanimous_task_regression = [
        task for task, ds in deltas.items()
        if task != "overall" and all(d < 0 for d in ds)
    ]
    rule3 = not unanimous_task_regression

    unanimous_ds_regression = []
    for ds in sorted(run["per_dataset"]):
        dd = [r - b for r, b in zip(run["per_dataset"][ds], base["per_dataset"][ds])]
        if all(d < 0 for d in dd):
            unanimous_ds_regression.append(ds)

    adopted = rule1 and rule2 and rule3
    return {
        "k": k,
        "fold_deltas": deltas,
        "overall": overall,
        "rule1_majority_improve": rule1,
        "rule2_gt_std": rule2,
        "rule3_no_task_regression": rule3,
        "unanimous_task_regression": unanimous_task_regression,
        "unanimous_dataset_regression": unanimous_ds_regression,
        "adopted": adopted,
    }


def render_comparison(run: Dict[str, Any], base: Dict[str, Any], result: Dict[str, Any]) -> str:
    lines = [
        f"\nPaired fold comparison (candidate {run['root']} vs baseline {base['root']})",
        f"{'segment':<42} {'mean_delta':>10} {'std':>7} {'worst':>8}  fold_deltas",
    ]
    for name in ["overall"] + sorted(n for n in result["fold_deltas"] if n != "overall"):
        s = summarize(result["fold_deltas"][name])
        lines.append(
            f"{name:<42} {s['mean']:>+10.4f} {s['std']:>7.4f} {s['worst']:>+8.4f}  "
            f"[{fmt_series(s['folds'])}]"
        )
    ds_warn = result["unanimous_dataset_regression"]
    lines += [
        "",
        f"  rule 1 (>= ceil(K/2) folds improve): {'PASS' if result['rule1_majority_improve'] else 'FAIL'}",
        f"  rule 2 (mean delta > fold std):      {'PASS' if result['rule2_gt_std'] else 'FAIL'}",
        f"  rule 3 (no unanimous task regression): {'PASS' if result['rule3_no_task_regression'] else 'FAIL — ' + ', '.join(result['unanimous_task_regression'])}",
    ]
    if ds_warn:
        lines.append(f"  dataset-level unanimous regressions (warning): {', '.join(ds_warn)}")
    lines.append(f"\n  ADOPTION: {'ADOPT — all three rules satisfied' if result['adopted'] else 'REJECT — treat as noise, roll back'}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate K-fold CV runs (mean/std/worst fold + adoption rules).")
    parser.add_argument("cv_root", type=str, help="Candidate CV root, e.g. outputs/cv/K3_candidate.")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Optional baseline CV root (same K/fold indices) for the paired 3-rule check.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path for the full result json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = load_run(Path(args.cv_root))
    print(render_single_run(run))

    result: Optional[Dict[str, Any]] = None
    if args.baseline:
        base = load_run(Path(args.baseline))
        result = evaluate_rules(run, base)
        print(render_comparison(run, base, result))

    if args.output_json:
        payload: Dict[str, Any] = {
            "candidate": {k: v for k, v in run.items() if k != "root"},
            "cv_root": run["root"],
        }
        if result is not None:
            payload["comparison"] = result
        write_json(Path(args.output_json), payload)
        print(f"\n[Info] Result written to {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
