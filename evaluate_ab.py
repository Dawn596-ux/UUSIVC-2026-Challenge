"""Patient-level paired bootstrap A/B comparison between two prediction runs.

Answers "is candidate B really better than baseline A, or is it holdout noise?"
without any retraining: both arms are scored on the SAME local val split
(single seed-2024 holdout), and only the sampling uncertainty of that holdout
is quantified. Training-side stochasticity is NOT covered here — borderline
results should be re-checked with the K-fold CV (aggregate_cv.py).

Input: two ``metrics_records.json`` files (written by ``predict_val.py`` or
``score_submission.py --records-json``). Samples are paired by their entry key;
uncertainty comes from resampling *patient groups* (``derive_group_key``) with
replacement, so correlated samples of one patient always move together.

Reported per task and per dataset (delta = B - A):

- ``delta``    point estimate on the full sample
- ``ci95``     percentile bootstrap CI of the delta
- ``p_pos``    fraction of resamples with delta > 0
- ``worst``    worst resample delta — pessimistic alignment with the official
               3-seed-worst ranking rule

Classification acc/auc are set-level metrics, so every resample recomputes
``0.5*(acc+auc)`` on the resampled patient pool; segmentation deltas are the
sample-pooled mean of per-sample ``0.7*DSC + 0.3*NSD`` scores. The overall row
re-derives the 5-task mean per resample, preserving cross-task correlation
(one shared group draw per iteration).

Note: local absolute scores are untrustworthy (domain gap) — only relative
deltas are meaningful.

Run: python -B evaluate_ab.py records_baseline.json records_candidate.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from score_submission import binary_auc_from_scores, load_json, write_json


def load_records(path: str) -> List[Dict[str, Any]]:
    data = load_json(Path(path))
    if isinstance(data, dict):  # tolerate a full metrics dict with embedded records
        data = data["records"]
    for r in data:
        for required in ("key", "task", "dataset", "group"):
            if required not in r:
                raise KeyError(f"{path}: record missing '{required}' — regenerate with the records export")
    return data


def index_records(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in records:
        key = (r["task"], r["key"])
        if key in indexed:
            raise ValueError(f"duplicate record key: {key}")
        indexed[key] = r
    return indexed


def seg_score(samples: List[Dict[str, Any]]) -> float:
    return float(np.mean([s["score"] for s in samples])) if samples else float("nan")


def cls_score(samples: List[Dict[str, Any]]) -> float:
    if not samples:
        return float("nan")
    gt = np.asarray([s["gt"] for s in samples], dtype=np.int64)
    pred = np.asarray([s["pred"] for s in samples], dtype=np.int64)
    prob = np.asarray([s["prob"] for s in samples], dtype=np.float64)
    acc = float((pred == gt).mean())
    auc = binary_auc_from_scores(prob, gt)
    return 0.5 * (acc + auc)


def sample_score(samples: List[Dict[str, Any]]) -> float:
    return cls_score(samples) if samples[0]["task"] in {"image_cls", "ceus_cls"} else seg_score(samples)


def build_segments(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Bucket records into report segments: per-dataset rows + per-task rows."""
    segments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        segments[f"{r['task']}/{r['dataset']}"].append(r)
    tasks = sorted({r["task"] for r in records})
    for task in tasks:
        segments[task].extend(r for r in records if r["task"] == task)
    return dict(segments), tasks


def paired_records(
    records_a: List[Dict[str, Any]], records_b: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    index_a = index_records(records_a)
    index_b = index_records(records_b)
    only_a = sorted(set(index_a) - set(index_b))
    only_b = sorted(set(index_b) - set(index_a))
    if only_a or only_b:
        raise ValueError(
            "Record key sets differ between arms — both runs must use the same val split.\n"
            f"  only in A: {only_a[:5]}{' ...' if len(only_a) > 5 else ''}\n"
            f"  only in B: {only_b[:5]}{' ...' if len(only_b) > 5 else ''}"
        )
    keys = sorted(index_a.keys())
    return [index_a[k] for k in keys], [index_b[k] for k in keys]


def bootstrap_deltas(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    n_boot: int,
    seed: int,
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    a_paired, b_paired = paired_records(records_a, records_b)

    segments_a, tasks = build_segments(a_paired)

    # Per-segment, per-group paired sample buckets. A record joins ONLY its own
    # dataset bucket and task bucket — never other segments (which would mix seg
    # and cls records and corrupt set-level cls metrics).
    seg_groups: Dict[str, Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]] = {
        name: defaultdict(lambda: ([], [])) for name in segments_a
    }
    for r_a, r_b in zip(a_paired, b_paired):
        assert r_a["group"] == r_b["group"]
        for name in (f"{r_a['task']}/{r_a['dataset']}", r_a["task"]):
            bucket = seg_groups[name][r_a["group"]]
            bucket[0].append(r_a)
            bucket[1].append(r_b)

    def segment_delta(name: str, counter: Dict[str, int]) -> float:
        sa: List[Dict[str, Any]] = []
        sb: List[Dict[str, Any]] = []
        for g, bucket in seg_groups[name].items():
            mult = counter.get(g, 0)
            for _ in range(mult):
                sa.extend(bucket[0])
                sb.extend(bucket[1])
        if not sa:
            return 0.0  # degenerate resample with no samples for this segment
        return sample_score(sb) - sample_score(sa)

    def all_deltas(counter: Dict[str, int]) -> Dict[str, float]:
        deltas = {name: segment_delta(name, counter) for name in segments_a}
        deltas["overall"] = float(np.mean([deltas[t] for t in tasks]))
        return deltas

    groups = sorted({r["group"] for r in a_paired})
    point = all_deltas({g: 1 for g in groups})  # full sample = every group once

    rng = random.Random(seed)
    point_by_name: Dict[str, List[float]] = defaultdict(list)
    worst: Dict[str, float] = {name: float("inf") for name in list(segments_a) + ["overall"]}
    for _ in range(n_boot):
        counter = Counter(rng.choices(groups, k=len(groups)))
        deltas = all_deltas(counter)
        for name, d in deltas.items():
            point_by_name[name].append(d)
            worst[name] = min(worst[name], d)

    rows: Dict[str, Dict[str, float]] = {}
    for name in list(segments_a) + ["overall"]:
        draws = np.asarray(point_by_name[name], dtype=np.float64)
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        rows[name] = {
            "n": len(segments_a[name]) if name != "overall" else len(a_paired),
            "delta": point[name],
            "boot_mean": float(draws.mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p_pos": float((draws > 0).mean()),
            "worst": float(worst[name]),
        }
    return rows


def print_report(
    rows: Dict[str, Dict[str, float]],
    name_a: str,
    name_b: str,
    n_boot: int,
    alpha: float,
) -> None:
    print(f"\nA/B paired bootstrap (delta = {name_b} - {name_a}; B={n_boot}, alpha={alpha})")
    print(f"{'segment':<40} {'n':>5} {'delta':>8} {'ci95_low':>9} {'ci95_high':>9} {'p_pos':>6} {'worst':>8}")
    order = sorted(rows, key=lambda n: (n != "overall", "/" not in n, n))
    for name in order:
        r = rows[name]
        print(
            f"{name:<40} {r['n']:>5} {r['delta']:>+8.4f} {r['ci_low']:>9.4f} {r['ci_high']:>9.4f} "
            f"{r['p_pos']:>6.2f} {r['worst']:>+8.4f}"
        )
    overall = rows["overall"]
    if overall["ci_low"] > 0 and overall["worst"] > 0:
        verdict = "IMPROVEMENT (CI excludes 0 AND worst-resample > 0)"
    elif overall["ci_high"] < 0 and overall["worst"] < 0:
        verdict = "REGRESSION (CI excludes 0 AND worst-resample < 0)"
    else:
        verdict = "INCONCLUSIVE (CI straddles 0 or worst-resample flips sign — verify with K-fold CV)"
    print(f"\noverall delta = {overall['delta']:+.4f}  ->  {verdict}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-level paired bootstrap A/B comparison of two metrics_records.json files."
    )
    parser.add_argument("records_a", type=str, help="Baseline arm records json (metrics_records.json).")
    parser.add_argument("records_b", type=str, help="Candidate arm records json (metrics_records.json).")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of patient-group resamples (default 1000).")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default 0).")
    parser.add_argument("--alpha", type=float, default=0.05, help="CI significance level (default 0.05).")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path for the full result json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_a = load_records(args.records_a)
    records_b = load_records(args.records_b)
    rows = bootstrap_deltas(records_a, records_b, n_boot=args.bootstrap, seed=args.seed, alpha=args.alpha)
    print_report(rows, Path(args.records_a).parent.name, Path(args.records_b).parent.name, args.bootstrap, args.alpha)
    if args.output_json:
        write_json(
            Path(args.output_json),
            {
                "records_a": str(args.records_a),
                "records_b": str(args.records_b),
                "n_boot": args.bootstrap,
                "seed": args.seed,
                "alpha": args.alpha,
                "rows": rows,
            },
        )
        print(f"[Info] Result written to {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
