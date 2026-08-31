#!/usr/bin/env python
"""Select augmentation operators whose fold0 overall score beats the baseline fold0.

Prints a comma-joined list of winning operators to stdout (empty string if none
improved). Used by scripts/run_aug_experiment.sh to auto-build the combined recipe.

Usage:
  python -B select_winners.py --cv-root outputs/cv --k 3 \
      --baseline-tag baseline --candidates "clahe elastic gaussnoise"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def metrics_path(fold_dir: Path) -> Path:
    sub = fold_dir / "predict" / "metrics.json"
    return sub if sub.exists() else fold_dir / "metrics.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv-root", default="outputs/cv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--baseline-tag", default="baseline")
    ap.add_argument("--candidates", default="clahe elastic gaussnoise")
    args = ap.parse_args()

    base_dir = Path(args.cv_root) / f"K{args.k}_{args.baseline_tag}" / "fold0"
    bp = metrics_path(base_dir)
    if not bp.exists():
        print(f"[select] ERROR: missing baseline fold0 metrics {bp}", file=sys.stderr)
        sys.exit(2)
    base = json.loads(bp.read_text())["overall_score"]

    winners = []
    for cand in args.candidates.split():
        cp = metrics_path(Path(args.cv_root) / f"K{args.k}_{cand}" / "fold0")
        if not cp.exists():
            print(f"[select] WARN: missing fold0 metrics for {cand} ({cp})", file=sys.stderr)
            continue
        s = json.loads(cp.read_text())["overall_score"]
        print(f"[select] {cand}: fold0={s:.4f} vs baseline={base:.4f} (delta {s - base:+.4f})",
              file=sys.stderr)
        if s > base:
            winners.append(cand)

    print(",".join(winners))


if __name__ == "__main__":
    main()
