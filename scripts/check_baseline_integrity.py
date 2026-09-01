"""Baseline integrity sentinel — verify K3 baseline scoring records are untouched.

Read-only checker for the K-fold baseline artifacts that every A/B comparison
and aggregate judgment depends on. Motivated by the 2026-09-01 incident where a
hardlink write-through silently rewrote K3_baseline metrics (see memory
uusivc2026-hardlink-copytree-clobber). Run this before trusting any
evaluate_ab / aggregate_cv verdict.

Usage (server):
    # regenerate the manifest (only after a verified restore/completion)
    python -B scripts/check_baseline_integrity.py --update \
        --baseline-dir /root/autodl-tmp/project/code/outputs/cv/K3_baseline

    # verify (default; non-zero exit on any mismatch)
    python -B scripts/check_baseline_integrity.py \
        --baseline-dir /root/autodl-tmp/project/code/outputs/cv/K3_baseline
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

RECORD_NAMES = ("metrics.json", "metrics_records.json", "predict_val_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify K3 baseline scoring-record integrity against a sha1 manifest.")
    parser.add_argument("--baseline-dir", type=str, default="outputs/cv/K3_baseline",
                        help="Baseline CV root containing fold*/predict/.")
    parser.add_argument("--k", type=int, default=3, help="Number of folds to cover.")
    parser.add_argument("--update", action="store_true",
                        help="(Re)generate the manifest from current file contents instead of verifying.")
    return parser.parse_args()


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_paths(baseline: Path, k: int) -> list:
    return [baseline / f"fold{i}" / "predict" / name for i in range(k) for name in RECORD_NAMES]


def main() -> None:
    args = parse_args()
    baseline = Path(args.baseline_dir)
    paths = expected_paths(baseline, args.k)
    manifest = baseline / "BASELINE_SHA1.txt"

    if args.update:
        missing = [p for p in paths if not p.is_file()]
        if missing:
            for p in missing:
                print(f"MISSING: {p}")
            raise SystemExit(f"cannot update manifest: {len(missing)} file(s) missing")
        lines = [f"{_sha1(p)}  {p.relative_to(baseline).as_posix()}" for p in paths]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[integrity] manifest written: {manifest} ({len(lines)} entries)")
        return

    if not manifest.is_file():
        raise SystemExit(f"manifest not found: {manifest} (run once with --update after a verified restore)")
    want: dict = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            want[rel.strip()] = digest

    failures = 0
    for p in paths:
        rel = p.relative_to(baseline).as_posix()
        if not p.is_file():
            print(f"FAIL missing  {rel}")
            failures += 1
        elif rel not in want:
            print(f"FAIL no-entry {rel} (manifest is older than this file)")
            failures += 1
        elif _sha1(p) != want[rel]:
            print(f"FAIL hash     {rel}")
            failures += 1
        else:
            print(f"PASS          {rel}")

    if failures:
        raise SystemExit(f"[integrity] {failures}/{len(paths)} FAILED — do not trust A/B or aggregate verdicts against this baseline")
    print(f"[integrity] all {len(paths)} baseline files verified OK")


if __name__ == "__main__":
    main()
