"""Temporal smoothing post-processing for video_seg predictions (offline, no torch).

Reads an existing predict directory (predict_val.py output layout), rewrites only
``video_seg/**/*.npz`` with temporally smoothed masks, and re-scores the whole
directory with ``score_submission.score_submission_dir`` so the result can be
paired against the source records via ``evaluate_ab.py``.

Background: predict.py only infers ``num_frames`` sampled frames per video and
copies the nearest sampled mask to every other frame index, so each npz holds a
block-constant ``fnum_mask`` dict with <=10 unique masks. Smoothing therefore
operates on the *unique* mask sequence (consecutive identical masks are merged by
content), then expands back to the original key set. Key set, string key forms,
dtype (uint8 0/255) and spatial size are preserved exactly — the scorer pairs
fnum keys against GT by exact string match.

ceus_seg npz files hold a single restored middle-frame mask (no temporal axis)
and are deliberately left untouched; everything else is hardlink-copied.

Usage (server):
    python -B scripts/smooth_video_preds.py \
        --src-dir outputs/cv/K3_baseline/fold0/predict \
        --dst-dir outputs/cv/K3_video_smooth/fold0/predict_median3 \
        --op median3 \
        --val-manifest outputs/uusivc2026_fixed/manifests_cv3/fold0/val_entries.json
Pairs are then compared with:
    python -B evaluate_ab.py <src>/metrics_records.json <dst>/metrics_records.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_submission import score_submission_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal smoothing for video_seg prediction npz files.")
    parser.add_argument("--src-dir", type=str, required=True, help="Source predict directory (read-only).")
    parser.add_argument("--dst-dir", type=str, required=True, help="Destination directory (created).")
    parser.add_argument("--op", choices=["none", "median", "vote"], default="median",
                        help="none=sanity no-op rewrite, median=sliding-window majority, vote=global majority.")
    parser.add_argument("--window", type=int, default=3, help="Odd sliding-window size for the median op.")
    parser.add_argument("--val-manifest", type=str, required=True, help="val_entries.json used for offline scoring.")
    parser.add_argument("--tolerance", type=int, default=1, help="NSD surface tolerance tau.")
    parser.add_argument("--no-score", action="store_true", help="Skip offline scoring (files only).")
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing an existing dst directory.")
    return parser.parse_args()


def _bin_u255(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def split_unique_sequence(masks: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], List[List[str]]]:
    """Merge the block-constant fnum dict into a unique mask sequence.

    Returns (unique_masks, key_groups) where key_groups[i] lists every original
    string key that maps to unique_masks[i]; groups follow ascending int(key).
    """
    ordered = sorted(masks.keys(), key=lambda k: int(k))
    unique: List[np.ndarray] = []
    groups: List[List[str]] = []
    for key in ordered:
        mask = masks[key]
        if unique and np.array_equal(unique[-1], mask):
            groups[-1].append(key)
        else:
            unique.append(mask)
            groups.append([key])
    return unique, groups


def smooth_median(unique: List[np.ndarray], window: int) -> List[np.ndarray]:
    """Sliding-window median over the unique sequence (binary majority for 0/255)."""
    if window % 2 == 0 or window < 1:
        raise ValueError(f"--window must be a positive odd number, got {window}")
    radius = window // 2
    n = len(unique)
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        stack = np.stack(unique[lo:hi], axis=0)
        smoothed.append(np.rint(np.median(stack, axis=0)).astype(np.uint8))
    return smoothed


def smooth_vote(unique: List[np.ndarray]) -> List[np.ndarray]:
    """Per-pixel strict majority across all unique masks (ties -> background)."""
    stack = np.stack(unique, axis=0)
    majority = (stack.sum(axis=0) * 2) > len(unique)
    return [majority.astype(np.uint8) * 255 for _ in unique]


def smooth_video_masks(masks: Dict[str, np.ndarray], op: str, window: int) -> Dict[str, np.ndarray]:
    binary = {key: _bin_u255(mask) for key, mask in masks.items()}
    unique, groups = split_unique_sequence(binary)
    if op == "none" or len(unique) == 1:
        smoothed_unique = unique
    elif op == "median":
        smoothed_unique = smooth_median(unique, window)
    elif op == "vote":
        smoothed_unique = smooth_vote(unique)
    else:
        raise ValueError(f"unknown op: {op}")
    out: Dict[str, np.ndarray] = {}
    for mask, keys in zip(smoothed_unique, groups):
        for key in keys:
            out[key] = mask
    assert set(out.keys()) == set(masks.keys()), "key set must be preserved"
    return out


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    src, dst = Path(args.src_dir), Path(args.dst_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"src dir not found: {src}")
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"dst already exists (use --overwrite): {dst}")

    # Baseline-integrity guard: snapshot src's scoring records (hash + content)
    # BEFORE any write, so the delta print never reads a file this run has
    # clobbered and any write-through to src is detected before exit.
    guard_names = ("metrics.json", "metrics_records.json")
    guard_hash = {n: _sha1(src / n) for n in guard_names if (src / n).exists()}
    src_metrics_path = src / "metrics.json"
    src_metrics = json.loads(src_metrics_path.read_text(encoding="utf-8")) if src_metrics_path.exists() else None

    # Hardlink-copy keeps disk cost at ~zero for unchanged files; video npz files
    # are unlinked before rewriting so the shared inode (src) is never modified.
    if not dst.exists():
        shutil.copytree(src, dst, copy_function=os.link)

    video_npzs = sorted((dst / "video_seg").rglob("*.npz")) if (dst / "video_seg").is_dir() else []
    changed = 0
    multi_unique = 0
    for npz_path in video_npzs:
        with np.load(npz_path, allow_pickle=True) as z:
            masks = {str(k): np.asarray(v) for k, v in z["fnum_mask"].item().items()}
        unique, _ = split_unique_sequence({k: _bin_u255(v) for k, v in masks.items()})
        multi_unique += int(len(unique) > 1)
        smoothed = smooth_video_masks(masks, args.op, args.window)
        identical = all(np.array_equal(smoothed[k], _bin_u255(masks[k])) for k in masks)
        if not identical:
            changed += 1
        npz_path.unlink()
        np.savez_compressed(npz_path, fnum_mask=smoothed)

    print(f"[smooth] op={args.op} window={args.window} videos={len(video_npzs)} "
          f"multi-unique={multi_unique} changed={changed}")

    if not args.no_score:
        with open(args.val_manifest, "r", encoding="utf-8") as f:
            val_entries = json.load(f)
        metrics = score_submission_dir(dst, val_entries, tolerance=args.tolerance, collect_records=True)
        records = metrics.pop("records")
        # copytree(os.link) hardlinked these to src's scoring records — unlink
        # first so write_json's in-place open("w") cannot clobber src through
        # the shared inode (that silently zeroed every src-vs-dst delta once).
        for name in ("metrics.json", "metrics_records.json"):
            p = dst / name
            if p.exists() or p.is_symlink():
                p.unlink()
        write_json(dst / "metrics.json", metrics)
        write_json(dst / "metrics_records.json", records)

        if src_metrics is not None:
            print("[smooth] per-task delta (dst - src):")
            for task, cur in metrics.get("per_task", {}).items():
                base = src_metrics.get("per_task", {}).get(task, {})
                delta = cur.get("score", 0.0) - base.get("score", 0.0)
                print(f"  {task}: {base.get('score', 0.0):.6f} -> {cur.get('score', 0.0):.6f} (delta {delta:+.6f})")
            overall_delta = metrics.get("overall_score", 0.0) - src_metrics.get("overall_score", 0.0)
            print(f"  overall: {src_metrics.get('overall_score', 0.0):.6f} -> "
                  f"{metrics.get('overall_score', 0.0):.6f} (delta {overall_delta:+.6f})")

    # Verify src was never modified — write-through via a hardlink inode is the
    # silent failure mode this guard exists for (2026-09-01 incident).
    for name, want in guard_hash.items():
        if _sha1(src / name) != want:
            print(f"[smooth] ERROR: src scoring record modified during the run: {src / name}", file=sys.stderr)
            raise SystemExit(2)
    print(f"[smooth] src integrity OK ({len(guard_hash)} files verified)")


if __name__ == "__main__":
    main()
