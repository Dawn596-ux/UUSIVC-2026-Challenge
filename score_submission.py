"""Score a generated submission directory against labeled local-val ground truth.

Reads the official submission layout (``classification.json`` + ``image_seg``/
``ceus_seg``/``video_seg`` mask files) produced by ``predict.py``/``predict_val.py``
and replicates the official metric conventions:

- segmentation per-sample score = ``0.7 * DSC + 0.3 * NSD(tau)`` (task-level score is
  the sample-pooled mean);
- classification per-dataset score = ``0.5 * Acc + 0.5 * AUC``;
- overall score = mean of the (up to) 5 task scores.

Because scoring runs on the *generated files*, inference-time changes (TTA,
morphological post-processing, prediction flipping) are reflected exactly as the
official scorer would see them — unlike loader-based eval, which re-runs the model.

Ground-truth conventions mirror the training-side datasets exactly:
- ``image_seg`` GT: PNG mask, RGB collapsed by channel-max, binarized at ``>0``;
- ``video_seg`` GT: npz with ``fnum_mask`` dict (paired by frame index) or ``mask``
  array (paired positionally against sorted prediction keys);
- ``ceus_seg`` GT: npz ``mask`` array; the official GT is frame 0 at original size,
  and the prediction is the restored full-size mask (same as ``predict_ceus_seg``);
- classification GT: ``class_label_index`` keyed by ``sample_id``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from utils.metrics import compute_binary_seg_score_np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_gt_mask_binary(path: Path) -> np.ndarray:
    """Load an image-seg GT mask: collapse RGB by channel-max, binarize at >0."""
    mask = np.asarray(Image.open(path).convert("RGB"))
    if mask.ndim == 3:
        mask = mask[..., :3].max(axis=-1)
    return (mask > 0).astype(np.uint8) * 255


def load_video_gt_masks(path: Path) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """Load a video-seg GT annotation npz.

    Returns ``(fnum_dict, stack)``: exactly one is non-None.
    - ``fnum_dict``: {frame_key_str -> 0/255 mask} when GT uses ``fnum_mask``;
    - ``stack``: [T, H, W] 0/255 array when GT uses ``mask``.
    """
    ann = np.load(path, allow_pickle=True)
    if "fnum_mask" in ann.files:
        frame_mask_dict = ann["fnum_mask"].item()
        return {str(k): (np.asarray(v) > 0).astype(np.uint8) * 255 for k, v in frame_mask_dict.items()}, None
    if "mask" in ann.files:
        masks = ann["mask"]
        if masks.ndim == 2:
            masks = masks[None, ...]
        return None, (masks > 0).astype(np.uint8) * 255
    raise KeyError(f"No supported mask key found in {path}; expected 'mask' or 'fnum_mask'")


def resize_mask_nearest(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = int(hw[0]), int(hw[1])
    if mask.shape[:2] == (h, w):
        return mask
    resized = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8) * 255


def dataset_of(entry: Dict[str, Any]) -> str:
    return str(entry.get("dataset_name") or entry.get("organ") or entry.get("task"))


def entry_key(entry: Dict[str, Any]) -> str:
    sample_id = entry.get("sample_id")
    if sample_id:
        return str(sample_id)
    stem = Path(str(entry.get("input_path_relative", ""))).stem
    return f"{entry.get('task')}/{stem}"


def score_seg_records(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {"n": 0, "dsc": 0.0, "nsd": 0.0, "score": 0.0}
    return {
        "n": len(records),
        "dsc": float(np.mean([r["dsc"] for r in records])),
        "nsd": float(np.mean([r["nsd"] for r in records])),
        "score": float(np.mean([r["score"] for r in records])),
    }


def binary_auc_from_scores(pos_scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC on positive-class scores (Mann-Whitney, ties = 0.5)."""
    labels = np.asarray(labels).astype(np.int64)
    pos = pos_scores[labels == 1]
    neg = pos_scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    comparisons = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(pos) * len(neg)))


def score_submission_dir(
    pred_dir: Path,
    val_entries: List[Dict[str, Any]],
    tolerance: int = 1,
) -> Dict[str, Any]:
    seg_records: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    task_records: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    cls_by_dataset: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"gt": [], "prob": [], "pred": []})

    classification_path = pred_dir / "classification.json"
    classification = load_json(classification_path) if classification_path.exists() else {}

    matched_cls = 0
    missing_cls = 0
    for entry in val_entries:
        task = entry.get("task")
        dataset = dataset_of(entry)
        root = Path(entry["_root"]) if "_root" in entry else None

        if task in {"image_seg", "video_seg", "ceus_seg"}:
            pred_rel = submission_rel_path(entry)
            pred_path = pred_dir / pred_rel
            if not pred_path.exists():
                raise FileNotFoundError(f"Prediction file missing: {pred_path}")
            if task == "image_seg":
                gt_path = root / str(entry["target_path_relative"]).replace("\\", "/").lstrip("/")
                pred = np.asarray(Image.open(pred_path))
                gt = load_gt_mask_binary(gt_path)
                pred = resize_mask_nearest(pred, gt.shape[:2])
                record = compute_binary_seg_score_np(pred, gt, tolerance=tolerance)
            elif task == "video_seg":
                pred_npz = np.load(pred_path, allow_pickle=True)
                pred_masks = {str(k): (np.asarray(v) > 0).astype(np.uint8) * 255 for k, v in pred_npz["fnum_mask"].item().items()}
                gt_fnum, gt_stack = load_video_gt_masks(root / str(entry["target_path_relative"]).replace("\\", "/").lstrip("/"))
                pairs: List[Tuple[np.ndarray, np.ndarray]] = []
                if gt_fnum is not None:
                    for key, gt_mask in gt_fnum.items():
                        if key in pred_masks:
                            pairs.append((pred_masks[key], gt_mask))
                else:
                    ordered_keys = sorted(pred_masks.keys(), key=lambda x: int(x))
                    for idx, key in enumerate(ordered_keys):
                        if idx >= gt_stack.shape[0]:
                            break
                        pairs.append((pred_masks[key], gt_stack[idx]))
                if not pairs:
                    raise ValueError(f"No aligned frames between prediction and GT for {pred_path}")
                frame_records = [compute_binary_seg_score_np(p, resize_mask_nearest(g, p.shape[:2]), tolerance=tolerance) for p, g in pairs]
                record = {k: float(np.mean([fr[k] for fr in frame_records])) for k in ("dsc", "nsd", "score")}
            else:  # ceus_seg
                pred_npz = np.load(pred_path, allow_pickle=True)
                pred = (np.asarray(pred_npz["mask"]) > 0).astype(np.uint8) * 255
                _, gt_stack = load_video_gt_masks(root / str(entry["target_path_relative"]).replace("\\", "/").lstrip("/"))
                gt = gt_stack[0]
                pred = resize_mask_nearest(pred, gt.shape[:2])
                record = compute_binary_seg_score_np(pred, gt, tolerance=tolerance)

            seg_records[f"{task}/{dataset}"].append(record)
            task_records[task].append(record)

        elif task in {"image_cls", "ceus_cls"}:
            key = entry_key(entry)
            label = entry.get("class_label_index")
            if key not in classification:
                missing_cls += 1
                continue
            matched_cls += 1
            pred = classification[key]
            bucket = cls_by_dataset[f"{task}/{dataset}"]
            bucket["gt"].append(int(label))
            bucket["pred"].append(int(pred["prediction"]))
            probs = list(pred.get("probability") or [])
            bucket["prob"].append(float(probs[1]) if len(probs) > 1 else float(probs[0]) if probs else 0.5)

    per_dataset: Dict[str, Dict[str, Any]] = {}
    for name, records in seg_records.items():
        per_dataset[name] = score_seg_records(records)
    for name, bucket in cls_by_dataset.items():
        gt = np.asarray(bucket["gt"], dtype=np.int64)
        pred = np.asarray(bucket["pred"], dtype=np.int64)
        prob = np.asarray(bucket["prob"], dtype=np.float64)
        acc = float((pred == gt).mean()) if len(gt) else 0.0
        auc = binary_auc_from_scores(prob, gt) if len(gt) else 0.5
        per_dataset[name] = {"n": int(len(gt)), "acc": acc, "auc": auc, "score": 0.5 * (acc + auc)}

    per_task: Dict[str, Dict[str, Any]] = {}
    for task, records in task_records.items():
        per_task[task] = score_seg_records(records)
    for task in {"image_cls", "ceus_cls"}:
        buckets = [v for k, v in per_dataset.items() if k.startswith(f"{task}/")]
        if buckets:
            n = int(sum(b["n"] for b in buckets))
            acc = sum(b["acc"] * b["n"] for b in buckets) / max(n, 1)
            auc = sum(b["auc"] * b["n"] for b in buckets) / max(n, 1)
            per_task[task] = {"n": n, "acc": acc, "auc": auc, "score": 0.5 * (acc + auc)}

    overall = float(np.mean([v["score"] for v in per_task.values()])) if per_task else 0.0

    return {
        "tolerance": tolerance,
        "classification_matched": matched_cls,
        "classification_missing": missing_cls,
        "per_dataset": per_dataset,
        "per_task": per_task,
        "overall_score": overall,
    }


def submission_rel_path(entry: Dict[str, Any]) -> Path:
    """Mirror predict.py's output path derivation for seg entries."""
    from predict import output_rel

    return output_rel(entry)


def print_table(metrics: Dict[str, Any]) -> None:
    print(f"\n{'dataset':<45} {'n':>5} {'dsc':>7} {'nsd':>7} {'acc':>7} {'auc':>7} {'score':>7}")
    for name, m in sorted(metrics["per_dataset"].items()):
        print(
            f"{name:<45} {m['n']:>5} "
            f"{m.get('dsc', float('nan')):>7.4f} {m.get('nsd', float('nan')):>7.4f} "
            f"{m.get('acc', float('nan')):>7.4f} {m.get('auc', float('nan')):>7.4f} "
            f"{m['score']:>7.4f}"
        )
    print(f"\n{'TASK':<45} {'n':>5} {'score':>7}")
    for task, m in sorted(metrics["per_task"].items()):
        print(f"{task:<45} {m['n']:>5} {m['score']:>7.4f}")
    print(f"\noverall_score = {metrics['overall_score']:.4f}  (tau={metrics['tolerance']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a submission directory against labeled local-val GT.")
    parser.add_argument("--pred-dir", type=str, required=True, help="Submission directory (classification.json + seg files).")
    parser.add_argument("--val-manifest", type=str, required=True, help="Path to val_entries.json (labeled local val split).")
    parser.add_argument("--tolerance", type=int, default=1, help="NSD surface tolerance tau (default 1).")
    parser.add_argument("--output-json", type=str, default=None, help="Metrics output path (default <pred-dir>/metrics.json).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir).resolve()
    val_entries = load_json(Path(args.val_manifest))
    metrics = score_submission_dir(pred_dir, val_entries, tolerance=args.tolerance)
    out_path = Path(args.output_json).resolve() if args.output_json else pred_dir / "metrics.json"
    write_json(out_path, metrics)
    print(f"[Info] Metrics written to {out_path}")
    print_table(metrics)


if __name__ == "__main__":
    main()
