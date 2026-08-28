"""Generate competition-format predictions + offline scores from the local val split.

Unlike ``predict.py`` (which reads the *unlabeled* public VAL package), this module
reads the *labeled* local validation split materialized by ``write_fixed_manifests``
(``val_entries.json``) and produces:

1. the official submission layout (``classification.json`` + ``image_seg``/``ceus_seg``/
   ``video_seg`` masks), identical in format to ``predict.py``;
2. an optional offline score (reuses ``test.py``'s evaluation on the same val split),
   so you can estimate leaderboard metrics without consuming Codabench submissions.

The split is deterministic (same ``local_val_fraction``/``split_seed``/``split_mode`` as
training), so the entries here match exactly what the training loop held out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from datasets.uusivc2026_paths import expand_fixed_uusivc_data_cfg, resolve_data_root
from predict import (
    build_ceus_processor,
    classification_key,
    load_model,
    load_yaml,
    predict_ceus_cls,
    predict_ceus_seg,
    predict_image_cls,
    predict_image_seg,
    predict_video_seg,
    resolve_checkpoint_reference,
    resolve_device,
    resolve_submission_checkpoint,
    write_json,
    write_submission_zip,
)
from score_submission import print_table, score_submission_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate competition-format predictions + offline scores from the local val split."
    )
    parser.add_argument("--config", type=str, default="configs/stage2_cls.yaml",
                        help="Model config path. Defaults to the final Stage-2 config.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Full-model checkpoint path or directory. If omitted, the best Stage-2 checkpoint is used.")
    parser.add_argument("--secondary-checkpoint", type=str, default=None,
                        help="Optional second full-model checkpoint used for video_seg and image_cls (mixed-checkpoint recipe).")
    parser.add_argument("--which", choices=["best", "latest"], default="best",
                        help="Checkpoint choice when --checkpoint is not provided.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Inference/eval device.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="UUSIVC2026 public data root containing TRAIN/ and VAL/ packages. If omitted, uses UUSIVC2026_DATA_ROOT.")
    parser.add_argument("--val-manifest", type=str, default=None,
                        help="Path to val_entries.json. If omitted, the split is (re)generated from the config's local_val_fraction/split_seed/split_mode.")
    parser.add_argument("--tolerance", type=int, default=1,
                        help="NSD surface tolerance tau used by the offline scorer (default 1).")
    parser.add_argument("--stage", type=str, default=None, choices=["stage1_seg", "stage2_cls"],
                        help="Deprecated: file-based scoring covers all tasks; kept for backward compatibility.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for the submission files and scores.")
    parser.add_argument("--zip-path", type=str, default=None,
                        help="Optional upload-ready zip path (official submission files only).")
    parser.add_argument("--no-zip", action="store_true", help="Write the submission directory only.")
    parser.add_argument("--no-score", action="store_true", help="Skip offline scoring (files only).")
    return parser.parse_args()


def load_val_entries(args: argparse.Namespace, cfg: Dict[str, Any]) -> tuple[list[Dict[str, Any]], Path]:
    """Return the val entries (full metadata) and the path they were read from.

    Either read a pre-generated ``val_entries.json`` (``--val-manifest``) or materialize
    the split deterministically from the config and read the freshly written file.
    """
    if args.val_manifest:
        path = Path(args.val_manifest)
        if not path.exists():
            raise FileNotFoundError(f"val_entries.json not found: {path}")
    else:
        expanded_data = expand_fixed_uusivc_data_cfg(cfg["data"])
        manifest_dir = Path(expanded_data.get("manifest_cache_dir", "./outputs/uusivc2026_fixed/manifests"))
        path = manifest_dir / "val_entries.json"
        if not path.exists():
            raise FileNotFoundError(f"val_entries.json not generated at {path}")

    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return entries, path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.data_root:
        cfg.setdefault("data", {})["data_root"] = args.data_root

    data_root = resolve_data_root(cfg["data"].get("data_root") or args.data_root)
    device = resolve_device(args.device)
    checkpoint_path = resolve_submission_checkpoint(cfg, args.checkpoint, args.which)

    val_entries, val_manifest_path = load_val_entries(args, cfg)

    out_dir = Path(args.output_dir or "outputs/predict_val").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Device: {device}")
    print(f"[Info] Data root: {data_root}")
    print(f"[Info] Config: {args.config}")
    print(f"[Info] Checkpoint: {checkpoint_path}")
    print(f"[Info] Val entries: {len(val_entries)} (from {val_manifest_path})")

    model = load_model(cfg, checkpoint_path, device)
    secondary_model = None
    if args.secondary_checkpoint:
        secondary_path = resolve_checkpoint_reference(args.secondary_checkpoint, prefix=None)
        secondary_model = load_model(cfg, secondary_path, device)
        print(f"[Info] Secondary Checkpoint: {secondary_path}")

    ceus_processor = build_ceus_processor(cfg.get("data", {}))
    num_frames = int(cfg.get("data", {}).get("num_frames", 10))

    # 混合 checkpoint：video_seg / image_cls 用 secondary（旧 encoder 未漂移），其余用主 checkpoint
    def _model_for(task: str):
        if secondary_model is not None and task in {"video_seg", "image_cls"}:
            return secondary_model
        return model

    # 1) Generate competition-format prediction files.
    classification: Dict[str, Dict[str, Any]] = {}
    counts = {"classification": 0, "segmentation": 0}
    for entry in tqdm(val_entries, desc="Predict"):
        task = entry.get("task")
        phase_root = Path(entry["_root"])
        if task == "image_seg":
            predict_image_seg(_model_for(task), entry, phase_root, out_dir, device)
            counts["segmentation"] += 1
        elif task == "video_seg":
            predict_video_seg(_model_for(task), entry, phase_root, out_dir, device, num_frames)
            counts["segmentation"] += 1
        elif task == "ceus_seg":
            predict_ceus_seg(_model_for(task), entry, phase_root, out_dir, device, ceus_processor)
            counts["segmentation"] += 1
        elif task == "image_cls":
            classification[classification_key(entry)] = predict_image_cls(_model_for(task), entry, phase_root, device)
            counts["classification"] += 1
        elif task == "ceus_cls":
            classification[classification_key(entry)] = predict_ceus_cls(_model_for(task), entry, phase_root, device, num_frames)
            counts["classification"] += 1

    write_json(out_dir / "classification.json", classification)

    # 2) Offline scoring on the *generated files* (inference-time changes are
    #    measured exactly as the official scorer would see them).
    metrics: Dict[str, Any] = {}
    if not args.no_score:
        metrics = score_submission_dir(out_dir, val_entries, tolerance=args.tolerance)
        write_json(out_dir / "metrics.json", metrics)
        print_table(metrics)

    # 3) Zip the official submission files.
    zip_path = None
    if not args.no_zip:
        zip_path = Path(args.zip_path).resolve() if args.zip_path else out_dir.with_suffix(".zip")
        write_submission_zip(out_dir, zip_path)

    summary = {
        "data_root": str(data_root),
        "val_manifest": str(val_manifest_path),
        "output_dir": str(out_dir),
        "zip_path": str(zip_path) if zip_path else None,
        "config": args.config,
        "checkpoint": checkpoint_path,
        "secondary_checkpoint": args.secondary_checkpoint,
        "tolerance": args.tolerance,
        "classification_samples": counts["classification"],
        "classification_keys": len(classification),
        "segmentation_files": counts["segmentation"],
        "scored": not args.no_score,
        "metrics": metrics,
        "format": "competition submission (from labeled local val split)",
    }
    write_json(out_dir / "predict_val_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
