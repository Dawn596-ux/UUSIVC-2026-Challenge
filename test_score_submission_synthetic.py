"""Synthetic self-check for score_submission.py (no real dataset needed).

Builds a fake submission directory + fake val_entries.json with known GT, then
asserts the scorer returns the expected DSC/NSD/Acc/AUC/overall numbers.
Run: python -B test_score_submission_synthetic.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from score_submission import (
    binary_auc_from_scores,
    load_video_gt_masks,
    score_submission_dir,
)


def make_dir_shape(h: int, w: int, r: int) -> np.ndarray:
    """Solid disk GT at original resolution."""
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= r**2).astype(np.uint8) * 255


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="score_selfcheck_"))
    root = tmp / "data_root"
    pred = tmp / "pred"
    (root / "masks").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (pred / "masks").mkdir(parents=True)
    (pred / "annotations").mkdir(parents=True)

    # --- image_seg: GT disk 64x64 r=10; perfect pred -> dsc=nsd=1; empty pred -> 0 ---
    # Predictions for labeled val entries are written at output_rel(entry) == the
    # GT-relative path (target_path_relative takes precedence in output_rel).
    gt = make_dir_shape(64, 64, 10)
    Image.fromarray(gt).save(root / "masks" / "seg_mask_a.png")
    Image.fromarray(gt).save(pred / "masks" / "seg_mask_a.png")
    Image.fromarray(np.zeros((64, 64), np.uint8)).save(pred / "masks" / "seg_mask_b.png")
    Image.fromarray(gt).save(root / "masks" / "seg_mask_b.png")

    # --- video_seg: GT fnum_mask with 2 frames; pred perfect on frame 10, empty on 20 ---
    f10 = make_dir_shape(50, 50, 8)
    f20 = make_dir_shape(50, 50, 8)
    np.savez_compressed(
        root / "annotations" / "seg_annotation_v.npz",
        fnum_mask={"10": f10, "20": f20},
    )
    np.savez_compressed(
        pred / "annotations" / "seg_annotation_v.npz",
        fnum_mask={"10": f10, "20": np.zeros((50, 50), np.uint8)},
    )

    # --- ceus_seg: GT mask stack; official GT = frame 0; pred perfect ---
    g0 = make_dir_shape(40, 40, 6)
    g1 = make_dir_shape(40, 40, 6)
    np.savez_compressed(root / "annotations" / "seg_annotation_c.npz", mask=np.stack([g0, g1]))
    np.savez_compressed(pred / "annotations" / "seg_annotation_c.npz", mask=g0)

    # --- classification: 4 binary samples, perfect AUC=1, acc=0.75 (one wrong) ---
    cls = {
        "s1": {"prediction": 1, "probability": [0.1, 0.9]},
        "s2": {"prediction": 0, "probability": [0.8, 0.2]},
        "s3": {"prediction": 1, "probability": [0.2, 0.8]},
        "s4": {"prediction": 1, "probability": [0.4, 0.6]},  # wrong (gt=0)
    }
    (pred / "classification.json").write_text(json.dumps(cls), encoding="utf-8")

    entries = [
        {"task": "image_seg", "dataset_name": "DS_IMG", "sample_id": "a",
         "input_path_relative": "imgs/seg_img_a.png", "target_path_relative": "masks/seg_mask_a.png",
         "_root": str(root)},
        {"task": "image_seg", "dataset_name": "DS_IMG", "sample_id": "b",
         "input_path_relative": "imgs/seg_img_b.png", "target_path_relative": "masks/seg_mask_b.png",
         "_root": str(root)},
        {"task": "video_seg", "dataset_name": "DS_VID", "sample_id": "v",
         "input_path_relative": "videos/seg_video_v.npy", "target_path_relative": "annotations/seg_annotation_v.npz",
         "_root": str(root)},
        {"task": "ceus_seg", "dataset_name": "DS_CEUS", "sample_id": "c",
         "input_path_relative": "videos/seg_video_c.npy", "target_path_relative": "annotations/seg_annotation_c.npz",
         "_root": str(root)},
        {"task": "image_cls", "dataset_name": "DS_CLS", "sample_id": "s1", "class_label_index": 1, "_root": str(root)},
        {"task": "image_cls", "dataset_name": "DS_CLS", "sample_id": "s2", "class_label_index": 0, "_root": str(root)},
        {"task": "image_cls", "dataset_name": "DS_CLS", "sample_id": "s3", "class_label_index": 1, "_root": str(root)},
        {"task": "image_cls", "dataset_name": "DS_CLS", "sample_id": "s4", "class_label_index": 0, "_root": str(root)},
    ]

    metrics = score_submission_dir(pred, entries, tolerance=1)

    # image_seg: perfect -> 1.0 ; empty-vs-disk -> dsc=0, nsd=0
    img = metrics["per_dataset"]["image_seg/DS_IMG"]
    assert img["n"] == 2
    assert abs(img["dsc"] - 0.5) < 1e-6, img
    assert abs(img["nsd"] - 0.5) < 1e-6, img
    assert abs(img["score"] - 0.5) < 1e-6, img

    # video_seg: frame10 perfect (1.0) + frame20 empty (0.0) -> 0.5 each metric
    vid = metrics["per_dataset"]["video_seg/DS_VID"]
    assert vid["n"] == 1
    assert abs(vid["dsc"] - 0.5) < 1e-6, vid
    assert abs(vid["score"] - 0.5) < 1e-6, vid

    # ceus_seg: pred == GT frame0 -> 1.0
    ceus = metrics["per_dataset"]["ceus_seg/DS_CEUS"]
    assert ceus["n"] == 1
    assert abs(ceus["dsc"] - 1.0) < 1e-6, ceus
    assert abs(ceus["score"] - 1.0) < 1e-6, ceus

    # classification: acc = 3/4 = 0.75; perfect separation -> auc = 1.0
    cls_m = metrics["per_dataset"]["image_cls/DS_CLS"]
    assert cls_m["n"] == 4
    assert abs(cls_m["acc"] - 0.75) < 1e-6, cls_m
    assert abs(cls_m["auc"] - 1.0) < 1e-6, cls_m
    assert abs(cls_m["score"] - 0.875) < 1e-6, cls_m

    # per_task + overall
    assert abs(metrics["per_task"]["image_seg"]["score"] - 0.5) < 1e-6
    assert abs(metrics["per_task"]["image_cls"]["score"] - 0.875) < 1e-6
    expected_overall = np.mean([0.5, 0.5, 1.0, 0.875])
    assert abs(metrics["overall_score"] - expected_overall) < 1e-6, metrics["overall_score"]

    # AUC edge cases
    assert abs(binary_auc_from_scores(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0])) - 1.0) < 1e-9
    assert abs(binary_auc_from_scores(np.array([0.1, 0.2, 0.9, 0.8]), np.array([1, 1, 0, 0])) - 0.0) < 1e-9
    assert abs(binary_auc_from_scores(np.array([0.5, 0.5]), np.array([1, 0])) - 0.5) < 1e-9  # all ties
    assert binary_auc_from_scores(np.array([0.5]), np.array([1])) == 0.5  # single-class fallback

    # load_video_gt_masks: both branches
    fnum, stack = load_video_gt_masks(root / "annotations" / "seg_annotation_v.npz")
    assert fnum is not None and stack is None and set(fnum.keys()) == {"10", "20"}
    fnum2, stack2 = load_video_gt_masks(root / "annotations" / "seg_annotation_c.npz")
    assert fnum2 is None and stack2 is not None and stack2.shape == (2, 40, 40)

    print("[PASS] score_submission synthetic self-check: all assertions OK")
    print(json.dumps(metrics, indent=2)[:1200])


if __name__ == "__main__":
    main()
