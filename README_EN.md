# UUSIVC-2026-Challenge

This repository is a baseline for the **Universal Ultrasound Image & Video Analysis Challenge 2026**. It provides one possible implementation path for participating in the challenge. Participants are encouraged to explore their own approaches and develop more competitive, state-of-the-art research algorithms.

## Expected Data Layout

Pass the public data root to the scripts. The resolver expects the released `TRAIN` and `VAL` packages under one root:

```text
<root>/
  TRAIN/
    Challenge_Data_Public/
    Challenge_Data_Private_v2_fully_anonymized/
      Train/
    dataset_json_fingerprints_v4/
      public_all_ground_truth.json
      private_train_ground_truth.json
  VAL/
    Challenge_Data_Private_v2_fully_anonymized/
      Val/
    dataset_json_fingerprints_v4/
      private_val_for_participants.json
```

The released `TRAIN` package includes labels and segmentation masks. The released `VAL` package contains participant metadata and input files only; it does not include labels or segmentation masks. If no path is supplied, the code checks `UUSIVC2026_DATA_ROOT`. Otherwise pass `--data-root` explicitly.

## Install

A minimal environment needs Python. Install dependencies with:

```powershell
pip install -r requirements.txt
```

The pretrained [Swin-Tiny checkpoint](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) is expected at:

```text
pretrained_ckpt/swin_tiny_patch4_window7_224.pth
```

Model checkpoints and generated results are written under `outputs/` at runtime. The `outputs/` directory is not required in the public repository and should not be uploaded.

## Train

Training uses the labeled `TRAIN` package. By default, the dataloader builds a deterministic local holdout from `TRAIN` for per-epoch validation and best-checkpoint selection, because the released `VAL` package has no labels.

Stage 1 trains segmentation tasks with a local holdout:

```powershell
python -B train.py --stage stage1_seg --data-root "<UUSIVC2026_DATA_ROOT>"
```

Stage 2 trains classification tasks and initializes from a Stage 1 checkpoint. Choose the checkpoint explicitly with `--init-checkpoint`.

If Stage 1 was trained with local validation, initialize Stage 2 from the selected best Stage 1 checkpoint:

```powershell
python -B train.py `
  --stage stage2_cls `
  --data-root "<UUSIVC2026_DATA_ROOT>" `
  --init-checkpoint outputs\stage1_seg\best_checkpoints\best_stage1_seg_rank1.pth
```

For final full-data training, use all labeled `TRAIN` data and skip validation. If Stage 1 was also trained with `--full-train`, initialize from its latest checkpoint:

```powershell
python -B train.py `
  --stage stage2_cls `
  --data-root "<UUSIVC2026_DATA_ROOT>" `
  --init-checkpoint outputs\stage1_seg\latest_stage1_seg.pth `
  --full-train
```

In full-train mode, the trainer saves `latest_stage*_*.pth` checkpoints only. It does not compute validation metrics or save best checkpoints.

The local holdout ratio is controlled by:

```yaml
data:
  local_val_fraction: 0.1
  split_seed: 2024
train:
  require_validation: true
```

You can override the holdout ratio from the command line:

```powershell
python -B train.py `
  --stage stage2_cls `
  --data-root "<UUSIVC2026_DATA_ROOT>" `
  --local-val-fraction 0.05
```

## Evaluate Locally

`test.py` evaluates the labeled local holdout split created from `TRAIN`. It does not evaluate the released `VAL` package, because `VAL` labels and masks are not public. Use the same `--local-val-fraction` value as training if you override it.

```powershell
python -B test.py `
  --stage stage2_cls `
  --checkpoint outputs\stage2_cls\best_checkpoints\best_stage2_cls_rank1.pth `
  --split val `
  --data-root "<UUSIVC2026_DATA_ROOT>"
```

## Generate Validation Submission

`predict.py` is the model inference entry for the released `VAL` package. It loads one full-model checkpoint, runs inference on participant metadata, then writes the official upload directory and zip. By default it uses the best Stage 2 checkpoint, and `--checkpoint` can point to any trained full-model weight for score comparison.

```powershell
python -B predict.py `
  --data-root "<UUSIVC2026_DATA_ROOT>" `
  --phase val `
  --checkpoint outputs\stage2_cls\best_checkpoints\best_stage2_cls_rank1.pth `
  --output-dir outputs\submission_val `
  --zip-path outputs\submission_val.zip
```

The upload zip contains exactly the official submission files:

```text
classification.json
image_seg/<Dataset>/masks/*.png
ceus_seg/<Dataset>/annotations/*.npz
video_seg/<Dataset>/annotations/*.npz
```

The output directory also includes `submission_summary.json` for local inspection; it is not added to the zip.

## Metrics Reference

Standalone reference implementations are provided in `metrics_reference/`:

```text
metrics_reference/segmentation_metrics_reference.py
metrics_reference/classification_metrics_reference.py
```

These files are for participant reference only and are not imported by the training or inference pipeline. The active project implementations are in `utils/metrics.py` and `utils/auc_utils.py`.

## Code Map

```text
configs/                      Compact training configs
datasets/uusivc2026_paths.py  Fixed challenge data protocol
datasets/                     Image/video dataset classes and loaders
metrics_reference/            Standalone metric reference implementations
models/                       Unified image/video model
trainers/                     Stage 1 segmentation and Stage 2 classification loops
train.py                      Training entry
test.py                       Local holdout evaluation entry
predict.py                    Competition-format prediction entry
utils/                        Metrics, checkpointing, logging, visualization
```


Dataset management is not a user-facing task. This baseline assumes the official UUSIVC2026 `TRAIN`/`VAL` package layout and keeps path expansion inside `datasets/uusivc2026_paths.py`.

This repository is built upon the [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) codebase. We thank the authors for making their work publicly available.
