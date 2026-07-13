# UUSIVC-2026-Challenge

本仓库是 **Universal Ultrasound Image & Video Analysis Challenge 2026（通用超声图像与视频分析挑战赛 2026）** 的基线实现，为参赛者提供一种可行的实现路径。鼓励参赛者探索自己的方法，开发更具竞争力的前沿研究算法。

## 预期数据布局

将公开数据根目录传递给脚本。解析器期望发布的 `TRAIN` 和 `VAL` 包位于同一根目录下：

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

发布的 `TRAIN` 包包含标签和分割掩码。发布的 `VAL` 包仅包含参赛者元数据和输入文件，不包含标签或分割掩码。如果未提供路径，代码会检查环境变量 `UUSIVC2026_DATA_ROOT`；否则请通过 `--data-root` 显式指定。

## 安装

最小环境需要 Python。使用以下命令安装依赖：

```powershell
pip install -r requirements.txt
```

预训练的 [Swin-Tiny 检查点](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) 应放置在：

```text
pretrained_ckpt/swin_tiny_patch4_window7_224.pth
```

模型检查点和生成的结果在运行时会写入 `outputs/` 目录。`outputs/` 目录不要求在公开仓库中存在，也不应上传。

## 训练

训练使用带标签的 `TRAIN` 包。默认情况下，数据加载器会从 `TRAIN` 中构建一个确定性的本地留出验证集，用于每个 epoch 的验证和最佳检查点选择，因为发布的 `VAL` 包没有标签。

阶段 1 使用本地留出验证集训练分割任务：

```powershell
python -B train.py --stage stage1_seg --data-root "<UUSIVC2026_DATA_ROOT>"
```

阶段 2 训练分类任务，并从阶段 1 的检查点初始化。请通过 `--init-checkpoint` 显式选择检查点。

如果阶段 1 使用本地验证训练，则从选定的阶段 1 最佳检查点初始化阶段 2：

```powershell
python -B train.py `
  --stage stage2_cls `
  --data-root "<UUSIVC2026_DATA_ROOT>" `
  --init-checkpoint outputs\stage1_seg\best_checkpoints\best_stage1_seg_rank1.pth
```

对于最终的全量数据训练，使用所有带标签的 `TRAIN` 数据并跳过验证。如果阶段 1 也使用了 `--full-train` 训练，则从其最新检查点初始化：

```powershell
python -B train.py `
  --stage stage2_cls `
  --data-root "<UUSIVC2026_DATA_ROOT>" `
  --init-checkpoint outputs\stage1_seg\latest_stage1_seg.pth `
  --full-train
```

在全量训练模式下，训练器仅保存 `latest_stage*_*.pth` 检查点，不计算验证指标，也不保存最佳检查点。

本地留出验证集比例由以下配置控制：

```yaml
data:
  local_val_fraction: 0.1
  split_seed: 2024
train:
  require_validation: true
```

可以通过命令行覆盖留出比例：

```powershell
python -B train.py `
  --stage stage2_cls `
  --data-root "<UUSIVC2026_DATA_ROOT>" `
  --local-val-fraction 0.05
```

## 本地评估

`test.py` 评估从 `TRAIN` 创建的带标签本地留出验证集。它不会评估发布的 `VAL` 包，因为 `VAL` 的标签和掩码未公开。如果覆盖了训练时的 `--local-val-fraction`，请使用相同的值。

```powershell
python -B test.py `
  --stage stage2_cls `
  --checkpoint outputs\stage2_cls\best_checkpoints\best_stage2_cls_rank1.pth `
  --split val `
  --data-root "<UUSIVC2026_DATA_ROOT>"
```

## 生成验证提交文件

`predict.py` 是针对发布 `VAL` 包的模型推理入口。它加载一个完整的模型检查点，对参赛者元数据运行推理，然后生成官方的上传目录和 zip 文件。默认使用阶段 2 的最佳检查点，`--checkpoint` 可以指向任意已训练的全模型权重以进行分数对比。

```powershell
python -B predict.py `
  --data-root "<UUSIVC2026_DATA_ROOT>" `
  --phase val `
  --checkpoint outputs\stage2_cls\best_checkpoints\best_stage2_cls_rank1.pth `
  --output-dir outputs\submission_val `
  --zip-path outputs\submission_val.zip
```

上传的 zip 文件恰好包含官方的提交文件：

```text
classification.json
image_seg/<Dataset>/masks/*.png
ceus_seg/<Dataset>/annotations/*.npz
video_seg/<Dataset>/annotations/*.npz
```

输出目录还包含 `submission_summary.json` 供本地查看，该文件不会添加到 zip 中。

## 指标参考

`metrics_reference/` 目录中提供了独立的参考实现：

```text
metrics_reference/segmentation_metrics_reference.py
metrics_reference/classification_metrics_reference.py
```

这些文件仅供参赛者参考，不会被训练或推理流程导入。项目中实际使用的实现在 `utils/metrics.py` 和 `utils/auc_utils.py` 中。

## 代码结构

```text
configs/                      精简的训练配置文件
datasets/uusivc2026_paths.py  固定的挑战赛数据协议
datasets/                     图像/视频数据集类及数据加载器
metrics_reference/            独立的指标参考实现
models/                       统一的图像/视频模型
trainers/                     阶段1分割与阶段2分类训练循环
train.py                      训练入口
test.py                       本地留出验证集评估入口
predict.py                    竞赛格式预测入口
utils/                        指标、检查点、日志、可视化工具
```

数据集管理不是面向用户的任务。本基线假设采用官方 UUSIVC2026 `TRAIN`/`VAL` 包布局，并将路径扩展逻辑封装在 `datasets/uusivc2026_paths.py` 中。

本仓库基于 [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) 代码库构建。我们感谢作者将其工作公开发布。
