# 调试模式验证日志 — Debug Mode Verification Log

**日期**：2026-08-10  
**分支**：`feature-add-logic`  
**服务器**：AutoDL GPU (connect.cqa1.seetacloud.com:22366)  
**环境**：Python 3.12.3 / PyTorch 2.3.0+cu121 / GPU: RTX 4090D

---

## 概述

对 `train.py` 新增的三级调试模式（`--debug fast|full|deep`）进行端到端验证。
共执行 8 项测试，全部通过。同时发现并修复 1 个 Bug。

---

## 代码版本

| 提交 | Hash | 说明 |
|------|------|------|
| feat: add three-level debug mode | `ea7edb9` | 核心功能：CLI 参数、apply_debug_config、DiagnosticsHook、manifest 截断、早停 |
| fix: disable balanced sampler | `a064494` | 修复：debug full/deep 模式下自动禁用 balanced sampler |

---

## 测试 1：fast 模式 — Stage1 快速冒烟 ✅

**命令**：
```bash
python train.py --stage stage1_seg --debug fast --debug-steps 3 --device cuda
```

**验证要点**：
- `[Debug] Mode active: levels=['fast']`
- `[Debug]   max_steps=3`
- 训练精确执行 3 步后自动退出
- 不触发 validation
- 不保存 checkpoint

**结果**：
```
[Debug] Fast mode stopped after 3 steps. avg loss=0.6257
```

✅ 通过

---

## 测试 2：fast 模式 — Stage2 快速冒烟 ✅

**命令**：
```bash
python train.py --stage stage2_cls --debug fast --debug-steps 3 --device cuda
```

**验证要点**：
- Stage2 同样 3 步早停
- Stage1 预训练权重加载正常（`init_checkpoint: best_stage1_seg_rank1.pth`）
- 冻结机制生效（29,276,774 frozen / 2,398,473 trainable）

**结果**：
```
[Debug] Fast mode stopped after 3 steps. avg loss=0.7892
```

✅ 通过

---

## 测试 3：full 模式 — Stage1 端到端管线 ✅

**命令**：
```bash
python train.py --stage stage1_seg --debug full --debug-samples 20 --debug-epochs 2 --device cuda
```

**验证要点**：
- `[Debug] Mode active: levels=['full']`
- `[Debug]   max_epochs=2`
- `[Debug]   max_samples_per_task=20`
- manifest 截断至 20 行/任务（验证通过：5 个 train_*.txt 各 20 行）
- 每 epoch 90 步（vs 正常模式 4994 步 — 截断生效）
- 完整管线：train → val → save checkpoint → export visuals
- `keep_best=1`，只保留 1 个 best checkpoint

**训练指标**：
```
Epoch 2/2: train_loss=0.2806 val_mean_score=0.4650
  image_seg_dsc: 0.6060 | score: 0.4669
  cardiac_video_seg_dsc: 0.8378 | score: 0.6237
  video_seg_ceus_dsc: 0.4141 | score: 0.3044
```

✅ 通过

---

## 测试 4：full 模式 — Stage2 端到端管线 ✅ (Bug 修复后)

**命令**：
```bash
python train.py --stage stage2_cls --debug full --debug-samples 20 --debug-epochs 2 --device cuda
```

**发现的 Bug**：

首次运行失败：
```
ValueError: Classification dataset must contain at least 2 classes, got class_count=[20]
```

**根因**：Stage2 配置中 `use_balanced_sampler=true`，manifest 截断至 20 条时全部来自同一类别，`WeightedRandomSampler` 要求至少 2 类。

**修复**（commit `a064494`）：在 `apply_debug_config()` 中自动禁用 balanced sampler：
```python
cfg.setdefault(data, {})[use_balanced_sampler_image_cls] = False
cfg.setdefault(data, {})[use_balanced_sampler_video_cls] = False
```

**修复后训练指标**：
```
Epoch 2/2: train_loss=0.1448 val_mean_score=0.5220
  image_cls_acc: 0.9000
  video_cls_ceus_acc: 0.4000 | score: 0.5385
```

✅ 通过

---

## 测试 5：deep 模式 — 深度诊断 ✅

**命令**：
```bash
python train.py --stage stage1_seg --debug deep --debug-samples 20 --debug-epochs 2 --device cuda
```

**验证要点**：
- 梯度统计：每步输出 Top-5 / Bottom-5 梯度范数，NaN/Inf 检测
- 激活值统计：关键层输出 mean/std/min/max
- 性能剖析：data loading 耗时 + compute 耗时 + GPU 显存

**诊断输出示例**：
```
=== [Deep Debug] Step 1 | Gradients ===
  Top-5 gradient norms:
    video_model.temporal_router.simple_video_seg_head.frame_head.classifier.weight: norm=0.940459
    video_model.temporal_router.simple_video_seg_head.frame_head.block.2.block.0.weight: norm=0.511184
    image_model.backbone.seg_smooth.1.0.weight: norm=0.374116
    ...
  Bottom-5 gradient norms:
    image_model.backbone.encoder.backbone.features.7.1.attn.relative_position_bias_table: norm=0.000282
    ...
  Layers with grad: 193 | NaN: 0 | Inf: 0

=== [Deep Debug] Step 1 | Activations ===
  video_model.temporal_router.simple_video_seg_head [1,10,2,224,224]: mean=-0.1578, std=0.3020, min=-2.8633, max=2.5508

=== [Deep Debug] Step 1 | Performance ===
  data:32μs | compute:707.3ms | total:707.3ms
  GPU: allocated=0.47GB, reserved=1.33GB
```

**关键发现**：
- 梯度最大的层：`seg_head` 的 `classifier.weight`（0.94）
- 梯度最小的层：encoder 深层 `norm` / `bias`（< 0.001） — 符合预期（浅层分割头训练更活跃）
- **NaN: 0 / Inf: 0** 全程无异常
- 第一步 compute 707ms（含 CUDA kernel 编译），稳定后 ~143ms/step
- GPU 显存：allocated 0.47GB, reserved 1.36GB（Swin-Tiny 极小模型，显存充裕）

✅ 通过

---

## 测试 6：deep + fast 叠加 ✅

**命令**：
```bash
python train.py --stage stage1_seg --debug deep fast --debug-steps 5 --device cuda
```

**验证要点**：
- 两级正确叠加：`levels=['deep', 'fast']`
- fast 步数限制生效（max_steps=5）
- deep 诊断每步输出

**输出**：
```
[Debug] Mode active: levels=['deep', 'fast']
[Debug]   max_steps=5
[Debug]   diagnostics=grad+act+perf
...
[Debug] Fast mode stopped after 5 steps. avg loss=0.6104
```

✅ 通过

---

## 测试 7：CPU 兼容性 ✅

**命令**：
```bash
python train.py --stage stage1_seg --debug deep fast --debug-steps 3 --device cpu
```

**验证要点**：
- `[Info] Using device: cpu`
- 诊断输出正常（梯度、激活值）
- 性能输出**无** `GPU:` 行（自动跳过 `torch.cuda.*` 查询）
- 无任何 CUDA 相关异常

✅ 通过

---

## 测试 8：回归测试 — 无 debug 正常训练 ✅

**命令**：
```bash
python train.py --stage stage1_seg --device cuda
```

**验证要点**：
- 输出中**无任何** `[Debug]` 行
- 4994 步/epoch（manifest 未被截断）
- `log_interval=20`（未被覆写）
- `keep_best=3`（未被覆写）
- 行为与改动前完全一致

✅ 通过

---

## 测试结果汇总

| # | 测试 | 命令 | 结果 |
|---|------|------|:--:|
| 1 | fast stage1 | `--debug fast --debug-steps 3` | ✅ |
| 2 | fast stage2 | `--debug fast --debug-steps 3` | ✅ |
| 3 | full stage1 | `--debug full --debug-samples 20 --debug-epochs 2` | ✅ |
| 4 | full stage2 | `--debug full --debug-samples 20 --debug-epochs 2` | ✅ |
| 5 | deep stage1 | `--debug deep --debug-samples 20 --debug-epochs 2` | ✅ |
| 6 | deep+fast | `--debug deep fast --debug-steps 5` | ✅ |
| 7 | CPU | `--debug deep fast --debug-steps 3 --device cpu` | ✅ |
| 8 | 回归 | 无 `--debug` | ✅ |

---

## Bug 记录

| ID | 发现 | 严重程度 | 修复 |
|----|------|---------|------|
| 1 | Stage2 `use_balanced_sampler=true` + manifest 截断导致单类别崩溃 | 中 | `apply_debug_config()` 中自动禁用 balanced sampler |

---

## 日志文件位置

- 训练日志：`outputs/stage1_seg/train_stage1_seg.log`
- 训练日志：`outputs/stage2_cls/train_stage2_cls.log`
- Checkpoint：`outputs/stage1_seg/best_checkpoints/`
- Checkpoint：`outputs/stage2_cls/best_checkpoints/`
- 可视化：`outputs/stage1_seg/best_epoch_outputs/`
- 分类预测：`outputs/stage2_cls/best_epoch_outputs/`

---

> 📅 日志生成时间：2026-08-10 19:00 CST | 分支：`feature-add-logic`
