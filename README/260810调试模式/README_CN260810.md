# 开发日志 — 调试模式 (Debug Mode)

**日期**：2026-08-10  
**分支**：`feature-add-logic`  
**作者**：Claude Code (DeepSeek V4 Pro)  
**任务**：为 `train.py` 添加三级调试模式

---

## 背景

`train.py` 此前没有正式的调试模式。开发者改完代码后只能通过 `--device cpu` 手动跑几步再 Ctrl+C 来验证管线是否通畅，缺乏系统化的调试手段。

## 需求设计

通过 `--debug` CLI 参数启用的三级调试系统：

| 级别 | 命令 | 数据策略 | 训练长度 | 诊断输出 |
|------|------|---------|---------|---------|
| **fast** | `--debug fast` | 完整 DataLoader | 跑 10 步后停止 | 无 |
| **full** | `--debug full` | 每任务 100 样本（manifest 截断） | 3 epoch，走完整 train→val→save 流程 | 无 |
| **deep** | `--debug deep` | 默认=full，也可 `deep fast` 叠加 | 同数据策略 | 梯度统计 + 激活值统计 + 性能剖析 |

### 可覆盖参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--debug-steps` | 10 | fast 模式的最大训练步数 |
| `--debug-samples` | 100 | full/deep 模式每任务最大样本数 |
| `--debug-epochs` | 3 | full/deep 模式的最大 epoch 数 |

### 关键设计决策

- **架构复用**：debug 配置覆写模式与现有 `--full-train` 一致，在 `load_yaml()` 之后、Trainer 构建之前通过 `apply_debug_config()` 覆写 cfg，不修改任何 YAML 文件
- **Manifest 截断**：full/deep 模式在 `write_fixed_manifests()` 写文件前截断样本行数，对 4 种 Dataset 类完全透明
- **诊断层可插拔**：deep 模式的诊断钩子通过 `DiagnosticsHook` 上下文管理器注入，正常训练零开销
- **deep + fast 可叠加**：`--debug deep fast` 在 fast 的步数限制下同时输出完整诊断信息

---

## 改动文件

### 1. `train.py`（+60 行）

**新增 CLI 参数**：`--debug`（`nargs='+'`，支持多级组合）、`--debug-steps`、`--debug-samples`、`--debug-epochs`

**新增函数** `apply_debug_config(cfg, args) -> Optional[Dict]`：
- 解析 debug 级别（`fast`/`full`/`deep`）
- 构建 `debug_cfg` 字典（含 `max_steps`、`max_epochs`、`max_samples_per_task`、`diagnostics` 子配置）
- 覆写 cfg 中的 `train.max_epochs`、`trainer.keep_best`、`trainer.log_interval`、`data.debug_max_samples_per_task`
- 正常训练时返回 `None`

**修改 `main()`**：调用 `apply_debug_config()`、输出 debug 信息、将 `debug_cfg` 传入 Trainer

### 2. `datasets/uusivc2026_paths.py`（+6 行）

**`write_fixed_manifests()`**：新增 `max_samples_per_task` 参数，写文件前截断 `lines[:max_samples_per_task]`

**`expand_fixed_uusivc_data_cfg()`**：从 cfg 读取 `debug_max_samples_per_task` 并透传

### 3. `utils/diagnostics.py`（新文件，~210 行）

**`DiagnosticsConfig`**（dataclass）：配置各诊断开关
- `grad_monitor: bool` — 梯度范数/NaN/Inf 检测
- `act_monitor: bool` — 关键层激活值统计
- `perf_monitor: bool` — 耗时 + GPU 显存

**`DiagnosticsHook`**（上下文管理器）：
- `__enter__`：通过 `get_submodule` 发现关键子模块（encoder、backbone、seg_head、cls_head、temporal_router 等），注册 forward hook
- 梯度统计：遍历 `named_parameters()`，输出 top-5 / bottom-5 梯度范数，标记 NaN/Inf
- 激活值统计：捕获 hook 输出，输出 shape + mean/std/min/max
- 性能剖析：两段计时（data loading + compute），GPU 显存查询
- `__exit__`：移除所有 hook，清理状态
- CPU 模式下自动跳过 GPU 显存查询

### 4. `trainers/trainer_stage1_seg.py`（+45 行）

**`__init__`**：新增 `debug_cfg` 参数，解析并存储 `_max_steps`、`_debug_diagnostics`、`_global_step`、`_early_stop`

**`train_one_epoch`**：
- 新增 `steps_done` 局部计数器，确保早停时平均 loss 计算准确
- 早停检查（两处：循环顶部 + 循环底部，覆盖跨 epoch 场景）
- 诊断钩子注入：循环前 `DiagnosticsHook.__enter__()`，循环中每步触发诊断输出，循环后 `.__exit__()`
- 性能计时：`start_step_timer()` → `record_data_loading_done()` → `record_forward_done()`

**`fit`**：早停时跳过 validation、checkpoint 保存、best model 逻辑，打印汇总信息

### 5. `trainers/trainer_stage2_cls.py`（+45 行）

与 Stage1 完全相同的改动模式。复用同一个 `DiagnosticsHook`，因 cls trainer 更简单（无 seg loss），改动稍少。

---

## 未改动文件

- `configs/stage1_seg.yaml` / `configs/stage2_cls.yaml` — 不改 YAML，纯代码覆写
- `datasets/us_image_cls_dataset.py` 等 4 个 Dataset 类 — manifest 截断对下游透明
- `datasets/build_loader.py` — 不需要改动
- `predict.py` / `test.py` — debug 仅限训练
- `utils/freeze.py` / `utils/logger.py` — 不涉及

---

## 验证状态

| 检查项 | 状态 |
|--------|------|
| AST 语法解析（5 个文件） | ✅ 全部通过 |
| 关键符号完整性（parse_args / apply_debug_config / main / DiagnosticsConfig / DiagnosticsHook / Stage1SegTrainer / Stage2ClsTrainer） | ✅ 全部存在 |
| 端到端运行测试 | ⏳ 待 AutoDL GPU 服务器上验证 |

### 建议验证命令

```bash
# 1. fast 模式 — 跑 3 步停
python train.py --stage stage1_seg --debug fast --debug-steps 3 --device cpu

# 2. full 模式 — 20 样本 + 2 epoch
python train.py --stage stage1_seg --debug full --debug-samples 20 --debug-epochs 2 --device cpu

# 3. deep 模式
python train.py --stage stage1_seg --debug deep --debug-samples 20 --debug-epochs 2 --device cpu

# 4. deep + fast 叠加
python train.py --stage stage1_seg --debug deep fast --debug-steps 5 --device cpu

# 5. stage2 同理
python train.py --stage stage2_cls --debug fast --debug-steps 3 --device cpu
```

---

## 使用示例输出

### fast 模式
```
[Debug] Mode active: levels=['fast']
[Debug]   max_steps=10
...[训练 10 步]...
[Debug] Fast mode stopped after 10 steps. avg loss=0.6931
```

### deep 模式
```
[Debug] Mode active: levels=['deep']
[Debug]   max_epochs=3
[Debug]   max_samples_per_task=100
[Debug]   diagnostics=grad+act+perf

=== [Deep Debug] Step 1 | Gradients ===
  Top-5 gradient norms:
    image_model.cls_head.fc.weight: norm=0.045632
    video_model.temporal_router.video_cls_head.attention.q_proj.weight: norm=0.032145
    ...
  Bottom-5 gradient norms:
    image_model.backbone.encoder.features.0.norm.weight: norm=0.000123
    ...
  Layers with grad: 142 | NaN: 0 | Inf: 0

=== [Deep Debug] Step 1 | Activations ===
  image_model.backbone.encoder [4,768,7,7]: mean=0.0234, std=0.8912, min=-2.3401, max=3.1204
  image_model.backbone [4,96,56,56]: mean=0.0150, std=0.4560, min=-1.2300, max=1.8900
  ...

=== [Deep Debug] Step 1 | Performance ===
  data:152.3ms | compute:283.1ms | total:435.4ms
  GPU: allocated=2.34GB, reserved=3.12GB
```
