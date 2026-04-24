# 变更日志

本文档记录 **tmdc-project**（MoS₂ moiré FFT + CNN 管线）面向复现与协作的重要变更。语义化版本见 `pyproject.toml` 的 `version` 字段。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [SemVer](https://semver.org/lang/zh-CN/) 的宽松用法（0.x 阶段 API 仍可能调整）。

---

## [0.2.0] - 2026-04-17

### 新增

- **`angle_cnn/core/train_utils.py`**：`weighted_huber_loss`（按归一化标签加权 Huber）、`ema_update`（参数 EMA）。
- **`train_cnn.py` v5 CLI**：`--ema-decay`（0 关闭；开启时验证 / 早停 / `best_model.pt` / 测试均基于 EMA 权重）、`--angle-loss-weight`（略加大高 θ 样本在 loss 中的权重）。
- **Checkpoint 元数据**：可选写入 `ema_decay`、`angle_loss_weight`、`huber_delta`；`--eval-only` 会从 `best_model.pt` 读取 `angle_loss_weight` / `huber_delta` 以保持 loss 口径一致。
- **`angle_cnn/tests/test_train_utils.py`**：加权 Huber 与 EMA 的单元测试（可用 `python -c` 导入执行，亦支持 `pytest`）。

### 修复 / 物理与评测

- **固定物理视野（nm）**：`core/moire_sim.py` 等处的合成坐标与论文「固定 fov」表述一致；FFT 侧周期像素数与 `eval_compare.py`、`dataset_generator.py`、`graded_eval.py`、`physics.py` 等对齐。
- **`theta_from_period` 向量化**：`core/physics.py` 支持数组 `L_nm`，修复 `eval_compare` 等路径上的 `TypeError`。

### 文档

- 根目录 **`README.md`**：补充 v5 训练参数、WSL 省内存示例、版本说明指向本文件。
- **`angle_cnn/README.md`**：从本目录运行的最短路径与常见故障。

---

## [0.4.0] - 2026-04-25

### 新增

- **`inference_real_data.py`**：真实 AFM 数据的端到端推理管线。支持 `.npz` / `.ibw` / `.spm` / `.tiff` / `.png` 输入，同时运行 CNN（含 MC Dropout 不确定性）+ FFT 角度提取 + 自适应融合，输出 CSV 结果和每样本预览图。
- **`docs/sim_to_real_roadmap.md`**：Sim-to-Real 优化路线图文档。记录当前性能水位、已失败的方向，以及真实数据到来后根据不同结果（FFT fail / CNN fail / 都不行）的三套针对性优化方案。
- **`tests/test_metrics.py`**：`compute_stratified_metrics` 和 `compute_calibration_metrics` 的 15 个单元测试。

### 变更

- **`train_cnn.py`**：实验证明 `--scale-augment`（RandomZoom）对转角回归有害（MAE 恶化至 0.0987°），已完全移除相关代码。
- **`check_regression.py`**：退化检测列表新增 `small_angle_mae`、`large_angle_mae` 分层指标。
- **`core/config.py`**：新增 12 个 `DEFAULT_*_RANGE` 退化参数常量，消除 `dataset_generator.py` 与 `eval_compare.py` 间的常量重复定义。
- **`core/eval_fft.py`**：从 `eval_compare.py` 抽取 FFT 工具函数（`extract_angle_fft_robust`、`fft_predict_batch_512`），消除 130 行重复代码。
- **`eval_compare.py`**：移除重复 FFT 函数，改用 `core.eval_fft`；添加残差相关性分析。
- **`dataset_generator.py`**：退化常量改为从 `core.config` 导入。

### 文档

- **`README.md`**：添加 `inference_real_data.py` 到项目结构和工作流；添加 Sim-to-Real 路线图说明。
- **`docs/sim_to_real_roadmap.md`**：后续 AI 或协作者的开发路线图（见上）。

### 测试

- 单元测试总数从 67 增至 **82**，涵盖 `core/metrics.py` 的全部分层指标和校准函数。

---

## [0.3.0] - 2026-04-25

### 新增

- **`check_regression.py`**：CNN 精度回归测试，加载 `best_model.pt` 在测试集上评估，与保存的 baseline 比对（MAE 恶化超 10% 返回非零退出码），支持 `--save-baseline` / `--tolerance`。
- **`robustness_sweep.py`**：四维度退化鲁棒性扫描（噪声、模糊、仿射畸变、探针卷积），从训练分布内到 OOD，量化 FFT vs CNN 崩溃边界，输出 `robustness_sweep.png` / `.csv`。
- **`core/eval_fft.py`**：从 `eval_compare.py` 抽取的 FFT 推理模块（`extract_angle_fft_robust` 6 策略重试）。
- **`dataset_generator.py --theta-sampling`**：支持 `uniform` / `log` / `inv_square` 三种采样策略（`--log-theta` 已弃用）。
- **`io_utils.py`**：`validate_npz_dataset()` 数据集预检函数。
- **`eval_compare.py`**：`--eval-mode paired`（同场景配对评估）、τ 参数扫描（`sweep_fusion_tau` / `plot_tau_sweep`）。
- **`train_cnn.py --swa`**：SWA（Stochastic Weight Averaging）支持。
- **`CLAUDE.md` / `.cursorrules`**：面向 AI 助手的项目级指令（回归测试、训练参数约束、禁止事项）。

### 变更

- **`train_cnn.py` 默认参数**：LR 从 `1e-3` 降至 `5e-4`，warmup 从 5 epoch 增至 8。
- **`core/degrade.py`**：`_cached_affine_grid` / `_cached_tilt_grid` 改用 `functools.lru_cache`。
- **`core/metrics.py`**：`print()` → `logging`；修复 `percentiles` 可变默认参数。
- **`core/io_utils.py`**：简化 `torch.load` 权重加载。
- **`core/train_utils.py`**：修复 EMA BN running stats 同步 bug（直接拷贝而非 EMA 平均）。
- **Remote URL**：HTTPS → SSH（解决 WSL2 网络问题）。

### 回归保护

- `check_regression.py` 纳入 `CLAUDE.md` / `.cursorrules`：**每次修改 `train_cnn.py` 或 `core/` 后必须运行并通过**。
- 当前基准：Test MAE = 0.054°（2026-04-25）。

---

## [0.2.0] - 2026-04-17

- 初始公开结构：`dataset_generator`、`train_cnn`、`eval_compare`、`moire_pipeline`、`export_thesis_table`、分级/畸变/FFT 失效分析脚本、`core/*` 与论文目录等。详见根目录 `README.md` 中的项目结构与工作流。

<!-- 发布打 tag 后可在此补充 Release 链接 -->
