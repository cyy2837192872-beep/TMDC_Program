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

## [0.1.0] - 此前基线

- 初始公开结构：`dataset_generator`、`train_cnn`、`eval_compare`、`moire_pipeline`、`export_thesis_table`、分级/畸变/FFT 失效分析脚本、`core/*` 与论文目录等。详见根目录 `README.md` 中的项目结构与工作流。

<!-- 发布打 tag 后可在此补充 Release 链接 -->
