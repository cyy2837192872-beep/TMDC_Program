# MoS₂ Moiré 转角提取（FFT + CNN）

从 MoS₂ **转角同质结** moiré 图像中估计扭转角 θ 的完整管线：**物理仿真 → 数据集 → CNN 训练 → FFT/CNN/融合评估 → 论文与实验文档**。

**版本**：PyPI/包版本见 [`pyproject.toml`](pyproject.toml) 的 `version`。**更新说明**见 [`CHANGELOG.md`](CHANGELOG.md)。**仅跑 `angle_cnn` 脚本**时也可先看 [`angle_cnn/README.md`](angle_cnn/README.md)。

| 方法 | 说明 |
|------|------|
| **FFT** | 频域峰与倒格矢差模型，`moire_pipeline.py` |
| **CNN** | ResNet 回归 + 可选 FFT 附加通道 + Mixup / Huber / MC Dropout，`train_cnn.py` |
| **融合** | 基于 CNN 不确定度在 FFT 与 CNN 间加权（`eval_compare.py`，需 `--mc-samples`） |

**仓库**：<https://github.com/cyy2837192872-beep/TMDC_Program>（本地默认分支为 `master`，远程为 `main`，推送示例见文末）

---

## 实验与仪器参照（仿真对标）

- **探针**：TITAN 70（tip radius ≈ 7 nm）  
- **模式**：Tapping（Height / Phase / Amplitude 多通道仿真）  
- **设备参照**：Cypher ES（Asylum Research / Oxford Instruments）

文献中 AFM / 扫描探针观测 moiré 的先例与本地图片索引：

- `angle_cnn/docs/literature_image_benchmark.md`  
- `angle_cnn/docs/figures/literature/README.md`  
- AFM 沟通用需求说明：`angle_cnn/docs/afm_experiment_requirements.md`  
- 实验记录模板：`angle_cnn/docs/experiment_record_template.md`

---

## 项目结构（与仓库一致）

```
tmdc-project/
├── angle_cnn/                      # 算法与仿真主代码
│   ├── train_cnn.py                # CNN 训练（主线，当前实现见文件头 v5 说明）
│   ├── README.md                   # 本模块最短运行路径（给 AI / 协作者）
│   ├── train_cnn_v2.py             # 可选：双流等实验入口
│   ├── dataset_generator.py        # 仿真数据集 .npz（多通道 + 退化）
│   ├── moire_pipeline.py           # FFT 角度提取 + 与仿真衔接
│   ├── eval_compare.py             # FFT vs CNN；legacy / paired；融合与 CSV 摘要
│   ├── export_thesis_table.py      # 合并 train/compare 摘要 → thesis_table_metrics.csv
│   ├── graded_eval.py              # 分级退化下 FFT vs CNN
│   ├── distortion_sweep.py         # 仿射畸变扫描
│   ├── analyze_fft_failure.py      # FFT 失效样本分析
│   ├── preprocess_afm_data.py      # 实验 AFM → 推理用 .npz
│   ├── outputs/                    # 训练/评估输出（见 .gitignore，默认不提交）
│   ├── docs/                       # 文献图、AFM 说明、实验模板
│   ├── tests/                      # pytest（physics / degrade / moire_sim 等）
│   └── core/
│       ├── config.py               # 物理常数、默认 θ 范围、IMG_SIZE / SIM_SIZE
│       ├── physics.py              # 物理与周期相关工具
│       ├── moire_sim.py            # 单/多通道 moiré 仿真
│       ├── degrade.py              # 探针卷积、1/f、行噪声、仿射等
│       ├── cnn.py                  # 模型、FFT 通道、MC Dropout 推理
│       ├── metrics.py              # 分层指标、校准、报告生成
│       ├── eval_utils.py           # 评测辅助
│       ├── augment.py              # 训练期增强
│       ├── train_utils.py          # 加权 Huber、EMA 更新（train_cnn 使用）
│       ├── dual_stream.py          # 双流实验用
│       ├── io_utils.py             # npz / checkpoint
│       ├── io_afm.py               # Cypher .ibw 等（需 igor2）
│       ├── fonts.py                # Matplotlib 中文字体
│       └── seed.py                 # 可复现种子
├── thesis/hust_bachelor_cse/       # 本科论文 LaTeX（xelatex + bibtex）
├── data/                           # 默认数据集输出目录（.gitignore，需本地生成）
├── pyproject.toml                  # 包元数据、依赖、版本号、控制台入口
├── requirements.txt                # pip 依赖（与 pyproject 对齐）
├── CHANGELOG.md                    # 版本与重要变更（协作 / AI 对齐用）
└── README.md
```

安装为可编辑包后，可使用入口命令（等价于 `python -m angle_cnn.xxx` 或直接运行脚本）：

| 命令 | 模块 |
|------|------|
| `moire-generate` | `dataset_generator` |
| `moire-train` | `train_cnn` |
| `moire-eval` | `eval_compare` |

---

## 环境安装

```bash
cd tmdc-project
pip install -r requirements.txt
pip install -e .                    # 启用 moire-* 命令

# 开发 / 测试
pip install -e ".[dev]"
pytest angle_cnn/tests -q

# 读取 Cypher ES .ibw（可选）
pip install -e ".[afm]"             # 或 pip install igor2
```

**说明**：`requirements.txt` / `pyproject.toml` 中的 PyTorch 版本区间面向较新 GPU；若仅 CPU 或旧卡，请按 [PyTorch 官网](https://pytorch.org/) 选择合适 wheel 再安装。

**WSL 内存**：全量数据集 + 大 batch 训练可能触发 OOM；可增大 Windows `.wslconfig` 内存/swap，或减小 `--batch-size`、`--num-workers`、`--prefetch-factor`，并考虑 `--fp16-data`。

---

## 工作流速查

### 1. 生成数据集

```bash
cd angle_cnn

# 默认：50000 样本，3 通道，输出 128×128，仿真网格 --n-sim 默认 512
python dataset_generator.py

python dataset_generator.py --log-theta          # 小角度更密
python dataset_generator.py --n-sim 256          # 更快、略粗
python dataset_generator.py --channels 1       # 仅 height
python dataset_generator.py --samples 20000 --workers 8
```

输出默认：`../data/moire_dataset.npz`（相对 `angle_cnn/`）。**大文件已被 .gitignore**，需本地生成或自备。

与 `core/config.py` 对齐的常用默认：**θ ∈ [0.5°, 5°]**，**CNN 裁剪 128**，推荐仿真 **512**（与 `eval_compare` 高分辨率 FFT 口径一致）。**若你曾在 2026-04 之前的代码版本上生成过 `.npz`，请用当前 `core/moire_sim.py`（固定 nm 视野）重新生成数据集后再训练**，以免与论文中「固定物理视野」表述不一致。

### 2. 训练 CNN（`train_cnn.py` v5）

```bash
python train_cnn.py
python train_cnn.py --bf16
python train_cnn.py --arch resnet34 --batch-size 512
python train_cnn.py --fft-channel --bf16
python train_cnn.py --mc-samples 30
python train_cnn.py --eval-only                 # 仅评估已有 best_model.pt
python train_cnn.py --no-compile --no-amp       # 调试

# WSL / 低内存环境（OOM 时优先尝试）
python train_cnn.py --fp16-data --num-workers 0 --batch-size 256 --no-compile

# v5 可选：EMA（验证与 best_model.pt 存 EMA 权重）+ 归一化标签加权 Huber（略强调高 θ）
python train_cnn.py --ema-decay 0.999 --angle-loss-weight 0.25
```

详见 `python train_cnn.py --help`。**实验向**：`train_cnn_v2.py` + `core/dual_stream.py`。

**Checkpoint**：`best_model.pt` 除 `model_state_dict` 外可含 `arch`、`n_channels`、`add_fft_channel`、`dropout`、`huber_delta`、`ema_decay`、`angle_loss_weight` 等；下游 `eval_compare.py` 只依赖权重与通道相关字段。

### 3. FFT vs CNN 评估（两种口径 + 融合）

```bash
# legacy：CNN 用 .npz 测试集图；FFT 按标签单独重算 512（与历史表格兼容）
python eval_compare.py --mc-samples 30

# paired：同一随机退化场景 → CNN 中心 128 + 同场景 height@512（更公平）
python eval_compare.py --eval-mode paired --paired-seed 12345 --mc-samples 30 --fusion-unc-scale 0.5
```

- **`--mc-samples 0`**：不做 MC Dropout，**不输出融合**（融合需要 σ_cnn）。  
- **融合**：\( \theta = w\,\theta_{\mathrm{fft}} + (1-w)\,\theta_{\mathrm{cnn}} \)，\(w = \sigma/(\sigma+\tau)\)，`τ` = `--fusion-unc-scale`（度）。

默认输出目录：`angle_cnn/outputs/`（`compare_report.txt`、`compare_scatter.png`、`compare_error.png`、`compare_summary.csv` 等）。

### 4. 合并论文用指标表

```bash
cd angle_cnn
python export_thesis_table.py
# → outputs/thesis_table_metrics.csv
```

依赖：先运行 `train_cnn.py`（生成 `train_test_summary.csv`）与 `eval_compare.py`（生成 `compare_summary.csv`）。

### 5. 其它评测与工具

```bash
python graded_eval.py --mc-samples 30
python distortion_sweep.py
python analyze_fft_failure.py
python preprocess_afm_data.py --input path/to/scan --output ../data/afm_processed/
```

### 6. 毕业论文编译

```bash
cd thesis/hust_bachelor_cse
xelatex -interaction=nonstopmode thesis.tex
bibtex thesis
xelatex -interaction=nonstopmode thesis.tex
xelatex -interaction=nonstopmode thesis.tex
```

参考文献：`Bibs/mybib.bib`，正文中 `\bibliography{Bibs/mybib}`。**附录 `verbatim` 请尽量使用 ASCII**，避免等宽字体缺字警告。

---

## 多通道与退化（摘要）

| 通道 | 含义（仿真侧） |
|------|------------------|
| Height | moiré 包络形貌 |
| Phase | 畴对比 / 耗散相关对比度 |
| Amplitude | 梯度相关对比度 |

退化链包括：探针卷积、仿射、模糊、行噪声、背景倾斜、1/f、反馈振荡、扫描方向偏移、加性噪声等（见 `degrade.py` 与 `dataset_generator.py`）。

---

## 输出文件（默认均在 `angle_cnn/outputs/`）

| 文件 | 说明 |
|------|------|
| `best_model.pt` | 最优权重与元数据（通道数、arch、dropout、FFT 通道、`huber_delta`、可选 `ema_decay` / `angle_loss_weight` 等） |
| `train_log.csv` / `train_log.png` | 训练曲线 |
| `train_test_summary.csv` | 测试集摘要（供 `export_thesis_table.py`） |
| `compare_report.txt` / `compare_summary.csv` | FFT/CNN（及融合）对比 |
| `compare_scatter.png` / `compare_error.png` | 对比图 |
| `graded_eval.*` / `distortion_sweep.*` | 分级与畸变扫描结果 |

---

## Git 与远程

远程主分支名为 **`main`**，本地常用 **`master`** 时，推送可执行：

```bash
git push origin HEAD:main
```

若已设置 `master` 跟踪 `origin/main`，在配置允许时也可直接使用 `git push`。

---

## 参考文献（节选）

- Weston et al. — 连续晶格重构与 moiré 物理图像  
- 晶格常数：MoS₂ **a = 0.316 nm**（见 `core/config.py`）

---

## 版本与更新日志

- **[CHANGELOG.md](CHANGELOG.md)**：按版本列出 API、训练选项、物理/评测修复与文档变更，便于他人或 AI 对齐环境。
- 升级自 **0.1.x** 时：若曾用旧版仿真生成 `moire_dataset.npz`，请按 `CHANGELOG` 与上文「固定物理视野」说明 **重新生成数据** 后再训练/对比。

---

## License

MIT License
