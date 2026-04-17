# MoS₂ Moiré 转角提取项目

本项目实现了从 MoS₂ 转角同质结 moiré 图像中提取扭转角度的两种方法：

- **FFT 方法**：基于快速傅里叶变换的传统图像处理方法（baseline）
- **CNN 方法**：基于深度学习的卷积神经网络回归方法（ResNet 骨干 + 可选 FFT 附加通道）

## 实验配置

针对以下 AFM 配置进行了优化：

- **探针**：TITAN 70（tip radius ≈ 7 nm）
- **模式**：Tapping Mode（AC Mode）
- **设备**：Cypher ES（Asylum Research / Oxford Instruments）

## 物理背景

MoS₂ 晶格常数 a = 0.316 nm，两层旋转 θ 角后形成 moiré 超周期结构：

```
L = a√3 / (4 sin(θ/2))    ← moiré 条纹间距
θ = 2 arcsin(a√3 / (4L))  ← 反推公式
```

## 实验可达性与文献先例（避免“仅仿真”质疑）

本项目当前以物理仿真与算法验证为主，但 AFM/扫描探针路线在文献中已有明确实证基础，可作为后续实验验证的参照：

- **Liao et al., Nat. Commun. 2020**（twisted MoS₂ homostructures）  
  含 AFM 形貌图（Fig. 2），说明 MoS₂ 扭转样品可进行 AFM 表征：  
  <https://www.nature.com/articles/s41467-020-16056-4/figures/2>  
  开放全文（PMC）：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7195481/>

- **McGilly et al., Nat. Nanotechnol. 2020**（Visualization of moiré superlattices）  
  使用 PFM（AFM 家族模式）可视化 moiré 超晶格：  
  <https://www.nature.com/articles/s41565-020-0708-3>

- **Engelke et al., PNAS 2024**（Torsional force microscopy）  
  AFM 衍生模式（TFM）实现 vdW moiré / 原子晶格成像：  
  <https://www.pnas.org/doi/10.1073/pnas.2314083121>

说明：上述工作证明“扫描探针可观测 moiré”的物理可达性；本项目后续将补充 Tapping 模式下 Height/Phase/Amplitude 的实测-仿真对照与 SHG 定标误差分析。

文献图像对照与答辩用说明已整理在：
`angle_cnn/docs/literature_image_benchmark.md`
  
已下载的文献图片本体（含本地渲染快照）见：
`angle_cnn/docs/figures/literature/README.md`

## 项目结构

```
tmdc-project/
├── angle_cnn/                      # 主要代码目录
│   ├── train_cnn.py                # CNN 训练（当前主线：v4，见下文）
│   ├── train_cnn_v2.py             # 可选：双流架构 + 实验性选项
│   ├── dataset_generator.py        # 数据集生成（多通道 + TITAN 70 退化模型）
│   ├── preprocess_afm_data.py      # 实验 AFM 数据 → .npz（供推理）
│   ├── moire_pipeline.py           # FFT baseline + moiré 仿真
│   ├── eval_compare.py             # FFT vs CNN 对比评估
│   ├── graded_eval.py              # 分级退化测试
│   ├── distortion_sweep.py         # 畸变压力测试
│   ├── analyze_fft_failure.py      # FFT 失效分析
│   ├── outputs/                    # 输出目录（模型、日志、图表）
│   ├── docs/                       # 实验记录与 AFM 需求说明
│   ├── tests/                      # pytest（core 模块等）
│   └── core/                       # 共享模块
│       ├── cnn.py                  # 模型构建、FFT 通道、MC Dropout 推理、LR 调度
│       ├── dual_stream.py          # 频域双流 CNN（供 train_cnn_v2 等使用）
│       ├── augment.py              # 训练期数据增强
│       ├── metrics.py              # 分层指标与评估报告
│       ├── moire_sim.py            # moiré 仿真（单/多通道）
│       ├── degrade.py              # 图像退化（含 TITAN 70 探针卷积）
│       ├── io_afm.py               # Cypher ES 数据导入 + AFM 预处理
│       ├── io_utils.py             # 数据集 / checkpoint I/O
│       ├── fonts.py                # CJK 字体设置
│       └── seed.py                 # 随机种子与 DataLoader worker
├── data/                           # 数据集目录（`moire_dataset.npz` 默认输出于此）
├── thesis/hust_bachelor_cse/       # 毕业论文 LaTeX 源文件
├── requirements.txt                # Python 依赖（与 pyproject 对齐）
├── pyproject.toml                  # 包元数据、可选依赖、控制台入口
└── README.md
```

安装为可编辑包后，可使用 `pyproject.toml` 中注册的命令（等价于在 `angle_cnn/` 下直接 `python` 调用对应模块的 `main`）：

- `moire-generate` → `dataset_generator`
- `moire-train` → `train_cnn`
- `moire-eval` → `eval_compare`

## 快速开始

### 1. 安装依赖

```bash
cd tmdc-project
pip install -r requirements.txt

# 可选：以可编辑方式安装包（启用上述 moire-* 命令）
pip install -e .

# 读取 Cypher ES 实验 .ibw 时（可选）
pip install igor2
# 或：pip install -e ".[afm]"
```

`requirements.txt` 中 PyTorch 版本针对较新 NVIDIA GPU（如 Blackwell）与 `torch.compile` 等特性留有区间；核心依赖还包括 `tqdm`、`psutil`（数据集生成与训练进度 / 资源监控）。

### 2. 生成数据集

```bash
cd angle_cnn

# 默认：3 通道，20000 样本，n_sim=256，多进程，TITAN 70 退化
python dataset_generator.py

# 对数均匀 θ 采样（小角度区域更密）
python dataset_generator.py --log-theta

# 高精度仿真分辨率 512×512
python dataset_generator.py --n-sim 512

# 单通道（向后兼容）
python dataset_generator.py --channels 1

# 自定义
python dataset_generator.py --samples 50000 --tip-radius 7.0 --channels 3 --workers 8
```

默认输出：`../data/moire_dataset.npz`（相对 `angle_cnn/` 目录）。

### 3. 训练 CNN（train_cnn.py v4）

主线脚本为 **v4**：多骨干（ResNet-18/34/50）、可选 **BF16** AMP、默认启用 **torch.compile**（无可用 C 编译器或失败时会自动跳过）、梯度裁剪、较大默认 batch、分层评估报告等。

```bash
# 默认：resnet18，CUDA 上 FP16 AMP，compile（若环境允许）
python train_cnn.py

# Blackwell / Ampere 等：BF16 AMP（需 --bf16；sm_80+ 原生 BF16）
python train_cnn.py --bf16

# 更大模型或更大 batch
python train_cnn.py --arch resnet34
python train_cnn.py --batch-size 1024

# FFT magnitude 额外通道（推荐与数据集通道数匹配使用）
python train_cnn.py --fft-channel --bf16

# MC Dropout 不确定性（示例）
python train_cnn.py --mc-samples 30

# 仅评估已有权重
python train_cnn.py --eval-only

# 调试：禁用 compile 或 AMP
python train_cnn.py --no-compile
python train_cnn.py --no-amp

# WSL 内存紧张时可：--fp16-data、减小 --batch-size / --num-workers / --prefetch-factor
```

**可选**：`train_cnn_v2.py` 提供 **双流**（`core/dual_stream.py`）等实验向选项，与主线默认流程不同，需单独阅读其参数说明。

### 4. FFT vs CNN 对比与退化测试

```bash
# 默认 legacy：CNN 用测试集 .npz，FFT 按标签重算 512（与旧表兼容）
python eval_compare.py --mc-samples 30

# 更公平的 paired：同一随机退化场景 → CNN 中心 128 裁剪 + FFT 用同场景 height@512
python eval_compare.py --eval-mode paired --paired-seed 12345 --mc-samples 30 --fusion-unc-scale 0.5

python graded_eval.py --mc-samples 30
```

### 5. 实验数据预处理（推理用 .npz）

将实验室导出的 AFM 文件转为项目统一的 `(N, C, 128, 128)` `.npz`，便于用训练好的模型做推理：

```bash
python preprocess_afm_data.py --input path/to/sample.spm --output ../data/afm_processed/
python preprocess_afm_data.py --input-dir ../data/afm_raw/ --output ../data/afm_processed/ --preview
```

支持格式以脚本内说明为准（如 `.spm`、`.ibw`、常见图像与表格等）。

### 6. 在代码中导入 Cypher ES 数据（.ibw）

```python
from core.io_afm import load_cypher_image, preprocess_afm_image, find_channel

channels = load_cypher_image("scan_001.ibw")
height = find_channel(channels, "height")
phase = find_channel(channels, "phase")
height_clean = preprocess_afm_image(height)
```

需安装 `igor2`。

### 7. 运行测试（可选）

```bash
pip install -e ".[dev]"
pytest angle_cnn/tests -q
```

## 主要功能

### 多通道仿真（Tapping Mode）

| 通道 | 物理含义 | 仿真模型 |
|------|----------|----------|
| Height | 表面形貌（moiré 包络） | R_sharp × cos(Φ_recon) |
| Phase | 能量耗散（AB/BA 畴对比） | 连续畴符号 × 重构强度 |
| Amplitude | 反馈误差（表面梯度） | 1 - ‖∇height‖ / max |

### 退化模型（TITAN 70 + Cypher ES）

1. **探针卷积** — TITAN 70 球形针尖（R ≈ 7 nm）
2. **高斯噪声**、**1/f 噪声**、**高斯模糊**、**仿射畸变**、**行噪声**、**背景倾斜**
3. **反馈振荡**（Tapping Mode）、**扫描方向偏移**（trace/retrace）

### MC Dropout 不确定性

推理时保持 Dropout，多次前向得到均值与方差，用于角度不确定度估计。

### AFM 导入管线（io_afm）

`.ibw` 读取、平面校正、行对齐、异常行处理、归一化等（依赖 `igor2`）。

## 超参数与默认值（摘要）

### dataset_generator.py（与脚本内常量一致）

- 样本数、图像 128×128、仿真分辨率 `N_SIM`、`N_CHANNELS`、θ 范围、`TIP_RADIUS_NM`、`--log-theta` 等见脚本 `--help` 与文件头部说明。

### train_cnn.py（v4 典型默认值）

| 项 | 默认值（摘录） |
|----|------------------|
| batch_size | 512 |
| epochs | 200 |
| lr | 1e-3 |
| patience | 30 |
| dropout | 0.3 |
| Huber δ | 0.02 |
| MC 采样 | 30 |
| arch | resnet18 |
| Warmup | 5 epoch + Cosine |
| Mixup α | 0.2 |

AMP：默认在 CUDA 上使用 **FP16**；加 **`--bf16`** 时使用 bfloat16（适合 Ampere 及以上）。**`--fft-channel`** 为可选输入通道，与 `core/cnn.py` 中 `compute_fft_channel` 一致。

### 架构要点（与当前 `core/cnn.py` 一致）

- `conv1`：3×3 stride 2；`maxpool` 换为 `Identity`
- 回归头：全连接 + Dropout；可选 FFT 通道；多骨干 `resnet18` / `resnet34` / `resnet50`

## 输出文件

结果默认在 `angle_cnn/outputs/`：

| 文件 | 说明 |
|------|------|
| `best_model.pt` | 最佳权重（含 arch、通道数、FFT、dropout 等元数据） |
| `train_log.csv` / `train_log.png` | 训练日志与曲线 |
| `compare_scatter.png` / `compare_error.png` / `compare_report.txt` | FFT vs CNN 对比 |
| `graded_eval.png` / `graded_report.txt` 等 | 分级退化 |
| `distortion_sweep.png` / `distortion_sweep.csv` | 畸变扫描 |

## 技术细节

### 晶格重构模型

小角度（θ < 2°）下考虑 AB/BA 畴与连续强度插值，见 `core/moire_sim.py` 与论文引用。

### 探针卷积与 TITAN 70

Fourier 域低通近似针尖滤波；θ 较大、moiré 周期与针尖半径可比时退化更明显（详见原 README 表格与物理讨论）。

## 参考文献

- Weston, A. et al. Nat. Nanotechnol. 2020 — 晶格重构效应
- MoS₂ 晶格常数：a = 0.316 nm

## License

MIT License
