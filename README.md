# MoS₂ Moiré 转角提取项目

本项目实现了从 MoS₂ 转角同质结 moiré 图像中提取扭转角度的两种方法：
- **FFT 方法**：基于快速傅里叶变换的传统图像处理方法（baseline）
- **CNN 方法**：基于深度学习的卷积神经网络回归方法

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

## 项目结构

```
tmdc-project/
├── angle_cnn/                 # 主要代码目录
│   ├── train_cnn.py          # CNN 训练（多通道 + MC Dropout）
│   ├── dataset_generator.py  # 数据集生成（多通道 + TITAN 70 退化模型）
│   ├── moire_pipeline.py     # FFT baseline + moiré 仿真
│   ├── eval_compare.py       # FFT vs CNN 对比评估
│   ├── graded_eval.py        # 分级退化测试
│   ├── distortion_sweep.py   # 畸变压力测试
│   ├── analyze_fft_failure.py # FFT 失效分析
│   ├── outputs/              # 输出目录（模型、日志、图表）
│   └── core/                 # 共享工具模块
│       ├── moire_sim.py      # moiré 仿真（单/多通道）
│       ├── degrade.py        # 图像退化（含 TITAN 70 探针卷积）
│       ├── io_afm.py         # Cypher ES 数据导入 + AFM 预处理
│       ├── io_utils.py       # I/O 工具
│       ├── fonts.py          # CJK 字体设置
│       └── seed.py           # 随机种子管理
├── data/                      # 数据集目录（自动生成）
├── thesis/                    # 论文 LaTeX 源文件
├── requirements.txt           # Python 依赖
├── pyproject.toml             # 项目配置
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd tmdc-project
pip install -r requirements.txt

# 如需导入 Cypher ES 实验数据（.ibw 文件）：
pip install igor2
```

### 2. 生成数据集

```bash
cd angle_cnn

# 默认：3ch，20000样本，n_sim=256（快速），自动多进程，TITAN 70
python dataset_generator.py

# 对数均匀 θ 采样（小角度区域更密集）
python dataset_generator.py --log-theta

# 高精度模式（仿真 512×512）
python dataset_generator.py --n-sim 512

# 单通道模式（向后兼容）
python dataset_generator.py --channels 1

# 自定义参数
python dataset_generator.py --samples 50000 --tip-radius 7.0 --channels 3 --workers 8
```

### 3. 训练 CNN 模型

```bash
# v3 默认配置（3x3 conv1, no maxpool, Huber δ=0.02, Mixup, Warmup+Cosine）
python train_cnn.py

# 启用 FFT magnitude 额外通道（推荐）
python train_cnn.py --fft-channel

# 带 MC Dropout 不确定性评估
python train_cnn.py --fft-channel --mc-samples 30

# 自定义超参数
python train_cnn.py --epochs 200 --lr 5e-4 --batch-size 32 --dropout 0.3 --huber-delta 0.02
```

### 4. FFT vs CNN 对比评估

```bash
# 含 MC Dropout 不确定性
python eval_compare.py --mc-samples 30

# 分级退化测试
python graded_eval.py --mc-samples 30
```

### 5. 导入 Cypher ES 实验数据

```python
from core.io_afm import load_cypher_image, preprocess_afm_image, find_channel

# 加载 .ibw 文件
channels = load_cypher_image("scan_001.ibw")

# 获取各通道
height = find_channel(channels, "height")
phase = find_channel(channels, "phase")

# AFM 标准预处理
height_clean = preprocess_afm_image(height)
```

## 主要功能

### 多通道仿真（Tapping Mode）

支持 3 个 Tapping Mode 成像通道：
| 通道 | 物理含义 | 仿真模型 |
|------|----------|----------|
| Height | 表面形貌（moiré 包络） | R_sharp × cos(Φ_recon) |
| Phase | 能量耗散（AB/BA 畴对比） | 连续畴符号 × 重构强度 |
| Amplitude | 反馈误差（表面梯度） | 1 - \|∇height\| / max |

### 退化模型（TITAN 70 + Cypher ES）

模拟真实 AFM 图像退化：
1. **探针卷积** — TITAN 70 球形针尖（R ≈ 7 nm）的分辨率限制
2. **高斯噪声** — 电子噪声
3. **1/f 噪声** — 电子学漂移 + 热漂移
4. **高斯模糊** — 有限空间分辨率
5. **仿射畸变** — 热漂移、压电非线性
6. **行噪声** — 扫描线不均匀
7. **背景倾斜** — 样品倾斜
8. **反馈振荡** — Tapping Mode Z-feedback ringing
9. **扫描方向偏移** — Trace/retrace 残余

### MC Dropout 不确定性量化

推理时保持 Dropout 层激活，多次前向传播估计预测不确定性：
- 均值 → 角度预测值
- 标准差 → 认识不确定性（epistemic uncertainty）

### AFM 数据导入管线

支持 Cypher ES / Asylum Research 输出格式：
- `.ibw` 文件读取（需 `igor2` 库）
- 平面校正（plane levelling）
- 行对齐（line-by-line flattening）
- 异常行修复（scar removal）
- 归一化

## 超参数配置

### dataset_generator.py
```python
TOTAL_SAMPLES = 20000     # 样本总数
IMG_SIZE = 128            # CNN 输入尺寸
N_SIM = 256               # 仿真分辨率（256=快速，512=高精度）
N_CHANNELS = 3            # 通道数 (1/2/3)
TIP_RADIUS_NM = 7.0       # TITAN 70 探针半径
THETA_MIN = 0.5           # 最小转角（度）
THETA_MAX = 5.0           # 最大转角（度）
LOG_THETA = False         # 对数均匀 θ 采样
```

### train_cnn.py (v3)
```python
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
DROPOUT = 0.3             # MC Dropout 率
MC_SAMPLES = 30           # MC 采样次数
HUBER_DELTA = 0.02        # Huber loss delta
MIXUP_ALPHA = 0.2         # Mixup 强度
WARMUP_EPOCHS = 5         # LR warmup 轮数
FFT_CHANNEL = False       # FFT magnitude 额外通道
```

### v3 架构改进
- **conv1**: 7×7 → 3×3（保留 moiré 高频信息）
- **maxpool**: 移除（避免 128×128 输入过度下采样）
- **输出层**: 移除 Sigmoid（消除边界梯度消失）
- **FFT 通道**: 可选的 log FFT magnitude 输入通道
- **Mixup**: 数据增强（α=0.2）
- **LR 调度**: Warmup 5ep + Cosine Annealing
- **在线增强**: 亮度/对比度抖动（±10%）

## 输出文件

训练和评估结果保存在 `angle_cnn/outputs/`：

| 文件 | 说明 |
|------|------|
| `best_model.pt` | 最佳模型权重（含通道数、FFT通道、dropout率等 v3 元数据） |
| `train_log.csv` | 训练日志 |
| `train_log.png` | 训练曲线 |
| `compare_scatter.png` | FFT vs CNN 散点对比（含不确定性误差棒） |
| `compare_error.png` | 误差分布分析 |
| `compare_report.txt` | 数值报告（含 MC Dropout 不确定性） |
| `graded_eval.png` | 分级退化测试 |
| `distortion_sweep.png` | 畸变鲁棒性测试 |

## 技术细节

### 晶格重构模型

小角度（θ < 2°）时考虑晶格重构效应：
- AB/BA 堆叠畴形成三角图案
- 使用连续插值 `strength = clip(1 - θ/2, 0, 1)` 消除硬切换

### 探针卷积模型

TITAN 70 球形针尖的分辨率限制通过 Fourier 空间低通滤波近似：
- 截止频率 ≈ 1/(2R)，其中 R = 7 nm
- 对于 θ > 3°（moiré 周期 < 6 nm），探针卷积显著影响图像质量
- 对于 θ < 1°（moiré 周期 > 18 nm），影响可忽略

### TITAN 70 分辨率限制

| θ (°) | moiré 周期 L (nm) | L / tip_radius | 可分辨性 |
|--------|-------------------|----------------|----------|
| 0.5    | 31.4              | 4.5            | 良好 |
| 1.0    | 15.8              | 2.3            | 可用 |
| 2.0    | 7.9               | 1.1            | 勉强 |
| 3.0    | 5.3               | 0.8            | 困难 |
| 5.0    | 3.2               | 0.5            | 几乎不可分辨 |

## 参考文献

- Weston, A. et al. Nat. Nanotechnol. 2020 — 晶格重构效应
- MoS₂ 晶格常数：a = 0.316 nm

## License

MIT License
