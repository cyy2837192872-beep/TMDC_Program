# MoS₂ moiré 转角提取：Sim-to-Real 优化路线图

## 项目状态（2026-04-25）

### 当前性能

| 方法 | 模拟数据 MAE | 状态 |
|------|-------------|------|
| FFT (512×512) | 0.0274° | ✅ 接近理论极限 |
| CNN (128×128, ResNet18) | 0.0544° | ✅ 可用基线 |
| 融合 (τ=0.01°) | 0.0275° | ✅ 已找到最优参数 |

### 已尝试但无效的方向

| 尝试 | 结果 | 原因 |
|------|------|------|
| `--scale-augment` (RandomZoom 0.75–1.25x) | CNN MAE 恶化至 0.0987° | 随机缩放改变 moiré 条纹像素周期但标签不变，破坏了物理对应关系 |
| 更多训练 epoch (200→更长) | 早停在 ~80 epoch | 模型已收敛，更多 epoch 无帮助 |
| 更大的 model (resnet34/50) | 收益甚微 | 任务复杂度低，更大模型过拟合 |

### 核心瓶颈

**没有真实 AFM 数据。** 所有训练和评估都在模拟数据上闭环。模拟数据和真实数据之间的 domain gap 是当前最大不确定性。

---

## 阶段 1：真实数据推理（已就绪）

收到真实 AFM 数据后，直接用以下命令跑推理：

```bash
# 预处理后的数据
python angle_cnn/inference_real_data.py --input-dir data/afm_processed/ --output real_results/

# 或原始 AFM 文件
python angle_cnn/inference_real_data.py --input-dir data/afm_raw/ --output real_results/ --raw --fov-nm 300
```

输出：
- `real_data_results.csv` — 每张图的 CNN/FFT/融合角度
- `preview/*_preview.png` — 每张图的三栏可视化

---

## 阶段 2：根据真实数据结果做针对性优化

真实数据到来后，先分析 CNN 和 FFT 各自的表现，然后根据出现的以下情况进行处理。

### 情况 A：FFT 失败率高，CNN 尚可

**现象**：大量真实图像的 FFT 角度为 NaN（FFT 峰检测失败），但 CNN 给出了合理预测。

**原因**：真实 AFM 噪声分布、扫描伪影与模拟退化不完全匹配，FFT 对周期性要求较高。

**方案：CNN 伪标签自训练 (Self-Training)**

1. **伪标签生成**：用当前 CNN 模型对无标签真实数据做 MC Dropout 推理，将均值作为伪标签
2. **筛选可靠样本**：只保留 MC 不确定性低于阈值（如 ±0.3°）的样本
3. **联合微调**：将伪标签样本与原始模拟数据混合，以小学习率（1e-5）微调 CNN

实现位置：`inference_real_data.py` 添加 `--self-train` 子命令。

```
python angle_cnn/inference_real_data.py --input-dir data/afm_processed/ \
    --self-train --self-train-lr 1e-5 --self-train-epochs 20 \
    --unc-threshold 0.3
```

**风险**：伪标签质量直接决定微调效果。若 CNN 本身在真实数据上 MAE 已经很高，伪标签会放大偏差。

---

### 情况 B：CNN 因 domain shift 显著劣化，FFT 仍可用

**现象**：CNN 的 MC 不确定性在真实数据上普遍较高（>0.5°），而 FFT 仍能稳定提取角度。

**原因**：模拟数据与真实 AFM 图像在底层特征分布（噪声纹理、探针卷积、背景倾斜）上存在差异。

**方案：扩宽退化范围 + 重训练**

1. **调整退化参数**：修改 `core/config.py` 中的 `DEFAULT_*_RANGE` 常量，扩展至能覆盖真实数据中观察到的退化程度

   ```
   # 例如，若真实数据噪声更大：
   DEFAULT_NOISE_RANGE = (0.0, 0.15)  →  (0.0, 0.3)
   DEFAULT_BLUR_RANGE  = (0.0, 0.5)   →  (0.0, 1.0)
   ```

2. **重新生成数据集**：
   ```bash
   python angle_cnn/dataset_generator.py --theta-sampling uniform --seed 42
   ```

3. **重新训练**：
   ```bash
   python angle_cnn/train_cnn.py
   ```

**优势**：改动最小，不需要新代码。

**缺点**：需要重建 8.2 GB 数据集 + 完整训练，时间成本高。过度扩宽退化范围可能反而降低模拟数据上的精度。

---

### 情况 C：两者都有明显偏差

**现象**：CNN 和 FFT 在真实数据上都表现不佳，且各自偏误差方向不同。

**方案：自适应融合 (Adaptive Fusion)**

不再使用固定的 τ 值，而是对每张图动态计算融合权重：

1. **FFT 置信度指标**：用 FFT 频谱中 moiré 峰的信噪比（peak SNR）衡量 FFT 可靠性
2. **CNN 置信度指标**：MC Dropout 标准差
3. **动态权重**：`w_cnn = f(unc_cnn, peak_snr_fft)`，其中 `f` 是一个简单的逻辑函数

实现位置：新建 `core/fusion.py`。

```python
# 示意逻辑
def adaptive_fusion(cnn_angle, cnn_unc, fft_angle, fft_peak_snr):
    if fft_angle is NaN or fft_peak_snr < 3.0:
        return cnn_angle, 1.0  # 完全信任 CNN
    w_cnn = cnn_unc / (cnn_unc + 0.01)  # CNN 不确定性越大权重越小
    w_fft = min(1 - w_cnn, fft_peak_snr / 20.0)  # FFT SNR 限制上限
    w_cnn = 1 - w_fft
    return w_cnn * cnn_angle + w_fft * fft_angle, w_cnn
```

**需要**：
- FFT 管道返回峰 SNR（修改 `moire_pipeline.py` 的 `extract_angle_fft`）
- 在真实数据上经验性地确定 SNR 阈值

---

## 阶段 3：论文与交付

当真实数据验证完成后，论文的核心主张应该是：

> **"在模拟数据上，FFT + CNN 融合达到 0.0275° MAE，接近 FFT 理论极限；同时建立了完整的真实 AFM → 转角预测管线（预处理 → CNN + FFT 并行推理 → 自适应融合），待真实数据验证后即可闭环。"**

### 关键数据表

| 指标 | FFT | CNN | 融合 (τ=0.01°) |
|------|-----|-----|----------------|
| MAE | 0.0274° | 0.0544° | 0.0275° |
| P95 | 0.0787° | 0.1443° | ~0.106° |
| 小角度 <1.5° | 0.0112° | 0.0403° | — |
| 大角度 >3.5° | 0.0414° | 0.0758° | — |
| CNN 残差 vs FFT 残差相关性 | — | 0.428 | — |

---

## 项目文件索引

| 文件 | 用途 |
|------|------|
| `train_cnn.py` | 主线 CNN 训练脚本 |
| `eval_compare.py` | FFT vs CNN 配对对比评估 |
| `inference_real_data.py` | **真实 AFM 数据推理管线（关键交付）** |
| `preprocess_afm_data.py` | 真实 AFM 数据预处理 |
| `core/io_afm.py` | Cypher ES .ibw 文件解析 + AFM 图像预处理 |
| `core/eval_fft.py` | FFT 评估工具（对比实验用） |
| `core/config.py` | 退化参数范围（sim-to-real 时修改这里） |
| `docs/afm_experiment_requirements.md` | 给导师的 AFM 实验需求 |

---

## 约束条件（请不要违反）

- 禁止修改的训练参数：`--no-compile`, `--ema-decay 0`（默认值）, `--bf16`（默认禁用）
- 数据集使用 `--theta-sampling uniform --seed 42`，不要用 `--theta-sampling log`
- 修改训练配置后必须运行 `python check_regression.py`
- 修改幅度不能使 baseline MAE 恶化超过 10%
