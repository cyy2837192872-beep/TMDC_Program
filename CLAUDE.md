# tmdc-project — MoS₂ 转角同质结角度提取

## 项目概述

MoS₂ 扭转角提取项目，结合 FFT 频域分析和 CNN 深度学习两种方法。
本科毕业设计，代码位于 `angle_cnn/`，论文 LaTeX 在 `thesis/`。

## 回归测试（必须遵守）

**每次修改 `train_cnn.py` 或 `angle_cnn/core/` 下的任何模块后，必须运行：**

```bash
cd angle_cnn
python check_regression.py
```

- 退出码 0 = 通过（允许 10% 以内的 MAE 恶化）
- 退出码 1 = 失败（精度退化超限，必须排查原因）
- 若改进了模型，运行 `python check_regression.py --save-baseline` 更新基准
- 当前基准 MAE ≤ 0.064°（2025-04-25）

**绝对不要做的事：**
- 不要重新生成数据集后不重新训练就提交
- 不要同时改多个变量（数据集 + 训练配置 + 模型架构），应逐个验证
- 不要忽略 check_regression 的失败结果

## 关键训练参数（不要随意修改）

- `--no-compile`：必须禁用 torch.compile（BF16 + reduce-overhead 会导致精度下降）
- `--ema-decay 0` 或不传：禁用 EMA（BF16 + EMA 导致数值不稳定）
- 不传 `--bf16`：使用 FP16 + GradScaler（BF16 尾数精度不足）
- 数据集：uniform 采样，seed=42，n_sim=512，3 通道

## 数据集

- 路径：`data/moire_dataset.npz`（~8.2 GB）
- 生成命令：`python angle_cnn/dataset_generator.py --theta-sampling uniform --seed 42`
- 不要用 `--theta-sampling log`，会导致大角度样本不足，CNN MAE 恶化

## 运行环境

- Python 3.13, PyTorch 2.11, RTX 5070 Ti (WSL2)
- 虚拟环境：`.venv/`
