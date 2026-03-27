#!/usr/bin/env python3
"""
eval_compare.py — MoS₂ moiré FFT vs CNN 角度提取对比评估
==========================================================

在同一批含退化的测试集图像上，同时运行 FFT 和 CNN 两种方法，
生成对比图和数值报告。

这是毕设的核心对比实验。

运行方式
--------
    python eval_compare.py

输出
----
    outputs/compare_scatter.png   — 预测值 vs 真实值散点图
    outputs/compare_error.png     — 误差分布对比图
    outputs/compare_report.txt    — 数值报告
"""

import os
import sys
import numpy as np
import torch

# ── 路径 ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'outputs')

DATASET_PATH = os.path.join(DATA_DIR, 'moire_dataset.npz')
MODEL_PATH   = os.path.join(OUT_DIR,  'best_model.pt')

# ── 导入自己的模块 ────────────────────────────────────────
sys.path.insert(0, SCRIPT_DIR)
from moire_pipeline import extract_angle_fft, A_NM   # MoS₂ a=0.316 nm
from train_cnn import build_model, THETA_MIN, THETA_MAX

# ── 中文字体 ──────────────────────────────────────────────
import matplotlib.font_manager as fm
from matplotlib import rcParams
import matplotlib.pyplot as plt

myfont_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(myfont_path)
font_name = fm.FontProperties(fname=myfont_path).get_name()
rcParams['font.sans-serif'] = [font_name]
rcParams['axes.unicode_minus'] = False


# ── CNN 推理 ──────────────────────────────────────────────

def cnn_predict(model, images, device, batch_size=64):
    """
    批量 CNN 推理

    Parameters
    ----------
    images : (N, H, W) float32 ndarray
    返回：(N,) float32，单位：度
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
            pred_norm = model(x).squeeze(1).cpu().numpy()
            pred_deg  = pred_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
            all_preds.append(pred_deg)

    return np.concatenate(all_preds)


# ── FFT 批量推理 ──────────────────────────────────────────

def fft_predict_batch(images, fovs):
    """
    对一批图像逐张运行 FFT 提取角度
    使用 MoS₂ 晶格常数（a=0.316 nm，从 moire_pipeline 导入）
    fovs : (N,) 每张图对应的真实视野大小（nm）
    """
    preds = []
    for i, (img, fov) in enumerate(zip(images, fovs)):
        th, unc, _ = extract_angle_fft(img, fov_nm=float(fov))
        if th is None:
            preds.append(np.nan)
        else:
            preds.append(float(th))

        if (i + 1) % 50 == 0:
            print(f'  FFT 进度: {i+1}/{len(images)}')

    return np.array(preds, dtype=np.float32)


# ── 误差统计 ──────────────────────────────────────────────

def error_stats(preds, labels, method_name):
    """计算并打印误差统计，返回有效误差数组"""
    valid  = ~np.isnan(preds)
    errors = np.abs(preds[valid] - labels[valid])
    fail_rate = (~valid).sum() / len(preds) * 100

    print(f'\n  [{method_name}]')
    print(f'    有效样本: {valid.sum()}/{len(preds)} '
          f'（失败率 {fail_rate:.1f}%）')
    print(f'    MAE:      {errors.mean():.4f}°')
    print(f'    中位误差: {np.median(errors):.4f}°')
    print(f'    std:      {errors.std():.4f}°')
    print(f'    90th pct: {np.percentile(errors, 90):.4f}°')
    print(f'    最大误差: {errors.max():.4f}°')

    return errors, valid


# ── 可视化 ────────────────────────────────────────────────

def plot_scatter(labels, fft_preds, cnn_preds):
    """预测值 vs 真实值散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r'MoS$_2$ moiré：FFT vs CNN 角度提取对比（测试集）', fontsize=13)

    theta_line = np.linspace(THETA_MIN, THETA_MAX, 100)

    # FFT
    valid_fft = ~np.isnan(fft_preds)
    ax1.scatter(labels[valid_fft], fft_preds[valid_fft],
                alpha=0.4, s=15, c='steelblue', label='FFT 提取')
    ax1.plot(theta_line, theta_line, 'r--', lw=1.5, label='理想 y=x')
    mae_fft = np.abs(fft_preds[valid_fft] - labels[valid_fft]).mean()
    ax1.set(xlabel='真实角度 θ (°)', ylabel='提取角度 (°)',
            title=f'FFT 方法  MAE={mae_fft:.3f}°')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.set_xlim(THETA_MIN - 0.2, THETA_MAX + 0.2)
    ax1.set_ylim(THETA_MIN - 0.5, THETA_MAX + 0.5)

    # CNN
    ax2.scatter(labels, cnn_preds,
                alpha=0.4, s=15, c='darkorange', label='CNN 提取')
    ax2.plot(theta_line, theta_line, 'r--', lw=1.5, label='理想 y=x')
    mae_cnn = np.abs(cnn_preds - labels).mean()
    ax2.set(xlabel='真实角度 θ (°)', ylabel='提取角度 (°)',
            title=f'CNN 方法  MAE={mae_cnn:.3f}°')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_xlim(THETA_MIN - 0.2, THETA_MAX + 0.2)
    ax2.set_ylim(THETA_MIN - 0.2, THETA_MAX + 0.2)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'compare_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\n  已保存: {path}')
    plt.close()


def plot_error_analysis(labels, fft_preds, cnn_preds):
    """误差分布 + 误差 vs θ 曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(r'MoS$_2$ moiré：FFT vs CNN 误差分析（测试集，含噪声+畸变+模糊）',
                 fontsize=13)

    valid_fft = ~np.isnan(fft_preds)
    err_fft = np.abs(fft_preds[valid_fft] - labels[valid_fft])
    err_cnn = np.abs(cnn_preds - labels)

    # 图1：误差直方图
    ax = axes[0]
    bins = np.linspace(0, max(err_fft.max(), err_cnn.max()) * 1.05, 40)
    ax.hist(err_fft, bins=bins, alpha=0.6, color='steelblue',
            label=f'FFT  MAE={err_fft.mean():.3f}°')
    ax.hist(err_cnn, bins=bins, alpha=0.6, color='darkorange',
            label=f'CNN  MAE={err_cnn.mean():.3f}°')
    ax.set(xlabel='|角度误差| (°)', ylabel='样本数', title='误差分布直方图')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # 图2：误差 vs θ（分箱统计）
    ax = axes[1]
    bins_theta    = np.linspace(THETA_MIN, THETA_MAX, 10)
    theta_centers = (bins_theta[:-1] + bins_theta[1:]) / 2

    fft_bin_mae, cnn_bin_mae = [], []
    for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
        mask_fft = valid_fft & (labels >= lo) & (labels < hi)
        fft_bin_mae.append(
            err_fft[mask_fft[valid_fft]].mean() if mask_fft.sum() > 0 else np.nan
        )
        mask_cnn = (labels >= lo) & (labels < hi)
        cnn_bin_mae.append(
            err_cnn[mask_cnn].mean() if mask_cnn.sum() > 0 else np.nan
        )

    ax.plot(theta_centers, fft_bin_mae, 'o-', c='steelblue',  ms=7, lw=2, label='FFT')
    ax.plot(theta_centers, cnn_bin_mae, 's-', c='darkorange', ms=7, lw=2, label='CNN')
    ax.axhline(0.1, color='red', ls='--', lw=1.2, label='0.1° 基准线')
    ax.set(xlabel='真实角度 θ (°)', ylabel='MAE (°)', title='各角度区间平均误差')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 图3：提升倍数
    ax = axes[2]
    improvement = np.array(fft_bin_mae) / np.array(cnn_bin_mae)
    ax.bar(theta_centers, improvement,
           width=(THETA_MAX - THETA_MIN) / 10 * 0.8,
           color='mediumseagreen', alpha=0.8)
    ax.axhline(1, color='gray', ls='--', lw=1)
    ax.set(xlabel='真实角度 θ (°)',
           ylabel='FFT误差 / CNN误差（倍）',
           title='CNN 相对 FFT 的精度提升倍数')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'compare_error.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  已保存: {path}')
    plt.close()


def save_report(labels, fft_preds, cnn_preds,
                fft_errors, cnn_errors, fft_valid):
    """保存数值报告"""
    path = os.path.join(OUT_DIR, 'compare_report.txt')
    with open(path, 'w') as f:
        f.write('MoS₂ moiré：FFT vs CNN 角度提取对比报告\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'材料：MoS₂ 转角同质结（a = {A_NM} nm）\n')
        f.write('测试集条件：含高斯噪声 + 仿射畸变 + 高斯模糊\n\n')

        f.write(f'{"指标":<20} {"FFT":>12} {"CNN":>12}\n')
        f.write('-' * 46 + '\n')
        f.write(f'{"有效样本数":<20} '
                f'{fft_valid.sum():>12} {len(labels):>12}\n')
        f.write(f'{"失败率":<20} '
                f'{(~fft_valid).sum()/len(labels)*100:>11.1f}% '
                f'{"0.0%":>12}\n')
        f.write(f'{"MAE (°)":<20} '
                f'{fft_errors.mean():>12.4f} '
                f'{cnn_errors.mean():>12.4f}\n')
        f.write(f'{"中位误差 (°)":<20} '
                f'{np.median(fft_errors):>12.4f} '
                f'{np.median(cnn_errors):>12.4f}\n')
        f.write(f'{"std (°)":<20} '
                f'{fft_errors.std():>12.4f} '
                f'{cnn_errors.std():>12.4f}\n')
        f.write(f'{"90th pct (°)":<20} '
                f'{np.percentile(fft_errors,90):>12.4f} '
                f'{np.percentile(cnn_errors,90):>12.4f}\n')
        f.write(f'{"最大误差 (°)":<20} '
                f'{fft_errors.max():>12.4f} '
                f'{cnn_errors.max():>12.4f}\n\n')

        improvement = fft_errors.mean() / cnn_errors.mean()
        f.write(f'CNN 比 FFT 精度提升：{improvement:.1f} 倍\n')

    print(f'  已保存: {path}')


# ── 主程序 ────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('MoS₂ moiré：FFT vs CNN 对比评估')
    print('=' * 60)

    # 1. 加载测试集
    print(f'\n加载数据集: {DATASET_PATH}')
    data = np.load(DATASET_PATH)
    images_test = data['images_test']
    labels_test = data['labels_test']
    fovs_test   = data['fovs_test']
    print(f'  测试集: {len(labels_test)} 样本')

    # 2. CNN 预测
    print('\n[1] CNN 推理...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    cnn_preds = cnn_predict(model, images_test, device)
    print(f'  完成，共 {len(cnn_preds)} 个预测')

    # 3. FFT 预测
    print('\n[2] FFT 提取（逐张，需要约1分钟）...')
    fft_preds = fft_predict_batch(images_test, fovs_test)
    n_fail = np.isnan(fft_preds).sum()
    print(f'  完成，失败 {n_fail}/{len(fft_preds)} 张')

    # 4. 误差统计
    print('\n[3] 误差统计:')
    fft_errors, fft_valid = error_stats(fft_preds, labels_test, 'FFT')
    cnn_errors, _         = error_stats(cnn_preds, labels_test, 'CNN')

    improvement = fft_errors.mean() / cnn_errors.mean()
    print(f'\n  >>> CNN 比 FFT 精度提升 {improvement:.1f} 倍 <<<')

    # 5. 画图
    print('\n[4] 生成对比图...')
    plot_scatter(labels_test, fft_preds, cnn_preds)
    plot_error_analysis(labels_test, fft_preds, cnn_preds)

    # 6. 保存报告
    save_report(labels_test, fft_preds, cnn_preds,
                fft_errors, cnn_errors, fft_valid)

    print('\n' + '=' * 60)
    print('完成！核心结论：')
    print(f'  FFT MAE: {fft_errors.mean():.3f}°')
    print(f'  CNN MAE: {cnn_errors.mean():.3f}°')
    print(f'  提升倍数: {improvement:.1f}x')
    print('=' * 60)