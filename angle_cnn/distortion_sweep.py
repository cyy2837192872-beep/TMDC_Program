#!/usr/bin/env python3
"""
distortion_sweep.py — FFT vs CNN 抗畸变能力扫描（修正版）
==========================================================

核心论文图：固定转角，逐步增大仿射畸变强度，
对比 FFT 和 CNN 的角度提取误差随畸变增强的变化曲线。

修正说明
--------
图像生成逻辑与 dataset_generator.py 完全一致（固定 FOV + 相同 ppp），
确保 CNN 推理时的图像尺度与训练分布匹配。

运行方式
--------
    python distortion_sweep.py

输出
----
    outputs/distortion_sweep.png
    outputs/distortion_sweep.csv
"""

import os
import sys
import numpy as np
import torch
import csv
from scipy.ndimage import gaussian_filter, map_coordinates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from moire_pipeline    import moire_period, extract_angle_fft, A_NM
from dataset_generator import generate_moire_raw, FIXED_FOV_NM, IMG_SIZE
from train_cnn         import build_model, THETA_MIN, THETA_MAX

import matplotlib.font_manager as fm
from matplotlib import rcParams
import matplotlib.pyplot as plt

myfont_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(myfont_path):
    fm.fontManager.addfont(myfont_path)
    rcParams['font.sans-serif'] = [fm.FontProperties(fname=myfont_path).get_name()]
rcParams['axes.unicode_minus'] = False

# ── 实验参数 ──────────────────────────────────────────────
TEST_THETAS  = [0.5, 1.0, 2.0, 3.0, 5.0]
SHEAR_LEVELS = np.linspace(0.0, 0.25, 12)
N_TRIALS     = 8
NOISE        = 0.15
BLUR         = 0.5


# ── 与训练完全一致的图像生成 ──────────────────────────────

def apply_affine(img, shear_x, shear_y, scale_x, scale_y):
    n = img.shape[0]
    cx, cy = n / 2, n / 2
    yi, xi = np.mgrid[0:n, 0:n]
    dx, dy = xi - cx, yi - cy
    xi_src = cx + scale_x * dx + shear_x * dy
    yi_src = cy + shear_y * dx + scale_y * dy
    coords = np.array([yi_src.ravel(), xi_src.ravel()])
    return map_coordinates(img, coords, order=1, mode='reflect').reshape(n, n)


def make_image(theta_deg, shear, seed):
    """
    生成与训练集完全一致的图像（固定 FOV + AB/BA 重构 + 受控畸变）
    返回：img_128 (128x128 float32), img_512 (512x512), fov_nm
    """
    rng = np.random.default_rng(seed)

    # 1. 固定 FOV -> ppp（与 dataset_generator 一致）
    actual_ppp = FIXED_FOV_NM / moire_period(theta_deg)
    actual_ppp = max(4.0, actual_ppp)

    img, fov_nm, L_nm = generate_moire_raw(theta_deg, ppp=actual_ppp, n=512)

    # 2. AB/BA 两态重构（theta < 2 度，与 dataset_generator 一致）
    if theta_deg < 2.0:
        theta_rad = np.radians(theta_deg)
        q = 2.0 * np.pi / L_nm
        fov = 512 * (L_nm / actual_ppp)
        x = np.linspace(0.0, fov, 512, endpoint=False)
        X, Y = np.meshgrid(x, x)
        psi = np.zeros((512, 512), dtype=complex)
        for k in range(3):
            phi = theta_rad / 2.0 + np.radians(60.0 * k)
            psi += np.exp(1j * (q * np.cos(phi) * X + q * np.sin(phi) * Y))
        strength    = np.clip(1.0 - theta_deg / 2.0, 0.0, 1.0)
        alpha       = 1.0 + strength * 8.0
        R           = np.abs(psi)
        Phi         = np.angle(psi)
        R_sharp     = np.tanh(alpha * R / R.max()) * R.max()
        domain_sign = np.sign(np.imag(psi))
        phase_quant = (domain_sign + 1) / 2 * np.pi
        Phi_recon   = (1 - strength) * Phi + strength * phase_quant
        img         = R_sharp * np.cos(Phi_recon)

    # 3. 仿射畸变（shear 幅度受控，方向随 seed 随机）
    sx = rng.uniform(-shear, shear) if shear > 0 else 0.0
    sy = rng.uniform(-shear, shear) if shear > 0 else 0.0
    sc = 1.0 + rng.uniform(0, shear * 0.4)
    img_dist = apply_affine(img, sx, sy, sc, 1.0 / sc)

    # 4. 模糊 + 噪声
    img_dist = gaussian_filter(img_dist, sigma=BLUR)
    ptp = img_dist.max() - img_dist.min()
    img_dist = img_dist + NOISE * ptp * rng.standard_normal(img_dist.shape)

    # 5. 归一化
    img_dist = (img_dist - img_dist.min()) / (img_dist.max() - img_dist.min() + 1e-9)
    img_dist = img_dist.astype(np.float32)

    # 6. 中心裁剪 -> 128x128（与训练一致）
    n = img_dist.shape[0]
    oy = (n - IMG_SIZE) // 2
    ox = (n - IMG_SIZE) // 2
    img_128 = img_dist[oy:oy+IMG_SIZE, ox:ox+IMG_SIZE]

    return img_128, img_dist, fov_nm, actual_ppp


def cnn_predict(model, img_128, device):
    x = torch.from_numpy(img_128).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_norm = model(x).item()
    return pred_norm * (THETA_MAX - THETA_MIN) + THETA_MIN


# ── 主扫描循环 ────────────────────────────────────────────

def run_sweep(model, device):
    results = {theta: {'fft': [], 'cnn': []} for theta in TEST_THETAS}
    total = len(TEST_THETAS) * len(SHEAR_LEVELS)
    done  = 0

    for theta in TEST_THETAS:
        for shear in SHEAR_LEVELS:
            fft_errs, cnn_errs = [], []

            for trial in range(N_TRIALS):
                seed = trial * 10000 + int(theta * 100) + int(shear * 1000)
                img_128, img_512, fov_nm, actual_ppp = make_image(theta, shear, seed)

                # FFT 在完整 512px 图上运行
                th_fft, _, _ = extract_angle_fft(img_512, fov_nm, ppp=actual_ppp)
                if th_fft is not None:
                    fft_errs.append(abs(th_fft - theta))

                # CNN 在 128px 裁剪图上运行
                th_cnn = cnn_predict(model, img_128, device)
                cnn_errs.append(abs(th_cnn - theta))

            fft_mae = np.median(fft_errs) if fft_errs else np.nan
            cnn_mae = np.median(cnn_errs)

            results[theta]['fft'].append(fft_mae)
            results[theta]['cnn'].append(cnn_mae)

            done += 1
            print(f'  [{done:>3}/{total}]  theta={theta}  '
                  f'shear={shear:.3f}  '
                  f'FFT={fft_mae:.3f}  CNN={cnn_mae:.3f}')

    return results


# ── 可视化 ────────────────────────────────────────────────

def plot_sweep(results):
    colors_fft = ['#1a6faf', '#3d9dd4', '#6bbceb', '#9dd4f5', '#c5e8fb']
    colors_cnn = ['#c0392b', '#e05a4e', '#f08070', '#f8a898', '#fdd0c8']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        r'MoS$_2$ moire: FFT vs CNN Anti-Distortion Comparison',
        fontsize=13, fontweight='bold'
    )
    ax1, ax2 = axes

    for i, theta in enumerate(TEST_THETAS):
        ax1.plot(SHEAR_LEVELS, results[theta]['fft'],
                 'o-', color=colors_fft[i], lw=2, ms=5,
                 label=f'FFT theta={theta}')
        ax1.plot(SHEAR_LEVELS, results[theta]['cnn'],
                 's--', color=colors_cnn[i], lw=2, ms=5,
                 label=f'CNN theta={theta}')

    ax1.axhline(0.1, color='gray', ls=':', lw=1.5, label='0.1 deg baseline')
    ax1.set(xlabel='Affine distortion strength (shear)',
            ylabel='Median angle error (deg)',
            title='Error vs distortion by twist angle')
    ax1.legend(fontsize=8, ncol=2, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(SHEAR_LEVELS[0], SHEAR_LEVELS[-1])

    improvement = []
    for si in range(len(SHEAR_LEVELS)):
        fft_m = np.nanmean([results[t]['fft'][si] for t in TEST_THETAS])
        cnn_m = np.nanmean([results[t]['cnn'][si] for t in TEST_THETAS])
        improvement.append(fft_m / cnn_m if cnn_m > 1e-6 else np.nan)

    ax2.fill_between(SHEAR_LEVELS, 1, improvement, alpha=0.2, color='mediumseagreen')
    ax2.plot(SHEAR_LEVELS, improvement,
             'o-', color='mediumseagreen', lw=2.5, ms=7,
             label='CNN accuracy improvement over FFT')
    ax2.axhline(1, color='gray', ls='--', lw=1.2, label='FFT = CNN (no advantage)')

    peak_idx = int(np.nanargmax(improvement))
    ax2.annotate(
        f'Peak {improvement[peak_idx]:.1f}x\n(shear={SHEAR_LEVELS[peak_idx]:.2f})',
        xy=(SHEAR_LEVELS[peak_idx], improvement[peak_idx]),
        xytext=(SHEAR_LEVELS[peak_idx] + 0.02, improvement[peak_idx] * 0.88),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=10
    )

    ax2.set(xlabel='Affine distortion strength (shear)',
            ylabel='FFT error / CNN error (x)',
            title='CNN vs FFT accuracy improvement (averaged over angles)')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(SHEAR_LEVELS[0], SHEAR_LEVELS[-1])
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'distortion_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\n  saved: {path}')
    plt.close()
    return improvement


def save_csv(results, improvement):
    path = os.path.join(OUT_DIR, 'distortion_sweep.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['shear']
        for t in TEST_THETAS:
            header += [f'fft_{t}deg', f'cnn_{t}deg']
        header += ['improvement_x']
        writer.writerow(header)
        for i, shear in enumerate(SHEAR_LEVELS):
            row = [f'{shear:.4f}']
            for t in TEST_THETAS:
                row += [f'{results[t]["fft"][i]:.4f}',
                        f'{results[t]["cnn"][i]:.4f}']
            row += [f'{improvement[i]:.2f}']
            writer.writerow(row)
    print(f'  saved: {path}')


# ── 主程序 ────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('MoS2 moire: Distortion Sweep (fixed version)')
    print('=' * 60)
    print(f'angles: {TEST_THETAS}')
    print(f'shear:  {SHEAR_LEVELS[0]:.2f} -> {SHEAR_LEVELS[-1]:.2f} ({len(SHEAR_LEVELS)} steps)')
    print(f'trials: {N_TRIALS}  noise={NOISE}  blur={BLUR}px')
    print()

    MODEL_PATH = os.path.join(SCRIPT_DIR, 'outputs', 'best_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f'device: {device},  model: {MODEL_PATH}\n')

    results     = run_sweep(model, device)
    improvement = plot_sweep(results)
    save_csv(results, improvement)

    print('\n' + '=' * 60)
    print(f'improvement at zero distortion: {improvement[0]:.1f}x')
    print(f'peak improvement:               {max(improvement):.1f}x')
    print(f'improvement at max distortion:  {improvement[-1]:.1f}x')
    print('=' * 60)