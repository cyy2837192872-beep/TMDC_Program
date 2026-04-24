#!/usr/bin/env python3
"""
distortion_sweep.py — FFT vs CNN 抗畸变能力扫描（修正版）
==========================================================

图像生成逻辑与 dataset_generator 共享 core.moire_sim / core.degrade，
确保 CNN 推理时的图像尺度与训练分布匹配。

运行方式
--------
    python distortion_sweep.py

输出
----
    outputs/distortion_sweep.png
    outputs/distortion_sweep.csv
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from angle_cnn.core.degrade import apply_affine_distortion, apply_tip_convolution
from angle_cnn.core.fonts import setup_matplotlib_cjk_font
from angle_cnn.core.io_utils import require_file
from angle_cnn.core.config import IMG_SIZE
from angle_cnn.core.physics import FIXED_FOV_NM, pixels_per_moire_period
from angle_cnn.core.moire_sim import synthesize_multichannel_moire, synthesize_reconstructed_moire
from angle_cnn.core.eval_utils import load_model_from_checkpoint, cnn_predict_single
from moire_pipeline import extract_angle_fft

# Will be set at runtime from checkpoint metadata
_N_CHANNELS = 1
_BASE_N_CHANNELS = 1
_ADD_FFT_CHANNEL = False

import matplotlib.pyplot as plt

setup_matplotlib_cjk_font()

OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 实验参数 ──────────────────────────────────────────────
TEST_THETAS = [0.5, 1.0, 2.0, 3.0, 5.0]
SHEAR_LEVELS = np.linspace(0.0, 0.25, 12)
N_TRIALS = 8
NOISE = 0.04
BLUR = 0.15
TIP_RADIUS_NM = 7.0


def make_image(theta_deg, shear, seed, n_channels=1):
    """
    生成与训练集一致的图像（固定 FOV + 连续晶格重构 + 受控畸变）。

    返回
    ----
    img_cnn   : CNN 输入 — (C, 128, 128) float32 或 (128, 128) float32
    img_512   : FFT 输入 — (512, 512) float32 (height channel)
    fov_nm    : 视野大小
    actual_ppp: 每周期像素数
    """
    rng = np.random.default_rng(seed)

    if n_channels > 1:
        ch_names = ("height", "phase", "amplitude")[:n_channels]
        ch_dict, fov_nm = synthesize_multichannel_moire(theta_deg, FIXED_FOV_NM, n=512, channels=ch_names)
    else:
        raw, fov_nm = synthesize_reconstructed_moire(theta_deg, FIXED_FOV_NM, n=512)
        ch_dict = {"height": raw}

    actual_ppp = pixels_per_moire_period(512, theta_deg, FIXED_FOV_NM)
    pixel_size_nm = fov_nm / 512

    sx = rng.uniform(-shear, shear) if shear > 0 else 0.0
    sy = rng.uniform(-shear, shear) if shear > 0 else 0.0
    sc = 1.0 + rng.uniform(0, shear * 0.4)

    for name in list(ch_dict.keys()):
        img = ch_dict[name]
        img = apply_affine_distortion(img, sx, sy, sc, 1.0 / sc)
        img = apply_tip_convolution(img, TIP_RADIUS_NM, pixel_size_nm)
        img = gaussian_filter(img, sigma=BLUR)
        ptp = img.max() - img.min()
        img = img + NOISE * ptp * rng.standard_normal(img.shape)
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        ch_dict[name] = img.astype(np.float32)

    img_512_height = ch_dict["height"]

    n = 512
    oy = (n - IMG_SIZE) // 2
    ox = (n - IMG_SIZE) // 2
    if n_channels > 1:
        ch_names = ("height", "phase", "amplitude")[:n_channels]
        crops = [ch_dict[name][oy:oy + IMG_SIZE, ox:ox + IMG_SIZE] for name in ch_names]
        img_cnn = np.stack(crops, axis=0)
    else:
        img_cnn = ch_dict["height"][oy:oy + IMG_SIZE, ox:ox + IMG_SIZE]

    return img_cnn, img_512_height, fov_nm, actual_ppp


def run_sweep(model, device, add_fft_channel=False):
    results = {theta: {"fft": [], "cnn": []} for theta in TEST_THETAS}
    total = len(TEST_THETAS) * len(SHEAR_LEVELS)
    done = 0

    for theta in TEST_THETAS:
        for shear in SHEAR_LEVELS:
            fft_errs, cnn_errs = [], []

            for trial in range(N_TRIALS):
                seed = trial * 10000 + int(theta * 100) + int(shear * 1000)
                img_128, img_512, fov_nm, actual_ppp = make_image(
                    theta, shear, seed, n_channels=_BASE_N_CHANNELS
                )

                th_fft, _, _ = extract_angle_fft(img_512, fov_nm, ppp=actual_ppp)
                if th_fft is not None:
                    fft_errs.append(abs(th_fft - theta))

                th_cnn = cnn_predict_single(model, img_128, device, add_fft_channel=add_fft_channel)
                cnn_errs.append(abs(th_cnn - theta))

            fft_mae = np.median(fft_errs) if fft_errs else np.nan
            cnn_mae = np.median(cnn_errs)

            results[theta]["fft"].append(fft_mae)
            results[theta]["cnn"].append(cnn_mae)

            done += 1
            print(
                f"  [{done:>3}/{total}]  theta={theta}  "
                f"shear={shear:.3f}  "
                f"FFT={fft_mae:.3f}  CNN={cnn_mae:.3f}"
            )

    return results


def plot_sweep(results):
    colors_fft = ["#1a6faf", "#3d9dd4", "#6bbceb", "#9dd4f5", "#c5e8fb"]
    colors_cnn = ["#c0392b", "#e05a4e", "#f08070", "#f8a898", "#fdd0c8"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        r"MoS$_2$ moiré：FFT vs CNN 抗畸变能力扫描",
        fontsize=13,
        fontweight="bold",
    )
    ax1, ax2 = axes

    for i, theta in enumerate(TEST_THETAS):
        ax1.plot(
            SHEAR_LEVELS,
            results[theta]["fft"],
            "o-",
            color=colors_fft[i],
            lw=2,
            ms=5,
            label=f"FFT θ={theta}°",
        )
        ax1.plot(
            SHEAR_LEVELS,
            results[theta]["cnn"],
            "s--",
            color=colors_cnn[i],
            lw=2,
            ms=5,
            label=f"CNN θ={theta}°",
        )

    ax1.axhline(0.1, color="gray", ls=":", lw=1.5, label="0.1° 基准线")
    ax1.set(
        xlabel="仿射畸变强度 (shear)",
        ylabel="中位角度误差 (°)",
        title="各角度下误差 vs 畸变强度",
    )
    ax1.legend(fontsize=8, ncol=2, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(SHEAR_LEVELS[0], SHEAR_LEVELS[-1])

    improvement = []
    for si in range(len(SHEAR_LEVELS)):
        fft_m = np.nanmean([results[t]["fft"][si] for t in TEST_THETAS])
        cnn_m = np.nanmean([results[t]["cnn"][si] for t in TEST_THETAS])
        improvement.append(fft_m / cnn_m if cnn_m > 1e-6 else np.nan)

    ax2.fill_between(SHEAR_LEVELS, 1, improvement, alpha=0.2, color="mediumseagreen")
    ax2.plot(
        SHEAR_LEVELS,
        improvement,
        "o-",
        color="mediumseagreen",
        lw=2.5,
        ms=7,
        label="CNN 精度提升倍数",
    )
    ax2.axhline(1, color="gray", ls="--", lw=1.2, label="FFT = CNN（无优势）")

    peak_idx = int(np.nanargmax(improvement))
    ax2.annotate(
        f"峰值 {improvement[peak_idx]:.1f}x\n(shear={SHEAR_LEVELS[peak_idx]:.2f})",
        xy=(SHEAR_LEVELS[peak_idx], improvement[peak_idx]),
        xytext=(SHEAR_LEVELS[peak_idx] + 0.02, improvement[peak_idx] * 0.88),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )

    ax2.set(
        xlabel="仿射畸变强度 (shear)",
        ylabel="FFT误差 / CNN误差（倍）",
        title="CNN 相对 FFT 精度提升（各角度平均）",
    )
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(SHEAR_LEVELS[0], SHEAR_LEVELS[-1])
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "distortion_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  已保存: {path}")
    plt.close()
    return improvement


def save_csv(results, improvement):
    path = os.path.join(OUT_DIR, "distortion_sweep.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["shear"]
        for t in TEST_THETAS:
            header += [f"fft_{t}deg", f"cnn_{t}deg"]
        header += ["improvement_x"]
        writer.writerow(header)
        for i, shear in enumerate(SHEAR_LEVELS):
            row = [f"{shear:.4f}"]
            for t in TEST_THETAS:
                row += [
                    f'{results[t]["fft"][i]:.4f}',
                    f'{results[t]["cnn"][i]:.4f}',
                ]
            row += [f"{improvement[i]:.2f}"]
            writer.writerow(row)
    print(f"  saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("MoS2 moire: Distortion Sweep (fixed version)")
    print("=" * 60)
    print(f"angles: {TEST_THETAS}")
    print(
        f"shear:  {SHEAR_LEVELS[0]:.2f} -> {SHEAR_LEVELS[-1]:.2f} ({len(SHEAR_LEVELS)} steps)"
    )
    print(f"trials: {N_TRIALS}  noise={NOISE}  blur={BLUR}px")
    print()

    MODEL_PATH = os.path.join(SCRIPT_DIR, "outputs", "best_model.pt")
    require_file(MODEL_PATH, "Model checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = load_model_from_checkpoint(MODEL_PATH, device)
    _BASE_N_CHANNELS = meta["base_n_channels"]
    add_fft = meta["add_fft_channel"]
    fft_str = "+FFT" if add_fft else ""
    print(f"device: {device}, channels: {meta['n_channels']}{fft_str}, model: {MODEL_PATH}\n")

    results = run_sweep(model, device, add_fft_channel=add_fft)
    improvement = plot_sweep(results)
    save_csv(results, improvement)

    print("\n" + "=" * 60)
    print(f"improvement at zero distortion: {improvement[0]:.1f}x")
    print(f"peak improvement:               {max(improvement):.1f}x")
    print(f"improvement at max distortion:  {improvement[-1]:.1f}x")
    print("=" * 60)
