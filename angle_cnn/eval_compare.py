#!/usr/bin/env python3
"""
eval_compare.py — MoS₂ moiré FFT vs CNN 角度提取公平对比评估（v3）
===================================================================

公平对比方法：
  - CNN：使用数据集中存储的 128×128 三通道裁剪图像（与训练一致）
  - FFT：为每个测试角度重新生成 512×512 高度图像并施加训练分布内退化，
         确保 FFT 获得足够的频率分辨率（与 graded_eval.py 保持一致）

这消除了之前版本中 FFT 只能使用 128×128 裁剪图的不公平限制。

运行方式
--------
    python eval_compare.py
    python eval_compare.py --mc-samples 50

输出
----
    outputs/compare_scatter.png
    outputs/compare_error.png
    outputs/compare_report.txt
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.degrade import (  # noqa: E402
    apply_affine_distortion,
    apply_background_tilt,
    apply_feedback_ringing,
    apply_oneoverf_noise,
    apply_row_noise,
    apply_scan_direction_offset,
    apply_tip_convolution,
)
from core.fonts import setup_matplotlib_cjk_font  # noqa: E402
from core.io_utils import (  # noqa: E402
    load_model_checkpoint,
    load_npz_dataset,
    require_file,
    state_dict_from_checkpoint,
)
from core.moire_sim import synthesize_reconstructed_moire  # noqa: E402
from dataset_generator import (  # noqa: E402
    DEFAULT_BLUR_RANGE,
    DEFAULT_NOISE_RANGE,
    DEFAULT_ONEOVERF_RANGE,
    DEFAULT_RINGING_RANGE,
    DEFAULT_ROW_NOISE_RANGE,
    DEFAULT_SCALE_RANGE,
    DEFAULT_SCAN_OFFSET_RANGE,
    DEFAULT_SHEAR_X_RANGE,
    DEFAULT_SHEAR_Y_RANGE,
    DEFAULT_TIP_RADIUS_NM,
    DEFAULT_TIP_RADIUS_RANGE,
    DEFAULT_TILT_AMP_RANGE,
)
from moire_pipeline import A_NM, extract_angle_fft, moire_period  # noqa: E402
from core.cnn import (  # noqa: E402
    THETA_MAX,
    THETA_MIN,
    build_model,
    compute_fft_channel,
    detect_n_channels,
    predict_with_uncertainty,
)

FIXED_FOV_NM = 10 * moire_period(THETA_MIN)

setup_matplotlib_cjk_font()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

DATASET_PATH = os.path.join(DATA_DIR, "moire_dataset.npz")
MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")


def _generate_fft_image_512(
    theta_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Generate a 512×512 height image with training-distribution degradation for FFT.

    Returns (img_512, fov_nm, actual_ppp).
    """
    raw, fov_nm = synthesize_reconstructed_moire(theta_deg, FIXED_FOV_NM, n=512)
    pixel_size_nm = fov_nm / 512
    actual_ppp = max(4.0, FIXED_FOV_NM / moire_period(theta_deg))

    img = raw.copy()

    tilt_amp = rng.uniform(*DEFAULT_TILT_AMP_RANGE)
    if tilt_amp > 0.01:
        img = apply_background_tilt(img, tilt_amp, rng.uniform(-1, 1), rng.uniform(-1, 1))

    shear_y = rng.uniform(*DEFAULT_SHEAR_Y_RANGE)
    shear_x = rng.uniform(*DEFAULT_SHEAR_X_RANGE)
    scale_x = rng.uniform(*DEFAULT_SCALE_RANGE)
    scale_y = rng.uniform(*DEFAULT_SCALE_RANGE)
    if abs(shear_y) > 0.001 or abs(shear_x) > 0.001:
        img = apply_affine_distortion(img, shear_x, shear_y, scale_x, scale_y)

    tip_r = rng.uniform(*DEFAULT_TIP_RADIUS_RANGE)
    if tip_r > 0:
        img = apply_tip_convolution(img, tip_r, pixel_size_nm)

    blur = rng.uniform(*DEFAULT_BLUR_RANGE)
    if blur > 0.1:
        img = gaussian_filter(img, sigma=blur)

    scan_offset = rng.uniform(*DEFAULT_SCAN_OFFSET_RANGE)
    if scan_offset > 0.003:
        img = apply_scan_direction_offset(img, scan_offset, rng)

    row_noise = rng.uniform(*DEFAULT_ROW_NOISE_RANGE)
    if row_noise > 0.01:
        img = apply_row_noise(img, row_noise, rng)

    ringing = rng.uniform(*DEFAULT_RINGING_RANGE)
    if ringing > 0.005:
        img = apply_feedback_ringing(img, ringing, rng)

    oneoverf = rng.uniform(*DEFAULT_ONEOVERF_RANGE)
    if oneoverf > 0.005:
        img = apply_oneoverf_noise(img, oneoverf, rng.uniform(0.8, 1.5), rng)

    noise = rng.uniform(*DEFAULT_NOISE_RANGE)
    if noise > 0.01:
        ptp = img.max() - img.min()
        img = img + noise * ptp * rng.standard_normal(img.shape)

    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return img.astype(np.float32), fov_nm, actual_ppp


def cnn_predict(model, images, device, batch_size=64, add_fft_channel=False):
    """images : (N, [C,] H, W) float32 → (N,) float32 degrees"""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]
            if batch.ndim == 3:
                x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
            else:
                x = torch.from_numpy(batch).float().to(device)
            if add_fft_channel:
                x = compute_fft_channel(x)
            pred_norm = model(x).squeeze(1).clamp(0, 1).cpu().numpy()
            pred_deg = pred_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
            all_preds.append(pred_deg)
    return np.concatenate(all_preds)


def cnn_predict_with_uncertainty(model, images, device, batch_size=64,
                                 mc_samples=30, add_fft_channel=False):
    """MC Dropout prediction returning (mean_deg, std_deg) arrays."""
    all_means, all_stds = [], []
    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        if batch.ndim == 3:
            x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
        else:
            x = torch.from_numpy(batch).float().to(device)
        if add_fft_channel:
            x = compute_fft_channel(x)
        mean_deg, std_deg = predict_with_uncertainty(model, x, n_samples=mc_samples)
        all_means.append(mean_deg)
        all_stds.append(std_deg)
    return np.concatenate(all_means), np.concatenate(all_stds)


def fft_predict_batch_512(
    labels: np.ndarray,
    seed: int = 12345,
) -> np.ndarray:
    """Generate 512×512 images on-the-fly and run FFT extraction.

    This gives FFT a fair chance by providing full-resolution images
    (matching ``graded_eval.py``), rather than the 128×128 crops stored
    in the dataset which are too small for reliable FFT peak detection.
    """
    n = len(labels)
    preds = []
    rng = np.random.default_rng(seed)

    for i in range(n):
        theta_deg = float(labels[i])
        img_512, fov_nm, actual_ppp = _generate_fft_image_512(theta_deg, rng)
        th, _, _ = extract_angle_fft(img_512, fov_nm=fov_nm, ppp=actual_ppp)
        preds.append(float(th) if th is not None else float("nan"))
        if (i + 1) % 50 == 0:
            print(f"  FFT 进度: {i+1}/{n}")

    return np.array(preds, dtype=np.float32)


def error_stats(preds, labels, method_name):
    valid = ~np.isnan(preds)
    errors = np.abs(preds[valid] - labels[valid])
    fail_rate = (~valid).sum() / len(preds) * 100

    print(f"\n  [{method_name}]")
    print(f"    有效样本: {valid.sum()}/{len(preds)} （失败率 {fail_rate:.1f}%）")
    print(f"    MAE:      {errors.mean():.4f}°")
    print(f"    中位误差: {np.median(errors):.4f}°")
    print(f"    std:      {errors.std():.4f}°")
    print(f"    90th pct: {np.percentile(errors, 90):.4f}°")
    print(f"    最大误差: {errors.max():.4f}°")

    return errors, valid


def plot_scatter(labels, fft_preds, cnn_preds, cnn_stds=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r"MoS$_2$ moiré：FFT(512px) vs CNN(128px) 公平对比", fontsize=13)

    theta_line = np.linspace(THETA_MIN, THETA_MAX, 100)

    valid_fft = ~np.isnan(fft_preds)
    ax1.scatter(labels[valid_fft], fft_preds[valid_fft], alpha=0.4, s=15, c="steelblue", label="FFT 提取")
    ax1.plot(theta_line, theta_line, "r--", lw=1.5, label="理想 y=x")
    mae_fft = np.abs(fft_preds[valid_fft] - labels[valid_fft]).mean()
    ax1.set(xlabel="真实角度 θ (°)", ylabel="提取角度 (°)", title=f"FFT 方法  MAE={mae_fft:.3f}°")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(THETA_MIN - 0.2, THETA_MAX + 0.2)
    ax1.set_ylim(THETA_MIN - 0.5, THETA_MAX + 0.5)

    ax2.scatter(labels, cnn_preds, alpha=0.4, s=15, c="darkorange", label="CNN 提取")
    if cnn_stds is not None:
        ax2.errorbar(labels, cnn_preds, yerr=cnn_stds, fmt="none", ecolor="darkorange", alpha=0.1, lw=0.5)
    ax2.plot(theta_line, theta_line, "r--", lw=1.5, label="理想 y=x")
    mae_cnn = np.abs(cnn_preds - labels).mean()
    unc_label = f"  ±{cnn_stds.mean():.3f}°" if cnn_stds is not None else ""
    ax2.set(xlabel="真实角度 θ (°)", ylabel="提取角度 (°)", title=f"CNN 方法  MAE={mae_cnn:.3f}°{unc_label}")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(THETA_MIN - 0.2, THETA_MAX + 0.2)
    ax2.set_ylim(THETA_MIN - 0.2, THETA_MAX + 0.2)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "compare_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  已保存: {path}")
    plt.close()


def plot_error_analysis(labels, fft_preds, cnn_preds):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(r"MoS$_2$ moiré：FFT(512px) vs CNN(128px) 误差分析", fontsize=13)

    valid_fft = ~np.isnan(fft_preds)
    err_fft = np.abs(fft_preds[valid_fft] - labels[valid_fft])
    err_cnn = np.abs(cnn_preds - labels)

    ax = axes[0]
    bins = np.linspace(0, max(err_fft.max(), err_cnn.max()) * 1.05, 40)
    ax.hist(err_fft, bins=bins, alpha=0.6, color="steelblue", label=f"FFT  MAE={err_fft.mean():.3f}°")
    ax.hist(err_cnn, bins=bins, alpha=0.6, color="darkorange", label=f"CNN  MAE={err_cnn.mean():.3f}°")
    ax.set(xlabel="|角度误差| (°)", ylabel="样本数", title="误差分布直方图")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins_theta = np.linspace(THETA_MIN, THETA_MAX, 10)
    theta_centers = (bins_theta[:-1] + bins_theta[1:]) / 2

    fft_bin_mae, cnn_bin_mae = [], []
    for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
        mask_fft = valid_fft & (labels >= lo) & (labels < hi)
        fft_bin_mae.append(err_fft[mask_fft[valid_fft]].mean() if mask_fft.sum() > 0 else np.nan)
        mask_cnn = (labels >= lo) & (labels < hi)
        cnn_bin_mae.append(err_cnn[mask_cnn].mean() if mask_cnn.sum() > 0 else np.nan)

    ax.plot(theta_centers, fft_bin_mae, "o-", c="steelblue", ms=7, lw=2, label="FFT")
    ax.plot(theta_centers, cnn_bin_mae, "s-", c="darkorange", ms=7, lw=2, label="CNN")
    ax.axhline(0.1, color="red", ls="--", lw=1.2, label="0.1° 基准线")
    ax.set(xlabel="真实角度 θ (°)", ylabel="MAE (°)", title="各角度区间平均误差")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    fft_arr = np.array(fft_bin_mae, dtype=float)
    cnn_arr = np.array(cnn_bin_mae, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        improvement = np.where(
            (cnn_arr > 1e-9) & ~np.isnan(fft_arr) & ~np.isnan(cnn_arr),
            fft_arr / cnn_arr, np.nan,
        )
    valid_mask = ~np.isnan(improvement)
    ax.bar(theta_centers[valid_mask], improvement[valid_mask],
           width=(THETA_MAX - THETA_MIN) / 10 * 0.8, color="mediumseagreen", alpha=0.8)
    ax.axhline(1, color="gray", ls="--", lw=1)
    ax.set(xlabel="真实角度 θ (°)", ylabel="FFT误差 / CNN误差（倍）", title="CNN 相对 FFT 的精度提升倍数")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "compare_error.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  已保存: {path}")
    plt.close()


def save_report(labels, fft_preds, cnn_preds, fft_errors, cnn_errors, fft_valid, cnn_stds=None):
    path = os.path.join(OUT_DIR, "compare_report.txt")
    with open(path, "w") as f:
        f.write("MoS₂ moiré：FFT vs CNN 角度提取公平对比报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"材料：MoS₂ 转角同质结（a = {A_NM} nm）\n")
        f.write("对比方法：\n")
        f.write("  CNN：128×128 三通道数据集图像（与训练一致）\n")
        f.write("  FFT：512×512 高度图重新生成（训练分布内退化）\n")
        f.write("退化条件：高斯噪声 + 仿射畸变 + 高斯模糊 + 探针卷积 + 1/f噪声\n\n")

        f.write(f'{"指标":<20} {"FFT":>12} {"CNN":>12}\n')
        f.write("-" * 46 + "\n")
        f.write(f'{"有效样本数":<20} {fft_valid.sum():>12} {len(labels):>12}\n')
        f.write(f'{"失败率":<20} {(~fft_valid).sum()/len(labels)*100:>11.1f}% {"0.0%":>12}\n')
        f.write(f'{"MAE (°)":<20} {fft_errors.mean():>12.4f} {cnn_errors.mean():>12.4f}\n')
        f.write(f'{"中位误差 (°)":<20} {np.median(fft_errors):>12.4f} {np.median(cnn_errors):>12.4f}\n')
        f.write(f'{"std (°)":<20} {fft_errors.std():>12.4f} {cnn_errors.std():>12.4f}\n')
        f.write(f'{"90th pct (°)":<20} {np.percentile(fft_errors,90):>12.4f} {np.percentile(cnn_errors,90):>12.4f}\n')
        f.write(f'{"最大误差 (°)":<20} {fft_errors.max():>12.4f} {cnn_errors.max():>12.4f}\n\n')

        improvement = fft_errors.mean() / cnn_errors.mean()
        f.write(f"CNN 比 FFT 精度提升：{improvement:.1f} 倍\n")

        if cnn_stds is not None:
            f.write(f"\nMC Dropout 不确定性:\n")
            f.write(f"  平均不确定性: ±{cnn_stds.mean():.4f}°\n")
            f.write(f"  不确定性范围: [{cnn_stds.min():.4f}°, {cnn_stds.max():.4f}°]\n")

    print(f"  已保存: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-samples", type=int, default=30, help="MC Dropout 采样次数 (0=禁用)")
    cli_args = parser.parse_args()

    print("=" * 60)
    print("MoS₂ moiré：FFT(512px) vs CNN(128px) 公平对比评估（v3）")
    print("=" * 60)

    print(f"\n加载数据集: {DATASET_PATH}")
    data = load_npz_dataset(DATASET_PATH)
    images_test = data["images_test"]
    labels_test = data["labels_test"]
    fovs_test = data["fovs_test"]
    n_ch = detect_n_channels(images_test)
    print(f"  测试集: {len(labels_test)} 样本，{n_ch} 通道")

    print("\n[1] CNN 推理...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_file(MODEL_PATH, "Model checkpoint")

    ckpt = load_model_checkpoint(MODEL_PATH, map_location=device)
    ckpt_n_ch = ckpt.get("n_channels", 1) if isinstance(ckpt, dict) else 1
    ckpt_dropout = ckpt.get("dropout", 0.3) if isinstance(ckpt, dict) else 0.3
    add_fft = ckpt.get("add_fft_channel", False) if isinstance(ckpt, dict) else False
    model = build_model(n_channels=ckpt_n_ch, dropout=ckpt_dropout).to(device)
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
    if add_fft:
        print(f"  模型含 FFT 通道（n_channels={ckpt_n_ch}）")

    cnn_stds = None
    if cli_args.mc_samples > 0:
        cnn_preds, cnn_stds = cnn_predict_with_uncertainty(
            model, images_test, device, mc_samples=cli_args.mc_samples,
            add_fft_channel=add_fft,
        )
        print(f"  MC Dropout ({cli_args.mc_samples} 采样) 完成")
        print(f"  平均不确定性: ±{cnn_stds.mean():.4f}°")
    else:
        cnn_preds = cnn_predict(model, images_test, device, add_fft_channel=add_fft)
    print(f"  完成，共 {len(cnn_preds)} 个预测")

    print("\n[2] FFT 提取（512×512 重新生成，公平对比）...")
    print("  注：为每个测试角度重新生成 512×512 图像并施加训练分布内退化，")
    print("      确保 FFT 获得足够的频率分辨率，与 graded_eval 方法一致。")
    fft_preds = fft_predict_batch_512(labels_test)
    n_fail = np.isnan(fft_preds).sum()
    print(f"  完成，失败 {n_fail}/{len(fft_preds)} 张")

    print("\n[3] 误差统计:")
    fft_errors, fft_valid = error_stats(fft_preds, labels_test, "FFT")
    cnn_errors, _ = error_stats(cnn_preds, labels_test, "CNN")

    improvement = fft_errors.mean() / cnn_errors.mean()
    print(f"\n  >>> CNN 比 FFT 精度提升 {improvement:.1f} 倍 <<<")

    print("\n[4] 生成对比图...")
    plot_scatter(labels_test, fft_preds, cnn_preds, cnn_stds=cnn_stds)
    plot_error_analysis(labels_test, fft_preds, cnn_preds)
    save_report(labels_test, fft_preds, cnn_preds, fft_errors, cnn_errors, fft_valid, cnn_stds=cnn_stds)

    print("\n" + "=" * 60)
    print("完成！核心结论：")
    print(f"  FFT MAE: {fft_errors.mean():.3f}°")
    print(f"  CNN MAE: {cnn_errors.mean():.3f}°")
    print(f"  提升倍数: {improvement:.1f}x")
    if cnn_stds is not None:
        print(f"  CNN 不确定性: ±{cnn_stds.mean():.3f}°")
    print("=" * 60)
    print("\n下一步：python distortion_sweep.py")
