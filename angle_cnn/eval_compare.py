#!/usr/bin/env python3
"""
eval_compare.py — MoS₂ moiré FFT vs CNN 角度提取公平对比评估（v3）
===================================================================

两种评测口径（务必在论文中写清用的是哪一种）：

1) ``legacy``（默认）
  - CNN：使用 .npz 中已生成的 128×128 图像（与训练一致）
  - FFT：按标签 θ 单独重算 512×512 退化图再提角
  - 说明：CNN 与 FFT **不是同一张物理场景**，仅角度标签相同；便于快速复现旧表。

2) ``paired``（更公平）
  - 对每个测试角，用与数据集 **相同** 的 ``generate_sample`` 退化管线在 512 上生成同一场景，
    CNN 取 **中心裁剪** 128，FFT 用该场景的 **height 全幅 512**。
  - 说明：仍用真值 θ 计算 ``ppp = n·L/fov``（与 ``extract_angle_fft`` 定义一致）；若需完全无标签先验，应再实现
    “由图像自估计周期/网格”的 FFT 设置（可作为后续工作）。

融合（可选，需 ``--mc-samples > 0`` 以得到 σ_cnn）::

    θ_final = w(σ_cnn)·θ_fft + (1-w(σ_cnn))·θ_cnn,
    w = σ_cnn / (σ_cnn + τ)（σ 越大越信 FFT）

运行方式
--------
    python eval_compare.py
    python eval_compare.py --eval-mode paired --paired-seed 12345 --mc-samples 50

输出
----
    outputs/compare_scatter.png
    outputs/compare_error.png
    outputs/compare_report.txt
    outputs/compare_summary.csv
"""

from __future__ import annotations

import os
import csv

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from angle_cnn.core.eval_fft import (
    extract_angle_fft_robust,
    fft_predict_batch_512,
    generate_fft_image_512,
)
from angle_cnn.core.fonts import setup_matplotlib_cjk_font
from angle_cnn.core.io_utils import (
    load_model_checkpoint,
    load_npz_dataset,
    require_file,
    state_dict_from_checkpoint,
)
from dataset_generator import (
    _subseed,
    generate_sample_paired_cnn_fft,
)
from angle_cnn.core.config import DEFAULT_TIP_RADIUS_RANGE, THETA_MIN, THETA_MAX
from angle_cnn.core.physics import A_NM, FIXED_FOV_NM
from angle_cnn.core.cnn import (
    build_model,
    compute_fft_channel,
    detect_n_channels,
    predict_with_uncertainty,
)
from angle_cnn.core.metrics import (
    compute_calibration_metrics,
    compute_stratified_metrics,
)

setup_matplotlib_cjk_font()

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

DATASET_PATH = os.path.join(DATA_DIR, "moire_dataset.npz")
MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")



def fusion_predict(
    fft_preds: np.ndarray,
    cnn_preds: np.ndarray,
    cnn_stds: np.ndarray,
    unc_scale: float = 0.5,
) -> np.ndarray:
    """θ_final = w·θ_fft + (1-w)·θ_cnn，CNN 不确定度越大 w 越大（更偏向 FFT）。"""
    w = cnn_stds / (cnn_stds + unc_scale + 1e-9)
    w = np.clip(w, 0.0, 1.0)
    return np.where(
        np.isnan(fft_preds),
        cnn_preds,
        w * fft_preds + (1.0 - w) * cnn_preds,
    ).astype(np.float32)


def sweep_fusion_tau(
    fft_preds: np.ndarray,
    cnn_preds: np.ndarray,
    cnn_stds: np.ndarray,
    labels: np.ndarray,
    tau_min: float = 0.01,
    tau_max: float = 1.0,
    n_steps: int = 50,
) -> dict:
    """Sweep fusion τ and return optimal value + full sweep data for analysis."""
    taus = np.logspace(np.log10(tau_min), np.log10(tau_max), n_steps)
    maes, p95s = [], []
    best_tau, best_mae = float("nan"), float("inf")

    for tau in taus:
        fused = fusion_predict(fft_preds, cnn_preds, cnn_stds, unc_scale=float(tau))
        err = np.abs(fused - labels)
        mae = float(np.mean(err))
        p95 = float(np.percentile(err, 95))
        maes.append(mae)
        p95s.append(p95)
        if mae < best_mae:
            best_mae = mae
            best_tau = float(tau)

    return {
        "taus": taus,
        "maes": np.array(maes, dtype=float),
        "p95s": np.array(p95s, dtype=float),
        "best_tau": best_tau,
        "best_mae": best_mae,
    }


def plot_tau_sweep(sweep_data: dict) -> None:
    """Plot τ sweep: MAE and P95 vs τ."""
    taus = sweep_data["taus"]
    maes = sweep_data["maes"]
    p95s = sweep_data["p95s"]
    best_tau = sweep_data["best_tau"]
    best_mae = sweep_data["best_mae"]

    # Also compute standalone metrics for reference
    cnn_mae_only = float("nan")  # placeholder, filled by caller context
    fft_mae_only = float("nan")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle(r"MoS$_2$ moiré：融合参数 τ 敏感性分析", fontsize=13, fontweight="bold")

    ax1.semilogx(taus, maes, "b-o", lw=2, ms=5, label="Fusion MAE")
    ax1.axvline(best_tau, color="blue", ls="--", lw=1.5, alpha=0.7,
                label=f"最优 τ={best_tau:.3f} (MAE={best_mae:.4f}°)")
    ax1.set_xlabel(r"τ (融合权重参数 — 越大越偏向 FFT)")
    ax1.set_ylabel("MAE (°)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.semilogx(taus, p95s, "r--s", lw=1.5, ms=4, alpha=0.7, label="Fusion P95")
    ax2.set_ylabel("P95 (°)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tau_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  已保存: {path}")
    plt.close()


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


def plot_scatter(labels, fft_preds, cnn_preds, cnn_stds=None, fusion_preds=None):
    ncols = 3 if fusion_preds is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    ax1, ax2 = axes[0], axes[1]
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

    if fusion_preds is not None:
        ax3 = axes[2]
        ax3.scatter(labels, fusion_preds, alpha=0.4, s=15, c="seagreen", label="Fusion")
        ax3.plot(theta_line, theta_line, "r--", lw=1.5, label="理想 y=x")
        mae_f = np.abs(fusion_preds - labels).mean()
        ax3.set(xlabel="真实角度 θ (°)", ylabel="提取角度 (°)", title=f"融合  MAE={mae_f:.3f}°")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.set_xlim(THETA_MIN - 0.2, THETA_MAX + 0.2)
        ax3.set_ylim(THETA_MIN - 0.2, THETA_MAX + 0.2)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "compare_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  已保存: {path}")
    plt.close()


def plot_error_analysis(labels, fft_preds, cnn_preds, fusion_preds=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(r"MoS$_2$ moiré：FFT(512px) vs CNN(128px) 误差分析", fontsize=13)

    valid_fft = ~np.isnan(fft_preds)
    err_fft = np.abs(fft_preds[valid_fft] - labels[valid_fft])
    err_cnn = np.abs(cnn_preds - labels)
    err_fusion = np.abs(fusion_preds - labels) if fusion_preds is not None else None

    ax = axes[0]
    hi = max(err_fft.max(), err_cnn.max(), err_fusion.max() if err_fusion is not None else 0.0)
    bins = np.linspace(0, hi * 1.05, 40)
    ax.hist(err_fft, bins=bins, alpha=0.6, color="steelblue", label=f"FFT  MAE={err_fft.mean():.3f}°")
    ax.hist(err_cnn, bins=bins, alpha=0.6, color="darkorange", label=f"CNN  MAE={err_cnn.mean():.3f}°")
    if err_fusion is not None:
        ax.hist(err_fusion, bins=bins, alpha=0.5, color="seagreen", label=f"Fusion  MAE={err_fusion.mean():.3f}°")
    ax.set(xlabel="|角度误差| (°)", ylabel="样本数", title="误差分布直方图")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins_theta = np.linspace(THETA_MIN, THETA_MAX, 10)
    theta_centers = (bins_theta[:-1] + bins_theta[1:]) / 2

    fft_bin_mae, cnn_bin_mae, fusion_bin_mae = [], [], []
    for lo, hi in zip(bins_theta[:-1], bins_theta[1:]):
        mask_fft = valid_fft & (labels >= lo) & (labels < hi)
        fft_bin_mae.append(err_fft[mask_fft[valid_fft]].mean() if mask_fft.sum() > 0 else np.nan)
        mask_cnn = (labels >= lo) & (labels < hi)
        cnn_bin_mae.append(err_cnn[mask_cnn].mean() if mask_cnn.sum() > 0 else np.nan)
        if fusion_preds is not None:
            err_f = np.abs(fusion_preds - labels)
            mask_f = (labels >= lo) & (labels < hi)
            fusion_bin_mae.append(err_f[mask_f].mean() if mask_f.sum() > 0 else np.nan)

    ax.plot(theta_centers, fft_bin_mae, "o-", c="steelblue", ms=7, lw=2, label="FFT")
    ax.plot(theta_centers, cnn_bin_mae, "s-", c="darkorange", ms=7, lw=2, label="CNN")
    if fusion_preds is not None:
        ax.plot(theta_centers, fusion_bin_mae, "^-", c="seagreen", ms=7, lw=2, label="Fusion")
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


def _residual_correlation(
    labels: np.ndarray,
    fft_preds: np.ndarray,
    cnn_preds: np.ndarray,
    fft_valid: np.ndarray,
) -> float:
    """Pearson correlation between FFT residuals and CNN residuals.

    Near +1 → both methods make errors on the same samples (redundant).
    Near  0 → errors are uncorrelated (ensemble potential).
    """
    mask = fft_valid & np.isfinite(cnn_preds)
    if mask.sum() < 3:
        return float("nan")
    fft_resid = labels[mask] - fft_preds[mask]
    cnn_resid = labels[mask] - cnn_preds[mask]
    return float(np.corrcoef(fft_resid, cnn_resid)[0, 1])


def save_report(
    labels,
    fft_preds,
    cnn_preds,
    fft_errors,
    cnn_errors,
    fft_valid,
    cnn_stds=None,
    *,
    eval_mode: str = "legacy",
    fusion_preds: np.ndarray | None = None,
    fusion_unc_scale: float = 0.5,
):
    path = os.path.join(OUT_DIR, "compare_report.txt")
    fft_metrics = compute_stratified_metrics(fft_preds, labels, theta_min=THETA_MIN, theta_max=THETA_MAX)
    cnn_metrics = compute_stratified_metrics(cnn_preds, labels, theta_min=THETA_MIN, theta_max=THETA_MAX)
    fusion_metrics = None
    if fusion_preds is not None:
        fusion_metrics = compute_stratified_metrics(
            fusion_preds, labels, theta_min=THETA_MIN, theta_max=THETA_MAX,
        )
    cal = None
    if cnn_stds is not None:
        cal = compute_calibration_metrics(cnn_preds, labels, cnn_stds)

    with open(path, "w") as f:
        f.write("MoS₂ moiré：FFT vs CNN 角度提取公平对比报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"评测口径 eval_mode: {eval_mode}  （legacy=旧口径；paired=同场景配对）\n")
        f.write(f"材料：MoS₂ 转角同质结（a = {A_NM} nm）\n")
        f.write("对比方法：\n")
        if eval_mode == "paired":
            f.write("  CNN：与训练相同的退化管线，同一场景 128×128 中心裁剪\n")
            f.write("  FFT：同一场景 height 通道 512×512（训练分布内退化）\n")
        else:
            f.write("  CNN：数据集中 128×128 图像（与训练一致）\n")
            f.write("  FFT：按标签单独重算 512×512 高度图（训练分布内退化）\n")
        f.write("退化条件：高斯噪声 + 仿射畸变 + 高斯模糊 + 探针卷积 + 1/f噪声 + …\n\n")

        f.write(f'{"指标":<20} {"FFT":>12} {"CNN":>12}\n')
        f.write("-" * 46 + "\n")
        f.write(f'{"有效样本数":<20} {fft_valid.sum():>12} {len(labels):>12}\n')
        f.write(f'{"失败率":<20} {(~fft_valid).sum()/len(labels)*100:>11.1f}% {"0.0%":>12}\n')
        f.write(f'{"MAE (°)":<20} {fft_errors.mean():>12.4f} {cnn_errors.mean():>12.4f}\n')
        f.write(f'{"中位误差 (°)":<20} {np.median(fft_errors):>12.4f} {np.median(cnn_errors):>12.4f}\n')
        f.write(f'{"std (°)":<20} {fft_errors.std():>12.4f} {cnn_errors.std():>12.4f}\n')
        f.write(f'{"90th pct (°)":<20} {np.percentile(fft_errors,90):>12.4f} {np.percentile(cnn_errors,90):>12.4f}\n')
        f.write(f'{"最大误差 (°)":<20} {fft_errors.max():>12.4f} {cnn_errors.max():>12.4f}\n\n')

        ratio = fft_errors.mean() / cnn_errors.mean()
        f.write(f"MAE 比值 MAE_FFT / MAE_CNN = {ratio:.3f}（<1 表示 FFT 平均误差更小）\n")

        # 残差相关性分析
        resid_corr = _residual_correlation(labels, fft_preds, cnn_preds, fft_valid)
        if np.isfinite(resid_corr):
            f.write(f"\n残差相关性:\n")
            f.write(f"  FFT vs CNN 残差相关系数: {resid_corr:.4f}\n")
            _corr_comment = (
                "（高度正相关 → 两者误差模式相似，融合冗余度高）"
                if resid_corr > 0.5 else
                "（弱相关/不相关 → 两者误差互补，融合有潜力）"
            )
            f.write(f"  {_corr_comment}\n")

        if fusion_preds is not None and fusion_metrics is not None:
            fusion_err = np.abs(fusion_preds - labels)
            f.write(
                f"融合 MAE（τ={fusion_unc_scale}°）: {fusion_err.mean():.4f}°  "
                f"P95={fusion_metrics.percentile_errors.get(95, float('nan')):.4f}°\n"
            )

        f.write("\n分层指标（按角度区间）\n")
        f.write("-" * 50 + "\n")
        f.write(f'{"指标":<20} {"FFT":>12} {"CNN":>12}\n')
        f.write(f'{"P95 (°)":<20} {fft_metrics.percentile_errors.get(95, np.nan):>12.4f} {cnn_metrics.percentile_errors.get(95, np.nan):>12.4f}\n')
        f.write(f'{"P99 (°)":<20} {fft_metrics.percentile_errors.get(99, np.nan):>12.4f} {cnn_metrics.percentile_errors.get(99, np.nan):>12.4f}\n')
        f.write(f'{"小角度MAE<1.5°":<20} {fft_metrics.small_angle_mae:>12.4f} {cnn_metrics.small_angle_mae:>12.4f}\n')
        f.write(f'{"大角度MAE>3.5°":<20} {fft_metrics.large_angle_mae:>12.4f} {cnn_metrics.large_angle_mae:>12.4f}\n')

        f.write("\n角度分箱 MAE（中心角: FFT / CNN）\n")
        f.write("-" * 50 + "\n")
        centers = sorted(set(fft_metrics.angle_bin_mae.keys()) | set(cnn_metrics.angle_bin_mae.keys()))
        for c in centers:
            f.write(
                f"  {c:>4.2f}°: "
                f"{fft_metrics.angle_bin_mae.get(c, np.nan):>7.4f} / "
                f"{cnn_metrics.angle_bin_mae.get(c, np.nan):>7.4f}\n"
            )

        if cnn_stds is not None:
            f.write(f"\nMC Dropout 不确定性:\n")
            f.write(f"  平均不确定性: ±{cnn_stds.mean():.4f}°\n")
            f.write(f"  不确定性范围: [{cnn_stds.min():.4f}°, {cnn_stds.max():.4f}°]\n")
            if cal is not None:
                f.write("  校准指标:\n")
                f.write(f"    1σ 覆盖率: {cal['within_1sigma_pct']:.2f}% (理想 68.27%)\n")
                f.write(f"    2σ 覆盖率: {cal['within_2sigma_pct']:.2f}% (理想 95.45%)\n")
                f.write(f"    3σ 覆盖率: {cal['within_3sigma_pct']:.2f}%\n")
                f.write(f"    误差-不确定性相关系数: {cal['error_uncertainty_correlation']:.3f}\n")
                f.write(f"    校准MSE: {cal['calibration_mse']:.5f}\n")

    print(f"  已保存: {path}")

    # Also export one-line summary CSV for thesis tables.
    csv_path = os.path.join(OUT_DIR, "compare_summary.csv")
    row = {
        "eval_mode": eval_mode,
        "n_samples": int(len(labels)),
        "fft_valid": int(fft_valid.sum()),
        "fft_fail_rate_pct": float((~fft_valid).sum() / len(labels) * 100),
        "fft_mae_deg": float(fft_errors.mean()),
        "cnn_mae_deg": float(cnn_errors.mean()),
        "mae_ratio_fft_over_cnn": float(fft_errors.mean() / cnn_errors.mean()),
        "fft_p95_deg": float(fft_metrics.percentile_errors.get(95, np.nan)),
        "cnn_p95_deg": float(cnn_metrics.percentile_errors.get(95, np.nan)),
        "fft_p99_deg": float(fft_metrics.percentile_errors.get(99, np.nan)),
        "cnn_p99_deg": float(cnn_metrics.percentile_errors.get(99, np.nan)),
        "fft_small_lt1p5_mae_deg": float(fft_metrics.small_angle_mae),
        "cnn_small_lt1p5_mae_deg": float(cnn_metrics.small_angle_mae),
        "fft_large_gt3p5_mae_deg": float(fft_metrics.large_angle_mae),
        "cnn_large_gt3p5_mae_deg": float(cnn_metrics.large_angle_mae),
        "residual_correlation": resid_corr if np.isfinite(resid_corr) else float("nan"),
        "fusion_unc_scale": float(fusion_unc_scale),
    }
    if fusion_preds is not None and fusion_metrics is not None:
        fusion_err = np.abs(fusion_preds - labels)
        row.update({
            "fusion_mae_deg": float(fusion_err.mean()),
            "fusion_p95_deg": float(fusion_metrics.percentile_errors.get(95, np.nan)),
            "fusion_p99_deg": float(fusion_metrics.percentile_errors.get(99, np.nan)),
        })
    if cal is not None:
        row.update({
            "cnn_within_1sigma_pct": float(cal["within_1sigma_pct"]),
            "cnn_within_2sigma_pct": float(cal["within_2sigma_pct"]),
            "cnn_within_3sigma_pct": float(cal["within_3sigma_pct"]),
            "cnn_error_unc_corr": float(cal["error_uncertainty_correlation"]),
            "cnn_calibration_mse": float(cal["calibration_mse"]),
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"  已保存: {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-samples", type=int, default=30, help="MC Dropout 采样次数 (0=禁用)")
    parser.add_argument(
        "--eval-mode",
        choices=("legacy", "paired"),
        default="legacy",
        help="legacy：CNN 用 .npz 图像、FFT 单独重算；paired：同一场景中心裁 CNN + 整幅 FFT",
    )
    parser.add_argument("--paired-seed", type=int, default=12345, help="paired 模式下每样本子种子基准")
    parser.add_argument(
        "--fusion-unc-scale",
        type=float,
        default=0.5,
        help="融合权重 τ：w=σ/(σ+τ)，σ 为 CNN MC 标准差（度）",
    )
    parser.add_argument("--save-predictions", action="store_true",
                        help="保存预测数组到 outputs/compare_predictions.npz（供 generate_thesis_figures.py 使用）")
    cli_args = parser.parse_args()

    print("=" * 60)
    print("MoS₂ moiré：FFT(512px) vs CNN(128px) 公平对比评估（v3）")
    print("=" * 60)

    print(f"\n加载数据集: {DATASET_PATH}")
    data = load_npz_dataset(DATASET_PATH)
    images_test = data["images_test"]
    labels_test = data["labels_test"]
    n_ch = detect_n_channels(images_test)
    print(f"  测试集: {len(labels_test)} 样本，{n_ch} 通道（eval_mode={cli_args.eval_mode}）")

    print("\n[1] 加载模型并准备输入（paired 或 legacy）…")
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

    if ckpt_n_ch != n_ch and cli_args.eval_mode == "legacy":
        print(f"  警告：checkpoint 期望 {ckpt_n_ch} 通道，数据集为 {n_ch}，请检查模型与数据是否匹配。")

    fft_preds: np.ndarray
    cnn_input: np.ndarray

    if cli_args.eval_mode == "paired":
        print("\n[paired] 生成同场景配对样本（较慢）并 FFT 提角…")
        paired_list: list[np.ndarray] = []
        fft_list: list[float] = []
        for i in range(len(labels_test)):
            theta_deg = float(labels_test[i])
            rng = np.random.default_rng(_subseed(cli_args.paired_seed, i))
            img_cnn, h512, fov_nm, actual_ppp = generate_sample_paired_cnn_fft(
                theta_deg,
                rng,
                img_size=128,
                fixed_fov_nm=FIXED_FOV_NM,
                n_channels=ckpt_n_ch,
                tip_radius_range=DEFAULT_TIP_RADIUS_RANGE,
                n_sim=512,
            )
            paired_list.append(img_cnn)
            fft_list.append(extract_angle_fft_robust(h512, fov_nm=fov_nm, actual_ppp=actual_ppp))
            if (i + 1) % 500 == 0:
                print(f"  paired 进度: {i + 1}/{len(labels_test)}")
        cnn_input = np.stack(paired_list, axis=0)
        fft_preds = np.array(fft_list, dtype=np.float32)
    else:
        cnn_input = images_test
        print("\n[2] FFT 提取（512×512 重新生成，legacy 口径）...")
        print("  注：为每个测试角度重新生成 512×512 图像并施加训练分布内退化，")
        print("      确保 FFT 获得足够的频率分辨率，与 graded_eval 方法一致。")
        fft_preds = fft_predict_batch_512(labels_test)
        n_fail = np.isnan(fft_preds).sum()
        print(f"  完成，失败 {n_fail}/{len(fft_preds)} 张")

    print("\n[2] CNN 推理…")
    cnn_stds = None
    if cli_args.mc_samples > 0:
        cnn_preds, cnn_stds = cnn_predict_with_uncertainty(
            model, cnn_input, device, mc_samples=cli_args.mc_samples,
            add_fft_channel=add_fft,
        )
        print(f"  MC Dropout ({cli_args.mc_samples} 采样) 完成")
        print(f"  平均不确定性: ±{cnn_stds.mean():.4f}°")
    else:
        cnn_preds = cnn_predict(model, cnn_input, device, add_fft_channel=add_fft)
    print(f"  CNN 完成，共 {len(cnn_preds)} 个预测")

    if cli_args.eval_mode == "paired":
        n_fail = np.isnan(fft_preds).sum()
        print(f"\n  paired FFT 失败 {n_fail}/{len(fft_preds)} 张")

    fusion_preds = None
    if cnn_stds is not None:
        fusion_preds = fusion_predict(
            fft_preds, cnn_preds, cnn_stds, unc_scale=cli_args.fusion_unc_scale,
        )
        print(
            f"\n  融合（τ={cli_args.fusion_unc_scale}°） MAE: "
            f"{np.abs(fusion_preds - labels_test).mean():.4f}°",
        )

        print("\n[2.5] τ 参数扫描（融合敏感性分析）...")
        sweep_data = sweep_fusion_tau(fft_preds, cnn_preds, cnn_stds, labels_test)
        print(
            f"  最优 τ = {sweep_data['best_tau']:.4f}°  "
            f"(融合 MAE = {sweep_data['best_mae']:.4f}°)"
        )
        print(f"  扫描范围: [{sweep_data['taus'][0]:.4f}, {sweep_data['taus'][-1]:.4f}]° "
              f"共 {len(sweep_data['taus'])} 点")
        plot_tau_sweep(sweep_data)

    print("\n[3] 误差统计:")
    fft_errors, fft_valid = error_stats(fft_preds, labels_test, "FFT")
    cnn_errors, _ = error_stats(cnn_preds, labels_test, "CNN")

    ratio = fft_errors.mean() / cnn_errors.mean()
    print(f"\n  >>> MAE_FFT / MAE_CNN = {ratio:.3f}（<1 表示 FFT 平均误差更小）<<<")

    print("\n[4] 生成对比图...")
    plot_scatter(labels_test, fft_preds, cnn_preds, cnn_stds=cnn_stds, fusion_preds=fusion_preds)
    plot_error_analysis(labels_test, fft_preds, cnn_preds, fusion_preds=fusion_preds)
    save_report(
        labels_test, fft_preds, cnn_preds, fft_errors, cnn_errors, fft_valid, cnn_stds=cnn_stds,
        eval_mode=cli_args.eval_mode,
        fusion_preds=fusion_preds,
        fusion_unc_scale=cli_args.fusion_unc_scale,
    )

    print("\n" + "=" * 60)
    print("完成！核心结论：")
    print(f"  FFT MAE: {fft_errors.mean():.3f}°")
    print(f"  CNN MAE: {cnn_errors.mean():.3f}°")
    print(f"  MAE_FFT/MAE_CNN: {ratio:.3f}")
    if fusion_preds is not None:
        print(f"  Fusion MAE: {np.abs(fusion_preds - labels_test).mean():.3f}°")
    if cnn_stds is not None:
        print(f"  CNN 不确定性: ±{cnn_stds.mean():.3f}°")

    if cli_args.save_predictions:
        pred_save_path = os.path.join(OUT_DIR, "compare_predictions.npz")
        np.savez_compressed(
            pred_save_path,
            labels=labels_test,
            fft_preds=fft_preds,
            cnn_preds=cnn_preds,
            cnn_stds=cnn_stds if cnn_stds is not None else np.array([]),
            fusion_preds=fusion_preds if fusion_preds is not None else np.array([]),
            eval_mode=cli_args.eval_mode,
        )
        print(f"  预测已保存: {pred_save_path}")

    print("=" * 60)
    print("\n下一步：python distortion_sweep.py")
