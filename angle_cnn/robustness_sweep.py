#!/usr/bin/env python3
"""
robustness_sweep.py — FFT vs CNN 多维度退化鲁棒性扫描
====================================================

从训练分布内推到训练分布外（OOD），量化两种方法各自的崩溃边界。

退化维度：
  noise   — 高斯噪声
  blur    — 高斯模糊
  shear   — 仿射畸变
  tip     — 探针卷积 (TITAN 70)

运行方式
--------
    python robustness_sweep.py
    python robustness_sweep.py --mc-samples 50 --trials 20

输出
----
    outputs/robustness_sweep.png
    outputs/robustness_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from angle_cnn.core.config import IMG_SIZE, THETA_MAX, THETA_MIN
from angle_cnn.core.physics import FIXED_FOV_NM
from angle_cnn.core.eval_utils import load_model_from_checkpoint
from angle_cnn.core.physics import pixels_per_moire_period
from dataset_generator import (
    DEFAULT_BLUR_RANGE,
    DEFAULT_NOISE_RANGE,
    DEFAULT_ONEOVERF_RANGE,
    DEFAULT_RINGING_RANGE,
    DEFAULT_ROW_NOISE_RANGE,
    DEFAULT_SCALE_RANGE,
    DEFAULT_SCAN_OFFSET_RANGE,
    DEFAULT_SHEAR_X_RANGE,
    DEFAULT_SHEAR_Y_RANGE,
    DEFAULT_TILT_AMP_RANGE,
    DEFAULT_TIP_RADIUS_RANGE,
    generate_sample_paired_cnn_fft,
)

# 复用 eval_compare 的鲁棒 FFT 和 CNN 推理
from eval_compare import _extract_angle_fft_robust
from angle_cnn.core.cnn import predict_with_uncertainty

from angle_cnn.core.fonts import setup_matplotlib_cjk_font

setup_matplotlib_cjk_font()
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")

# ── 配置 ──────────────────────────────────────────────────────

TEST_THETAS = [0.5, 1.0, 2.0, 3.0, 5.0]

SWEEP_DIMS: Dict[str, dict] = {
    "noise": {
        "levels": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
        "train_range": (DEFAULT_NOISE_RANGE[0], DEFAULT_NOISE_RANGE[1]),
        "xlabel": r"噪声幅度 (noise, ptp比)",
        "param": "noise_range",
    },
    "blur": {
        "levels": [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        "train_range": (DEFAULT_BLUR_RANGE[0], DEFAULT_BLUR_RANGE[1]),
        "xlabel": r"高斯模糊 $\sigma$ (px)",
        "param": "blur_range",
    },
    "shear": {
        "levels": [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25],
        "train_range": (DEFAULT_SHEAR_Y_RANGE[0], DEFAULT_SHEAR_Y_RANGE[1]),
        "xlabel": "仿射畸变 shear",
        "param": "shear_range",
    },
    "tip": {
        "levels": [0.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
        "train_range": (DEFAULT_TIP_RADIUS_RANGE[0], DEFAULT_TIP_RADIUS_RANGE[1]),
        "xlabel": "探针半径 (nm)",
        "param": "tip_radius_nm",
    },
}


def _build_degradation_kwargs(dim_name: str, level: float) -> dict:
    """构建 generate_sample_paired_cnn_fft 的退化参数。

    固定非目标维度为训练分布中位数，目标维度设为固定值。
    """
    # 各维度的中位默认值
    noise_mid = (DEFAULT_NOISE_RANGE[0] + DEFAULT_NOISE_RANGE[1]) / 2
    blur_mid = (DEFAULT_BLUR_RANGE[0] + DEFAULT_BLUR_RANGE[1]) / 2
    shear_mid_x = (DEFAULT_SHEAR_X_RANGE[0] + DEFAULT_SHEAR_X_RANGE[1]) / 2
    shear_mid_y = (DEFAULT_SHEAR_Y_RANGE[0] + DEFAULT_SHEAR_Y_RANGE[1]) / 2
    tilt_mid = (DEFAULT_TILT_AMP_RANGE[0] + DEFAULT_TILT_AMP_RANGE[1]) / 2
    row_mid = (DEFAULT_ROW_NOISE_RANGE[0] + DEFAULT_ROW_NOISE_RANGE[1]) / 2
    oof_mid = (DEFAULT_ONEOVERF_RANGE[0] + DEFAULT_ONEOVERF_RANGE[1]) / 2
    ring_mid = (DEFAULT_RINGING_RANGE[0] + DEFAULT_RINGING_RANGE[1]) / 2
    scan_mid = (DEFAULT_SCAN_OFFSET_RANGE[0] + DEFAULT_SCAN_OFFSET_RANGE[1]) / 2
    scale_mid = (DEFAULT_SCALE_RANGE[0] + DEFAULT_SCALE_RANGE[1]) / 2
    tip_mid = (DEFAULT_TIP_RADIUS_RANGE[0] + DEFAULT_TIP_RADIUS_RANGE[1]) / 2

    # 目标维度设为固定值（range 的上下界都设为 level）
    if dim_name == "noise":
        noise_range = (level, level)
    else:
        noise_range = (noise_mid, noise_mid)

    if dim_name == "blur":
        blur_range = (level, level)
    else:
        blur_range = (blur_mid, blur_mid)

    if dim_name == "shear":
        shear_x_range = (-level, level)
        shear_y_range = (-level, level)
    else:
        shear_x_range = (shear_mid_x, shear_mid_x)
        shear_y_range = (shear_mid_y, shear_mid_y)

    if dim_name == "tip":
        tip_radius_nm = level
        tip_radius_range = None
    else:
        tip_radius_nm = tip_mid
        tip_radius_range = (tip_mid, tip_mid)

    return dict(
        tip_radius_nm=tip_radius_nm,
        tip_radius_range=tip_radius_range,
        noise_range=noise_range,
        blur_range=blur_range,
        scale_range=(scale_mid, scale_mid),
        shear_x_range=shear_x_range,
        shear_y_range=shear_y_range,
        tilt_amp_range=(tilt_mid, tilt_mid),
        row_noise_range=(row_mid, row_mid),
        oneoverf_range=(oof_mid, oof_mid),
        ringing_range=(ring_mid, ring_mid),
        scan_offset_range=(scan_mid, scan_mid),
    )


def run_sweep(
    model: torch.nn.Module,
    device: torch.device,
    n_trials: int = 10,
    mc_samples: int = 30,
    base_n_channels: int = 3,
    add_fft_channel: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """对每个退化维度运行 FFT vs CNN 扫描。

    返回 {dim_name: {"levels": [...], "fft_mae": [...], "cnn_mae": [...], "cnn_std": [...]}}
    """
    results: Dict[str, dict] = {}

    for dim_name, dim_cfg in SWEEP_DIMS.items():
        print(f"\n{'='*50}")
        print(f"退化维度: {dim_name}  ({len(dim_cfg['levels'])} 级 × {len(TEST_THETAS)} 角 × {n_trials} 次)")
        print(f"{'='*50}")

        levels = dim_cfg["levels"]
        fft_maes, fft128_maes, cnn_maes, cnn_stds_all = [], [], [], []

        for li, level in enumerate(levels):
            fft_errs, fft128_errs, cnn_errs = [], [], []
            cnn_stds_list = []

            for theta in TEST_THETAS:
                for trial in range(n_trials):
                    seed = trial * 100000 + int(theta * 1000) + li * 10000
                    rng = np.random.default_rng(seed)

                    degrad_kwargs = _build_degradation_kwargs(dim_name, level)
                    img_cnn, h512, fov_nm, actual_ppp = generate_sample_paired_cnn_fft(
                        theta, rng,
                        img_size=IMG_SIZE,
                        fixed_fov_nm=FIXED_FOV_NM,
                        n_channels=base_n_channels,
                        n_sim=512,
                        **degrad_kwargs,
                    )

                    # FFT-512
                    th_fft = _extract_angle_fft_robust(h512, fov_nm=fov_nm, actual_ppp=actual_ppp)
                    if np.isfinite(th_fft):
                        fft_errs.append(abs(th_fft - theta))

                    # FFT-128: same center crop as CNN
                    n_sim = 512
                    oy = (n_sim - IMG_SIZE) // 2
                    ox = (n_sim - IMG_SIZE) // 2
                    h128 = h512[oy:oy + IMG_SIZE, ox:ox + IMG_SIZE]
                    scale = IMG_SIZE / n_sim
                    fov_128 = fov_nm * scale
                    ppp_128 = pixels_per_moire_period(IMG_SIZE, theta, fov_128)
                    th_fft128 = _extract_angle_fft_robust(h128, fov_nm=fov_128, actual_ppp=ppp_128)
                    if np.isfinite(th_fft128):
                        fft128_errs.append(abs(th_fft128 - theta))

                    # CNN
                    x = torch.from_numpy(img_cnn).unsqueeze(0).float().to(device)
                    if add_fft_channel:
                        from angle_cnn.core.cnn import compute_fft_channel
                        x = compute_fft_channel(x)
                    mean_deg, std_deg = predict_with_uncertainty(model, x, n_samples=mc_samples)
                    cnn_errs.append(abs(mean_deg[0] - theta))
                    cnn_stds_list.append(std_deg[0])

            fft_mae = float(np.median(fft_errs)) if fft_errs else float("nan")
            fft128_mae = float(np.median(fft128_errs)) if fft128_errs else float("nan")
            cnn_mae = float(np.median(cnn_errs))
            cnn_std_mean = float(np.mean(cnn_stds_list))
            fft_fail = len(TEST_THETAS) * n_trials - len(fft_errs)
            fft128_fail = len(TEST_THETAS) * n_trials - len(fft128_errs)

            fft_maes.append(fft_mae)
            fft128_maes.append(fft128_mae)
            cnn_maes.append(cnn_mae)
            cnn_stds_all.append(cnn_std_mean)

            status = "OOD" if level > dim_cfg["train_range"][1] else "in-distr"
            fft128_str = f"FFT128={fft128_mae:.4f}° (fail={fft128_fail})" if not np.isnan(fft128_mae) else "FFT128=fail"
            print(f"  [{li+1}/{len(levels)}] {dim_name}={level:<6}  "
                  f"FFT512={fft_mae:.4f}° (fail={fft_fail})  "
                  f"{fft128_str}  "
                  f"CNN={cnn_mae:.4f}°  CNN_std={cnn_std_mean:.4f}°  [{status}]")

        results[dim_name] = {
            "levels": levels,
            "fft_mae": fft_maes,
            "fft128_mae": fft128_maes,
            "cnn_mae": cnn_maes,
            "cnn_std": cnn_stds_all,
            "train_range": dim_cfg["train_range"],
            "xlabel": dim_cfg["xlabel"],
        }

    return results


def plot_sweep(results: Dict[str, dict]) -> None:
    """绘制 4 个退化维度的子图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        r"MoS$_2$ moiré：FFT-512 vs FFT-128 vs CNN 退化鲁棒性边界扫描",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for idx, (dim_name, dim_data) in enumerate(results.items()):
        ax = axes[idx // 2][idx % 2]
        levels = dim_data["levels"]
        fft_mae = np.array(dim_data["fft_mae"])
        fft128_mae = np.array(dim_data["fft128_mae"])
        cnn_mae = np.array(dim_data["cnn_mae"])
        cnn_std = np.array(dim_data["cnn_std"])
        train_lo, train_hi = dim_data["train_range"]

        # 训练分布范围
        ax.axvspan(train_lo, train_hi, alpha=0.12, color="gray", label="训练分布范围")

        # 0.1° 基准线
        ax.axhline(0.1, color="red", ls="--", lw=1.2, alpha=0.7, label="0.1° 基准线")

        # FFT-512
        ax.plot(levels, fft_mae, "o-", color="steelblue", lw=2, ms=6, label="FFT (512px)")

        # FFT-128
        ax.plot(levels, fft128_mae, "D--", color="lightsteelblue", lw=1.8, ms=5, label="FFT (128px)")

        # CNN + 不确定性带
        ax.plot(levels, cnn_mae, "s-", color="darkorange", lw=2, ms=6, label="CNN (128px)")
        ax.fill_between(levels, np.maximum(cnn_mae - cnn_std, 0), cnn_mae + cnn_std,
                         alpha=0.2, color="darkorange", label="CNN ±1σ (MC Dropout)")

        ax.set(xlabel=dim_data["xlabel"], ylabel="中位角度误差 (°)",
               title=dim_name.capitalize())
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)

        # 标注 OOD 区域
        if len(levels) > 0 and levels[-1] > train_hi:
            ax.text(train_hi, ax.get_ylim()[1] * 0.9, " ← OOD",
                    fontsize=9, color="gray", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "robustness_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  已保存: {path}")
    plt.close()


def save_csv(results: Dict[str, dict]) -> None:
    path = os.path.join(OUT_DIR, "robustness_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dim", "level", "train_lo", "train_hi",
                         "fft512_mae_deg", "fft128_mae_deg", "cnn_mae_deg", "cnn_mc_std_deg",
                         "fft512_over_cnn_ratio", "is_ood"])
        for dim_name, dim_data in results.items():
            train_lo, train_hi = dim_data["train_range"]
            for i, level in enumerate(dim_data["levels"]):
                fft_m = dim_data["fft_mae"][i]
                fft128_m = dim_data["fft128_mae"][i]
                cnn_m = dim_data["cnn_mae"][i]
                cnn_s = dim_data["cnn_std"][i]
                ratio = fft_m / cnn_m if cnn_m > 1e-9 and np.isfinite(fft_m) else float("nan")
                is_ood = level > train_hi
                writer.writerow([dim_name, f"{level:.4f}", f"{train_lo:.4f}", f"{train_hi:.4f}",
                                 f"{fft_m:.4f}", f"{fft128_m:.4f}", f"{cnn_m:.4f}", f"{cnn_s:.4f}",
                                 f"{ratio:.3f}", is_ood])
    print(f"  已保存: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFT vs CNN 多维度退化鲁棒性扫描")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--trials", type=int, default=10, help="每个条件的重复次数")
    cli = parser.parse_args()

    print("=" * 60)
    print("MoS₂ moiré：多维度退化鲁棒性扫描")
    print("=" * 60)
    print(f"角度: {TEST_THETAS}")
    print(f"维度: {list(SWEEP_DIMS.keys())}")
    print(f"每条件: {cli.trials} 次试验, MC Dropout {cli.mc_samples} 采样")
    print()

    from angle_cnn.core.io_utils import require_file
    require_file(MODEL_PATH, "Model checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = load_model_from_checkpoint(MODEL_PATH, device)
    base_n_ch = meta["base_n_channels"]
    add_fft = meta["add_fft_channel"]
    fft_str = "+FFT" if add_fft else ""
    print(f"device: {device}, channels: {meta['n_channels']}{fft_str}\n")

    results = run_sweep(model, device, n_trials=cli.trials,
                        mc_samples=cli.mc_samples,
                        base_n_channels=base_n_ch,
                        add_fft_channel=add_fft)

    print("\n[汇总]")
    for dim_name, dim_data in results.items():
        in_data = [(dim_data["fft_mae"][i], dim_data["fft128_mae"][i], dim_data["cnn_mae"][i])
                   for i in range(len(dim_data["levels"]))
                   if not (dim_data["levels"][i] > dim_data["train_range"][1])]
        ood_data = [(dim_data["fft_mae"][i], dim_data["fft128_mae"][i], dim_data["cnn_mae"][i])
                    for i in range(len(dim_data["levels"]))
                    if dim_data["levels"][i] > dim_data["train_range"][1]]
        in_fft = np.nanmedian([f for f, _, _ in in_data]) if in_data else float("nan")
        in_fft128 = np.nanmedian([f128 for _, f128, _ in in_data]) if in_data else float("nan")
        in_cnn = np.nanmedian([c for _, _, c in in_data]) if in_data else float("nan")
        ood_fft = np.nanmedian([f for f, _, _ in ood_data]) if ood_data else float("nan")
        ood_fft128 = np.nanmedian([f128 for _, f128, _ in ood_data]) if ood_data else float("nan")
        ood_cnn = np.nanmedian([c for _, _, c in ood_data]) if ood_data else float("nan")
        print(f"  {dim_name:6s}  分布内  FFT512={in_fft:.4f}°  FFT128={in_fft128:.4f}°  CNN={in_cnn:.4f}°  |  "
              f"OOD  FFT512={ood_fft:.4f}°  FFT128={ood_fft128:.4f}°  CNN={ood_cnn:.4f}°")

    plot_sweep(results)
    save_csv(results)

    print("\n" + "=" * 60)
    print("完成！输出: outputs/robustness_sweep.png, outputs/robustness_sweep.csv")
    print("=" * 60)
