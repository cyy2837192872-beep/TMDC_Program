#!/usr/bin/env python3
"""
graded_eval.py — 分级退化测试：FFT vs CNN 公平对比（v2 多通道 + TITAN 70）
===========================================================================

在不同退化强度下对比FFT和CNN的角度提取精度，展示：
  - FFT在轻度退化下仍然有效
  - CNN在中重度退化下的优势区间
  - MC Dropout 不确定性随退化级别的变化

运行方式
--------
    python graded_eval.py
    python graded_eval.py --mc-samples 50

输出
----
    outputs/graded_eval.png    — 分级对比图
    outputs/graded_eval.csv    — 数值结果
    outputs/graded_report.txt  — 详细报告
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

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
from core.io_utils import require_file  # noqa: E402
from core.config import IMG_SIZE, THETA_MIN, THETA_MAX  # noqa: E402
from core.physics import A_NM, angle_uncertainty, FIXED_FOV_NM, pixels_per_moire_period  # noqa: E402
from core.moire_sim import synthesize_multichannel_moire, synthesize_reconstructed_moire  # noqa: E402
from moire_pipeline import extract_angle_fft  # noqa: E402
from core.cnn import predict_with_uncertainty  # noqa: E402
from core.eval_utils import load_model_from_checkpoint, cnn_predict_single  # noqa: E402

setup_matplotlib_cjk_font()

import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_THETAS = [0.5, 1.0, 2.0, 3.0, 5.0]
N_TRIALS = 20

# (name, noise, blur, shear_y_max, tilt_amp, row_noise, 1/f_amp, ringing_amp, tip_radius)
# 校准到 Cypher ES + TITAN 70 + Tapping Mode 实际条件
# L0-L4 在训练分布内，L5-L6 为分布外鲁棒性测试
DEGRADATION_LEVELS = [
    ("L0_理想",  0.0,   0.0,  0.0,   0.0,  0.0,   0.0,   0.0,   0.0),
    ("L1_极轻",  0.02,  0.1,  0.003, 0.01, 0.005, 0.01,  0.005, 7.0),
    ("L2_轻度",  0.04,  0.15, 0.005, 0.02, 0.01,  0.015, 0.01,  7.0),
    ("L3_中度",  0.06,  0.2,  0.008, 0.03, 0.015, 0.02,  0.015, 7.0),
    ("L4_较重",  0.08,  0.3,  0.01,  0.05, 0.02,  0.03,  0.02,  7.0),
    ("L5_重度",  0.12,  0.5,  0.02,  0.08, 0.04,  0.05,  0.03,  7.0),
    ("L6_极端",  0.20,  1.0,  0.05,  0.15, 0.08,  0.08,  0.05,  7.0),
]


def generate_test_image(
    theta_deg: float,
    noise: float,
    blur: float,
    shear_y_max: float,
    tilt_amp: float,
    row_noise_amp: float,
    oneoverf_amp: float,
    ringing_amp: float,
    tip_radius_nm: float,
    rng: np.random.Generator,
    n_channels: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate a test image pair (CNN input + FFT input)."""
    if n_channels > 1:
        ch_names = ("height", "phase", "amplitude")[:n_channels]
        ch_dict, fov_nm = synthesize_multichannel_moire(theta_deg, FIXED_FOV_NM, n=512, channels=ch_names)
    else:
        raw, fov_nm = synthesize_reconstructed_moire(theta_deg, FIXED_FOV_NM, n=512)
        ch_dict = {"height": raw}

    actual_ppp = pixels_per_moire_period(512, theta_deg, FIXED_FOV_NM)
    pixel_size_nm = fov_nm / 512

    # Apply degradations to all channels
    for name in list(ch_dict.keys()):
        img = ch_dict[name]
        if name == "height" and tilt_amp > 0.01:
            img = apply_background_tilt(img, tilt_amp, rng.uniform(-1, 1), rng.uniform(-1, 1))

        if shear_y_max > 0.001:
            shear_x_max = shear_y_max / 3
            img = apply_affine_distortion(
                img,
                rng.uniform(-shear_x_max, shear_x_max),
                rng.uniform(-shear_y_max, shear_y_max),
                rng.uniform(0.9, 1.1),
                rng.uniform(0.9, 1.1),
            )

        img = apply_tip_convolution(img, tip_radius_nm, pixel_size_nm)

        if blur > 0.1:
            img = gaussian_filter(img, sigma=blur)

        if row_noise_amp > 0.01:
            img = apply_row_noise(img, row_noise_amp, rng)

        if ringing_amp > 0.005:
            img = apply_feedback_ringing(img, ringing_amp, rng)

        if oneoverf_amp > 0.005:
            img = apply_oneoverf_noise(img, oneoverf_amp, 1.0, rng)

        if noise > 0.01:
            ptp = img.max() - img.min()
            img = img + noise * ptp * rng.standard_normal(img.shape)

        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        ch_dict[name] = img.astype(np.float32)

    # FFT input: full 512x512 height
    img_512_height = ch_dict["height"]

    # CNN input: center-cropped multi-channel
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


def cnn_predict_with_unc(model, img_cnn, device, mc_samples=30, add_fft_channel=False):
    if img_cnn.ndim == 2:
        x = torch.from_numpy(img_cnn).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        x = torch.from_numpy(img_cnn).unsqueeze(0).float().to(device)
    if add_fft_channel:
        x = compute_fft_channel(x)
    mean_deg, std_deg = predict_with_uncertainty(model, x, n_samples=mc_samples)
    return float(mean_deg[0]), float(std_deg[0])


def run_graded_eval(model, device, n_channels=1, mc_samples=0, add_fft_channel=False):
    results: Dict[str, Dict[str, List]] = {}

    total_configs = len(DEGRADATION_LEVELS) * len(TEST_THETAS)
    current = 0

    print(f"\n测试角度: {TEST_THETAS}")
    print(f"每个配置测试次数: {N_TRIALS}")
    print(f"通道数: {n_channels}")
    if mc_samples > 0:
        print(f"MC Dropout 采样: {mc_samples}")
    print()

    for level_name, noise, blur, shear_y, tilt, row_noise, oneoverf, ringing, tip_r in DEGRADATION_LEVELS:
        results[level_name] = {"fft_mae": [], "fft_fail": [], "cnn_mae": [], "cnn_unc": []}

        for theta in TEST_THETAS:
            current += 1
            fft_errors, cnn_errors, cnn_uncs = [], [], []
            fft_fail_count = 0

            for trial in range(N_TRIALS):
                seed = hash(f"{level_name}_{theta}_{trial}") % (2 ** 31)
                rng = np.random.default_rng(seed)

                img_cnn, img_512, fov_nm, actual_ppp = generate_test_image(
                    theta, noise, blur, shear_y, tilt, row_noise, oneoverf, ringing, tip_r, rng,
                    n_channels=n_channels,
                )

                th_fft, _, _ = extract_angle_fft(img_512, fov_nm, ppp=actual_ppp)
                if th_fft is not None:
                    fft_errors.append(abs(th_fft - theta))
                else:
                    fft_fail_count += 1

                if mc_samples > 0:
                    th_cnn, unc = cnn_predict_with_unc(model, img_cnn, device, mc_samples,
                                                       add_fft_channel=add_fft_channel)
                    cnn_uncs.append(unc)
                else:
                    th_cnn = cnn_predict_single(model, img_cnn, device,
                                                add_fft_channel=add_fft_channel)
                cnn_errors.append(abs(th_cnn - theta))

            fft_mae = np.mean(fft_errors) if fft_errors else np.nan
            cnn_mae = np.mean(cnn_errors)
            fft_fail_rate = fft_fail_count / N_TRIALS * 100
            mean_unc = np.mean(cnn_uncs) if cnn_uncs else 0.0

            results[level_name]["fft_mae"].append(fft_mae)
            results[level_name]["fft_fail"].append(fft_fail_rate)
            results[level_name]["cnn_mae"].append(cnn_mae)
            results[level_name]["cnn_unc"].append(mean_unc)

            unc_str = f" unc=±{mean_unc:.3f}°" if mc_samples > 0 else ""
            ratio_str = f"提升={fft_mae/cnn_mae:.1f}x" if not np.isnan(fft_mae) and cnn_mae > 1e-6 else "FFT失败"
            print(
                f"  [{current:>2}/{total_configs}] {level_name:8s} θ={theta}°  "
                f"FFT={fft_mae:.3f}° (fail={fft_fail_rate:.0f}%)  "
                f"CNN={cnn_mae:.3f}°{unc_str}  {ratio_str}"
            )

    return results


def compute_summary(results):
    summary = []
    for level_name, noise, blur, shear_y, tilt, row_noise, oneoverf, ringing, tip_r in DEGRADATION_LEVELS:
        fft_maes = [x for x in results[level_name]["fft_mae"] if not np.isnan(x)]
        cnn_maes = results[level_name]["cnn_mae"]

        fft_mae_avg = np.mean(fft_maes) if fft_maes else np.nan
        cnn_mae_avg = np.mean(cnn_maes)
        fft_fail_avg = np.mean(results[level_name]["fft_fail"])
        cnn_unc_avg = np.mean(results[level_name]["cnn_unc"])

        if not np.isnan(fft_mae_avg) and cnn_mae_avg > 1e-6:
            improvement = fft_mae_avg / cnn_mae_avg
        else:
            improvement = np.nan

        summary.append({
            "level": level_name,
            "noise": noise, "blur": blur, "shear_y": shear_y, "tip_r": tip_r,
            "fft_mae": fft_mae_avg, "fft_fail": fft_fail_avg,
            "cnn_mae": cnn_mae_avg, "cnn_unc": cnn_unc_avg,
            "improvement": improvement,
        })
    return summary


def plot_results(results, summary):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    level_names = [s["level"] for s in summary]
    x = np.arange(len(level_names))

    ax = axes[0, 0]
    fft_maes = [s["fft_mae"] for s in summary]
    cnn_maes = [s["cnn_mae"] for s in summary]
    width = 0.35
    ax.bar(x - width / 2, fft_maes, width, label="FFT", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, cnn_maes, width, label="CNN", color="darkorange", alpha=0.8)
    cnn_uncs = [s["cnn_unc"] for s in summary]
    if any(u > 0 for u in cnn_uncs):
        ax.errorbar(x + width / 2, cnn_maes, yerr=cnn_uncs, fmt="none", ecolor="red", capsize=3, lw=1.5)
    ax.axhline(0.1, color="red", ls="--", lw=1.5, label="0.1° 基准线")
    ax.set_xlabel("退化级别")
    ax.set_ylabel("MAE (°)")
    ax.set_title("各退化级别下的角度提取误差")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in level_names], fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(fft_maes) * 1.1 if any(np.isfinite(fft_maes)) else 1)

    ax = axes[0, 1]
    improvements = [s["improvement"] for s in summary]
    colors = ["mediumseagreen" if not np.isnan(imp) and imp < 20 else "crimson" for imp in improvements]
    bars = ax.bar(x, improvements, color=colors, alpha=0.8)
    ax.axhline(1, color="gray", ls="--", lw=1.2, label="FFT=CNN")
    ax.set_xlabel("退化级别")
    ax.set_ylabel("FFT误差 / CNN误差 (倍)")
    ax.set_title("CNN相对FFT的精度提升倍数")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in level_names], fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, imp in zip(bars, improvements):
        if np.isfinite(imp):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{imp:.1f}x", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    for i, theta in enumerate(TEST_THETAS):
        fft_by_level = [results[level]["fft_mae"][i] for level in level_names]
        ax.plot(x, fft_by_level, "o-", label=f"θ={theta}°", lw=2, ms=6)
    ax.axhline(0.1, color="red", ls="--", lw=1.2)
    ax.set_xlabel("退化级别")
    ax.set_ylabel("FFT MAE (°)")
    ax.set_title("FFT在各角度下的表现")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in level_names], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    ax = axes[1, 1]
    for i, theta in enumerate(TEST_THETAS):
        cnn_by_level = [results[level]["cnn_mae"][i] for level in level_names]
        ax.plot(x, cnn_by_level, "s-", label=f"θ={theta}°", lw=2, ms=6)
    ax.axhline(0.1, color="red", ls="--", lw=1.2)
    ax.set_xlabel("退化级别")
    ax.set_ylabel("CNN MAE (°)")
    ax.set_title("CNN在各角度下的表现")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in level_names], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max([max(results[level]["cnn_mae"]) for level in level_names]) * 1.2)

    plt.suptitle(r"MoS$_2$ moiré：FFT vs CNN 分级退化对比（含 TITAN 70 探针卷积）", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "graded_eval.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n已保存: {path}")
    plt.close()


def save_csv(summary):
    path = os.path.join(OUT_DIR, "graded_eval.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "级别", "噪声", "模糊(px)", "剪切(shear_y)", "探针半径(nm)",
            "FFT_MAE(°)", "FFT失败率(%)", "CNN_MAE(°)", "CNN_unc(°)", "提升倍数",
        ])
        for s in summary:
            writer.writerow([
                s["level"],
                f"{s['noise']:.2f}", f"{s['blur']:.1f}", f"{s['shear_y']:.2f}", f"{s['tip_r']:.1f}",
                f"{s['fft_mae']:.4f}" if np.isfinite(s["fft_mae"]) else "N/A",
                f"{s['fft_fail']:.1f}",
                f"{s['cnn_mae']:.4f}",
                f"{s['cnn_unc']:.4f}" if s["cnn_unc"] > 0 else "N/A",
                f"{s['improvement']:.1f}" if np.isfinite(s["improvement"]) else "N/A",
            ])
    print(f"已保存: {path}")


def save_report(summary):
    path = os.path.join(OUT_DIR, "graded_report.txt")
    unc_theory = angle_uncertainty(FIXED_FOV_NM)

    with open(path, "w", encoding="utf-8") as f:
        f.write("MoS₂ moiré：FFT vs CNN 分级退化对比报告（v2 TITAN 70 + Tapping Mode）\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"材料：MoS₂ 转角同质结（a = {A_NM} nm）\n")
        f.write(f"固定视野：{FIXED_FOV_NM:.1f} nm\n")
        f.write(f"探针：TITAN 70（tip_radius = 7 nm）\n")
        f.write(f"测试角度：{TEST_THETAS}\n")
        f.write(f"每配置测试次数：{N_TRIALS}\n")
        f.write(f"FFT理论不确定度（无畸变）：{unc_theory:.4f}°\n\n")

        f.write("-" * 70 + "\n")
        f.write(f"{'级别':<12} {'噪声':>6} {'模糊':>6} {'剪切':>6} {'探针':>6} "
                f"{'FFT_MAE':>10} {'CNN_MAE':>10} {'CNN_unc':>10} {'提升':>8}\n")
        f.write("-" * 70 + "\n")

        for s in summary:
            fft_str = f"{s['fft_mae']:.4f}" if np.isfinite(s["fft_mae"]) else "    N/A"
            imp_str = f"{s['improvement']:.1f}x" if np.isfinite(s["improvement"]) else "  N/A"
            unc_str = f"±{s['cnn_unc']:.4f}" if s["cnn_unc"] > 0 else "    N/A"
            f.write(f"{s['level']:<12} {s['noise']:>6.2f} {s['blur']:>6.1f} {s['shear_y']:>6.2f} "
                    f"{s['tip_r']:>6.1f} {fft_str:>10} {s['cnn_mae']:>10.4f} {unc_str:>10} {imp_str:>8}\n")

        f.write("-" * 70 + "\n\n")
        f.write("【实验配置说明】\n\n")
        f.write("退化模型包含 TITAN 70 探针卷积（7 nm tip radius）、1/f噪声、\n")
        f.write("反馈环路振荡等 Cypher ES Tapping Mode 特有退化。\n\n")

        ideal = summary[0]
        moderate = summary[3]
        extreme = summary[-1]

        f.write("【关键发现】\n\n")
        f.write(f"1. 理想图像（无退化，无探针卷积）：\n")
        if np.isfinite(ideal["fft_mae"]):
            f.write(f"   FFT MAE = {ideal['fft_mae']:.4f}°，CNN MAE = {ideal['cnn_mae']:.4f}°\n")
            f.write(f"   CNN提升 {ideal['improvement']:.1f}x\n\n")

        f.write(f"2. 中度退化（含 TITAN 70 探针卷积）：\n")
        if np.isfinite(moderate["fft_mae"]):
            f.write(f"   FFT MAE = {moderate['fft_mae']:.4f}°，CNN MAE = {moderate['cnn_mae']:.4f}°\n")
            f.write(f"   CNN提升 {moderate['improvement']:.1f}x\n\n")

        f.write(f"3. 极端退化：\n")
        if np.isfinite(extreme["fft_mae"]):
            f.write(f"   FFT MAE = {extreme['fft_mae']:.4f}°，CNN MAE = {extreme['cnn_mae']:.4f}°\n\n")

    print(f"已保存: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-samples", type=int, default=0, help="MC Dropout 采样次数 (0=禁用)")
    args = parser.parse_args()

    print("=" * 70)
    print("MoS₂ moiré: FFT vs CNN 分级退化对比评估（v2 TITAN 70 + Tapping Mode）")
    print("=" * 70)

    MODEL_PATH = os.path.join(OUT_DIR, "best_model.pt")
    require_file(MODEL_PATH, "Model checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = load_model_from_checkpoint(MODEL_PATH, device)
    base_n_ch = meta["base_n_channels"]
    add_fft = meta["add_fft_channel"]
    fft_str = "（含FFT通道）" if add_fft else ""
    print(f"\n模型加载完成，设备: {device}，通道数: {meta['n_channels']}{fft_str}")

    results = run_graded_eval(model, device, n_channels=base_n_ch,
                              mc_samples=args.mc_samples, add_fft_channel=add_fft)

    summary = compute_summary(results)

    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    header = f"{'级别':<12} {'FFT_MAE':>10} {'CNN_MAE':>10}"
    if args.mc_samples > 0:
        header += f" {'CNN_unc':>10}"
    header += f" {'提升':>8}"
    print(header)
    print("-" * len(header))
    for s in summary:
        fft_str = f"{s['fft_mae']:.4f}" if np.isfinite(s["fft_mae"]) else "    N/A"
        imp_str = f"{s['improvement']:.1f}x" if np.isfinite(s["improvement"]) else "  N/A"
        line = f"{s['level']:<12} {fft_str:>10} {s['cnn_mae']:>10.4f}"
        if args.mc_samples > 0:
            line += f" ±{s['cnn_unc']:>9.4f}"
        line += f" {imp_str:>8}"
        print(line)

    plot_results(results, summary)
    save_csv(summary)
    save_report(summary)

    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
