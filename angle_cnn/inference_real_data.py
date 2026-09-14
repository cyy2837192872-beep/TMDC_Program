#!/usr/bin/env python3
"""
inference_real_data.py — 真实 AFM 数据推断管线
=================================================

对真实 AFM 图像同时运行 CNN + FFT + 融合，输出转角预测。

流程
----
1. 加载 AFM 图像（预处理的 .npz 或原始 .ibw/.png/.tiff）
2. CNN 推理（128×128，含 MC Dropout 不确定性）
3. FFT 角度提取（将 height 通道插值到 512×512）
4. 融合（τ 扫描最优权重，默认 τ=0.01°）
5. 输出 CSV 结果 + 单样本可视化

用法
----
    # 处理单个预处理文件
    python inference_real_data.py --input sample.npz --output results/

    # 批量处理目录
    python inference_real_data.py --input-dir data/afm_processed/ --output results/

    # 直接处理原始 AFM 文件（自动预处理）
    python inference_real_data.py --input-dir data/afm_raw/ --output results/ --raw

    # 选择模型
    python inference_real_data.py --input sample.npz --output results/ \\
        --model path/to/best_model.pt --tau 0.01 --mc-samples 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

from angle_cnn.core.cnn import (
    build_model,
    compute_fft_channel,
    detect_n_channels,
    predict_with_uncertainty,
)
from angle_cnn.core.config import THETA_MIN, THETA_MAX
from angle_cnn.core.eval_fft import extract_angle_fft_robust
from angle_cnn.core.io_utils import load_model_checkpoint, state_dict_from_checkpoint
from angle_cnn.core.physics import FIXED_FOV_NM, pixels_per_moire_period
from angle_cnn.core.afm_preprocess import prepare_afm_channels
from angle_cnn.core.fusion import fuse_angles


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoS₂ moiré 真实 AFM 数据推断管线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--input", type=str, help="单个输入文件（.npz 或图像）")
    file_group.add_argument("--input-dir", type=str, help="输入目录")

    parser.add_argument("--output", type=str, default="real_results",
                        help="输出目录")
    parser.add_argument("--model", type=str, default=None,
                        help="CNN 权重路径（默认 outputs/best_model.pt）")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="融合阈值 τ（°）；应在独立验证集上预先选定")
    parser.add_argument("--mc-samples", type=int, default=30,
                        help="MC Dropout 采样次数")
    parser.add_argument("--fov-nm", type=float, default=None,
                        help="实测扫描视野边长（nm）；覆盖文件元数据，缺失则报错")
    parser.add_argument("--raw", action="store_true",
                        help="输入为原始 AFM 文件（非预处理 npz），需要 igor2/PIL 支持")
    parser.add_argument("--no-preview", action="store_true",
                        help="跳过单样本预览图生成")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def _progress(iterable, desc: str = "", **kwargs):
    if tqdm is not None:
        return tqdm(iterable, desc=desc, file=sys.stderr, dynamic_ncols=True, **kwargs)
    return iterable


# ── 加载器 ────────────────────────────────────────────────────


def load_preprocessed_npz(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """加载 preprocess_afm_data.py 输出的 .npz 文件。

    Returns
    -------
    image : (3, 128, 128) float32, 归一化到 [0, 1]
    meta  : metadata dict
    """
    with np.load(path) as data:
        if "height_fft" not in data:
            raise ValueError("旧版 NPZ 缺少原分辨率 height_fft；请重新运行 preprocess_afm_data")
        image = data["image"]
        height_fft = data["height_fft"]
        meta = json.loads(str(data.get("metadata", "{}")))
    if image.shape != (3, 128, 128) or height_fft.ndim != 2 or height_fft.shape[0] != height_fft.shape[1]:
        raise ValueError("NPZ image/height_fft shape invalid; rerun preprocessing")
    if height_fft.shape[0] < 256 or not np.all(np.isfinite(image)) or not np.all(np.isfinite(height_fft)):
        raise ValueError("NPZ scan is undersampled or contains non-finite pixels")
    return image, height_fft, meta


def load_raw_as_npz(filepath: str, fov_nm: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """直接加载原始 AFM 文件并预处理为 (3, 128, 128) 格式。

    优先使用 core.io_afm 的 Cypher ES 解析器；回退到 PIL。
    """
    from angle_cnn.core.io_afm import (
        find_channel,
        load_cypher_image,
    )
    from angle_cnn.preprocess_afm_data import (
        load_ibw_file,
        load_image_file,
        load_spm_file,
    )

    ext = Path(filepath).suffix.lower()
    channels = {}
    meta = {}

    if ext == ".ibw":
        try:
            result = load_cypher_image(filepath, return_metadata=True)
            if isinstance(result, tuple):
                channels, meta = result
            else:
                channels = result
        except Exception:
            channels, metadata = load_ibw_file(filepath)
            meta = vars(metadata)
    elif ext == ".spm":
        channels, metadata = load_spm_file(filepath)
        meta = vars(metadata)
    elif ext in (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"):
        channels, metadata = load_image_file(filepath)
        meta = vars(metadata)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    if not channels:
        raise RuntimeError(f"无法加载文件: {filepath}")
    selected = {}
    for name in ("height", "phase", "amplitude"):
        found = find_channel(channels, name)
        if found is not None:
            selected[name] = found
    channels = selected

    measured_fov = fov_nm if fov_nm is not None else meta.get("scan_size_nm")
    image, height_fft = prepare_afm_channels(channels, float(measured_fov or 0))
    meta["scan_size_nm"] = float(measured_fov)
    return image, height_fft, meta


def collect_input_files(args) -> list[tuple[str, str]]:
    """收集输入文件列表。返回 [(filepath, output_stem), ...]."""
    files: list[tuple[str, str]] = []

    if args.input:
        p = Path(args.input)
        if p.is_file():
            files.append((str(p), p.stem))
        else:
            raise FileNotFoundError(f"输入文件不存在: {args.input}")
    elif args.input_dir:
        patterns = ["*.npz"] if not args.raw else ["*.ibw", "*.spm", "*.tiff", "*.tif", "*.png", "*.jpg"]
        for pat in patterns:
            for p in sorted(Path(args.input_dir).rglob(pat)):
                files.append((str(p), p.stem))

    if not files:
        raise FileNotFoundError(f"未在输入路径中找到匹配的文件")

    return files


# ── 推理 ──────────────────────────────────────────────────────


def load_cnn_model(model_path: str, device) -> tuple:
    """加载 CNN 模型并返回 (model, add_fft_channel, n_channels, meta)."""
    import torch
    ckpt = load_model_checkpoint(model_path, map_location=device)
    meta = ckpt if isinstance(ckpt, dict) else {}

    add_fft = bool(meta.get("add_fft_channel", False))
    n_channels = int(meta.get("n_channels", 3))
    arch = str(meta.get("arch", "resnet18"))
    dropout = float(meta.get("dropout", 0.3))

    model = build_model(n_channels=n_channels, dropout=dropout, arch=arch)
    _tgt = model
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    sd = state_dict_from_checkpoint(ckpt)
    _tgt.load_state_dict(sd)
    model.eval()

    return model, add_fft, n_channels, meta


def run_cnn_inference(
    model, image: np.ndarray, add_fft: bool, device, mc_samples: int,
) -> tuple[float, float, np.ndarray]:
    """运行 CNN 推理。

    Returns
    -------
    angle_deg : float
    uncertainty_deg : float
    mc_samples_raw : np.ndarray  所有 MC 采样的角度值
    """
    import torch

    img_t = torch.from_numpy(image).unsqueeze(0).float().to(device)
    if add_fft:
        img_t = compute_fft_channel(img_t)

    mean_deg, std_deg = predict_with_uncertainty(model, img_t, n_samples=mc_samples)
    return float(mean_deg[0]), float(std_deg[0]), mean_deg


def run_fft_inference(height_ch: np.ndarray, fov_nm: float) -> float:
    """对 height 通道运行 FFT 角度提取。

    将图像插值到约 512×512（FFT 方法的工作分辨率）。
    """
    h, w = height_ch.shape
    if h != w or h < 256:
        raise ValueError("FFT requires an original square scan of at least 256 pixels")
    img_512 = height_ch.astype(np.float64)
    n_px = h

    # 归一化
    img_512 = (img_512 - img_512.min()) / (img_512.max() - img_512.min() + 1e-9)

    ppp_est = pixels_per_moire_period(n_px, 2.0, fov_nm)  # fixed, label-independent reference
    angle = extract_angle_fft_robust(img_512, fov_nm=fov_nm, actual_ppp=max(4.0, ppp_est))
    return angle


# ── 融合 ──────────────────────────────────────────────────────


def fusion_estimate(cnn_angle: float, fft_angle: float, cnn_unc: float,
                    tau: float = 0.01) -> tuple[float, float]:
    """不确定性加权融合。

    Parameters
    ----------
    tau : float  权重过渡阈值（°），τ=0.01 完全信任 FFT

    Returns
    -------
    fused : float  融合角度
    w_cnn : float  CNN 权重
    """
    fused, w_fft = fuse_angles(fft_angle, cnn_angle, cnn_unc, tau)
    return fused, 1.0 - w_fft


# ── 可视化 ────────────────────────────────────────────────────


def save_preview(image: np.ndarray, cnn_angle: float, cnn_unc: float,
                 fft_angle: float, fused_angle: float, save_path: str):
    """单样本预览图——模型预测 + 原始图像 + FFT 频谱。"""
    try:
        from angle_cnn.core.fonts import setup_matplotlib_cjk_font
        setup_matplotlib_cjk_font()
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 左：height 通道
        ax = axes[0]
        ax.imshow(image[0], cmap="afmhot")
        ax.set_title("Height 通道")
        ax.axis("off")

        # 中：phase 通道
        ax = axes[1]
        ax.imshow(image[1] if image[1].std() > 0 else image[0], cmap="afmhot")
        ax.set_title("Phase 通道")
        ax.axis("off")

        # 右：预测结果
        ax = axes[2]
        ax.axis("off")
        lines = [
            f"CNN:  {cnn_angle:.3f}°  ±{cnn_unc:.3f}°",
        ]
        if not np.isnan(fft_angle):
            lines.append(f"FFT:  {fft_angle:.3f}°")
        lines.append(f"融合:  {fused_angle:.3f}°")
        y0 = 0.7
        for line in lines:
            ax.text(0.1, y0, line, fontsize=11, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
            y0 -= 0.15

        fig.suptitle(f"MoS₂ moiré 转角预测", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logging.warning(f"预览图生成失败: {e}")


# ── 主程序 ────────────────────────────────────────────────────


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 模型路径
    if args.model is not None:
        model_path = args.model
    else:
        model_path = os.path.join(SCRIPT_DIR, "outputs", "best_model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}")

    # 输出目录
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    preview_dir = os.path.join(out_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)

    # 收集输入文件
    input_files = collect_input_files(args)
    print(f"找到 {len(input_files)} 个输入文件")

    # 加载 CNN 模型
    model, add_fft, n_channels, ckpt_meta = load_cnn_model(model_path, device)
    base_channels = int(ckpt_meta.get("base_n_channels", n_channels - int(add_fft)))
    if base_channels not in (1, 3):
        raise ValueError(f"不支持的基础通道数: {base_channels}")
    if add_fft:
        print(f"  CNN 使用 FFT 额外通道")
    print(f"  CNN 架构: {ckpt_meta.get('arch', 'resnet18')}, 输入通道: {n_channels}")

    # 结果收集
    results: list[dict] = []

    for filepath, stem in _progress(input_files, desc="推理"):
        try:
            # 加载
            if args.raw or filepath.endswith(".npz"):
                if filepath.endswith(".npz"):
                    image, height_fft, meta = load_preprocessed_npz(filepath)
                    if args.fov_nm is not None and not np.isclose(args.fov_nm, float(meta.get("scan_size_nm", 0))):
                        raise ValueError("--fov-nm 与已预处理 NPZ 的视野不一致；请重新预处理原始图像")
                else:
                    image, height_fft, meta = load_raw_as_npz(filepath, args.fov_nm)
            else:
                image, height_fft, meta = load_raw_as_npz(filepath, args.fov_nm)

            # CNN
            cnn_angle, cnn_unc, _ = run_cnn_inference(
                model, image[:base_channels], add_fft, device, args.mc_samples,
            )

            # FFT on height channel
            fov_nm = args.fov_nm
            if fov_nm is None and "scan_size_nm" in meta:
                fov_nm = float(meta["scan_size_nm"])
            if fov_nm is None or not np.isfinite(fov_nm) or fov_nm <= 0:
                raise ValueError("缺少实测 scan_size_nm；请提供 --fov-nm")
            fft_angle = run_fft_inference(height_fft, fov_nm)

            # 融合
            fused, w_cnn = fusion_estimate(cnn_angle, fft_angle, cnn_unc, tau=args.tau)

            # 预览
            if not args.no_preview:
                preview_path = os.path.join(preview_dir, f"{stem}_preview.png")
                save_preview(image, cnn_angle, cnn_unc, fft_angle, fused, preview_path)

            row = {
                "filename": os.path.basename(filepath),
                "cnn_angle_deg": round(cnn_angle, 4),
                "cnn_uncertainty_deg": round(cnn_unc, 4),
                "fft_angle_deg": round(fft_angle, 4) if not np.isnan(fft_angle) else "NaN",
                "fusion_angle_deg": round(fused, 4),
                "fov_nm": round(fov_nm, 1),
            }
            results.append(row)

            line = f"  {stem}: CNN={cnn_angle:.3f}°±{cnn_unc:.3f}"
            if not np.isnan(fft_angle):
                line += f"  FFT={fft_angle:.3f}°"
            line += f"  融合={fused:.3f}°"
            print(line)

        except Exception as e:
            logging.error(f"处理失败 {filepath}: {e}")
            results.append({
                "filename": os.path.basename(filepath),
                "cnn_angle_deg": "ERROR",
                "cnn_uncertainty_deg": "ERROR",
                "fft_angle_deg": "ERROR",
                "fusion_angle_deg": "ERROR",
                "fov_nm": "",
            })

    # ── 输出 CSV ──
    csv_path = os.path.join(out_dir, "real_data_results.csv")
    fieldnames = ["filename", "cnn_angle_deg", "cnn_uncertainty_deg",
                  "fft_angle_deg", "fusion_angle_deg", "fov_nm"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果 CSV: {csv_path}")

    # ── 汇总 ──
    valid = [r for r in results if isinstance(r.get("fusion_angle_deg"), (int, float))]
    print(f"\n{'='*50}")
    print(f"处理完成: {len(valid)}/{len(results)} 成功")
    if valid:
        cnn_angles = [r["cnn_angle_deg"] for r in valid]
        fft_angles = [r["fft_angle_deg"] for r in valid if isinstance(r["fft_angle_deg"], (int, float))]
        fused = [r["fusion_angle_deg"] for r in valid]
        print(f"  CNN 范围:     {min(cnn_angles):.3f}° – {max(cnn_angles):.3f}°")
        if fft_angles:
            print(f"  FFT 范围:     {min(fft_angles):.3f}° – {max(fft_angles):.3f}°")
        print(f"  融合角度范围: {min(fused):.3f}° – {max(fused):.3f}°")
    print(f"预览图: {preview_dir}/")
    print(f"下一步：分析真实数据结果并对比模拟数据基线")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
