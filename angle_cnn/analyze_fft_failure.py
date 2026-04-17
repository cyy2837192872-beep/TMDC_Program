#!/usr/bin/env python3
"""
analyze_fft_failure.py — 深度分析：为什么原测试集FFT误差高达6.34°
=====================================================================

对比两种FFT评估方式的差异：
1. eval_compare.py：使用128×128测试集图像 → FFT误差6.34°
2. graded_eval.py：使用512×512完整图像 → FFT误差~0.1°

核心假设：图像尺寸差异导致FFT峰检测失败
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

from core.degrade import apply_affine_distortion, apply_background_tilt, apply_row_noise
from core.fonts import setup_matplotlib_cjk_font
from core.io_utils import load_npz_dataset, require_file
from core.config import THETA_MIN, THETA_MAX  # noqa: E402
from core.physics import A_NM, angle_uncertainty, moire_period, FIXED_FOV_NM  # noqa: E402
from core.moire_sim import synthesize_reconstructed_moire
from moire_pipeline import extract_angle_fft
from core.cnn import build_model
from core.eval_utils import load_model_from_checkpoint

setup_matplotlib_cjk_font()

import matplotlib.pyplot as plt

OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def analyze_fft_on_test_set():
    """分析原始测试集上的FFT行为"""
    print("=" * 70)
    print("分析原始测试集上的FFT行为")
    print("=" * 70)
    
    # 加载测试集
    DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "moire_dataset.npz")
    data = load_npz_dataset(DATA_PATH)
    images_test = data["images_test"]  # (N, 128, 128)
    labels_test = data["labels_test"]
    fovs_test = data["fovs_test"]
    
    print(f"\n测试集信息：")
    print(f"  图像尺寸: {images_test.shape[1]}×{images_test.shape[2]}")
    print(f"  样本数: {len(labels_test)}")
    print(f"  视野范围: {fovs_test.min():.1f} ~ {fovs_test.max():.1f} nm")
    
    # 分析FFT参数计算
    print("\n--- FFT参数分析 ---")
    sample_idx = 0
    img = images_test[sample_idx]
    label = labels_test[sample_idx]
    fov = fovs_test[sample_idx]
    
    img_px = img.shape[0]  # 128
    scale = img_px / 512
    
    print(f"\n样本 {sample_idx}:")
    print(f"  图像尺寸: {img_px}×{img_px}")
    print(f"  真实角度: {label:.2f}°")
    print(f"  记录的视野(fov): {fov:.1f} nm")
    
    # eval_compare.py 中的计算方式
    fov_actual = fov * scale
    ppp_512 = max(4.0, FIXED_FOV_NM / moire_period(label))
    actual_ppp = ppp_512 * scale
    
    print(f"\n  eval_compare.py 计算方式：")
    print(f"    scale = {img_px} / 512 = {scale:.4f}")
    print(f"    fov_actual = {fov} × {scale} = {fov_actual:.1f} nm")
    print(f"    ppp_512 = {ppp_512:.1f}")
    print(f"    actual_ppp = {ppp_512:.1f} × {scale} = {actual_ppp:.2f}")
    
    # FFT峰位置
    r_peak = img_px / actual_ppp
    r_min = max(2.0, 0.4 * r_peak)
    r_max = min(img_px // 3, 2.5 * r_peak)
    print(f"\n  FFT搜索参数：")
    print(f"    理论峰位 r_peak = {img_px} / {actual_ppp:.2f} = {r_peak:.1f} px")
    print(f"    搜索范围: [{r_min:.1f}, {r_max:.1f}] px")
    
    if r_peak < 3:
        print(f"    ⚠️ 警告：r_peak太小，FFT可能找不到峰！")
    
    # 尝试FFT提取
    th, unc, info = extract_angle_fft(img, fov_nm=fov_actual, ppp=actual_ppp)
    if th is not None:
        err = abs(th - label)
        print(f"\n  FFT结果: θ = {th:.3f}°, 误差 = {err:.3f}°")
    else:
        print(f"\n  FFT结果: 失败，无法检测峰")
    
    return images_test, labels_test, fovs_test


def compare_image_sizes():
    """对比不同图像尺寸下FFT的表现"""
    print("\n" + "=" * 70)
    print("对比不同图像尺寸下FFT的表现")
    print("=" * 70)
    
    test_angles = [0.5, 1.0, 2.0, 3.0, 5.0]
    noise = 0.15
    blur = 0.5
    shear_y = 0.05
    
    results = {theta: {"128": [], "512": []} for theta in test_angles}
    
    for theta in test_angles:
        for trial in range(10):
            seed = hash(f"size_test_{theta}_{trial}") % (2**31)
            rng = np.random.default_rng(seed)
            
            # 生成512×512图像
            img_512, fov_nm = synthesize_reconstructed_moire(theta, FIXED_FOV_NM, n=512)
            
            # 应用退化
            img_512 = apply_background_tilt(img_512, 0.1, rng.uniform(-1, 1), rng.uniform(-1, 1))
            img_512 = apply_affine_distortion(img_512, shear_y/3, shear_y, 1.0, 1.0)
            img_512 = gaussian_filter(img_512, sigma=blur)
            ptp = img_512.max() - img_512.min()
            img_512 = img_512 + noise * ptp * rng.standard_normal(img_512.shape)
            img_512 = (img_512 - img_512.min()) / (img_512.max() - img_512.min() + 1e-9)
            
            # 裁剪到128×128
            n = img_512.shape[0]
            oy = (n - 128) // 2
            ox = (n - 128) // 2
            img_128 = img_512[oy:oy+128, ox:ox+128]
            
            # FFT on 512×512
            actual_ppp_512 = max(4.0, FIXED_FOV_NM / moire_period(theta))
            th_512, _, _ = extract_angle_fft(img_512, fov_nm=fov_nm, ppp=actual_ppp_512)
            if th_512 is not None:
                results[theta]["512"].append(abs(th_512 - theta))
            
            # FFT on 128×128 (eval_compare方式)
            scale = 128 / 512
            fov_128 = fov_nm * scale
            actual_ppp_128 = actual_ppp_512 * scale
            th_128, _, _ = extract_angle_fft(img_128, fov_nm=fov_128, ppp=actual_ppp_128)
            if th_128 is not None:
                results[theta]["128"].append(abs(th_128 - theta))
    
    # 打印结果
    print(f"\n{'θ (°)':<8} {'128×128 FFT MAE':>18} {'512×512 FFT MAE':>18} {'差异倍数':>12}")
    print("-" * 60)
    
    for theta in test_angles:
        mae_128 = np.mean(results[theta]["128"]) if results[theta]["128"] else np.nan
        mae_512 = np.mean(results[theta]["512"]) if results[theta]["512"] else np.nan
        
        if np.isfinite(mae_128) and np.isfinite(mae_512) and mae_512 > 1e-6:
            ratio = mae_128 / mae_512
        else:
            ratio = np.nan
        
        print(f"{theta:<8.1f} {mae_128:>18.4f} {mae_512:>18.4f} {ratio:>12.1f}x")
    
    return results


def analyze_ppp_effect():
    """分析ppp参数对FFT的影响"""
    print("\n" + "=" * 70)
    print("分析ppp（每周期像素数）对FFT的影响")
    print("=" * 70)
    
    theta = 2.0  # 测试角度
    
    # 不同ppp下的FFT峰位置
    print(f"\n对于θ={theta}°的moire图案：")
    L = moire_period(theta)
    print(f"  moiré周期 L = {L:.1f} nm")
    
    print(f"\n  {'图像尺寸':>10} {'ppp':>8} {'r_peak (px)':>12} {'搜索范围':>20}")
    print("  " + "-" * 55)
    
    for n_img in [512, 256, 128, 64]:
        # 计算ppp和r_peak
        ppp_values = []
        if n_img == 512:
            ppp = FIXED_FOV_NM / L
        else:
            ppp = (n_img / 512) * (FIXED_FOV_NM / L)
        
        r_peak = n_img / ppp
        r_min = max(2.0, 0.4 * r_peak)
        r_max = min(n_img // 3, 2.5 * r_peak)
        
        status = "✓" if r_peak >= 5 else "⚠️ 可能失效"
        print(f"  {n_img:>10}×{n_img:<4} {ppp:>8.2f} {r_peak:>12.1f} [{r_min:.1f}, {r_max:.1f}] {status}")


def visualize_fft_comparison():
    """可视化对比128×128和512×512下的FFT功率谱"""
    print("\n" + "=" * 70)
    print("生成FFT对比图")
    print("=" * 70)
    
    theta = 2.0
    seed = 42
    rng = np.random.default_rng(seed)
    
    # 生成图像
    img_512, fov_nm = synthesize_reconstructed_moire(theta, FIXED_FOV_NM, n=512)
    
    # 应用退化
    noise, blur, shear_y = 0.15, 0.5, 0.05
    img_512 = apply_background_tilt(img_512, 0.1, rng.uniform(-1, 1), rng.uniform(-1, 1))
    img_512 = apply_affine_distortion(img_512, shear_y/3, shear_y, 1.0, 1.0)
    img_512 = gaussian_filter(img_512, sigma=blur)
    ptp = img_512.max() - img_512.min()
    img_512 = img_512 + noise * ptp * rng.standard_normal(img_512.shape)
    img_512 = (img_512 - img_512.min()) / (img_512.max() - img_512.min() + 1e-9)
    
    # 裁剪到128
    n = img_512.shape[0]
    oy = (n - 128) // 2
    ox = (n - 128) // 2
    img_128 = img_512[oy:oy+128, ox:ox+128]
    
    # FFT
    actual_ppp_512 = max(4.0, FIXED_FOV_NM / moire_period(theta))
    th_512, unc_512, info_512 = extract_angle_fft(img_512, fov_nm=fov_nm, ppp=actual_ppp_512)
    
    scale = 128 / 512
    fov_128 = fov_nm * scale
    actual_ppp_128 = actual_ppp_512 * scale
    th_128, unc_128, info_128 = extract_angle_fft(img_128, fov_nm=fov_128, ppp=actual_ppp_128)
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：512×512
    axes[0, 0].imshow(img_512, cmap='afmhot')
    axes[0, 0].set_title(f"512×512 原图\nθ={theta}°")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.log1p(info_512['power']), cmap='viridis')
    axes[0, 1].set_title(f"512×512 FFT功率谱\nr_peak={512/actual_ppp_512:.1f}px")
    if len(info_512.get('coords', [])) > 0:
        for y, x in info_512['coords']:
            axes[0, 1].plot(x, y, 'r+', ms=10, mew=2)
    
    axes[0, 2].text(0.5, 0.5, 
                   f"FFT结果: θ = {th_512:.3f}°\n误差 = {abs(th_512-theta):.4f}°\n\nr_peak = {512/actual_ppp_512:.1f} px\n搜索范围正常",
                   ha='center', va='center', fontsize=14, transform=axes[0, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[0, 2].axis('off')
    
    # 第二行：128×128
    axes[1, 0].imshow(img_128, cmap='afmhot')
    axes[1, 0].set_title(f"128×128 裁剪图\nθ={theta}°")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.log1p(info_128['power']) if info_128 else np.zeros((128,128)), cmap='viridis')
    r_peak_128 = 128 / actual_ppp_128
    axes[1, 1].set_title(f"128×128 FFT功率谱\nr_peak={r_peak_128:.1f}px")
    if info_128 and len(info_128.get('coords', [])) > 0:
        for y, x in info_128['coords']:
            axes[1, 1].plot(x, y, 'r+', ms=10, mew=2)
    
    result_text = f"FFT结果: θ = {th_128:.3f}°\n误差 = {abs(th_128-theta):.4f}°\n\nr_peak = {r_peak_128:.1f} px"
    if r_peak_128 < 5:
        result_text += "\n\n⚠️ r_peak太小！"
    color = 'lightyellow' if abs(th_128-theta) < 0.5 else 'lightcoral'
    
    axes[1, 2].text(0.5, 0.5, result_text,
                   ha='center', va='center', fontsize=14, transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    axes[1, 2].axis('off')
    
    plt.suptitle(f"FFT对比：图像尺寸对角度提取的影响 (θ={theta}°)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(OUT_DIR, "fft_size_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"已保存: {path}")
    plt.close()


def main():
    print("=" * 70)
    print("FFT误差分析：为什么原测试集FFT误差高达6.34°")
    print("=" * 70)
    
    # 1. 分析原始测试集
    images_test, labels_test, fovs_test = analyze_fft_on_test_set()
    
    # 2. 对比不同图像尺寸
    compare_image_sizes()
    
    # 3. 分析ppp效应
    analyze_ppp_effect()
    
    # 4. 可视化
    visualize_fft_comparison()
    
    # 总结
    print("\n" + "=" * 70)
    print("分析结论")
    print("=" * 70)
    print("""
【根本原因】FFT在128×128小图上失效的原因：

1. **每周期像素数（ppp）不足**
   - 512×512图像：ppp ≈ 20，r_peak ≈ 25.6 px
   - 128×128图像：ppp ≈ 5，r_peak ≈ 25.6 px（理论值）
   - 但实际FFT峰检测需要足够的频率分辨率

2. **视野缩小4倍**
   - 512×512：fov ≈ 313.6 nm
   - 128×128：fov ≈ 78.4 nm
   - FFT频率分辨率 δf = 1/fov，分辨率下降4倍

3. **FFT理论不确定度**
   - δθ = a√3 / (2·fov)
   - fov缩小4倍 → δθ增大4倍
   - 128×128图像的理论极限约 0.2°

4. **采样与混叠**
   - 小图每周期只有~5个像素，接近Nyquist极限
   - moiré峰可能被混叠或分辨率模糊

【解决方案】

选项A：评估FFT时使用完整512×512图像
  - 优点：FFT发挥最佳性能
  - 缺点：与CNN输入尺寸不一致，比较不完全公平

选项B：训练CNN时也使用512×512图像
  - 优点：完全公平对比
  - 缺点：训练更慢，CNN可能过度依赖分辨率

选项C：保持现有设置，明确说明适用场景
  - FFT：适合高分辨率、大视野图像
  - CNN：适合小视野、已裁剪图像

【当前graded_eval.py已经正确实现】
使用512×512图像评估FFT，128×128评估CNN
这反映了两种方法的实际应用场景。
""")


if __name__ == "__main__":
    main()