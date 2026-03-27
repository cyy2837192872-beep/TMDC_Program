#!/usr/bin/env python3
"""
dataset_generator.py — MoS₂ moiré CNN 训练数据集生成器
========================================================

生成 (image, θ) 对，保存为 .npz 文件供 CNN 训练使用。

数据集设计
----------
- 材料：MoS₂ 转角同质结（晶格常数 a = 0.316 nm）
- θ 范围：0.5° ~ 5.0°（均匀采样）
- 图像尺寸：128×128 px（CNN 输入，比仿真的 512px 小，训练更快）
- 每张图独立随机化：噪声强度、模糊半径、仿射畸变
- 三种 split：train / val / test = 8:1:1

退化模型 v2（基于文献真实 AFM 图像特征改进）
----------------------------------------------
原有退化：
1. 各向同性高斯噪声  — 电子噪声
2. 高斯模糊          — 有限空间分辨率（针尖卷积）
3. 随机仿射畸变      — 热漂移、压电非线性
4. 随机裁剪          — 视野不完整

v2 新增退化（参照文献真实 AFM 特征）：
5. 线性背景倾斜      — 样品未完全水平导致低频背景
   参数依据：Nat. Commun. 2020（Liao et al.）等文献中 tMoS₂ AFM
             图像均需平面校正，倾斜幅度约 10–30% peak-to-peak
6. 方向性热漂移      — 慢扫描轴（y方向）漂移远大于快扫描轴（x方向）
   参数依据：CVD TB-MoS₂ AFM 图（PMC10794196）明确指出非真空环境
             热漂移导致 FFT 图案变形，y方向漂移约为 x方向 3倍
7. 扫描行噪声        — AFM 逐行扫描引入的水平条纹
   参数依据：小角度 tMoS₂ AFM 图像（moiré 高度调制 ~0.157 nm，
             Nat. Commun. 2024 MoS₂/WSe₂ 1.1°样品）中行噪声
             幅度约为 moiré 信号的 5–20%

退化施加顺序（对应真实成像物理过程）：
  仿真 → 背景倾斜 → 方向性仿射 → 模糊 → 扫描行噪声 → 各向同性噪声 → 归一化 → 裁剪

运行方式
--------
    python dataset_generator.py

输出
----
    ~/tmdc-project/data/moire_dataset.npz
        images_train : (N_train, 128, 128) float32
        labels_train : (N_train,)          float32  单位：度
        images_val   : (N_val,   128, 128) float32
        labels_val   : (N_val,)            float32
        images_test  : (N_test,  128, 128) float32
        labels_test  : (N_test,)           float32
        fovs_train/val/test : (N,)         float32  单位：nm
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import os
import sys
import time

# ── 路径设置 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── 从 moire_pipeline 复用物理函数（含 MoS₂ 晶格常数）──
sys.path.insert(0, SCRIPT_DIR)
from moire_pipeline import moire_period, A_NM

# ── 数据集参数 ────────────────────────────────────────────
TOTAL_SAMPLES = 5000       # 总样本数
IMG_SIZE      = 128        # CNN 输入图像尺寸（px）
PPP           = 20         # pixels per moiré period（生成时用，之后裁剪到 IMG_SIZE）
THETA_MIN     = 0.5        # 最小转角（度）
THETA_MAX     = 5.0        # 最大转角（度）
FIXED_FOV_NM  = 10 * moire_period(THETA_MIN)   # 固定物理视野（基于 MoS₂ 参数）
SPLIT_RATIO   = (0.8, 0.1, 0.1)   # train / val / test

# ── 退化参数范围（均匀采样）────────────────────────────────
# 原有参数
NOISE_RANGE  = (0.0, 0.5)    # 各向同性高斯噪声幅度（相对 peak-to-peak）
BLUR_RANGE   = (0.0, 2.0)    # 高斯模糊（px），模拟针尖半径卷积
SCALE_RANGE  = (0.9, 1.1)    # 各向异性缩放

# v2：方向性热漂移（替代原来的各向同性 SHEAR_RANGE）
# 文献依据：非真空 AFM 热漂移主要沿慢扫描轴（y方向）累积
# PMC10794196 (CVD TB-MoS₂): "thermal drift is a typical challenge
# for AFM in non-vacuum environments"，y方向漂移约为 x方向 3×
SHEAR_X_RANGE = (-0.05, 0.05)   # 快扫描轴（x）：小
SHEAR_Y_RANGE = (-0.15, 0.15)   # 慢扫描轴（y）：大，约 3× x方向

# v2 新增：线性背景倾斜
# 样品未完全水平时图像叠加低频斜面背景，moiré 信号在其上
# 参数范围：倾斜幅度 0–30% peak-to-peak，方向随机
TILT_AMP_RANGE = (0.0, 0.3)

# v2 新增：扫描行噪声幅度（相对 peak-to-peak）
# 文献依据：tMoS₂/WSe₂ 1.1° 样品 moiré 高度调制 ~157 pm
# （Nat. Commun. 2024, doi:10.1038/s41467-024-53083-x）
# 行噪声约为 moiré 信号的 5–20%，对应归一化图像 0.0–0.15
ROW_NOISE_RANGE = (0.0, 0.15)


# ── 物理仿真（简化版，专为数据集生成优化）────────────────

def generate_moire_raw(theta_deg, ppp=PPP, n=512,
                       a_nm=A_NM, seed=None):
    """
    生成 MoS₂ moiré 图案（大图，之后再裁剪到 IMG_SIZE）

    直接生成 moiré 超晶格，避免原子晶格混叠问题。
    使用 MoS₂ 晶格常数 a = 0.316 nm。
    """
    rng = np.random.default_rng(seed)
    theta_rad = np.radians(theta_deg)
    L_nm      = moire_period(theta_deg, a_nm)
    nm_per_px = L_nm / ppp
    fov_nm    = n * nm_per_px

    x = np.linspace(0.0, fov_nm, n, endpoint=False)
    X, Y = np.meshgrid(x, x)

    q_m = 2.0 * np.pi / L_nm
    img = np.zeros((n, n), dtype=np.float64)
    for k in range(3):
        phi = theta_rad / 2.0 + np.radians(60.0 * k)
        img += np.cos(q_m * np.cos(phi) * X + q_m * np.sin(phi) * Y)

    return img, fov_nm, L_nm


def apply_affine_distortion(img, shear_x, shear_y, scale_x, scale_y):
    """
    仿射畸变：模拟 AFM 热漂移和压电非线性

    v2 改进：shear_y（慢扫描轴）参数范围设为 shear_x 的 3 倍，
    反映真实 AFM 热漂移的方向性特征。

    这是 FFT 方法最怕的退化类型：畸变后 moiré 条纹不再严格周期，
    FFT 峰展宽，频率估算偏移。
    """
    n = img.shape[0]
    cx, cy = n / 2, n / 2

    yi, xi = np.mgrid[0:n, 0:n]
    dx = xi - cx
    dy = yi - cy
    xi_src = cx + scale_x * dx + shear_x * dy
    yi_src = cy + shear_y * dx + scale_y * dy

    coords = np.array([yi_src.ravel(), xi_src.ravel()])
    img_dist = map_coordinates(img, coords, order=1,
                               mode='reflect').reshape(n, n)
    return img_dist


def apply_background_tilt(img, tilt_amp, ax, ay):
    """
    线性背景倾斜（v2 新增）

    模拟样品未完全水平时 AFM 图像叠加的低频斜面背景。
    实验中需要做平面校正（plane subtraction）才能看到 moiré 信号，
    此退化模拟校正前的原始图像状态。

    参数
    ----
    tilt_amp : 倾斜幅度（相对 peak-to-peak，0–0.3）
    ax, ay   : 倾斜方向（均匀分布在 [-1, 1]，归一化后使用）

    文献依据
    --------
    tMoS₂ 文献中 AFM 图像普遍需要平面校正，参见：
    Liao et al., Nat. Commun. 2020（doi:10.1038/s41467-020-16056-4）
    """
    n = img.shape[0]
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x)
    # 归一化方向向量，保证幅度可控
    norm = max(abs(ax) + abs(ay), 1e-6)
    tilt = tilt_amp * (ax / norm * X + ay / norm * Y)
    return img + tilt


def apply_row_noise(img, row_noise_amp, rng):
    """
    扫描行噪声（v2 新增）

    AFM 逐行扫描时，每行有独立的随机偏置，产生水平条纹。
    这是普通 AFM 图像中最常见的系统性噪声，在 2D 材料
    moiré 成像中尤为明显。

    参数
    ----
    row_noise_amp : 行噪声幅度（相对 peak-to-peak，0–0.15）

    文献依据
    --------
    MoS₂/WSe₂ 1.1° 样品 moiré 高度调制 ~157 pm
    （Nat. Commun. 2024, doi:10.1038/s41467-024-53083-x）
    行噪声通常为 moiré 信号的 5–20%，对应归一化图像 0–0.15
    """
    if row_noise_amp < 1e-4:
        return img
    row_offsets = rng.standard_normal(img.shape[0]) * row_noise_amp
    return img + row_offsets[:, np.newaxis]   # 每行加独立偏置（水平条纹）


def generate_sample(theta_deg, rng, img_size=IMG_SIZE):
    """
    生成单个训练样本：仿真 + 随机退化 + 裁剪到目标尺寸

    使用固定物理视野（FIXED_FOV_NM），确保不同角度图像的
    条纹密度随角度变化，CNN 能从中学习角度信息。

    v2 退化施加顺序（对应真实成像物理过程）：
      仿真 → 背景倾斜 → 方向性仿射 → 模糊 → 扫描行噪声 → 各向同性噪声 → 归一化 → 裁剪
    """
    # ── 1. 物理仿真（固定视野，ppp 随角度变化）──────────────
    actual_ppp = FIXED_FOV_NM / moire_period(theta_deg)
    actual_ppp = max(4.0, actual_ppp)
    img, fov_nm, L_nm = generate_moire_raw(theta_deg, ppp=actual_ppp, n=512)

    # ── 2. 小角度 AB/BA 两态晶格重构（θ < 2°）──────────────
    # 文献依据：Weston et al., Nat. Nanotechnol. 2020
    # （doi:10.1038/s41565-020-0682-9）
    # θ < 2° 时出现三角形畴重构，AB/BA 两类堆叠域交替排列。
    # 用 Im(ψ) 符号区分两类域，相位量化到 {0, π} 两态。
    if theta_deg < 2.0:
        theta_rad_s = np.radians(theta_deg)
        L_s         = moire_period(theta_deg)
        q_s         = 2.0 * np.pi / L_s
        fov_s       = 512 * (L_s / actual_ppp)
        x_s         = np.linspace(0.0, fov_s, 512, endpoint=False)
        X_s, Y_s    = np.meshgrid(x_s, x_s)
        psi = np.zeros((512, 512), dtype=complex)
        for k in range(3):
            phi = theta_rad_s / 2.0 + np.radians(60.0 * k)
            psi += np.exp(1j * (q_s * np.cos(phi) * X_s + q_s * np.sin(phi) * Y_s))
        strength    = np.clip(1.0 - theta_deg / 2.0, 0.0, 1.0)
        alpha       = 1.0 + strength * 8.0
        R           = np.abs(psi)
        Phi         = np.angle(psi)
        R_sharp     = np.tanh(alpha * R / R.max()) * R.max()
        # AB/BA 两态模型：Im(ψ) 符号区分两类堆叠域
        domain_sign = np.sign(np.imag(psi))           # +1=AB, -1=BA
        phase_quant = (domain_sign + 1) / 2 * np.pi  # AB→0, BA→π
        Phi_recon   = (1 - strength) * Phi + strength * phase_quant
        img         = R_sharp * np.cos(Phi_recon)

    # ── 3. 背景倾斜（v2 新增，在仿射之前施加）───────────────
    # 物理顺序：样品倾斜是静态背景，先于扫描引入的漂移
    tilt_amp = rng.uniform(*TILT_AMP_RANGE)
    if tilt_amp > 0.01:
        ax = rng.uniform(-1.0, 1.0)
        ay = rng.uniform(-1.0, 1.0)
        img = apply_background_tilt(img, tilt_amp, ax, ay)

    # ── 4. 方向性仿射畸变（v2 改进：y 方向漂移更大）────────
    # 快扫描轴（x）: ±0.05，慢扫描轴（y）: ±0.15（约 3× x方向）
    shear_x = rng.uniform(*SHEAR_X_RANGE)
    shear_y = rng.uniform(*SHEAR_Y_RANGE)
    scale_x = rng.uniform(*SCALE_RANGE)
    scale_y = rng.uniform(*SCALE_RANGE)
    img = apply_affine_distortion(img, shear_x, shear_y, scale_x, scale_y)

    # ── 5. 高斯模糊（针尖卷积，在噪声之前施加）─────────────
    # 物理顺序：有限分辨率对信号模糊，发生在检测电子噪声之前
    blur = rng.uniform(*BLUR_RANGE)
    if blur > 0.1:
        img = gaussian_filter(img, sigma=blur)

    # ── 6. 扫描行噪声（v2 新增，在各向同性噪声之前）────────
    # 物理顺序：行噪声是系统性偏置（每行独立），不被模糊
    row_noise_amp = rng.uniform(*ROW_NOISE_RANGE)
    img = apply_row_noise(img, row_noise_amp, rng)

    # ── 7. 各向同性高斯噪声（电子噪声）─────────────────────
    noise = rng.uniform(*NOISE_RANGE)
    if noise > 0.01:
        ptp = img.max() - img.min()
        img = img + noise * ptp * rng.standard_normal(img.shape)

    # ── 8. 归一化 ────────────────────────────────────────────
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)

    # ── 9. 随机裁剪到目标尺寸 ────────────────────────────────
    n = img.shape[0]
    if n > img_size:
        max_offset = n - img_size
        oy = rng.integers(0, max_offset + 1)
        ox = rng.integers(0, max_offset + 1)
        img = img[oy:oy + img_size, ox:ox + img_size]

    return img.astype(np.float32), fov_nm


# ── 主生成逻辑 ────────────────────────────────────────────

def generate_dataset(n_total=TOTAL_SAMPLES, seed=42):
    """
    生成完整数据集

    θ 均匀采样后打乱，确保 train/val/test 各角度分布均匀。
    """
    rng = np.random.default_rng(seed)

    print(f'生成 MoS₂ moiré 数据集：{n_total} 样本，IMG={IMG_SIZE}×{IMG_SIZE}')
    print(f'晶格常数 a = {A_NM} nm，固定视野 = {FIXED_FOV_NM:.1f} nm')
    print(f'θ 范围：{THETA_MIN}° ~ {THETA_MAX}°')
    print(f'退化 v2：noise={NOISE_RANGE}, blur={BLUR_RANGE}')
    print(f'        shear_x={SHEAR_X_RANGE}, shear_y={SHEAR_Y_RANGE}（方向性热漂移）')
    print(f'        tilt={TILT_AMP_RANGE}（背景倾斜）, row_noise={ROW_NOISE_RANGE}（扫描行噪声）')
    print()

    thetas = np.linspace(THETA_MIN, THETA_MAX, n_total)
    idx    = rng.permutation(n_total)
    thetas = thetas[idx]

    images = np.zeros((n_total, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    fovs   = np.zeros(n_total, dtype=np.float32)
    labels = thetas.astype(np.float32)

    t0 = time.time()
    for i in range(n_total):
        images[i], fovs[i] = generate_sample(thetas[i], rng, IMG_SIZE)

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (n_total - i - 1) / rate
            print(f'  [{i+1:>5}/{n_total}]  '
                  f'θ={thetas[i]:5.2f}°  '
                  f'速度={rate:.1f} img/s  '
                  f'ETA={eta:.0f}s')

    print(f'\n生成完成，耗时 {time.time()-t0:.1f}s')
    return images, labels, fovs


def split_dataset(images, labels, fovs, ratio=SPLIT_RATIO):
    """按比例切分 train/val/test"""
    n = len(images)
    n_train = int(n * ratio[0])
    n_val   = int(n * ratio[1])

    return (images[:n_train],          labels[:n_train],          fovs[:n_train],
            images[n_train:n_train+n_val], labels[n_train:n_train+n_val], fovs[n_train:n_train+n_val],
            images[n_train+n_val:],    labels[n_train+n_val:],    fovs[n_train+n_val:])


def print_stats(name, labels):
    print(f'  {name:10s}: {len(labels):5d} 样本  '
          f'θ ∈ [{labels.min():.2f}°, {labels.max():.2f}°]  '
          f'均值={labels.mean():.2f}°  std={labels.std():.2f}°')


# ── 主程序 ────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('MoS₂ moiré CNN 数据集生成器')
    print('=' * 60)

    # 1. 生成
    images, labels, fovs = generate_dataset(TOTAL_SAMPLES)

    # 2. 切分
    (img_train, lbl_train, fov_train,
     img_val,   lbl_val,   fov_val,
     img_test,  lbl_test,  fov_test) = split_dataset(images, labels, fovs)

    print('\n数据集划分：')
    print_stats('train', lbl_train)
    print_stats('val',   lbl_val)
    print_stats('test',  lbl_test)

    # 3. 保存
    save_path = os.path.join(DATA_DIR, 'moire_dataset.npz')
    np.savez_compressed(
        save_path,
        images_train=img_train, labels_train=lbl_train, fovs_train=fov_train,
        images_val=img_val,     labels_val=lbl_val,     fovs_val=fov_val,
        images_test=img_test,   labels_test=lbl_test,   fovs_test=fov_test,
    )

    size_mb = os.path.getsize(save_path) / 1024**2
    print(f'\n已保存: {save_path}')
    print(f'文件大小: {size_mb:.1f} MB')

    # 4. 预览
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 8, figsize=(20, 5))
        fig.suptitle(r'MoS$_2$ 数据集样本预览（前8个 train 样本）', fontsize=12)

        for i in range(8):
            axes[0, i].imshow(img_train[i], cmap='afmhot', vmin=0, vmax=1)
            axes[0, i].set_title(f'θ={lbl_train[i]:.2f}°', fontsize=9)
            axes[0, i].axis('off')

            axes[1, i].imshow(img_train[i + 8], cmap='afmhot', vmin=0, vmax=1)
            axes[1, i].set_title(f'θ={lbl_train[i+8]:.2f}°', fontsize=9)
            axes[1, i].axis('off')

        preview_path = os.path.join(DATA_DIR, 'dataset_preview.png')
        plt.tight_layout()
        plt.savefig(preview_path, dpi=120, bbox_inches='tight')
        print(f'预览图: {preview_path}')

    except Exception as e:
        print(f'预览图生成跳过: {e}')

    print('\n下一步：python train_cnn.py')