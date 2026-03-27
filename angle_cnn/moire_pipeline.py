#!/usr/bin/env python3
"""
moire_pipeline.py — MoS₂ moiré 图案仿真与 FFT 角度提取
==========================================================

Step 1 of 3：不用 AI 也能提取角度（baseline）

物理关系
--------
MoS₂ 晶格常数 a = 0.316 nm，两层旋转 θ 角后，
moiré 超周期由六角倒格矢之差决定：

    L = a√3 / (4 sin(θ/2))          ← FFT 可见的 moiré 条纹间距
    θ = 2 arcsin(a√3 / (4L))        ← 反推公式

仿真策略：直接生成 moiré 超晶格包络（不仿真原子晶格）
----------------------------------------------------------
两层六角晶格叠加时，差矢 ΔGᵢ = Gᵢ⁽²⁾ - Gᵢ⁽¹⁾ 产生 moiré 调制：
    I(r) = Σᵢ₌₁³ cos(ΔGᵢ · r)
    |ΔGᵢ| = 2π/L，方向 φₖ = θ/2 + k·60°

直接仿真 moiré 包络的好处：
  - 避免仿真原子晶格带来的混叠（a=0.316nm << moiré 视野）
  - FFT 中 moiré 峰强且位置精确，峰在像素距离 N/PPP 处
  - 更接近 AFM/STM 高度图的物理含义（高度调制 = moiré 包络）

FFT 分辨率决定的角度不确定度（误差传播）：
    δf = 1/fov_nm  →  δL = L²/fov_nm
    δθ_rad = a√3/(2·fov_nm)   （与 θ 无关！）
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
import os
OUT = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT, exist_ok=True)

# ── 中文字体 ──────────────────────────────────────────────────────────────────
myfont_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(myfont_path)
font_name = fm.FontProperties(fname=myfont_path).get_name()
rcParams['font.sans-serif'] = [font_name]
rcParams['axes.unicode_minus'] = False

# ── 全局参数 ──────────────────────────────────────────────────────────────────
A_NM = 0.316   # MoS₂ 晶格常数 (nm)
N    = 512     # 图像像素数（正方形）
PPP  = 20      # Pixels Per moiré Period（图像空间每 moiré 周期占多少像素）
               # FFT 峰位于像素半径 r = N/PPP = 512/20 = 25.6 px 处


# ── 物理公式 ──────────────────────────────────────────────────────────────────

def moire_period(theta_deg, a_nm=A_NM):
    """
    从转角计算 moiré 周期

    推导：
      六角晶格第一壳层倒格矢大小 G = 4π/(a√3)
      两层旋转 θ 后，差矢大小 |ΔG| = 2G·sin(θ/2)
      L = 2π/|ΔG| = a√3 / (4·sin(θ/2))
    """
    if theta_deg == 0:
        raise ValueError("θ=0 无意义")
    theta_rad = np.radians(theta_deg)
    return a_nm * np.sqrt(3) / (4.0 * np.sin(theta_rad / 2.0))


def theta_from_period(L_nm, a_nm=A_NM):
    """从 moiré 周期反推转角（度）"""
    arg = np.clip(a_nm * np.sqrt(3) / (4.0 * L_nm), -1.0, 1.0)
    return 2.0 * np.degrees(np.arcsin(arg))


def angle_uncertainty(fov_nm, a_nm=A_NM):
    """
    FFT 频率分辨率传播到角度的理论不确定度（度）

    δθ_rad = a√3 / (2·fov_nm)，与 θ 无关
    物理含义：不确定度只由视野大小决定，与转角无关
    """
    return np.degrees(a_nm * np.sqrt(3) / (2.0 * fov_nm))


# ── moiré 图案仿真（超晶格包络）─────────────────────────────────────────────

def generate_moire(theta_deg, a_nm=A_NM, n=N, ppp=PPP,
                   noise=0.0, blur=0.0, seed=None):
    """
    生成 MoS₂ 转角同质结 moiré 超晶格强度图

    核心思路：直接叠加 3 个 moiré 平面波（ΔGᵢ），不仿真原子晶格
    这样 FFT 中 moiré 峰是唯一显著峰，且位置精确

    Parameters
    ----------
    theta_deg : 转角（度）
    ppp       : 每 moiré 周期的像素数（决定 nm/pixel）
    noise     : 高斯噪声幅度（相对 peak-to-peak）
    blur      : 高斯模糊（像素），模拟有限分辨率

    Returns
    -------
    image  : (n, n) float32，归一化到 [0,1]
    fov_nm : 视野（nm）
    L_nm   : moiré 周期（nm）
    """
    rng = np.random.default_rng(seed)
    theta_rad = np.radians(theta_deg)

    L_nm       = moire_period(theta_deg, a_nm)
    nm_per_pix = L_nm / ppp
    fov_nm     = n * nm_per_pix

    x = np.linspace(0.0, fov_nm, n, endpoint=False)
    X, Y = np.meshgrid(x, x)

    q_m = 2.0 * np.pi / L_nm   # moiré 波矢大小

    psi = np.zeros((n, n), dtype=complex)
    for k in range(3):
        phi = theta_rad / 2.0 + np.radians(60.0 * k)
        psi += np.exp(1j * (q_m * np.cos(phi) * X + q_m * np.sin(phi) * Y))

    # ── 连续晶格重构模型 ─────────────────────────────────────
    # 将原来的 if θ<2° / else 硬切换改为连续插值，消除 2° 处的人工不连续性。
    #
    # strength = clip(1 - θ/2, 0, 1) 是重构强度权重：
    #   θ = 0°  → strength = 1.0（完全 AB/BA 三角畴，强锐化边界）
    #   θ = 1°  → strength = 0.5（三角畴与正弦调制各半，过渡区）
    #   θ ≥ 2° → strength = 0.0（纯正弦调制，退化为 real(ψ)）
    #
    # 物理依据：Weston et al. Nat. Nanotechnol. 2020 表明晶格重构在
    # θ < 2° 时主导图案，随转角增大连续减弱，无实验证据支持硬切换。
    #
    # 数学验证（strength=0 时公式连续退化）：
    #   R_sharp → R，phase_quant 权重 → 0，Phi_recon → Phi
    #   img = R·cos(Phi) = real(ψ)  ✓ 与原 else 分支完全一致
    strength = np.clip(1.0 - theta_deg / 2.0, 0.0, 1.0)
    R        = np.abs(psi)
    Phi      = np.angle(psi)

    # R_sharp：strength=0 时退化为 R（无锐化），strength=1 时强锐化
    R_sharp = ((1.0 - strength) * R
               + strength * np.tanh(strength * 8.0 * R / (R.max() + 1e-9)) * R.max())

    # AB/BA 两态相位量化（strength=0 时权重为零，自然消失）
    # Im(ψ) 符号区分两类堆叠域：+1=AB（相位→0），-1=BA（相位→π）
    domain_sign = np.sign(np.imag(psi))
    phase_quant = (domain_sign + 1) / 2.0 * np.pi   # AB→0, BA→π
    Phi_recon   = (1.0 - strength) * Phi + strength * phase_quant

    img = R_sharp * np.cos(Phi_recon)

    # 退化模型（先模糊后加噪，符合实际成像顺序）
    if blur > 0:
        img = gaussian_filter(img, sigma=blur)
    if noise > 0:
        ptp = float(img.max() - img.min())
        img = img + noise * ptp * rng.standard_normal(img.shape)

    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return img.astype(np.float32), fov_nm, L_nm



# ── FFT 角度提取 ──────────────────────────────────────────────────────────────

def extract_angle_fft(image, fov_nm, a_nm=A_NM, ppp=PPP, n_peaks=6):
    """
    从 moiré 图像中提取转角（FFT 方法）

    FFT 峰位置分析：
      - 频率分辨率：df = 1/fov_nm（nm⁻¹/pixel）
      - moiré 频率：f_m = 1/L_nm
      - 峰的像素半径：r_peak = f_m / df = fov_nm/L_nm = n/ppp
      → 搜索范围：[0.4, 2.5] × (n/ppp)，远离DC也远离边界

    Parameters
    ----------
    image   : 归一化图像 (n, n)
    fov_nm  : 视野（nm）
    ppp     : 生成时用的 pixels-per-period，决定搜索中心

    Returns
    -------
    theta_mean : 提取角度均值（度）
    unc_deg    : 理论不确定度（度）
    info       : 中间结果字典（供可视化）
    """
    n_img = image.shape[0]
    c = n_img // 2

    power = np.abs(np.fft.fftshift(np.fft.fft2(image))) ** 2

    Yi, Xi = np.ogrid[:n_img, :n_img]
    r_map  = np.sqrt((Xi - c)**2 + (Yi - c)**2)

    # 理论峰位（像素）：r_peak = n_img / ppp
    r_peak = n_img / ppp
    r_min  = max(2.0, 0.4 * r_peak)
    r_max  = min(n_img // 3, 2.5 * r_peak)

    work = power * ((r_map >= r_min) & (r_map <= r_max))

    # 局部极大值窗口大小：约 r_peak/3（奇数）
    win = max(3, int(r_peak / 3) * 2 + 1)
    local_max = (work == maximum_filter(work, size=win))
    valid     = local_max & (work > 0.05 * work.max())
    coords    = np.argwhere(valid)

    if len(coords) == 0:
        return None, None, {}

    # 按功率排序，取前 n_peaks 个
    pwr_at  = work[tuple(coords.T)]
    top_idx = np.argsort(pwr_at)[::-1][:n_peaks]
    coords  = coords[top_idx]

    # 像素半径 → 空间频率 → moiré 周期 → 角度
    r_px       = np.sqrt((coords[:, 1] - c)**2 + (coords[:, 0] - c)**2)
    L_vals     = fov_nm / r_px
    theta_vals = theta_from_period(L_vals, a_nm)

    theta_mean = float(np.mean(theta_vals))
    unc_deg    = angle_uncertainty(fov_nm, a_nm)

    freqs = np.fft.fftshift(np.fft.fftfreq(n_img, d=fov_nm / n_img))
    info  = dict(power=work, coords=coords, r_px=r_px,
                 L_vals=L_vals, theta_vals=theta_vals,
                 freqs=freqs, c=c)
    return theta_mean, unc_deg, info


# ── 可视化 ────────────────────────────────────────────────────────────────────

def show_single(theta_deg, noise=0.0, blur=0.0, save=True):
    """单角度完整分析：moiré图 + FFT功率谱 + 峰提取汇总"""
    img, fov_nm, L_true = generate_moire(theta_deg, noise=noise, blur=blur)
    th, unc, info = extract_angle_fft(img, fov_nm)

    if th is None:
        print(f'  [θ={theta_deg}°] 峰检测失败')
        return None, None

    freqs = info['freqs']
    f_lim = 3.0 / L_true

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        r'MoS$_2$ moiré 分析    '
        f'θ_true = {theta_deg}°    L = {L_true:.1f} nm    '
        f'θ_ext = {th:.3f}° ± {unc:.3f}°',
        fontsize=12
    )

    # ── 实空间图像 ──
    ax1.imshow(img, cmap='afmhot',
               extent=[0, fov_nm, fov_nm, 0], origin='upper')
    ax1.set_title(f'moiré 图像（视野 {fov_nm:.0f} nm）', fontsize=11)
    ax1.set_xlabel('x (nm)'); ax1.set_ylabel('y (nm)')

    # ── FFT 功率谱 ──
    fe = [freqs[0], freqs[-1], freqs[-1], freqs[0]]
    ax2.imshow(np.log1p(info['power']), cmap='viridis',
               extent=fe, origin='upper')
    for row, col in info['coords']:
        ax2.plot(freqs[col], freqs[row], 'r+', ms=14, mew=2.5)
    t_ring = np.linspace(0, 2 * np.pi, 300)
    f_m = 1.0 / L_true
    ax2.plot(f_m * np.cos(t_ring), f_m * np.sin(t_ring),
             'w--', lw=1, alpha=0.6, label=f'f_m={f_m:.3f} nm$^{{-1}}$')
    ax2.set(title='FFT 功率谱（对数）',
            xlabel='fx (nm$^{-1}$)', ylabel='fy (nm$^{-1}$)',
            xlim=(-f_lim, f_lim), ylim=(-f_lim, f_lim))
    ax2.legend(fontsize=8, loc='upper right')

    # ── 各峰提取角度 ──
    n_p = len(info['theta_vals'])
    ax3.bar(range(n_p), info['theta_vals'], color='steelblue', alpha=0.8, zorder=3)
    ax3.axhline(theta_deg, c='crimson',    ls='--', lw=2.0, zorder=4,
                label=f'真实值 {theta_deg}°')
    ax3.axhline(th,        c='darkorange', ls='-',  lw=2.0, zorder=5,
                label=f'FFT均值 {th:.3f}°')
    ax3.fill_between([-0.5, n_p - 0.5], th - unc, th + unc,
                     alpha=0.25, color='darkorange',
                     label=f'±{unc:.3f}°（FFT分辨率限）')
    ax3.set(xlabel='峰编号', ylabel='θ (°)', title='各 FFT 峰的提取结果')
    ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)
    ax3.set_xticks(range(n_p))

    plt.tight_layout()
    if save:
        noise_tag = f'_noise{noise:.0%}' if noise > 0 else ''
        blur_tag  = f'_blur{blur:.0f}px' if blur > 0 else ''
        fname = os.path.join(OUT, f'single_{theta_deg}deg{noise_tag}{blur_tag}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'  已保存: {fname}')
    plt.show()

    err = th - theta_deg
    print(f'  θ_true={theta_deg:5.2f}°  θ_ext={th:.4f}°  '
          f'误差={err:+.4f}°  理论unc=±{unc:.4f}°')
    return th, unc


def show_gallery(thetas=None, save=True):
    """不同转角的 moiré 图案 gallery（论文图）"""
    if thetas is None:
        thetas = [0.5, 1.0, 2.0, 3.0, 5.0]

    fig, axes = plt.subplots(1, len(thetas), figsize=(4.2 * len(thetas), 4.5))
    fig.suptitle(r'MoS$_2$ moiré 图案 vs 转角（仿真）', fontsize=13, y=1.01)

    for ax, theta in zip(axes, thetas):
        FIXED_FOV = 10 * moire_period(min(thetas))
        img, fov_nm, L_nm = generate_moire(theta, ppp=N * moire_period(theta) / FIXED_FOV)
        ax.imshow(img, cmap='afmhot', extent=[0, fov_nm, fov_nm, 0])
        ax.set_title(f'θ = {theta}°\nL = {L_nm:.1f} nm', fontsize=11)
        ax.set_xlabel('x (nm)')
        if ax is axes[0]:
            ax.set_ylabel('y (nm)')
        else:
            ax.set_yticks([])

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUT, 'moire_gallery.png'), dpi=150, bbox_inches='tight')
        print(f'  已保存: {os.path.join(OUT, "moire_gallery.png")}')
    plt.show()


def validation_sweep(thetas=None, noise=0.0, save=True):
    """
    多角度误差扫描

    核心结论（来自误差传播）：
      δθ（绝对值）= a√3/(2·fov)，与 θ 无关
      相对误差 δθ/θ ≈ 1/FOV_periods，随视野内包含的周期数增加而减小
    """
    if thetas is None:
        thetas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    print(f'\n{"θ_true":>8}  {"θ_ext":>9}  {"误差":>9}  {"理论unc":>9}  {"相对误差":>8}')
    print('─' * 55)

    rows = []
    for theta in thetas:
        img, fov_nm, _ = generate_moire(theta, noise=noise)
        th, unc, _      = extract_angle_fft(img, fov_nm)
        if th is None:
            print(f'{theta:>8.2f}°  峰检测失败')
            continue
        err = th - theta
        rel = abs(err) / theta * 100
        rows.append([theta, th, err, unc])
        print(f'{theta:>8.2f}°  {th:>9.4f}°  {err:>+9.4f}°  '
              f'{unc:>9.4f}°  {rel:>7.2f}%')

    if not rows:
        print('  所有角度提取失败')
        return None

    R = np.array(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r'MoS$_2$ FFT 角度提取验证', fontsize=13)

    ax1.plot(R[:, 0], R[:, 1], 'o-', c='steelblue', ms=8, lw=2, label='FFT 提取')
    ax1.fill_between(R[:, 0], R[:, 1] - R[:, 3], R[:, 1] + R[:, 3],
                     alpha=0.2, color='steelblue', label='±理论不确定度')
    ax1.plot(R[:, 0], R[:, 0], 'k--', lw=1.5, label='理想 y=x')
    ax1.set(xlabel='真实角度 θ (°)', ylabel=r'提取角度 $\hat{\theta}$ (°)', title='提取精度')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.semilogy(R[:, 0], np.clip(np.abs(R[:, 2]), 1e-6, None), 's-', c='crimson',
                 ms=8, lw=2, label='|绝对误差|')
    ax2.semilogy(R[:, 0], R[:, 3], '^--', c='darkorange',
                 ms=8, lw=2, label=r'理论不确定度 $a\sqrt{3}/(2\cdot fov)$')
    ax2.set(xlabel='真实角度 θ (°)', ylabel='误差 (°)', title='误差分析')
    ax2.legend(); ax2.grid(alpha=0.3, which='both')

    plt.tight_layout()
    if save:
        tag = f'_noise{noise:.0%}' if noise > 0 else ''
        plt.savefig(os.path.join(OUT, f'moire_validation{tag}.png'), dpi=150, bbox_inches='tight')
        print(f'\n  已保存: {os.path.join(OUT, f"moire_validation{tag}.png")}')
    plt.show()
    return R


# ── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('MoS₂ moiré Pipeline — Step 1: FFT Baseline')
    print('=' * 60)
    print(f'图像尺寸: {N}×{N} px，每 moiré 周期 {PPP} px')
    print(f'FFT moiré 峰理论位置: r ≈ {N/PPP:.1f} px（从中心起）')

    # ── 1. 理论周期预览 ──
    print('\n理论 moiré 周期 L = a√3/(4sin(θ/2)):')
    print(f'  {"θ":>6}  {"L (nm)":>8}  {"视野(PPP=20)":>14}  {"±δθ":>10}')
    print('  ' + '─' * 46)
    for th in [0.5, 1.0, 2.0, 3.0, 5.0]:
        L   = moire_period(th)
        fov = N * (L / PPP)
        unc = angle_uncertainty(fov)
        print(f'  {th:>6.1f}°  {L:>8.1f}  {fov:>12.1f} nm  {unc:>+10.4f}°')

    # ── 2. Gallery ──
    print('\n--- 生成 gallery ---')
    show_gallery()

    # ── 3. 单角度精细分析 ──
    print('\n--- 单角度分析 (θ = 2°) ---')
    show_single(2.0)

    print('\n--- 单角度分析 (θ = 0.5°，小角度挑战) ---')
    show_single(0.5)

    # ── 4. 含噪声+模糊（模拟实验退化）──
    print('\n--- 含噪声/模糊 (θ = 2°，noise=0.2，blur=1px) ---')
    show_single(2.0, noise=0.2, blur=1.0)

    # ── 5. 全角度验证 ──
    print('\n--- 角度扫描验证（无噪声）---')
    validation_sweep()

    print('\n--- 角度扫描验证（noise=0.2）---')
    validation_sweep(noise=0.2)

    # ── 6. 噪声压力测试 ──
    print('\n--- 噪声压力测试 ---')
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    test_thetas  = [0.5, 1.0, 2.0, 5.0]

    results = {theta: [] for theta in test_thetas}

    for noise in noise_levels:
        for theta in test_thetas:
            estimates = []
            for trial in range(5):
                img, fov_nm, _ = generate_moire(theta, noise=noise, seed=trial)
                th, unc, _     = extract_angle_fft(img, fov_nm)
                if th is not None:
                    estimates.append(th)
            if estimates:
                err = abs(np.median(estimates) - theta)
            else:
                err = np.nan
            results[theta].append(err)

    print(f'\n  {"noise":>8}', end='')
    for theta in test_thetas:
        print(f'  θ={theta}°误差', end='')
    print()
    print('  ' + '─' * (8 + 14 * len(test_thetas)))
    for i, noise in enumerate(noise_levels):
        print(f'  {noise:>8.1f}', end='')
        for theta in test_thetas:
            err = results[theta][i]
            marker = ' ❌' if np.isnan(err) or err > 1.0 else ''
            print(f'  {err:>8.4f}°{marker}  ', end='')
        print()

    fig, ax = plt.subplots(figsize=(9, 5))
    for theta in test_thetas:
        ax.plot(noise_levels, results[theta], 'o-', ms=6, lw=2, label=f'θ = {theta}°')
    ax.axhline(0.1, color='red', ls='--', lw=1.5, label='失效阈值 0.1°')
    ax.set(xlabel='噪声幅度（相对 peak-to-peak）',
           ylabel='|角度误差| (°)',
           title=r'MoS$_2$ FFT 方法噪声鲁棒性测试')
    ax.legend(); ax.grid(alpha=0.3)
    path = os.path.join(OUT, 'noise_stress_test.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\n  已保存: {path}')
    plt.show()