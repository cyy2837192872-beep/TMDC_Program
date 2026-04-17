#!/usr/bin/env python3
"""
dataset_generator.py — MoS₂ moiré CNN 训练数据集生成器（v2：多通道 + TITAN 70 退化模型）
=======================================================================================

生成 (image, θ) 对，保存为 .npz 文件供 CNN 训练使用。

v2 新增
-------
- 多通道支持：height / phase / amplitude（Tapping Mode 三通道）
- TITAN 70 探针卷积效应（tip_radius ≈ 7 nm）
- 1/f 噪声（AFM 电子学 + 热漂移）
- 反馈环路振荡（Tapping Mode ringing）
- 扫描方向偏移（trace/retrace 残余）
- 默认样本数从 5000 增大到 20000

数据集设计
----------
- 材料：MoS₂ 转角同质结（晶格常数 a = 0.316 nm）
- θ 范围：0.5° ~ 5.0°（均匀采样）
- 图像尺寸：128×128 px（CNN 输入）
- 通道数：1（仅 height）或 3（height + phase + amplitude）
- 三种 split：train / val / test = 8:1:1

运行方式
--------
    python dataset_generator.py                          # 3通道，20000样本
    python dataset_generator.py --channels 1             # 单通道（向后兼容）
    python dataset_generator.py --samples 50000 --workers 4
    python dataset_generator.py --tip-radius 7.0         # TITAN 70 探针
    python dataset_generator.py --no-progress            # 无进度条（日志/CI）
    python dataset_generator.py --worker-monitor         # 多进程时周期性打印子进程 CPU

输出
----
    ~/tmdc-project/data/moire_dataset.npz
        images_train : (N_train, [C,] 128, 128) float32
        labels_train : (N_train,) float32  单位：度
        fovs_train   : (N_train,) float32  单位：nm
        ...（val, test 同理）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.config import A_NM, DEFAULT_PPP, IMG_SIZE, SIM_SIZE, THETA_MIN, THETA_MAX  # noqa: E402
from core.physics import moire_period, pixels_per_moire_period  # noqa: E402
from core.fonts import cjk_fontproperties  # noqa: E402
from core.moire_sim import synthesize_multichannel_moire, synthesize_reconstructed_moire  # noqa: E402
from core.degrade import (  # noqa: E402
    apply_affine_distortion,
    apply_background_tilt,
    apply_feedback_ringing,
    apply_gaussian_blur,
    apply_isotropic_gaussian_noise,
    apply_multichannel_degradation,
    apply_oneoverf_noise,
    apply_row_noise,
    apply_scan_direction_offset,
    apply_tip_convolution,
)

logger = logging.getLogger(__name__)

# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_TOTAL_SAMPLES = 50000  # GPU 快速训练：更多数据 → 更好泛化
DEFAULT_SEED = 42
DEFAULT_SPLIT_RATIO = (0.8, 0.1, 0.1)
DEFAULT_N_CHANNELS = 3
DEFAULT_TIP_RADIUS_NM = 7.0  # TITAN 70
DEFAULT_TIP_RADIUS_RANGE = (0.0, 7.0)  # randomise: 0 = no probe, 7 = TITAN 70

# ── 默认退化参数范围（校准到 Cypher ES + TITAN 70 + Tapping Mode）────
# 扩展范围以提升 CNN 对分布外退化的鲁棒性
DEFAULT_NOISE_RANGE = (0.0, 0.15)
DEFAULT_BLUR_RANGE = (0.0, 0.5)
DEFAULT_SCALE_RANGE = (0.97, 1.03)
DEFAULT_SHEAR_X_RANGE = (-0.008, 0.008)
DEFAULT_SHEAR_Y_RANGE = (-0.015, 0.015)
DEFAULT_TILT_AMP_RANGE = (0.0, 0.08)
DEFAULT_ROW_NOISE_RANGE = (0.0, 0.04)
DEFAULT_ONEOVERF_RANGE = (0.0, 0.05)
DEFAULT_RINGING_RANGE = (0.0, 0.03)
DEFAULT_SCAN_OFFSET_RANGE = (0.0, 0.015)


CHANNEL_NAMES = ("height", "phase", "amplitude")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoS₂ moiré CNN 数据集生成器（v2：多通道 + TITAN 70）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_TOTAL_SAMPLES, help="总样本数")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="输出图像尺寸（像素）")
    parser.add_argument("--theta-min", type=float, default=THETA_MIN, help="最小转角（度）")
    parser.add_argument("--theta-max", type=float, default=THETA_MAX, help="最大转角（度）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行 worker 数（默认 auto = cpu_count/2）")
    parser.add_argument("--n-sim", type=int, default=512,
                        help="仿真分辨率（256=快速，512=高精度）")
    parser.add_argument("--log-theta", action="store_true",
                        help="对数均匀 θ 采样（小角度区域更密集）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--channels", type=int, default=DEFAULT_N_CHANNELS, choices=[1, 2, 3],
                        help="通道数 (1=height, 2=height+phase, 3=height+phase+amplitude)")
    parser.add_argument("--tip-radius", type=float, default=DEFAULT_TIP_RADIUS_NM,
                        help="探针尖端半径 (nm)，TITAN 70 ≈ 7 nm，0 禁用")
    parser.add_argument("--noise-max", type=float, default=DEFAULT_NOISE_RANGE[1], help="最大高斯噪声")
    parser.add_argument("--blur-max", type=float, default=DEFAULT_BLUR_RANGE[1], help="最大模糊半径")
    parser.add_argument("--shear-max", type=float, default=DEFAULT_SHEAR_Y_RANGE[1], help="最大剪切畸变")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIO[0], help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIO[1], help="验证集比例")
    parser.add_argument("--no-preview", action="store_true", help="不生成预览图")
    parser.add_argument("--no-progress", action="store_true",
                        help="禁用 tqdm 进度条（非 TTY 时默认也会关闭）")
    parser.add_argument("--worker-monitor", action="store_true",
                        help="多进程时每 8s 在进度条下方打印子进程 CPU%%（需 psutil）")
    parser.add_argument("--chunksize", type=int, default=16,
                        help="多进程任务分块大小（增大可降低进程通信开销）")
    return parser.parse_args()


def generate_moire_raw(theta_deg, ppp=DEFAULT_PPP, n=SIM_SIZE, a_nm=A_NM, seed=None):
    """Legacy single-channel generator (backward-compatible)."""
    rng = np.random.default_rng(seed)
    theta_rad = np.radians(theta_deg)
    L_nm = moire_period(theta_deg, a_nm)
    nm_per_px = L_nm / ppp
    fov_nm = n * nm_per_px
    x = np.linspace(0.0, fov_nm, n, endpoint=False)
    X, Y = np.meshgrid(x, x)
    q_m = 2.0 * np.pi / L_nm
    img = np.zeros((n, n), dtype=np.float64)
    for k in range(3):
        phi = theta_rad / 2.0 + np.radians(60.0 * k)
        img += np.cos(q_m * np.cos(phi) * X + q_m * np.sin(phi) * Y)
    return img, fov_nm, L_nm


def generate_sample(
    theta_deg: float,
    rng: np.random.Generator,
    img_size: int,
    fixed_fov_nm: float,
    n_channels: int = 1,
    tip_radius_nm: float = 0.0,
    tip_radius_range: Tuple[float, float] | None = None,
    n_sim: int = 256,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    blur_range: Tuple[float, float] = DEFAULT_BLUR_RANGE,
    scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    shear_x_range: Tuple[float, float] = DEFAULT_SHEAR_X_RANGE,
    shear_y_range: Tuple[float, float] = DEFAULT_SHEAR_Y_RANGE,
    tilt_amp_range: Tuple[float, float] = DEFAULT_TILT_AMP_RANGE,
    row_noise_range: Tuple[float, float] = DEFAULT_ROW_NOISE_RANGE,
    oneoverf_range: Tuple[float, float] = DEFAULT_ONEOVERF_RANGE,
    ringing_range: Tuple[float, float] = DEFAULT_RINGING_RANGE,
    scan_offset_range: Tuple[float, float] = DEFAULT_SCAN_OFFSET_RANGE,
):
    """Generate a single training sample with multi-channel support.

    Returns
    -------
    img : (C, img_size, img_size) float32 if n_channels > 1, else (img_size, img_size)
    fov_nm : float
    """
    requested = CHANNEL_NAMES[:n_channels]

    if n_channels > 1:
        ch_dict, fov_nm = synthesize_multichannel_moire(
            theta_deg, fixed_fov_nm, n=n_sim, channels=requested,
        )
    else:
        raw, fov_nm = synthesize_reconstructed_moire(theta_deg, fixed_fov_nm, n=n_sim)
        ch_dict = {"height": raw}

    ch_dict = {k: v.astype(np.float32, copy=False) for k, v in ch_dict.items()}

    pixel_size_nm = fov_nm / n_sim

    actual_tip_r = rng.uniform(*tip_radius_range) if tip_radius_range is not None else tip_radius_nm

    shear_x = rng.uniform(*shear_x_range)
    shear_y = rng.uniform(*shear_y_range)
    scale_x = rng.uniform(*scale_range)
    scale_y = rng.uniform(*scale_range)
    tilt_amp = rng.uniform(*tilt_amp_range)
    tilt_ax = rng.uniform(-1.0, 1.0)
    tilt_ay = rng.uniform(-1.0, 1.0)

    ch_dict = apply_multichannel_degradation(
        ch_dict, rng,
        tip_radius_nm=actual_tip_r,
        pixel_size_nm=pixel_size_nm,
        noise_amp=rng.uniform(*noise_range),
        blur_sigma=rng.uniform(*blur_range),
        oneoverf_amp=rng.uniform(*oneoverf_range),
        oneoverf_alpha=rng.uniform(0.8, 1.5),
        row_noise_amp=rng.uniform(*row_noise_range),
        ringing_amp=rng.uniform(*ringing_range),
        scan_offset_amp=rng.uniform(*scan_offset_range),
        tilt_amp=tilt_amp, tilt_ax=tilt_ax, tilt_ay=tilt_ay,
        shear_x=shear_x, shear_y=shear_y,
        scale_x=scale_x, scale_y=scale_y,
    )

    crops: list[np.ndarray] = []
    oy = ox = None
    for name in requested:
        arr = ch_dict[name]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        n = arr.shape[0]
        if n > img_size:
            if oy is None:
                oy = rng.integers(0, n - img_size + 1)
                ox = rng.integers(0, n - img_size + 1)
            arr = arr[oy: oy + img_size, ox: ox + img_size]
        crops.append(arr.astype(np.float32))

    if n_channels == 1:
        return crops[0], fov_nm
    return np.stack(crops, axis=0), fov_nm


def generate_sample_paired_cnn_fft(
    theta_deg: float,
    rng: np.random.Generator,
    img_size: int,
    fixed_fov_nm: float,
    n_channels: int = 1,
    tip_radius_nm: float = 0.0,
    tip_radius_range: Tuple[float, float] | None = None,
    n_sim: int = 512,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    blur_range: Tuple[float, float] = DEFAULT_BLUR_RANGE,
    scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    shear_x_range: Tuple[float, float] = DEFAULT_SHEAR_X_RANGE,
    shear_y_range: Tuple[float, float] = DEFAULT_SHEAR_Y_RANGE,
    tilt_amp_range: Tuple[float, float] = DEFAULT_TILT_AMP_RANGE,
    row_noise_range: Tuple[float, float] = DEFAULT_ROW_NOISE_RANGE,
    oneoverf_range: Tuple[float, float] = DEFAULT_ONEOVERF_RANGE,
    ringing_range: Tuple[float, float] = DEFAULT_RINGING_RANGE,
    scan_offset_range: Tuple[float, float] = DEFAULT_SCAN_OFFSET_RANGE,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """同一场景配对：与训练一致的退化后，CNN 取中心裁剪，FFT 用整幅 height。

    解决 eval_compare 中「CNN 用数据集图像、FFT 另起炉灶重算」导致的分布不一致。
    返回
    ----
    img_cnn : (H,W) 或 (C,H,W)，与 ``generate_sample`` 相同布局
    height_512 : (512, 512) 归一化后的 height，供 FFT
    fov_nm : float
    actual_ppp : float  供 ``extract_angle_fft`` 的 ppp 下界（仍依赖真值 θ；见评测说明）
    """
    requested = CHANNEL_NAMES[:n_channels]

    if n_channels > 1:
        ch_dict, fov_nm = synthesize_multichannel_moire(
            theta_deg, fixed_fov_nm, n=n_sim, channels=requested,
        )
    else:
        raw, fov_nm = synthesize_reconstructed_moire(theta_deg, fixed_fov_nm, n=n_sim)
        ch_dict = {"height": raw}

    ch_dict = {k: v.astype(np.float32, copy=False) for k, v in ch_dict.items()}
    pixel_size_nm = fov_nm / n_sim

    actual_tip_r = rng.uniform(*tip_radius_range) if tip_radius_range is not None else tip_radius_nm

    shear_x = rng.uniform(*shear_x_range)
    shear_y = rng.uniform(*shear_y_range)
    scale_x = rng.uniform(*scale_range)
    scale_y = rng.uniform(*scale_range)
    tilt_amp = rng.uniform(*tilt_amp_range)
    tilt_ax = rng.uniform(-1.0, 1.0)
    tilt_ay = rng.uniform(-1.0, 1.0)

    ch_dict = apply_multichannel_degradation(
        ch_dict, rng,
        tip_radius_nm=actual_tip_r,
        pixel_size_nm=pixel_size_nm,
        noise_amp=rng.uniform(*noise_range),
        blur_sigma=rng.uniform(*blur_range),
        oneoverf_amp=rng.uniform(*oneoverf_range),
        oneoverf_alpha=rng.uniform(0.8, 1.5),
        row_noise_amp=rng.uniform(*row_noise_range),
        ringing_amp=rng.uniform(*ringing_range),
        scan_offset_amp=rng.uniform(*scan_offset_range),
        tilt_amp=tilt_amp, tilt_ax=tilt_ax, tilt_ay=tilt_ay,
        shear_x=shear_x, shear_y=shear_y,
        scale_x=scale_x, scale_y=scale_y,
    )

    actual_ppp = pixels_per_moire_period(n_sim, theta_deg, fixed_fov_nm)

    crops: list[np.ndarray] = []
    for name in requested:
        arr = ch_dict[name]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        n = arr.shape[0]
        if n > img_size:
            oy = (n - img_size) // 2
            ox = (n - img_size) // 2
            arr = arr[oy: oy + img_size, ox: ox + img_size]
        crops.append(arr.astype(np.float32))

    h = ch_dict["height"]
    h_norm = (h - h.min()) / (h.max() - h.min() + 1e-9)

    if n_channels == 1:
        img_cnn = crops[0]
    else:
        img_cnn = np.stack(crops, axis=0)

    return img_cnn, h_norm.astype(np.float32), fov_nm, float(actual_ppp)


def _subseed(master: int, index: int) -> int:
    return int((master * 2654435761 + index * 104729) % (2**32 - 1))


_worker_config: dict = {}


def _init_worker(config: dict):
    global _worker_config
    _worker_config = config


def _parallel_sample_worker(args: Tuple[int, float, int]) -> Tuple[int, np.ndarray, float]:
    """返回 (样本下标, 图像, fov_nm)，便于乱序完成时写回正确位置。"""
    idx, theta_deg, sub_seed = args
    rng = np.random.default_rng(sub_seed)
    im, fv = generate_sample(
        theta_deg, rng,
        img_size=_worker_config["img_size"],
        fixed_fov_nm=_worker_config["fixed_fov_nm"],
        n_channels=_worker_config["n_channels"],
        tip_radius_nm=_worker_config["tip_radius_nm"],
        tip_radius_range=_worker_config.get("tip_radius_range"),
        n_sim=_worker_config["n_sim"],
        noise_range=_worker_config["noise_range"],
        blur_range=_worker_config["blur_range"],
        scale_range=_worker_config["scale_range"],
        shear_x_range=_worker_config["shear_x_range"],
        shear_y_range=_worker_config["shear_y_range"],
        tilt_amp_range=_worker_config["tilt_amp_range"],
        row_noise_range=_worker_config["row_noise_range"],
        oneoverf_range=_worker_config["oneoverf_range"],
        ringing_range=_worker_config["ringing_range"],
        scan_offset_range=_worker_config["scan_offset_range"],
    )
    return idx, im, fv


def _stderr_is_tty() -> bool:
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _want_progress(no_progress_flag: bool) -> bool:
    return (not no_progress_flag) and _stderr_is_tty() and tqdm is not None


def _worker_monitor_loop(stop: threading.Event, interval: float = 8.0) -> None:
    """在独立线程中周期性打印子进程 CPU（不阻塞主线程）。"""
    if psutil is None:
        return
    try:
        me = psutil.Process()
    except (psutil.Error, OSError):
        return
    while not stop.wait(interval):
        try:
            kids = me.children(recursive=True)
        except (psutil.Error, OSError):
            continue
        if not kids:
            continue
        cpus = []
        for c in kids:
            try:
                cpus.append(c.cpu_percent(interval=None))
            except (psutil.Error, OSError):
                pass
        if not cpus:
            continue
        line = (
            f"[worker-monitor] 子进程数={len(cpus)}  "
            f"CPU min/avg/max={min(cpus):.0f}% / {sum(cpus)/len(cpus):.0f}% / {max(cpus):.0f}%"
        )
        if tqdm is not None:
            tqdm.write(line, file=sys.stderr)
        else:
            print(line, file=sys.stderr)


def generate_dataset(
    n_total: int = DEFAULT_TOTAL_SAMPLES,
    seed: int = DEFAULT_SEED,
    parallel_workers: int | None = None,
    img_size: int = IMG_SIZE,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
    n_channels: int = DEFAULT_N_CHANNELS,
    tip_radius_nm: float = DEFAULT_TIP_RADIUS_NM,
    tip_radius_range: Tuple[float, float] | None = DEFAULT_TIP_RADIUS_RANGE,
    n_sim: int = 256,
    log_theta: bool = False,
    noise_range: Tuple[float, float] = DEFAULT_NOISE_RANGE,
    blur_range: Tuple[float, float] = DEFAULT_BLUR_RANGE,
    scale_range: Tuple[float, float] = DEFAULT_SCALE_RANGE,
    shear_x_range: Tuple[float, float] = DEFAULT_SHEAR_X_RANGE,
    shear_y_range: Tuple[float, float] = DEFAULT_SHEAR_Y_RANGE,
    tilt_amp_range: Tuple[float, float] = DEFAULT_TILT_AMP_RANGE,
    row_noise_range: Tuple[float, float] = DEFAULT_ROW_NOISE_RANGE,
    oneoverf_range: Tuple[float, float] = DEFAULT_ONEOVERF_RANGE,
    ringing_range: Tuple[float, float] = DEFAULT_RINGING_RANGE,
    scan_offset_range: Tuple[float, float] = DEFAULT_SCAN_OFFSET_RANGE,
    no_progress: bool = False,
    worker_monitor: bool = False,
    chunksize: int = 16,
):
    if parallel_workers is None:
        parallel_workers = int(os.environ.get("DATASET_NUM_WORKERS", "0"))

    rng = np.random.default_rng(seed)
    fixed_fov_nm = 10 * moire_period(theta_min)

    ch_label = f"{n_channels}ch ({', '.join(CHANNEL_NAMES[:n_channels])})"
    print(f"生成 MoS₂ moiré 数据集：{n_total} 样本，IMG={img_size}×{img_size}，{ch_label}")
    print(f"晶格常数 a = {A_NM} nm，固定视野 = {fixed_fov_nm:.1f} nm")
    print(f"θ 范围：{theta_min}° ~ {theta_max}°")
    if tip_radius_range is not None:
        print(f"探针卷积：tip_radius ∈ [{tip_radius_range[0]:.1f}, {tip_radius_range[1]:.1f}] nm (随机)")
    else:
        print(f"探针卷积：tip_radius = {tip_radius_nm:.1f} nm {'(TITAN 70)' if tip_radius_nm > 0 else '(禁用)'}")
    print(f"仿真分辨率：{n_sim}×{n_sim}")
    sampling = "对数均匀（小角度密集）" if log_theta else "线性均匀"
    print(f"θ 采样：{sampling}")
    print(f"退化参数：noise={noise_range}, blur={blur_range}, 1/f={oneoverf_range}")
    print(f"         shear_y={shear_y_range}, ringing={ringing_range}, scan_offset={scan_offset_range}")
    if parallel_workers and parallel_workers > 1:
        print(f"并行 workers: {parallel_workers}")
    print()

    if log_theta:
        thetas = np.exp(np.linspace(
            np.log(theta_min), np.log(theta_max), n_total
        ))
    else:
        thetas = np.linspace(theta_min, theta_max, n_total)
    idx = rng.permutation(n_total)
    thetas = thetas[idx]

    if n_channels > 1:
        images = np.zeros((n_total, n_channels, img_size, img_size), dtype=np.float32)
    else:
        images = np.zeros((n_total, img_size, img_size), dtype=np.float32)
    fovs = np.zeros(n_total, dtype=np.float32)
    labels = thetas.astype(np.float32)

    t0 = time.time()

    config = {
        "img_size": img_size,
        "fixed_fov_nm": fixed_fov_nm,
        "n_channels": n_channels,
        "tip_radius_nm": tip_radius_nm,
        "tip_radius_range": tip_radius_range,
        "n_sim": n_sim,
        "noise_range": noise_range,
        "blur_range": blur_range,
        "scale_range": scale_range,
        "shear_x_range": shear_x_range,
        "shear_y_range": shear_y_range,
        "tilt_amp_range": tilt_amp_range,
        "row_noise_range": row_noise_range,
        "oneoverf_range": oneoverf_range,
        "ringing_range": ringing_range,
        "scan_offset_range": scan_offset_range,
    }

    use_tqdm = _want_progress(no_progress)
    if worker_monitor and psutil is None:
        print("警告：未安装 psutil，--worker-monitor 无效。请 pip install psutil", file=sys.stderr)

    if parallel_workers and parallel_workers > 1:
        tasks = ((i, float(thetas[i]), _subseed(seed, i)) for i in range(n_total))
        _init_worker(config)
        monitor_stop = threading.Event()
        mon_th: threading.Thread | None = None
        if worker_monitor and psutil is not None:
            mon_th = threading.Thread(
                target=_worker_monitor_loop, args=(monitor_stop,), daemon=True,
            )
        with ProcessPoolExecutor(
            max_workers=parallel_workers,
            initializer=_init_worker,
            initargs=(config,),
        ) as pool:
            result_iter = pool.map(_parallel_sample_worker, tasks, chunksize=max(1, int(chunksize)))
            if mon_th is not None:
                mon_th.start()
            pbar = None
            if use_tqdm and tqdm is not None:
                pbar = tqdm(
                    result_iter,
                    total=n_total,
                    mininterval=0.25,
                    unit="img",
                    desc="生成样本",
                    file=sys.stderr,
                    dynamic_ncols=True,
                )
                result_iter = pbar
            completed = 0
            try:
                for idx, im, fv in result_iter:
                    images[idx] = im
                    fovs[idx] = fv
                    completed += 1
                    if pbar is not None:
                        pbar.set_postfix(θ=f"{thetas[idx]:.2f}°", refresh=False)
                    elif completed == 1 or completed % 500 == 0:
                        elapsed = time.time() - t0
                        rate = completed / elapsed
                        eta = (n_total - completed) / rate if rate > 0 else 0
                        print(
                            f"  [{completed:>5}/{n_total}]  θ={thetas[idx]:5.2f}°  "
                            f"速度={rate:.1f} img/s  ETA={eta:.0f}s",
                            flush=True,
                        )
            finally:
                monitor_stop.set()
                if mon_th is not None:
                    mon_th.join(timeout=2.0)
            if pbar is not None:
                pbar.close()
    else:
        seq = range(n_total)
        pbar_seq = None
        if use_tqdm and tqdm is not None:
            pbar_seq = tqdm(
                seq, total=n_total, unit="img", desc="生成样本",
                file=sys.stderr, dynamic_ncols=True,
            )
            seq = pbar_seq
        for i in seq:
            images[i], fovs[i] = generate_sample(
                thetas[i], rng,
                img_size=img_size,
                fixed_fov_nm=fixed_fov_nm,
                n_channels=n_channels,
                tip_radius_nm=tip_radius_nm,
                tip_radius_range=tip_radius_range,
                n_sim=n_sim,
                noise_range=noise_range,
                blur_range=blur_range,
                scale_range=scale_range,
                shear_x_range=shear_x_range,
                shear_y_range=shear_y_range,
                tilt_amp_range=tilt_amp_range,
                row_noise_range=row_noise_range,
                oneoverf_range=oneoverf_range,
                ringing_range=ringing_range,
                scan_offset_range=scan_offset_range,
            )
            if pbar_seq is not None:
                pbar_seq.set_postfix(θ=f"{thetas[i]:.2f}°", refresh=False)
            elif (i + 1) % 500 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_total - i - 1) / rate
                print(
                    f"  [{i+1:>5}/{n_total}]  θ={thetas[i]:5.2f}°  "
                    f"速度={rate:.1f} img/s  ETA={eta:.0f}s",
                    flush=True,
                )
        if pbar_seq is not None:
            pbar_seq.close()

    print(f"\n生成完成，耗时 {time.time()-t0:.1f}s")
    return images, labels, fovs


def split_dataset(images, labels, fovs, ratio=DEFAULT_SPLIT_RATIO):
    n = len(images)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    return (
        images[:n_train], labels[:n_train], fovs[:n_train],
        images[n_train:n_train + n_val], labels[n_train:n_train + n_val], fovs[n_train:n_train + n_val],
        images[n_train + n_val:], labels[n_train + n_val:], fovs[n_train + n_val:],
    )


def print_stats(name, labels):
    print(
        f"  {name:10s}: {len(labels):5d} 样本  "
        f"θ ∈ [{labels.min():.2f}°, {labels.max():.2f}°]  "
        f"均值={labels.mean():.2f}°  std={labels.std():.2f}°"
    )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("MoS₂ moiré CNN 数据集生成器 v2（Tapping Mode + TITAN 70）")
    print("=" * 60)
    workers = args.workers
    if workers is None:
        env_w = os.environ.get("DATASET_NUM_WORKERS", "").strip()
        if env_w.isdigit() and int(env_w) > 0:
            workers = int(env_w)
        else:
            cpu = os.cpu_count() or 4
            # 数据生成是纯 CPU 任务，留 2 个核给系统和 GPU 调度
            workers = max(1, cpu - 2)

    noise_range = (0.0, args.noise_max)
    blur_range = (0.0, args.blur_max)
    shear_x_range = (-args.shear_max / 3, args.shear_max / 3)
    shear_y_range = (-args.shear_max, args.shear_max)
    split_ratio = (args.train_ratio, args.val_ratio, 1 - args.train_ratio - args.val_ratio)

    print(f"配置:")
    print(f"  样本数:      {args.samples}")
    print(f"  图像尺寸:    {args.img_size}×{args.img_size}")
    print(f"  通道数:      {args.channels} ({', '.join(CHANNEL_NAMES[:args.channels])})")
    print(f"  θ 范围:      {args.theta_min}° ~ {args.theta_max}°")
    print(f"  探针半径:    {args.tip_radius:.1f} nm")
    print(f"  仿真分辨率:  {args.n_sim}×{args.n_sim}")
    print(f"  θ 采样:      {'对数均匀' if args.log_theta else '线性均匀'}")
    print(f"  并行 workers: {workers}")
    print(f"  随机种子:    {args.seed}")
    if args.no_progress:
        _prog_desc = "关闭"
    elif _stderr_is_tty() and tqdm is not None:
        _prog_desc = "tqdm 进度条"
    else:
        _prog_desc = "文本行（约每 500 条）"
    print(f"  进度显示:    {_prog_desc}")
    print(f"  进程监控:    {'开启' if args.worker_monitor else '关闭'}")

    tip_radius_range = (0.0, args.tip_radius)

    images, labels, fovs = generate_dataset(
        n_total=args.samples,
        seed=args.seed,
        parallel_workers=workers,
        img_size=args.img_size,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        n_channels=args.channels,
        tip_radius_nm=args.tip_radius,
        tip_radius_range=tip_radius_range,
        n_sim=args.n_sim,
        log_theta=args.log_theta,
        noise_range=noise_range,
        blur_range=blur_range,
        shear_x_range=shear_x_range,
        shear_y_range=shear_y_range,
        no_progress=args.no_progress,
        worker_monitor=args.worker_monitor,
        chunksize=args.chunksize,
    )

    (
        img_train, lbl_train, fov_train,
        img_val, lbl_val, fov_val,
        img_test, lbl_test, fov_test,
    ) = split_dataset(images, labels, fovs, ratio=split_ratio)

    print("\n数据集划分：")
    print_stats("train", lbl_train)
    print_stats("val", lbl_val)
    print_stats("test", lbl_test)

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, "moire_dataset.npz")
    np.savez_compressed(
        save_path,
        images_train=img_train, labels_train=lbl_train, fovs_train=fov_train,
        images_val=img_val, labels_val=lbl_val, fovs_val=fov_val,
        images_test=img_test, labels_test=lbl_test, fovs_test=fov_test,
        config={
            "n_samples": args.samples,
            "img_size": args.img_size,
            "n_channels": args.channels,
            "channel_names": list(CHANNEL_NAMES[:args.channels]),
            "theta_min": args.theta_min,
            "theta_max": args.theta_max,
            "tip_radius_nm": args.tip_radius,
            "tip_radius_range": list(tip_radius_range),
            "n_sim": args.n_sim,
            "log_theta": args.log_theta,
            "seed": args.seed,
        },
    )

    size_mb = os.path.getsize(save_path) / 1024 ** 2
    print(f"\n已保存: {save_path}")
    print(f"文件大小: {size_mb:.1f} MB")
    print(f"images shape: {img_train.shape}")

    if not args.no_preview:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # suptitle 常不继承 rcParams 的 sans-serif；显式传入 CJK 字体文件
            _fp = cjk_fontproperties()

            n_show = min(8, len(img_train))
            n_ch = args.channels
            fig, axes = plt.subplots(n_ch, n_show, figsize=(2.5 * n_show, 2.5 * n_ch))
            if n_ch == 1:
                axes = axes[np.newaxis, :]
            ch_names = list(CHANNEL_NAMES[:n_ch])
            # 化学式用 mathtext 下标，与 CJK 混排；不依赖字体是否含 U+2082
            _title_zh = rf"$\mathrm{{MoS}}_2$ 数据集样本预览（{n_ch}ch）"
            _title_en = rf"$\mathrm{{MoS}}_2$ dataset preview ({n_ch} ch)"
            if _fp is not None:
                fig.suptitle(_title_zh, fontsize=12, fontproperties=_fp)
            else:
                fig.suptitle(_title_en, fontsize=12)

            for col in range(n_show):
                for row, cname in enumerate(ch_names):
                    if n_ch > 1:
                        arr = img_train[col, row]
                    else:
                        arr = img_train[col]
                    axes[row, col].imshow(arr, cmap="afmhot", vmin=0, vmax=1)
                    if row == 0:
                        _tkw: dict = {"fontsize": 9}
                        if _fp is not None:
                            _tkw["fontproperties"] = _fp
                        axes[row, col].set_title(
                            f"θ={lbl_train[col]:.2f}°",
                            **_tkw,
                        )
                    if col == 0:
                        axes[row, col].set_ylabel(cname, fontsize=9)
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])

            preview_path = os.path.join(output_dir, "dataset_preview.png")
            plt.tight_layout()
            plt.savefig(preview_path, dpi=120, bbox_inches="tight")
            print(f"预览图: {preview_path}")
        except ImportError as e:
            logger.warning("预览图跳过（matplotlib 不可用）: %s", e)
        except OSError as e:
            logger.warning("预览图保存失败: %s", e)

    print("\n下一步：python train_cnn.py")


if __name__ == "__main__":
    main()
