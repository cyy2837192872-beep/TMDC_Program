"""FFT evaluation utilities for fair CNN vs FFT comparison.

Consolidates functions previously duplicated across eval_compare.py
and other scripts, importing through the package to avoid fragile
sibling-module dependencies.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from angle_cnn.core.config import (
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
)
from angle_cnn.core.degrade import (
    apply_affine_distortion,
    apply_background_tilt,
    apply_feedback_ringing,
    apply_oneoverf_noise,
    apply_row_noise,
    apply_scan_direction_offset,
    apply_tip_convolution,
)
from angle_cnn.core.moire_sim import synthesize_reconstructed_moire
from angle_cnn.core.physics import FIXED_FOV_NM, pixels_per_moire_period
from angle_cnn.core.physics import moire_period, theta_from_period
from angle_cnn.core.config import THETA_MIN, THETA_MAX
from angle_cnn.moire_pipeline import extract_angle_fft, _subpixel_peak_2d


def extract_angle_fft_blind(
    image: np.ndarray,
    fov_nm: float,
    theta_range: tuple[float, float] = (THETA_MIN, THETA_MAX),
) -> float:
    """Estimate angle without using the sample's true angle or a guessed period.

    The fixed protocol supplies only the measured FOV and supported angle range.
    Six strong local spectral peaks must form one approximately hexagonal ring;
    ambiguous or undersampled images return NaN and count as FFT failures.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or image.shape[0] < 128:
        raise ValueError("FFT input must be a square image of at least 128 pixels")
    if not np.all(np.isfinite(image)) or not np.isfinite(fov_nm) or fov_nm <= 0:
        raise ValueError("FFT image and measured FOV must be finite and FOV positive")
    lo, hi = theta_range
    if not (0 < lo < hi):
        raise ValueError("theta_range must be increasing and positive")

    n = image.shape[0]
    centered = (image - image.mean()) * np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    yy, xx = np.indices(power.shape)
    radius = np.hypot(xx - n // 2, yy - n // 2)
    r_min = max(2.0, 0.8 * fov_nm / moire_period(lo))
    r_max = min(n / 3, 1.2 * fov_nm / moire_period(hi))
    if r_max <= r_min:
        return float("nan")
    annulus = (radius >= r_min) & (radius <= r_max)
    candidates = np.argwhere((power == maximum_filter(power, size=7)) & annulus)
    if len(candidates) < 6:
        return float("nan")
    strengths = power[tuple(candidates.T)]
    chosen = candidates[np.argsort(strengths)[-6:]]
    refined = np.array([_subpixel_peak_2d(power, int(y), int(x)) for y, x in chosen])
    vectors = refined - n // 2
    radii = np.hypot(vectors[:, 0], vectors[:, 1])
    median_radius = float(np.median(radii))
    if np.max(np.abs(radii - median_radius)) > max(2.0, 0.12 * median_radius):
        return float("nan")

    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    sector = np.pi / 3
    turns = ((angles - angles[0]) % (2 * np.pi)) / sector
    rounded = np.rint(turns)
    if len(set((rounded.astype(int) % 6).tolist())) != 6:
        return float("nan")
    if np.max(np.abs(turns - rounded)) > 0.25:  # 15 degrees
        return float("nan")
    result = float(theta_from_period(fov_nm / float(np.mean(radii))))
    return result


def generate_fft_image_512(
    theta_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Generate a 512×512 height image with training-distribution degradation for FFT.

    Returns (img_512, fov_nm, actual_ppp).
    """
    raw, fov_nm = synthesize_reconstructed_moire(theta_deg, FIXED_FOV_NM, n=512)
    pixel_size_nm = fov_nm / 512
    actual_ppp = pixels_per_moire_period(512, theta_deg, FIXED_FOV_NM)

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


def extract_angle_fft_robust(img_512: np.ndarray, fov_nm: float, actual_ppp: float) -> float:
    """Robust FFT angle extraction with light retry strategy.

    Strategy:
    1) default ppp / n_peaks=6
    2) ppp jitter (±8%)
    3) widen peak candidates (n_peaks=8)
    4) mild denoise then retry
    """
    candidates = [
        (actual_ppp, 6, None),
        (actual_ppp * 0.92, 6, None),
        (actual_ppp * 1.08, 6, None),
        (actual_ppp, 8, None),
        (actual_ppp * 0.92, 8, 0.8),
        (actual_ppp * 1.08, 8, 0.8),
    ]
    for ppp_try, n_peaks, blur_sigma in candidates:
        img_try = img_512 if blur_sigma is None else gaussian_filter(img_512, sigma=blur_sigma)
        th, _, _ = extract_angle_fft(img_try, fov_nm=fov_nm, ppp=max(4.0, ppp_try), n_peaks=n_peaks)
        if th is not None and np.isfinite(th):
            return float(th)
    return float("nan")


def fft_predict_batch_512(
    labels: np.ndarray,
    seed: int = 12345,
) -> np.ndarray:
    """Generate 512×512 images on-the-fly and run FFT extraction.

    This gives FFT a fair chance by providing full-resolution images
    rather than the 128×128 crops stored in the dataset which are too
    small for reliable FFT peak detection.
    """
    n = len(labels)
    preds = []
    rng = np.random.default_rng(seed)

    for i in range(n):
        theta_deg = float(labels[i])
        img_512, fov_nm, actual_ppp = generate_fft_image_512(theta_deg, rng)
        th = extract_angle_fft_robust(img_512, fov_nm=fov_nm, actual_ppp=actual_ppp)
        preds.append(th)
        if (i + 1) % 50 == 0:
            print(f"  FFT 进度: {i+1}/{n}")

    return np.array(preds, dtype=np.float32)
