"""AFM-style degradations for Tapping Mode / Cypher ES / TITAN 70 probe.

Includes both legacy degradations (affine, tilt, blur, row noise, Gaussian noise)
and new instrument-specific models (tip convolution, 1/f noise, feedback ringing,
scan-direction offset).
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.ndimage import gaussian_filter, map_coordinates


_GRID_CACHE: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
_TILT_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _cached_affine_grid(n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    cached = _GRID_CACHE.get(n)
    if cached is None:
        yi, xi = np.mgrid[0:n, 0:n]
        cx, cy = n / 2, n / 2
        cached = (yi, xi, cx, cy)
        _GRID_CACHE[n] = cached
    return cached


def _cached_tilt_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _TILT_CACHE.get(n)
    if cached is None:
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        X, Y = np.meshgrid(x, x)
        cached = (X, Y)
        _TILT_CACHE[n] = cached
    return cached



# ---------------------------------------------------------------------------
# Legacy degradation functions (unchanged API)
# ---------------------------------------------------------------------------

def apply_affine_distortion(
    img: np.ndarray,
    shear_x: float,
    shear_y: float,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Affine warp via scipy map_coordinates (order=1, reflect)."""
    n = img.shape[0]
    yi, xi, cx, cy = _cached_affine_grid(n)
    dx, dy = xi - cx, yi - cy
    xi_src = cx + scale_x * dx + shear_x * dy
    yi_src = cy + shear_y * dx + scale_y * dy
    coords = np.array([yi_src.ravel(), xi_src.ravel()])
    return map_coordinates(img, coords, order=1, mode="reflect").reshape(n, n)


def apply_background_tilt(img: np.ndarray, tilt_amp: float, ax: float, ay: float) -> np.ndarray:
    """Linear ramp background (normalized direction)."""
    X, Y = _cached_tilt_grid(img.shape[0])
    norm = max(abs(ax) + abs(ay), 1e-6)
    return img + tilt_amp * (ax / norm * X + ay / norm * Y)


def apply_row_noise(img: np.ndarray, row_noise_amp: float, rng: Generator) -> np.ndarray:
    """Per-row bias (horizontal stripes)."""
    if row_noise_amp < 1e-4:
        return img
    row_bias = rng.standard_normal(img.shape[0]) * row_noise_amp
    return img + row_bias[:, None]


def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.1:
        return img
    return gaussian_filter(img, sigma=sigma)


def apply_isotropic_gaussian_noise(
    img: np.ndarray, noise_amp: float, rng: Generator
) -> np.ndarray:
    """Relative peak-to-peak scaled Gaussian noise."""
    if noise_amp <= 0.01:
        return img
    ptp = img.max() - img.min()
    return img + noise_amp * ptp * rng.standard_normal(img.shape)


# ---------------------------------------------------------------------------
# Instrument-specific degradations (TITAN 70 + Tapping Mode + Cypher ES)
# ---------------------------------------------------------------------------

def apply_tip_convolution(
    img: np.ndarray,
    tip_radius_nm: float,
    pixel_size_nm: float,
) -> np.ndarray:
    """Spherical AFM tip convolution via Fourier-space low-pass filter.

    For smooth moiré corrugation the lateral resolution of a parabolic tip
    scales as sqrt(2 R h) where h is the corrugation amplitude (~0.1-0.5 nm
    for MoS₂).  This is much finer than the tip radius itself.  We
    approximate with a Gaussian whose sigma = 0.15 R, giving realistic
    attenuation: features with period >> R are barely affected while features
    with period < R/2 are significantly smoothed.

    Parameters
    ----------
    tip_radius_nm : Tip apex radius in nm (TITAN 70 ≈ 7 nm).
    pixel_size_nm : Physical size of one pixel in nm (= fov_nm / n_pixels).
    """
    if tip_radius_nm <= 0 or pixel_size_nm <= 0:
        return img
    sigma_px = (tip_radius_nm * 0.15) / pixel_size_nm
    if sigma_px < 0.3:
        return img
    return gaussian_filter(img, sigma=sigma_px)


def apply_oneoverf_noise(
    img: np.ndarray,
    amplitude: float,
    alpha: float,
    rng: Generator,
) -> np.ndarray:
    """1/f^alpha (pink / red) noise common in AFM electronics and thermal drift.

    Parameters
    ----------
    amplitude : Noise strength relative to image peak-to-peak.
    alpha     : Spectral exponent (1.0 = pink, 2.0 = Brownian/red).
    """
    if amplitude <= 0.005:
        return img
    n = img.shape[0]
    white = rng.standard_normal((n, n))
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    f_mag = np.sqrt(fx ** 2 + fy ** 2)
    f_mag[0, 0] = 1.0
    power_filter = 1.0 / (f_mag ** (alpha / 2.0))
    power_filter[0, 0] = 0.0
    noise = np.real(np.fft.ifft2(np.fft.fft2(white) * power_filter))
    noise /= noise.std() + 1e-9
    ptp = img.max() - img.min()
    return img + amplitude * ptp * noise


def apply_feedback_ringing(
    img: np.ndarray,
    amplitude: float,
    rng: Generator,
    freq_range: tuple[float, float] = (0.15, 0.35),
    decay_range: tuple[float, float] = (3.0, 10.0),
) -> np.ndarray:
    """Tapping Mode feedback-loop ringing (damped oscillation along fast-scan axis).

    At steep features the Z-feedback overshoots, producing horizontal damped
    oscillation artefacts.  We convolve each row with a damped-sinusoid kernel
    whose parameters are randomised per call.
    """
    if amplitude <= 0.005:
        return img
    n = img.shape[0]
    freq = rng.uniform(*freq_range)
    decay = rng.uniform(*decay_range)
    t = np.arange(n, dtype=np.float64)
    kernel = np.exp(-t / decay) * np.cos(2.0 * np.pi * freq * t)
    kernel[0] = 0.0
    kernel /= np.abs(kernel).sum() + 1e-9

    gy = np.gradient(img, axis=1)
    gy_f = np.fft.rfft(gy, axis=1)
    k_f = np.fft.rfft(kernel, n=n)
    ring = np.fft.irfft(gy_f * k_f[np.newaxis, :], n=n, axis=1)
    ptp = img.max() - img.min()
    return img + amplitude * ptp * ring / (np.abs(ring).max() + 1e-9)


def apply_scan_direction_offset(
    img: np.ndarray,
    amplitude: float,
    rng: Generator,
) -> np.ndarray:
    """Trace / retrace mismatch — alternating-line vertical offset.

    Even with Cypher ES closed-loop scanner, a small residual offset between
    trace and retrace scan lines can remain.
    """
    if amplitude <= 0.002:
        return img
    ptp = img.max() - img.min()
    offset = amplitude * ptp * rng.uniform(-1.0, 1.0)
    out = img.copy()
    out[1::2] += offset
    return out


def apply_multichannel_degradation(
    channels: dict[str, np.ndarray],
    rng: Generator,
    *,
    tip_radius_nm: float = 0.0,
    pixel_size_nm: float = 1.0,
    noise_amp: float = 0.0,
    blur_sigma: float = 0.0,
    oneoverf_amp: float = 0.0,
    oneoverf_alpha: float = 1.0,
    row_noise_amp: float = 0.0,
    ringing_amp: float = 0.0,
    scan_offset_amp: float = 0.0,
    tilt_amp: float = 0.0,
    tilt_ax: float = 0.0,
    tilt_ay: float = 0.0,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> dict[str, np.ndarray]:
    """Apply a consistent degradation pipeline to all channels.

    Geometric distortions are applied identically across channels (same scan).
    Noise terms are drawn independently per channel (separate signal paths).
    """
    out: dict[str, np.ndarray] = {}
    for name, img in channels.items():
        x = img.copy()

        if name == "height" and tilt_amp > 0.01:
            x = apply_background_tilt(x, tilt_amp, tilt_ax, tilt_ay)

        if abs(shear_x) > 1e-4 or abs(shear_y) > 1e-4 or abs(scale_x - 1) > 1e-4 or abs(scale_y - 1) > 1e-4:
            x = apply_affine_distortion(x, shear_x, shear_y, scale_x, scale_y)

        x = apply_tip_convolution(x, tip_radius_nm, pixel_size_nm)
        x = apply_gaussian_blur(x, blur_sigma)
        x = apply_row_noise(x, row_noise_amp, rng)
        x = apply_feedback_ringing(x, ringing_amp, rng)
        x = apply_scan_direction_offset(x, scan_offset_amp, rng)
        x = apply_oneoverf_noise(x, oneoverf_amp, oneoverf_alpha, rng)
        x = apply_isotropic_gaussian_noise(x, noise_amp, rng)
        out[name] = x
    return out
