"""One fusion rule shared by synthetic evaluation and real-data inference."""

from __future__ import annotations

import numpy as np


def fuse_angles(fft_angle, cnn_angle, cnn_std, tau: float):
    """Blend angles with FFT weight sigma_cnn / (sigma_cnn + tau).

    A missing FFT estimate falls back to CNN. All other non-finite inputs
    are rejected so they cannot silently enter a reported measurement.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    fft, cnn, std = np.broadcast_arrays(
        np.asarray(fft_angle, dtype=float),
        np.asarray(cnn_angle, dtype=float),
        np.asarray(cnn_std, dtype=float),
    )
    if not np.all(np.isfinite(cnn)) or not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError("CNN angles and uncertainties must be finite; uncertainty must be non-negative")
    if np.any(np.isinf(fft)):
        raise ValueError("FFT angles must be finite or NaN")
    weight_fft = std / (std + tau)
    weight_fft = np.where(np.isnan(fft), 0.0, weight_fft)
    fused = weight_fft * np.where(np.isnan(fft), 0.0, fft) + (1.0 - weight_fft) * cnn
    if fused.ndim == 0:
        return float(fused), float(weight_fft)
    return fused, weight_fft
