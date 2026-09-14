"""Physical-scale-aware conversion of a square AFM scan to CNN and FFT inputs."""

from __future__ import annotations

import numpy as np

from angle_cnn.core.physics import FIXED_FOV_NM

TRAIN_SIM_PIXELS = 512
TRAIN_CROP_PIXELS = 128
TRAIN_CROP_FOV_NM = FIXED_FOV_NM * TRAIN_CROP_PIXELS / TRAIN_SIM_PIXELS


def prepare_afm_channels(
    channels: dict, fov_nm: float, crop_fov_nm: float = TRAIN_CROP_FOV_NM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (3,128,128) CNN crop and original-resolution normalized height.

    Training normalizes each full synthetic channel before cropping a 128-pixel
    window from a 512-pixel simulation. Preserve that physical window here.
    """
    from scipy.ndimage import zoom

    if not np.isfinite(fov_nm) or fov_nm <= 0:
        raise ValueError("scan_size_nm must be a positive, finite number")
    if not np.isfinite(crop_fov_nm) or crop_fov_nm <= 0:
        raise ValueError("crop_fov_nm must be a positive, finite number")
    height = channels.get("height")
    if height is None:
        raise ValueError("a calibrated height channel is required")
    height = np.asarray(height, dtype=np.float32)
    if height.ndim != 2 or height.shape[0] != height.shape[1]:
        raise ValueError("AFM scan must be a square 2D image with one square FOV")
    n = height.shape[0]
    crop_px = round(crop_fov_nm * n / fov_nm)
    if crop_px < TRAIN_CROP_PIXELS:
        raise ValueError("AFM pixel resolution is below the training resolution; upsampling cannot recover detail")
    if crop_px > n:
        raise ValueError("AFM field of view is smaller than the training crop FOV")

    def normalize(arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != height.shape or not np.all(np.isfinite(arr)):
            raise ValueError("all AFM channels must have the same square shape and finite values")
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / (hi - lo + 1e-9)

    normalized = {name: normalize(value) for name, value in channels.items()}
    start = (n - crop_px) // 2
    images = []
    for name in ("height", "phase", "amplitude"):
        full = normalized.get(name, normalized["height"])
        crop = full[start:start + crop_px, start:start + crop_px]
        if crop_px != TRAIN_CROP_PIXELS:
            crop = zoom(crop, TRAIN_CROP_PIXELS / crop_px, order=1)
        images.append(crop.astype(np.float32))
    return np.stack(images), normalized["height"].astype(np.float32)
