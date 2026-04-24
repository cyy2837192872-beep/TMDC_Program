"""Shared utilities for moiré CNN pipeline (fonts, I/O, simulation, degradations, AFM import)."""

from .config import (
    A_NM, IMG_SIZE, SIM_SIZE, DEFAULT_PPP, THETA_MIN, THETA_MAX,
    DEFAULT_BLUR_RANGE, DEFAULT_NOISE_RANGE, DEFAULT_ONEOVERF_RANGE,
    DEFAULT_RINGING_RANGE, DEFAULT_ROW_NOISE_RANGE, DEFAULT_SCALE_RANGE,
    DEFAULT_SCAN_OFFSET_RANGE, DEFAULT_SHEAR_X_RANGE, DEFAULT_SHEAR_Y_RANGE,
    DEFAULT_TILT_AMP_RANGE, DEFAULT_TIP_RADIUS_NM, DEFAULT_TIP_RADIUS_RANGE,
)
from .fonts import setup_matplotlib_cjk_font
from .io_utils import require_file, load_npz_dataset, load_model_checkpoint
from .physics import (
    moire_period,
    theta_from_period,
    angle_uncertainty,
    FIXED_FOV_NM,
    pixels_per_moire_period,
)
from .seed import set_global_seed

__all__ = [
    "A_NM", "IMG_SIZE", "SIM_SIZE", "DEFAULT_PPP",
    "THETA_MIN", "THETA_MAX",
    "DEFAULT_BLUR_RANGE", "DEFAULT_NOISE_RANGE", "DEFAULT_ONEOVERF_RANGE",
    "DEFAULT_RINGING_RANGE", "DEFAULT_ROW_NOISE_RANGE", "DEFAULT_SCALE_RANGE",
    "DEFAULT_SCAN_OFFSET_RANGE", "DEFAULT_SHEAR_X_RANGE", "DEFAULT_SHEAR_Y_RANGE",
    "DEFAULT_TILT_AMP_RANGE", "DEFAULT_TIP_RADIUS_NM", "DEFAULT_TIP_RADIUS_RANGE",
    "moire_period", "theta_from_period", "angle_uncertainty",
    "FIXED_FOV_NM", "pixels_per_moire_period",
    "setup_matplotlib_cjk_font",
    "require_file", "load_npz_dataset", "load_model_checkpoint",
    "set_global_seed",
]

# io_afm is imported on-demand (requires optional igor2 dependency)
