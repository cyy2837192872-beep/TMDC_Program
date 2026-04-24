"""Shared configuration constants for the MoS₂ moiré pipeline.

All physical constants and default simulation/training parameters
are defined here to avoid duplication across modules.
"""

# Physical constants
A_NM: float = 0.316  # MoS₂ lattice constant (nm)

# Default angle range for regression
THETA_MIN: float = 0.5  # degrees
THETA_MAX: float = 5.0  # degrees

# Default image dimensions
IMG_SIZE: int = 128       # CNN input size (pixels)
SIM_SIZE: int = 512       # Simulation resolution (pixels)
DEFAULT_PPP: int = 20     # Pixels Per moiré Period (for FFT pipeline)

# ── Default degradation parameter ranges (calibrated to Cypher ES + TITAN 70 + Tapping Mode) ──
# These are shared by dataset_generator.py, eval_compare.py, and evaluation scripts.
# Expansion ranges improve CNN robustness to out-of-distribution degradation.

DEFAULT_NOISE_RANGE: tuple[float, float] = (0.0, 0.15)
DEFAULT_BLUR_RANGE: tuple[float, float] = (0.0, 0.5)
DEFAULT_SCALE_RANGE: tuple[float, float] = (0.97, 1.03)
DEFAULT_SHEAR_X_RANGE: tuple[float, float] = (-0.008, 0.008)
DEFAULT_SHEAR_Y_RANGE: tuple[float, float] = (-0.015, 0.015)
DEFAULT_TILT_AMP_RANGE: tuple[float, float] = (0.0, 0.08)
DEFAULT_ROW_NOISE_RANGE: tuple[float, float] = (0.0, 0.04)
DEFAULT_ONEOVERF_RANGE: tuple[float, float] = (0.0, 0.05)
DEFAULT_RINGING_RANGE: tuple[float, float] = (0.0, 0.03)
DEFAULT_SCAN_OFFSET_RANGE: tuple[float, float] = (0.0, 0.015)
DEFAULT_TIP_RADIUS_NM: float = 7.0               # TITAN 70 probe (nm)
DEFAULT_TIP_RADIUS_RANGE: tuple[float, float] = (0.0, 7.0)
