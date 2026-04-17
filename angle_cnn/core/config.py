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
