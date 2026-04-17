"""MoS₂ moiré physics formulas.

Pure mathematical functions with no side effects.
FFT-pipeline-specific logic stays in moire_pipeline.py.
"""

from __future__ import annotations

import numpy as np

from .config import A_NM, THETA_MIN


def moire_period(theta_deg: float, a_nm: float = A_NM) -> float:
    """Moiré superlattice period from twist angle.

    Derivation:
      Hexagonal lattice first-shell reciprocal vector magnitude G = 4π/(a√3)
      Two layers twisted by θ → |ΔG| = 2G·sin(θ/2)
      L = 2π/|ΔG| = a√3 / (4·sin(θ/2))
    """
    if theta_deg == 0:
        raise ValueError("θ=0 is undefined")
    theta_rad = np.radians(theta_deg)
    return float(a_nm * np.sqrt(3) / (4.0 * np.sin(theta_rad / 2.0)))


def theta_from_period(L_nm: float, a_nm: float = A_NM) -> float:
    """Inverse: twist angle (degrees) from moiré period."""
    arg = np.clip(a_nm * np.sqrt(3) / (4.0 * L_nm), -1.0, 1.0)
    return float(2.0 * np.degrees(np.arcsin(arg)))


def angle_uncertainty(fov_nm: float, a_nm: float = A_NM) -> float:
    """FFT angle uncertainty (degrees) from field of view.

    δθ_rad = a√3 / (2·fov_nm), independent of θ.
    """
    return float(np.degrees(a_nm * np.sqrt(3) / (2.0 * fov_nm)))


# Default FOV: 10 moiré periods at minimum angle
FIXED_FOV_NM: float = 10.0 * moire_period(THETA_MIN)
