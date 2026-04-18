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


def theta_from_period(L_nm: float | np.ndarray, a_nm: float = A_NM) -> float | np.ndarray:
    """Inverse: twist angle (degrees) from moiré period (scalar or array)."""
    L = np.asarray(L_nm, dtype=np.float64)
    arg = np.clip(a_nm * np.sqrt(3) / (4.0 * L), -1.0, 1.0)
    out = 2.0 * np.degrees(np.arcsin(arg))
    if np.ndim(L_nm) == 0:
        return float(out)
    return out


def angle_uncertainty(fov_nm: float, a_nm: float = A_NM) -> float:
    """FFT angle uncertainty (degrees) from field of view.

    δθ_rad = a√3 / (2·fov_nm), independent of θ.
    """
    return float(np.degrees(a_nm * np.sqrt(3) / (2.0 * fov_nm)))


# Default FOV: 10 moiré periods at minimum angle
FIXED_FOV_NM: float = 10.0 * moire_period(THETA_MIN)


def pixels_per_moire_period(
    n_pixels: int, theta_deg: float, fov_nm: float, a_nm: float = A_NM
) -> float:
    """Pixels spanning one moiré period for an n×n raster over ``fov_nm`` (nm).

    Used as ``ppp`` in ``extract_angle_fft`` (``r_peak = n_img / ppp``).
    Enforces a small floor to avoid aliasing when angles are large (tiny L).
    """
    L = moire_period(theta_deg, a_nm)
    return float(max(4.0, n_pixels * L / fov_nm))
