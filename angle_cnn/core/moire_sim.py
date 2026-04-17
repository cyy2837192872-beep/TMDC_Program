"""
Moiré superlattice synthesis with continuous lattice reconstruction.

Supports single-channel (height only, legacy) and multi-channel output
(height + Tapping-Mode phase + amplitude) for TITAN 70 / Cypher ES workflows.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import A_NM, SIM_SIZE
from .physics import moire_period

_X_LINSPACE_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _cached_linspace_axis(fov_nm: float, n: int) -> np.ndarray:
    key = (n, int(round(float(fov_nm) * 1e6)))
    if key not in _X_LINSPACE_CACHE:
        _X_LINSPACE_CACHE[key] = np.linspace(0.0, fov_nm, n, endpoint=False)
    return _X_LINSPACE_CACHE[key]


def _compute_moire_fields(
    theta_deg: float,
    fixed_fov_nm: float,
    n: int,
    a_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Core computation shared by single- and multi-channel synthesis.

    ``fixed_fov_nm`` is the physical scan width/height (nm): coordinates run
    ``[0, fov_nm)`` so that moiré period L(θ) implies ~``fov_nm / L`` periods
    across the image — **independent of θ** when ``fixed_fov_nm`` is constant.

    Returns (psi, R_sharp, Phi_recon, domain_sign, strength, fov_nm).
    """
    L_nm = moire_period(theta_deg, a_nm)
    # Physical extent must match the caller's intent (dataset: FIXED_FOV_NM).
    # Older versions incorrectly set fov_nm = n·L²/fixed_fov, which shrank the
    # moiré period in pixel space as θ→0 and made patterns look artificially dense.
    fov_nm = float(fixed_fov_nm)

    theta_rad = np.radians(theta_deg)
    q = 2.0 * np.pi / L_nm
    x = _cached_linspace_axis(fov_nm, n)
    X, Y = np.meshgrid(x, x)
    psi = np.zeros((n, n), dtype=np.complex128)
    for k in range(3):
        phi = theta_rad / 2.0 + np.radians(60.0 * k)
        psi += np.exp(1j * (q * np.cos(phi) * X + q * np.sin(phi) * Y))

    strength = float(np.clip(1.0 - theta_deg / 2.0, 0.0, 1.0))
    R = np.abs(psi)
    Phi = np.angle(psi)
    R_sharp = (1.0 - strength) * R + strength * np.tanh(
        strength * 8.0 * R / (R.max() + 1e-9)
    ) * R.max()
    domain_sign = np.sign(np.imag(psi))
    phase_quant = (domain_sign + 1) / 2.0 * np.pi
    Phi_recon = (1.0 - strength) * Phi + strength * phase_quant
    return psi, R_sharp, Phi_recon, domain_sign, strength, float(fov_nm)


def synthesize_reconstructed_moire(
    theta_deg: float,
    fixed_fov_nm: float,
    n: int = 512,
    a_nm: float = A_NM,
) -> tuple[np.ndarray, float]:
    """Build 2D height image (n x n) — backward-compatible single-channel API."""
    _, R_sharp, Phi_recon, _, _, fov_nm = _compute_moire_fields(
        theta_deg, fixed_fov_nm, n, a_nm
    )
    img = (R_sharp * np.cos(Phi_recon)).astype(np.float64)
    return img, fov_nm


def synthesize_multichannel_moire(
    theta_deg: float,
    fixed_fov_nm: float,
    n: int = 512,
    a_nm: float = A_NM,
    channels: tuple[str, ...] = ("height", "phase", "amplitude"),
) -> tuple[dict[str, np.ndarray], float]:
    """Synthesise Tapping-Mode multi-channel moiré image.

    Parameters
    ----------
    channels : Which channels to compute.  Any subset of
               ``("height", "phase", "amplitude")``.

    Returns
    -------
    channel_dict : ``{name: (n, n) float64}``
    fov_nm       : Physical field of view (nm).

    Channel physics
    ---------------
    height    : R_sharp * cos(Phi_recon) — topographic corrugation.
    phase     : Tapping-Mode phase contrast.  AB/BA stacking domains have
                different tip–sample energy dissipation, producing binary-like
                contrast that weakens with increasing twist angle.
    amplitude : Tapping-Mode amplitude error, inversely related to the
                topographic gradient magnitude (feedback error signal).
    """
    psi, R_sharp, Phi_recon, domain_sign, strength, fov_nm = _compute_moire_fields(
        theta_deg, fixed_fov_nm, n, a_nm
    )
    result: dict[str, np.ndarray] = {}

    if "height" in channels:
        result["height"] = (R_sharp * np.cos(Phi_recon)).astype(np.float64)

    if "phase" in channels:
        phase_contrast = 0.3 + 0.7 * strength
        phase_raw = phase_contrast * domain_sign.astype(np.float64)
        dw_sigma = max(1.0, 3.0 * (1.0 - strength))
        phase_smooth = gaussian_filter(phase_raw, sigma=dw_sigma)
        result["phase"] = phase_smooth

    if "amplitude" in channels:
        height = R_sharp * np.cos(Phi_recon)
        gy, gx = np.gradient(height)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        gmax = grad_mag.max()
        amp = 1.0 - grad_mag / (gmax + 1e-9) if gmax > 1e-9 else np.ones_like(height)
        result["amplitude"] = amp.astype(np.float64)

    return result, fov_nm
