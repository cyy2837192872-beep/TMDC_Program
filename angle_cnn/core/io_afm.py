"""Cypher ES / Asylum Research AFM data import and preprocessing.

Supports Igor Binary Wave (.ibw) files via the ``igor2`` library and provides
standard AFM image preprocessing routines (plane levelling, line-by-line
flattening, scar removal).

Install the optional reader dependency with::

    pip install igor2

Typical workflow
----------------
>>> from core.io_afm import load_cypher_image, preprocess_afm_image
>>> channels = load_cypher_image("scan_001.ibw")
>>> height = preprocess_afm_image(channels["Height"])
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

logger = logging.getLogger(__name__)

# Channel name aliases used by Cypher ES Tapping Mode exports.
_HEIGHT_ALIASES = {"Height", "HeightRetrace", "ZSensor", "ZSensorRetrace"}
_PHASE_ALIASES = {"Phase", "PhaseRetrace", "Phase1", "Phase1Retrace"}
_AMPLITUDE_ALIASES = {"Amplitude", "AmplitudeRetrace", "Amp", "AmpRetrace"}


# ---------------------------------------------------------------------------
# .ibw loader
# ---------------------------------------------------------------------------

def load_ibw(path: str) -> dict:
    """Load an Igor Binary Wave (.ibw) file and return raw wave + notes.

    Returns
    -------
    dict with keys ``"data"`` (ndarray), ``"note"`` (str), ``"name"`` (str).

    Raises
    ------
    ImportError  if ``igor2`` is not installed.
    FileNotFoundError  if *path* does not exist.
    """
    try:
        from igor2 import binarywave
    except ImportError as exc:
        raise ImportError(
            "igor2 is required to read .ibw files. Install with: pip install igor2"
        ) from exc
    if not os.path.isfile(path):
        raise FileNotFoundError(f"IBW file not found: {path}")
    wave = binarywave.load(path)
    data = wave["wave"]["wData"]
    note = wave["wave"].get("note", b"")
    if isinstance(note, bytes):
        note = note.decode("utf-8", errors="replace")
    name = wave["wave"]["wave_header"].get("bname", b"").decode("utf-8", errors="replace")
    return {"data": np.asarray(data, dtype=np.float64), "note": note, "name": name}


def _parse_note_kv(note: str) -> dict[str, str]:
    """Parse Asylum Research note string into key-value pairs."""
    kv: dict[str, str] = {}
    for line in note.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            kv[k.strip()] = v.strip()
    return kv


def load_cypher_image(
    path: str,
    *,
    return_metadata: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load a Cypher ES image and split multi-channel data.

    Cypher ES Tapping Mode typically saves a 3D array (rows, cols, channels)
    with channel names embedded in the note section.

    Parameters
    ----------
    path : Path to ``.ibw`` file.
    return_metadata : If True, also return parsed note metadata.

    Returns
    -------
    channels : ``{channel_name: (H, W) float64}``
    metadata : (optional) dict of note key-value pairs.
    """
    raw = load_ibw(path)
    data = raw["data"]
    meta = _parse_note_kv(raw["note"])

    channels: dict[str, np.ndarray] = {}

    if data.ndim == 2:
        channels[raw["name"] or "Height"] = data
    elif data.ndim == 3:
        n_ch = data.shape[2] if data.shape[2] < data.shape[0] else data.shape[0]
        ch_axis = 2 if data.shape[2] < data.shape[0] else 0
        channel_names = _guess_channel_names(meta, n_ch)
        for i in range(n_ch):
            if ch_axis == 2:
                channels[channel_names[i]] = data[:, :, i]
            else:
                channels[channel_names[i]] = data[i]
    else:
        channels["data"] = data

    if return_metadata:
        return channels, meta
    return channels


def _guess_channel_names(meta: dict[str, str], n_ch: int) -> list[str]:
    """Try to extract channel names from Asylum metadata."""
    names_str = meta.get("ChannelName", "")
    if names_str:
        names = [n.strip() for n in names_str.split(";") if n.strip()]
        if len(names) >= n_ch:
            return names[:n_ch]
    defaults = ["Height", "Phase", "Amplitude", "ZSensor"]
    return defaults[:n_ch] if n_ch <= len(defaults) else [f"Ch{i}" for i in range(n_ch)]


def find_channel(
    channels: dict[str, np.ndarray],
    kind: str,
) -> Optional[np.ndarray]:
    """Look up a channel by kind ("height", "phase", or "amplitude").

    Matches against common Asylum Research naming conventions.
    """
    aliases = {"height": _HEIGHT_ALIASES, "phase": _PHASE_ALIASES, "amplitude": _AMPLITUDE_ALIASES}
    candidates = aliases.get(kind.lower(), set())
    for name in candidates:
        if name in channels:
            return channels[name]
    for name, arr in channels.items():
        if kind.lower() in name.lower():
            return arr
    return None


# ---------------------------------------------------------------------------
# AFM image preprocessing
# ---------------------------------------------------------------------------

def plane_level(img: np.ndarray) -> np.ndarray:
    """Remove best-fit plane (1st-order polynomial background)."""
    n, m = img.shape
    yi, xi = np.mgrid[:n, :m]
    A = np.column_stack([xi.ravel(), yi.ravel(), np.ones(n * m)])
    z = img.ravel()
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    plane = (coeffs[0] * xi + coeffs[1] * yi + coeffs[2])
    return img - plane


def line_flatten(img: np.ndarray, order: int = 1) -> np.ndarray:
    """Line-by-line polynomial flattening (removes scan-line offsets).

    Parameters
    ----------
    order : Polynomial order (1 = linear, 2 = quadratic).
    """
    out = img.copy()
    n, m = img.shape
    x = np.arange(m, dtype=np.float64)
    for row in range(n):
        coeffs = np.polyfit(x, out[row], order)
        out[row] -= np.polyval(coeffs, x)
    return out


def remove_scars(
    img: np.ndarray,
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """Replace outlier scan lines (scars) with median-filtered values.

    A scan line whose mean deviates more than *threshold_sigma* standard
    deviations from the global mean is considered a scar.
    """
    out = img.copy()
    row_means = img.mean(axis=1)
    mu, sigma = row_means.mean(), row_means.std()
    if sigma < 1e-12:
        return out
    bad_rows = np.abs(row_means - mu) > threshold_sigma * sigma
    if bad_rows.any():
        med = median_filter(img, size=(3, 1))
        out[bad_rows] = med[bad_rows]
    return out


def normalize_to_unit(img: np.ndarray) -> np.ndarray:
    """Scale image to [0, 1] range."""
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def preprocess_afm_image(
    img: np.ndarray,
    *,
    do_plane_level: bool = True,
    do_line_flatten: bool = True,
    flatten_order: int = 1,
    do_scar_removal: bool = True,
    do_normalize: bool = True,
    gaussian_sigma: float = 0.0,
) -> np.ndarray:
    """Full AFM preprocessing pipeline.

    Typical use for Cypher ES data before CNN inference::

        height = preprocess_afm_image(raw_height)
    """
    out = img.astype(np.float64, copy=True)
    if do_plane_level:
        out = plane_level(out)
    if do_scar_removal:
        out = remove_scars(out)
    if do_line_flatten:
        out = line_flatten(out, order=flatten_order)
    if gaussian_sigma > 0:
        out = gaussian_filter(out, sigma=gaussian_sigma)
    if do_normalize:
        out = normalize_to_unit(out)
    return out
