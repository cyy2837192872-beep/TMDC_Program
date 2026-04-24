"""Dataset and checkpoint loading with clear preflight errors."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import torch

REQUIRED_NPZ_KEYS = (
    "images_train",
    "labels_train",
    "fovs_train",
    "images_val",
    "labels_val",
    "fovs_val",
    "images_test",
    "labels_test",
    "fovs_test",
)


def require_file(path: str, description: str = "File") -> str:
    """Raise FileNotFoundError with a helpful message if path is missing."""
    if not path:
        raise FileNotFoundError(f"{description} path is empty.")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            f"  (cwd={os.getcwd()})"
        )
    return path


def load_npz_dataset(path: str) -> np.lib.npyio.NpzFile:
    """Load .npz and verify keys expected by training/eval scripts."""
    require_file(path, "Dataset")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise OSError(f"Failed to load dataset npz: {path}") from exc
    missing = [k for k in REQUIRED_NPZ_KEYS if k not in data.files]
    if missing:
        data.close()
        raise KeyError(
            f"Dataset {path} is missing keys: {missing}. "
            f"Expected keys include: {REQUIRED_NPZ_KEYS[:3]}..."
        )
    return data


def validate_npz_dataset(data: np.lib.npyio.NpzFile) -> list[str]:
    """Validate dataset array shapes, types, and value ranges.

    Returns a list of warnings; an empty list means the dataset is valid.
    """
    from .config import THETA_MIN, THETA_MAX

    issues: list[str] = []
    for split in ("train", "val", "test"):
        img_key = f"images_{split}"
        lbl_key = f"labels_{split}"
        fov_key = f"fovs_{split}"
        if img_key not in data:
            continue
        imgs = data[img_key]
        lbls = data[lbl_key]
        fovs = data[fov_key]
        if imgs.ndim not in (3, 4):
            issues.append(f"{img_key}: expected 3D or 4D, got {imgs.ndim}D shape {imgs.shape}")
        elif imgs.ndim == 4 and imgs.shape[1] not in (1, 2, 3):
            issues.append(f"{img_key}: unexpected channel count {imgs.shape[1]}")
        if imgs.shape[-2] != imgs.shape[-1]:
            issues.append(f"{img_key}: non-square images {imgs.shape}")
        if lbls.ndim != 1:
            issues.append(f"{lbl_key}: expected 1D, got {lbls.ndim}D shape {lbls.shape}")
        if len(lbls) != len(imgs):
            issues.append(f"{lbl_key} length {len(lbls)} != {img_key} length {len(imgs)}")
        if len(lbls) != len(fovs):
            issues.append(f"{lbl_key} length {len(lbls)} != {fov_key} length {len(fovs)}")
        if len(lbls) > 0:
            lbl_min, lbl_max = float(lbls.min()), float(lbls.max())
            if lbl_min < THETA_MIN - 0.1 or lbl_max > THETA_MAX + 0.1:
                issues.append(f"{lbl_key}: labels range [{lbl_min:.2f}, {lbl_max:.2f}] outside [{THETA_MIN}, {THETA_MAX}]")
        if fovs.ndim != 1:
            issues.append(f"{fov_key}: expected 1D, got {fovs.ndim}D shape {fovs.shape}")
    return issues


def load_model_checkpoint(
    path: str,
    map_location: Optional[Union[str, Any]] = None,
) -> Any:
    """Load torch checkpoint; supports dict checkpoints with model_state_dict."""
    import torch

    require_file(path, "Model checkpoint")
    kwargs: dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    # torch>=2.6 guarantees weights_only parameter exists; checkpoint dicts
    # contain non-tensor metadata (arch, dropout, epoch, etc.) alongside
    # model_state_dict, so we must pass weights_only=False.
    import inspect

    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _normalize_state_dict_keys(sd: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip wrappers so weights load into an unwrapped nn.Module.

    - ``torch.compile`` checkpoints often prefix keys with ``_orig_mod.``
    - DataParallel / DDP often prefix with ``module.``
    """
    keys = list(sd.keys())
    if not keys:
        return dict(sd)
    sample = keys[0]
    if sample.startswith("_orig_mod."):
        prefix = "_orig_mod."
    elif sample.startswith("module."):
        prefix = "module."
    else:
        return dict(sd)
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}


def state_dict_from_checkpoint(ckpt: Any) -> Any:
    """Normalize load result to a state_dict."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise TypeError("Checkpoint must be a state_dict or dict with 'model_state_dict'.")
    return _normalize_state_dict_keys(sd)
