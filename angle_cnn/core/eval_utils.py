"""Shared utilities for evaluation scripts (distortion_sweep, graded_eval, etc.)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .config import THETA_MAX, THETA_MIN
from .cnn import build_model, compute_fft_channel
from .io_utils import load_model_checkpoint, state_dict_from_checkpoint


def load_model_from_checkpoint(
    model_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict]:
    """Load model and metadata from checkpoint.

    Returns (model, metadata_dict) where metadata_dict contains
    n_channels, dropout, add_fft_channel, base_n_channels, arch.
    """
    ckpt = load_model_checkpoint(model_path, map_location=device)
    meta = ckpt if isinstance(ckpt, dict) else {}
    n_ch = int(meta.get("n_channels", 1))
    dropout = float(meta.get("dropout", 0.3))
    add_fft = bool(meta.get("add_fft_channel", False))
    base_n_ch = int(meta.get("base_n_channels", n_ch))
    arch = str(meta.get("arch", "resnet18"))

    model = build_model(n_channels=n_ch, dropout=dropout, arch=arch).to(device)
    model.load_state_dict(state_dict_from_checkpoint(ckpt))
    model.eval()

    metadata = {
        "n_channels": n_ch,
        "dropout": dropout,
        "add_fft_channel": add_fft,
        "base_n_channels": base_n_ch,
        "arch": arch,
    }
    return model, metadata


def cnn_predict_single(
    model: torch.nn.Module,
    img_cnn: np.ndarray,
    device: torch.device,
    add_fft_channel: bool = False,
) -> float:
    """Single-image CNN prediction returning angle in degrees."""
    if img_cnn.ndim == 2:
        x = torch.from_numpy(img_cnn).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        x = torch.from_numpy(img_cnn).unsqueeze(0).float().to(device)
    if add_fft_channel:
        x = compute_fft_channel(x)
    with torch.no_grad():
        pred_norm = max(0.0, min(1.0, model(x).item()))
    return pred_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
