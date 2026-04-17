"""CNN utilities shared across training and evaluation scripts.

This module intentionally avoids importing matplotlib or performing any
filesystem side-effects at import time, so it is safe to import from
evaluation scripts.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

from .config import THETA_MAX, THETA_MIN


def detect_n_channels(images: np.ndarray) -> int:
    """Auto-detect channel count from image array shape."""
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return int(images.shape[1])
    raise ValueError(f"Unexpected image shape: {images.shape}")


def compute_fft_channel(x: torch.Tensor) -> torch.Tensor:
    """Compute log FFT magnitude of channel 0 and append as extra channel.

    Works for both single-sample (C,H,W) and batched (B,C,H,W) input.
    """
    if x.dim() == 3:
        height = x[0:1]  # (1,H,W)
        fft2d = torch.fft.fft2(height)
        fft_mag = torch.log1p(torch.abs(torch.fft.fftshift(fft2d, dim=(-2, -1))))
        fft_mag = (fft_mag - fft_mag.mean()) / (fft_mag.std() + 1e-6)
        return torch.cat([x, fft_mag], dim=0)
    if x.dim() == 4:
        height = x[:, 0:1]  # (B,1,H,W)
        fft2d = torch.fft.fft2(height)
        fft_mag = torch.log1p(torch.abs(torch.fft.fftshift(fft2d, dim=(-2, -1))))
        b = int(fft_mag.shape[0])
        flat = fft_mag.reshape(b, -1)
        mean = flat.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        std = flat.std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        fft_mag = (fft_mag - mean) / (std + 1e-6)
        return torch.cat([x, fft_mag], dim=1)
    raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")


_ARCH_REGISTRY = {
    "resnet18": (resnet18, 512),
    "resnet34": (resnet34, 512),
    "resnet50": (resnet50, 2048),
}


def build_model(n_channels: int = 1, dropout: float = 0.3, arch: str = "resnet18") -> nn.Module:
    """ResNet backbone adapted for grayscale/multi-channel angle regression.

    arch : "resnet18" (default, ~11M params) | "resnet34" (~21M) | "resnet50" (~23M)
    """
    if arch not in _ARCH_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(_ARCH_REGISTRY)}")
    factory, hidden = _ARCH_REGISTRY[arch]
    model = factory(weights=None)
    model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
        nn.Linear(hidden, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(64, 1),
    )
    return model


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable Dropout layers while keeping BatchNorm in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """MC Dropout inference: run *n_samples* stochastic forward passes.

    Returns
    -------
    mean_deg : (batch,) predicted angle in degrees.
    std_deg  : (batch,) epistemic uncertainty in degrees.
    """
    model.eval()
    enable_mc_dropout(model)
    preds = []
    for _ in range(int(n_samples)):
        # torch.compile(reduce-overhead) 可启用 CUDA Graph：连续前向会复用输出缓冲区；
        # 若不 clone，stack 时会报 CUDAGraphs output overwritten。
        out = model(x).squeeze(1).detach().clone()
        preds.append(out)
    preds_t = torch.stack(preds, dim=0)  # (n_samples, batch)
    mean_norm = preds_t.mean(dim=0).clamp(0, 1).cpu().numpy()
    std_norm = preds_t.std(dim=0).cpu().numpy()
    scale = theta_max - theta_min
    return mean_norm * scale + theta_min, std_norm * scale


def warmup_cosine_lr(epoch: int, warmup: int, total: int, min_ratio: float = 0.01) -> float:
    """Warmup + cosine decay multiplier for LR schedulers."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))

