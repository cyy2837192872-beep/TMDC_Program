"""Small training helpers (losses, EMA) kept import-light for tests and train_cnn."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_huber_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    delta: float,
    angle_weight: float,
) -> torch.Tensor:
    """Huber loss; optionally up-weight high-angle samples in normalized label space.

    ``targets`` are normalized angles in [0, 1] (θ maps linearly from
    ``theta_min`` … ``theta_max``). When ``angle_weight > 0``,

        w = 1 + angle_weight * targets

    so labels near ``theta_max`` contribute more to the loss.
    """
    if angle_weight <= 0:
        return F.huber_loss(preds, targets, reduction="mean", delta=delta)
    per = F.huber_loss(preds, targets, reduction="none", delta=delta)
    w = 1.0 + float(angle_weight) * targets
    return (per * w).mean()


@torch.no_grad()
def ema_update(online: nn.Module, ema: nn.Module, decay: float) -> None:
    """In-place EMA: ``ema = decay * ema + (1 - decay) * online`` for all parameters.

    BatchNorm buffers (running_mean / running_var) are copied directly rather
    than EMA-averaged, because the online model already maintains them as
    running estimates.  Without this copy, the EMA model would use the
    initial values (mean=0, var=1) and produce garbage predictions.
    """
    d = float(decay)
    one_m = 1.0 - d
    for p_ema, p in zip(ema.parameters(), online.parameters()):
        p_ema.mul_(d).add_(p.detach(), alpha=one_m)
    for b_ema, b in zip(ema.buffers(), online.buffers()):
        b_ema.copy_(b)
