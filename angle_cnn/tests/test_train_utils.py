"""Tests for core.train_utils (weighted Huber, EMA)."""

from __future__ import annotations

import torch
import torch.nn as nn

from angle_cnn.core.cnn import build_model
from angle_cnn.core.train_utils import ema_update, weighted_huber_loss


def test_weighted_huber_matches_plain_when_weight_zero():
    preds = torch.tensor([[0.3], [0.7]], requires_grad=True)
    targets = torch.tensor([[0.25], [0.65]])
    delta = 0.02
    l0 = weighted_huber_loss(preds, targets, delta=delta, angle_weight=0.0)
    l_ref = torch.nn.functional.huber_loss(preds, targets, delta=delta, reduction="mean")
    assert torch.allclose(l0, l_ref)


def test_weighted_huber_high_angle_higher_than_plain():
    preds = torch.zeros(4, 1, requires_grad=True)
    targets = torch.tensor([[0.1], [0.2], [0.8], [0.9]])
    delta = 0.02
    l_w = weighted_huber_loss(preds, targets, delta=delta, angle_weight=1.0)
    l_u = weighted_huber_loss(preds, targets, delta=delta, angle_weight=0.0)
    assert l_w > l_u


def test_ema_update_moves_toward_online():
    online = build_model(n_channels=3, dropout=0.0, arch="resnet18")
    ema = build_model(n_channels=3, dropout=0.0, arch="resnet18")
    with torch.no_grad():
        for p in ema.parameters():
            p.zero_()
        for p in online.parameters():
            p.fill_(1.0)
    ema_update(online, ema, decay=0.9)
    sample = next(ema.parameters())
    assert sample.mean().item() > 0.09


if __name__ == "__main__":
    test_weighted_huber_matches_plain_when_weight_zero()
    test_weighted_huber_high_angle_higher_than_plain()
    test_ema_update_moves_toward_online()
    print("test_train_utils: ok")
