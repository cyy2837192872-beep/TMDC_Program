#!/usr/bin/env python3
"""
dual_stream.py — MoS₂ moiré 频域双流 CNN 架构
===============================================

设计理念
--------
Moiré 图像是周期性图案，频域特征（FFT peaks）对角度估计至关重要。
传统 CNN 在空间域操作，可能未能充分利用频域信息。

本模块实现双流架构：
1. 空间流 (Spatial Stream)：标准 CNN 提取空间特征
2. 频域流 (Frequency Stream)：从 FFT 幅度图提取频域特征
3. 特征融合：多种融合策略

架构变体
--------
- DualStreamNet：基础双流架构，简单拼接融合
- DualStreamNetWithAttention：带通道注意力的融合
- DualStreamNetWithCrossAttention：交叉注意力融合
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class FFTFeatureExtractor(nn.Module):
    """FFT 频域特征提取器。

    将输入图像转换为 FFT 幅度图，然后通过 CNN 提取频域特征。

    Parameters
    ----------
    out_channels : int
        输出特征通道数
    hidden_channels : int
        隐藏层通道数
    """

    def __init__(self, out_channels: int = 64, hidden_channels: int = 32):
        super().__init__()

        # 频域特征提取网络
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取 FFT 特征。

        Parameters
        ----------
        x : torch.Tensor
            输入图像，形状 (B, C, H, W)

        Returns
        -------
        torch.Tensor
            FFT 特征，形状 (B, out_channels, H/8, W/8)
        """
        # 取第一个通道（通常是高度图）
        if x.dim() == 4 and x.shape[1] > 1:
            x_fft = x[:, 0:1, :, :]  # (B, 1, H, W)
        else:
            x_fft = x.unsqueeze(1) if x.dim() == 3 else x

        # 计算 2D FFT
        fft2d = torch.fft.fft2(x_fft)
        fft_mag = torch.abs(torch.fft.fftshift(fft2d, dim=(-2, -1)))

        # Log 尺度压缩 + 归一化
        fft_mag = torch.log1p(fft_mag)
        b = fft_mag.shape[0]
        flat = fft_mag.reshape(b, -1)
        mean = flat.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        std = flat.std(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        fft_mag = (fft_mag - mean) / (std + 1e-6)

        # 通过 CNN 提取特征
        features = self.conv(fft_mag)

        return features


class ChannelAttention(nn.Module):
    """通道注意力模块 (SE-Block style)。

    Parameters
    ----------
    channels : int
        输入通道数
    reduction : int
        通道压缩比
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力模块。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class FeatureFusion(nn.Module):
    """特征融合模块。

    支持多种融合策略：
    - concat：简单拼接后卷积
    - add：逐元素相加
    - attention：注意力加权融合

    Parameters
    ----------
    spatial_channels : int
        空间流特征通道数
    freq_channels : int
        频域流特征通道数
    out_channels : int
        输出通道数
    fusion_type : str
        融合类型：'concat', 'add', 'attention'
    """

    def __init__(
        self,
        spatial_channels: int,
        freq_channels: int,
        out_channels: int,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Conv2d(spatial_channels + freq_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == "add":
            # 需要通道数匹配
            assert spatial_channels == freq_channels, "Add fusion requires matching channel counts"
            self.fusion = nn.Identity()
        elif fusion_type == "attention":
            self.spatial_attention = ChannelAttention(spatial_channels)
            self.freq_attention = ChannelAttention(freq_channels)
            self.fusion = nn.Sequential(
                nn.Conv2d(spatial_channels + freq_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor
    ) -> torch.Tensor:
        """融合空间和频域特征。

        Parameters
        ----------
        spatial_feat : torch.Tensor
            空间流特征
        freq_feat : torch.Tensor
            频域流特征

        Returns
        -------
        torch.Tensor
            融合后的特征
        """
        # 调整空间尺寸以匹配
        if spatial_feat.shape[-2:] != freq_feat.shape[-2:]:
            freq_feat = F.interpolate(
                freq_feat,
                size=spatial_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.fusion_type == "concat":
            combined = torch.cat([spatial_feat, freq_feat], dim=1)
            return self.fusion(combined)
        elif self.fusion_type == "add":
            return spatial_feat + freq_feat
        elif self.fusion_type == "attention":
            spatial_att = self.spatial_attention(spatial_feat)
            freq_att = self.freq_attention(freq_feat)
            combined = torch.cat([spatial_att, freq_att], dim=1)
            return self.fusion(combined)


class DualStreamNet(nn.Module):
    """双流 CNN 架构（空间流 + 频域流）。

    Parameters
    ----------
    n_channels : int
        输入图像通道数
    dropout : float
        Dropout 率
    arch : str
        骨干网络类型：'resnet18' 或 'resnet34'
    fusion_type : str
        特征融合类型
    freq_channels : int
        频域流输出通道数
    """

    def __init__(
        self,
        n_channels: int = 1,
        dropout: float = 0.3,
        arch: str = "resnet18",
        fusion_type: str = "concat",
        freq_channels: int = 64,
    ):
        super().__init__()

        # 骨干网络选择
        if arch == "resnet18":
            backbone_fn = resnet18
            hidden = 512
        elif arch == "resnet34":
            backbone_fn = resnet34
            hidden = 512
        else:
            raise ValueError(f"Unknown arch: {arch}")

        # 空间流：改造的 ResNet
        self.spatial_stream = backbone_fn(weights=None)
        self.spatial_stream.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.spatial_stream.maxpool = nn.Identity()

        # 移除原始 FC 层
        del self.spatial_stream.fc

        # 频域流：FFT 特征提取
        self.freq_stream = FFTFeatureExtractor(out_channels=freq_channels)

        # 特征融合
        self.fusion = FeatureFusion(
            spatial_channels=hidden,
            freq_channels=freq_channels,
            out_channels=hidden,
            fusion_type=fusion_type,
        )

        # 回归头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        self.arch = arch
        self.n_channels = n_channels
        self.fusion_type = fusion_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入图像，形状 (B, C, H, W)

        Returns
        -------
        torch.Tensor
            预测角度（归一化），形状 (B, 1)
        """
        # 空间流特征
        s1 = self.spatial_stream.conv1(x)
        s1 = self.spatial_stream.bn1(s1)
        s1 = self.spatial_stream.relu(s1)
        s1 = self.spatial_stream.layer1(s1)
        s1 = self.spatial_stream.layer2(s1)
        s1 = self.spatial_stream.layer3(s1)
        s1 = self.spatial_stream.layer4(s1)

        # 频域流特征
        f1 = self.freq_stream(x)

        # 特征融合
        fused = self.fusion(s1, f1)

        # 回归
        out = self.head(fused)

        return out


class DualStreamNetWithAttention(DualStreamNet):
    """带注意力机制的双流 CNN 架构。

    在特征融合前对两个流分别应用通道和空间注意力。
    """

    def __init__(
        self,
        n_channels: int = 1,
        dropout: float = 0.3,
        arch: str = "resnet18",
        fusion_type: str = "attention",
        freq_channels: int = 64,
    ):
        super().__init__(
            n_channels=n_channels,
            dropout=dropout,
            arch=arch,
            fusion_type=fusion_type,
            freq_channels=freq_channels,
        )

        # 为两个流添加注意力模块
        if arch in ["resnet18", "resnet34"]:
            hidden = 512
        else:
            hidden = 2048

        self.spatial_attention = nn.Sequential(
            ChannelAttention(hidden),
            SpatialAttention(),
        )
        self.freq_attention = nn.Sequential(
            ChannelAttention(freq_channels),
            SpatialAttention(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 空间流特征
        s1 = self.spatial_stream.conv1(x)
        s1 = self.spatial_stream.bn1(s1)
        s1 = self.spatial_stream.relu(s1)
        s1 = self.spatial_stream.layer1(s1)
        s1 = self.spatial_stream.layer2(s1)
        s1 = self.spatial_stream.layer3(s1)
        s1 = self.spatial_stream.layer4(s1)
        s1 = self.spatial_attention(s1)

        # 频域流特征
        f1 = self.freq_stream(x)
        f1 = self.freq_attention(f1)

        # 特征融合
        fused = self.fusion(s1, f1)

        # 回归
        out = self.head(fused)

        return out


class LightweightDualStreamNet(nn.Module):
    """轻量级双流网络。

    使用更小的卷积核和更少的参数，
    适合快速实验和资源受限场景。

    Parameters
    ----------
    n_channels : int
        输入通道数
    dropout : float
        Dropout 率
    base_channels : int
        基础通道数
    """

    def __init__(
        self,
        n_channels: int = 1,
        dropout: float = 0.3,
        base_channels: int = 32,
    ):
        super().__init__()

        # 空间流：轻量级 CNN
        self.spatial_stream = nn.Sequential(
            # Stage 1: 128 -> 64
            nn.Conv2d(n_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            # Stage 2: 64 -> 32
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            # Stage 3: 32 -> 16
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            # Stage 4: 16 -> 8
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # 频域流
        self.freq_stream = FFTFeatureExtractor(
            out_channels=base_channels * 4,
            hidden_channels=base_channels,
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 8 + base_channels * 4, base_channels * 8, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            ChannelAttention(base_channels * 8),
        )

        # 回归头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 空间流
        s_feat = self.spatial_stream(x)

        # 频域流
        f_feat = self.freq_stream(x)

        # 调整尺寸
        if s_feat.shape[-2:] != f_feat.shape[-2:]:
            f_feat = F.interpolate(
                f_feat, size=s_feat.shape[-2:], mode="bilinear", align_corners=False
            )

        # 融合
        combined = torch.cat([s_feat, f_feat], dim=1)
        fused = self.fusion(combined)

        # 回归
        out = self.head(fused)

        return out


def build_dual_stream_model(
    n_channels: int = 1,
    dropout: float = 0.3,
    arch: str = "resnet18",
    fusion_type: str = "concat",
    use_attention: bool = False,
    lightweight: bool = False,
) -> nn.Module:
    """构建双流模型。

    Parameters
    ----------
    n_channels : int
        输入通道数
    dropout : float
        Dropout 率
    arch : str
        骨干网络类型
    fusion_type : str
        融合类型
    use_attention : bool
        是否使用注意力机制
    lightweight : bool
        是否使用轻量级版本

    Returns
    -------
    nn.Module
        双流模型
    """
    if lightweight:
        return LightweightDualStreamNet(n_channels=n_channels, dropout=dropout)

    if use_attention:
        return DualStreamNetWithAttention(
            n_channels=n_channels,
            dropout=dropout,
            arch=arch,
            fusion_type=fusion_type,
        )

    return DualStreamNet(
        n_channels=n_channels,
        dropout=dropout,
        arch=arch,
        fusion_type=fusion_type,
    )