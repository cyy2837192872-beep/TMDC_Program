#!/usr/bin/env python3
"""
augment.py — MoS₂ moiré CNN 数据增强模块
==========================================

提供多尺度数据增强，解决固定 FOV 导致的尺度不变性缺失问题。

增强策略
--------
1. 多尺度随机裁剪 (MultiScaleCrop)
   - 从大视野仿真图中随机裁剪不同尺寸区域
   - resize 到固定网络输入尺寸
   - 模拟真实 AFM 中 FOV 可变的场景

2. 在线尺度增强 (OnlineScaleAugment)
   - 训练时随机缩放图像
   - 同时调整标签（如果尺度影响角度测量）

3. 频域增强 (FrequencyDomainAugment)
   - 随机添加/抑制特定频率分量
   - 模拟不同扫描条件下的频率响应差异
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


class MultiScaleCrop:
    """多尺度随机裁剪增强。

    从输入图像中随机裁剪不同尺寸的区域，然后 resize 到目标尺寸。
    这使 CNN 学习到尺度不变的特征表示。

    Parameters
    ----------
    target_size : int
        目标输出尺寸（正方形）
    scale_range : tuple(float, float)
        裁剪尺度范围，相对于 target_size
        例如 (0.7, 1.3) 表示裁剪 0.7*target_size 到 1.3*target_size 的区域
    """

    def __init__(
        self,
        target_size: int = 128,
        scale_range: Tuple[float, float] = (0.7, 1.3),
    ):
        self.target_size = target_size
        self.scale_range = scale_range

    def __call__(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """应用多尺度裁剪。

        Parameters
        ----------
        img : torch.Tensor
            输入图像，形状 (C, H, W) 或 (H, W)
        rng : np.random.Generator
            随机数生成器

        Returns
        -------
        torch.Tensor
            裁剪并 resize 后的图像，形状 (C, target_size, target_size)
        """
        if img.dim() == 2:
            img = img.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        _, h, w = img.shape
        assert h == w, "只支持正方形输入"

        # 随机选择裁剪尺度
        scale = rng.uniform(*self.scale_range)
        crop_size = int(self.target_size / scale)

        # 确保裁剪尺寸不超过原图
        crop_size = min(crop_size, h)

        # 随机裁剪位置
        top = rng.integers(0, h - crop_size + 1) if crop_size < h else 0
        left = rng.integers(0, w - crop_size + 1) if crop_size < w else 0

        # 裁剪
        cropped = img[:, top:top + crop_size, left:left + crop_size]

        # Resize 到目标尺寸
        if crop_size != self.target_size:
            cropped = cropped.unsqueeze(0)  # (1, C, crop_size, crop_size)
            cropped = F.interpolate(
                cropped,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
            cropped = cropped.squeeze(0)  # (C, target_size, target_size)

        if squeeze_output:
            cropped = cropped.squeeze(0)

        return cropped


class RandomZoom:
    """随机缩放增强。

    对图像进行随机放大/缩小，模拟不同 FOV 条件下的成像。

    Parameters
    ----------
    zoom_range : tuple(float, float)
        缩放因子范围，例如 (0.8, 1.2) 表示 0.8x 到 1.2x 缩放
    target_size : int
        目标输出尺寸
    """

    def __init__(
        self,
        zoom_range: Tuple[float, float] = (0.8, 1.2),
        target_size: int = 128,
    ):
        self.zoom_range = zoom_range
        self.target_size = target_size

    def __call__(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """应用随机缩放。

        Parameters
        ----------
        img : torch.Tensor
            输入图像，形状 (C, H, W) 或 (H, W)
        rng : np.random.Generator
            随机数生成器

        Returns
        -------
        torch.Tensor
            缩放后的图像
        """
        if img.dim() == 2:
            img = img.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        _, h, w = img.shape

        # 随机缩放因子
        zoom = rng.uniform(*self.zoom_range)

        # 计算缩放后的尺寸
        new_size = int(h * zoom)

        # 缩放
        img_batch = img.unsqueeze(0)  # (1, C, H, W)
        scaled = F.interpolate(
            img_batch,
            size=(new_size, new_size),
            mode="bilinear",
            align_corners=False,
        )
        scaled = scaled.squeeze(0)  # (C, new_size, new_size)

        # 如果缩放后尺寸变大，中心裁剪到目标尺寸
        # 如果缩放后尺寸变小，padding 到目标尺寸
        if new_size > self.target_size:
            # 中心裁剪
            start = (new_size - self.target_size) // 2
            scaled = scaled[:, start:start + self.target_size, start:start + self.target_size]
        elif new_size < self.target_size:
            # Padding
            pad = (self.target_size - new_size) // 2
            pad_left = pad
            pad_right = self.target_size - new_size - pad
            scaled = F.pad(scaled, (pad_left, pad_right, pad_left, pad_right), mode="reflect")

        if squeeze_output:
            scaled = scaled.squeeze(0)

        return scaled


class GaussianNoise:
    """高斯噪声增强。

    Parameters
    ----------
    std_range : tuple(float, float)
        噪声标准差范围（相对于图像强度范围）
    """

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.05)):
        self.std_range = std_range

    def __call__(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        std = rng.uniform(*self.std_range)
        if std > 0:
            noise = torch.randn_like(img) * std
            img = img + noise
            img = img.clamp(0, 1)
        return img


class RandomRotation90:
    """随机 90° 旋转（保持周期性图案的结构）。"""

    def __call__(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        k = rng.integers(0, 4)
        if k > 0:
            img = torch.rot90(img, k, dims=[-2, -1])
        return img


class RandomFlip:
    """随机水平/垂直翻转。"""

    def __call__(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if rng.random() > 0.5:
            img = torch.flip(img, dims=[-1])
        if rng.random() > 0.5:
            img = torch.flip(img, dims=[-2])
        return img


class Compose:
    """组合多个增强操作。"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        for t in self.transforms:
            img = t(img, rng)
        return img


def get_default_augmentation(
    target_size: int = 128,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    noise_std_range: Tuple[float, float] = (0.0, 0.03),
) -> Compose:
    """获取默认的数据增强组合。

    Parameters
    ----------
    target_size : int
        目标图像尺寸
    scale_range : tuple
        多尺度范围
    noise_std_range : tuple
        高斯噪声标准差范围

    Returns
    -------
    Compose
        组合的增强操作
    """
    return Compose([
        RandomFlip(),
        RandomRotation90(),
        RandomZoom(zoom_range=scale_range, target_size=target_size),
        GaussianNoise(std_range=noise_std_range),
    ])


# ── 评估增强效果的工具函数 ─────────────────────────────────────


def compute_scale_invariance_score(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    scales: list = [0.8, 0.9, 1.0, 1.1, 1.2],
) -> dict:
    """计算模型的尺度不变性分数。

    对同一图像在不同尺度下预测，评估预测的一致性。

    Parameters
    ----------
    model : torch.nn.Module
        训练好的 CNN 模型
    images : torch.Tensor
        测试图像，形状 (N, C, H, W)
    device : torch.device
        计算设备
    scales : list
        测试尺度列表

    Returns
    -------
    dict
        包含尺度不变性指标的字典
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for scale in scales:
            scaled_preds = []
            for i in range(len(images)):
                img = images[i:i+1]
                # 缩放
                h, w = img.shape[-2:]
                new_size = int(h * scale)
                scaled = F.interpolate(img, size=(new_size, new_size), mode="bilinear", align_corners=False)

                # 如果变大，中心裁剪；如果变小，padding
                if new_size > h:
                    start = (new_size - h) // 2
                    scaled = scaled[:, :, start:start+h, start:start+h]
                elif new_size < h:
                    pad = (h - new_size) // 2
                    scaled = F.pad(scaled, (pad, h - new_size - pad, pad, h - new_size - pad), mode="reflect")

                scaled = scaled.to(device)
                pred = model(scaled).squeeze().cpu().item()
                scaled_preds.append(pred)

            all_preds.append(scaled_preds)

    # 计算每个样本在不同尺度下的预测方差
    all_preds = np.array(all_preds)  # (num_scales, num_samples)
    per_sample_var = np.var(all_preds, axis=0)
    mean_var = np.mean(per_sample_var)

    return {
        "scale_invariance_score": 1.0 / (1.0 + mean_var),  # 越高越好
        "mean_prediction_variance": mean_var,
        "predictions_by_scale": all_preds,
    }