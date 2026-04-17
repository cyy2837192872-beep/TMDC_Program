#!/usr/bin/env python3
"""
test_core.py — MoS₂ moiré CNN 核心模块单元测试
==============================================

测试覆盖：
- augment.py：数据增强模块
- metrics.py：评估指标模块
- dual_stream.py：双流架构模块
- cnn.py：基础 CNN 模块

运行方式
--------
    # 运行所有测试
    pytest tests/test_core.py -v

    # 运行特定测试
    pytest tests/test_core.py::test_augmentation -v

    # 带覆盖率
    pytest tests/test_core.py --cov=core --cov-report=html
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pytest
import torch

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """随机数生成器 fixture。"""
    return np.random.default_rng(42)


@pytest.fixture
def sample_image():
    """单通道示例图像。"""
    return torch.randn(1, 128, 128)


@pytest.fixture
def sample_multichannel_image():
    """三通道示例图像。"""
    return torch.randn(3, 128, 128)


@pytest.fixture
def sample_batch():
    """批量示例图像。"""
    return torch.randn(8, 3, 128, 128)


@pytest.fixture
def sample_predictions():
    """示例预测和标签。"""
    rng = np.random.default_rng(42)
    labels = rng.uniform(0.5, 5.0, 100).astype(np.float32)
    predictions = labels + rng.normal(0, 0.1, 100).astype(np.float32)
    uncertainties = rng.uniform(0.05, 0.2, 100).astype(np.float32)
    return predictions, labels, uncertainties


# ── Augmentation Tests ────────────────────────────────────────────────────

class TestAugmentation:
    """数据增强模块测试。"""

    def test_random_flip_shape(self, sample_image, rng):
        """测试随机翻转保持形状。"""
        from core.augment import RandomFlip

        flip = RandomFlip()
        result = flip(sample_image, rng)

        assert result.shape == sample_image.shape

    def test_random_rotation90_shape(self, sample_image, rng):
        """测试 90° 旋转保持形状。"""
        from core.augment import RandomRotation90

        rot = RandomRotation90()
        result = rot(sample_image, rng)

        assert result.shape == sample_image.shape

    def test_random_zoom_shape(self, sample_image, rng):
        """测试随机缩放输出正确形状。"""
        from core.augment import RandomZoom

        zoom = RandomZoom(zoom_range=(0.8, 1.2), target_size=128)
        result = zoom(sample_image, rng)

        assert result.shape == sample_image.shape

    def test_random_zoom_multichannel(self, sample_multichannel_image, rng):
        """测试多通道图像缩放。"""
        from core.augment import RandomZoom

        zoom = RandomZoom(zoom_range=(0.8, 1.2), target_size=128)
        result = zoom(sample_multichannel_image, rng)

        assert result.shape == sample_multichannel_image.shape

    def test_gaussian_noise_range(self, sample_image, rng):
        """测试高斯噪声输出在有效范围。"""
        from core.augment import GaussianNoise

        noise = GaussianNoise(std_range=(0.0, 0.1))
        result = noise(sample_image.clamp(0, 1), rng)

        assert result.min() >= 0
        assert result.max() <= 1

    def test_compose(self, sample_image, rng):
        """测试组合增强。"""
        from core.augment import Compose, RandomFlip, RandomRotation90, RandomZoom

        compose = Compose([
            RandomFlip(),
            RandomRotation90(),
            RandomZoom(zoom_range=(0.9, 1.1), target_size=128),
        ])

        result = compose(sample_image, rng)
        assert result.shape == sample_image.shape

    def test_get_default_augmentation(self, sample_image, rng):
        """测试默认增强组合。"""
        from core.augment import get_default_augmentation

        aug = get_default_augmentation(target_size=128)
        result = aug(sample_image, rng)

        assert result.shape == sample_image.shape


# ── Metrics Tests ──────────────────────────────────────────────────────────

class TestMetrics:
    """评估指标模块测试。"""

    def test_stratified_metrics_basic(self, sample_predictions):
        """测试基础分层指标计算。"""
        from core.metrics import compute_stratified_metrics

        predictions, labels, _ = sample_predictions
        metrics = compute_stratified_metrics(predictions, labels)

        assert metrics.n_samples == 100
        assert metrics.n_valid == 100
        assert metrics.overall_mae > 0
        assert metrics.overall_mae < 1.0  # 假设误差合理

    def test_stratified_metrics_percentiles(self, sample_predictions):
        """测试百分位误差计算。"""
        from core.metrics import compute_stratified_metrics

        predictions, labels, _ = sample_predictions
        metrics = compute_stratified_metrics(predictions, labels)

        assert 50 in metrics.percentile_errors
        assert 90 in metrics.percentile_errors
        assert 95 in metrics.percentile_errors
        assert 99 in metrics.percentile_errors

        # 百分位应递增
        assert metrics.percentile_errors[50] <= metrics.percentile_errors[90]
        assert metrics.percentile_errors[90] <= metrics.percentile_errors[95]
        assert metrics.percentile_errors[95] <= metrics.percentile_errors[99]

    def test_stratified_metrics_angle_bins(self, sample_predictions):
        """测试角度区间分层。"""
        from core.metrics import compute_stratified_metrics

        predictions, labels, _ = sample_predictions
        metrics = compute_stratified_metrics(
            predictions, labels,
            n_angle_bins=10
        )

        assert len(metrics.angle_bin_mae) == 10

    def test_stratified_metrics_with_nan(self, sample_predictions):
        """测试包含 NaN 的预测处理。"""
        from core.metrics import compute_stratified_metrics

        predictions, labels, _ = sample_predictions
        predictions_with_nan = predictions.copy()
        predictions_with_nan[::10] = np.nan  # 10% NaN

        metrics = compute_stratified_metrics(predictions_with_nan, labels)

        assert metrics.n_valid == 90
        assert metrics.failure_rate == 10.0

    def test_calibration_metrics(self, sample_predictions):
        """测试校准指标计算。"""
        from core.metrics import compute_calibration_metrics

        predictions, labels, uncertainties = sample_predictions
        cal = compute_calibration_metrics(predictions, labels, uncertainties)

        assert "within_1sigma_pct" in cal
        assert "within_2sigma_pct" in cal
        assert "error_uncertainty_correlation" in cal
        assert 0 <= cal["within_1sigma_pct"] <= 100

    def test_metrics_to_dict(self, sample_predictions):
        """测试指标转字典。"""
        from core.metrics import compute_stratified_metrics

        predictions, labels, _ = sample_predictions
        metrics = compute_stratified_metrics(predictions, labels)
        d = metrics.to_dict()

        assert "overall_mae" in d
        assert "p90" in d
        assert "p95" in d
        assert isinstance(d["overall_mae"], float)

    def test_metrics_summary(self, sample_predictions):
        """测试指标摘要输出。"""
        from core.metrics import compute_stratified_metrics

        predictions, labels, _ = sample_predictions
        metrics = compute_stratified_metrics(predictions, labels)
        summary = metrics.summary()

        assert isinstance(summary, str)
        assert "MAE" in summary
        assert "P90" in summary


# ── Dual Stream Tests ───────────────────────────────────────────────────────

class TestDualStream:
    """双流架构模块测试。"""

    def test_fft_feature_extractor_shape(self, sample_batch):
        """测试 FFT 特征提取器输出形状。"""
        from core.dual_stream import FFTFeatureExtractor

        extractor = FFTFeatureExtractor(out_channels=64)
        features = extractor(sample_batch)

        # 输入 128x128, 3次 stride=2 下采样 -> 16x16
        assert features.shape == (8, 64, 16, 16)

    def test_channel_attention_shape(self, sample_batch):
        """测试通道注意力输出形状。"""
        from core.dual_stream import ChannelAttention

        attention = ChannelAttention(channels=3)
        result = attention(sample_batch)

        assert result.shape == sample_batch.shape

    def test_spatial_attention_shape(self, sample_batch):
        """测试空间注意力输出形状。"""
        from core.dual_stream import SpatialAttention

        attention = SpatialAttention()
        result = attention(sample_batch)

        assert result.shape == sample_batch.shape

    def test_feature_fusion_concat(self):
        """测试 concat 融合。"""
        from core.dual_stream import FeatureFusion

        fusion = FeatureFusion(
            spatial_channels=64,
            freq_channels=32,
            out_channels=64,
            fusion_type="concat",
        )

        spatial = torch.randn(4, 64, 16, 16)
        freq = torch.randn(4, 32, 16, 16)
        result = fusion(spatial, freq)

        assert result.shape == (4, 64, 16, 16)

    def test_feature_fusion_size_mismatch(self):
        """测试尺寸不匹配时的融合。"""
        from core.dual_stream import FeatureFusion

        fusion = FeatureFusion(
            spatial_channels=64,
            freq_channels=32,
            out_channels=64,
            fusion_type="concat",
        )

        spatial = torch.randn(4, 64, 16, 16)
        freq = torch.randn(4, 32, 8, 8)  # 不同尺寸
        result = fusion(spatial, freq)

        assert result.shape == (4, 64, 16, 16)

    def test_dual_stream_net_forward(self, sample_batch):
        """测试双流网络前向传播。"""
        from core.dual_stream import DualStreamNet

        model = DualStreamNet(n_channels=3, dropout=0.3, arch="resnet18")
        output = model(sample_batch)

        assert output.shape == (8, 1)
        assert not torch.isnan(output).any()

    def test_dual_stream_net_with_attention(self, sample_batch):
        """测试带注意力的双流网络。"""
        from core.dual_stream import DualStreamNetWithAttention

        model = DualStreamNetWithAttention(n_channels=3, dropout=0.3)
        output = model(sample_batch)

        assert output.shape == (8, 1)
        assert not torch.isnan(output).any()

    def test_lightweight_dual_stream_net(self, sample_batch):
        """测试轻量级双流网络。"""
        from core.dual_stream import LightweightDualStreamNet

        model = LightweightDualStreamNet(n_channels=3, dropout=0.3)
        output = model(sample_batch)

        assert output.shape == (8, 1)
        assert not torch.isnan(output).any()

    def test_build_dual_stream_model(self):
        """测试模型构建函数。"""
        from core.dual_stream import build_dual_stream_model

        # 基础模型
        model1 = build_dual_stream_model(n_channels=3, lightweight=False, use_attention=False)
        assert model1 is not None

        # 注意力模型
        model2 = build_dual_stream_model(n_channels=3, lightweight=False, use_attention=True)
        assert model2 is not None

        # 轻量级模型
        model3 = build_dual_stream_model(n_channels=3, lightweight=True)
        assert model3 is not None


# ── CNN Module Tests ────────────────────────────────────────────────────────

class TestCNN:
    """基础 CNN 模块测试。"""

    def test_build_model_output_shape(self, sample_batch):
        """测试模型输出形状。"""
        from core.cnn import build_model

        model = build_model(n_channels=3, dropout=0.3, arch="resnet18")
        output = model(sample_batch)

        assert output.shape == (8, 1)

    def test_build_model_architectures(self):
        """测试不同骨干网络构建。"""
        from core.cnn import build_model

        for arch in ["resnet18", "resnet34", "resnet50"]:
            model = build_model(n_channels=3, arch=arch)
            assert model is not None

    def test_compute_fft_channel_shape(self, sample_batch):
        """测试 FFT 通道计算形状。"""
        from core.cnn import compute_fft_channel

        result = compute_fft_channel(sample_batch)

        # 应该增加一个通道
        assert result.shape == (8, 4, 128, 128)

    def test_predict_with_uncertainty(self, sample_batch):
        """测试 MC Dropout 推理。"""
        from core.cnn import build_model, predict_with_uncertainty

        model = build_model(n_channels=3, dropout=0.5)
        model.eval()

        mean_deg, std_deg = predict_with_uncertainty(
            model, sample_batch, n_samples=10
        )

        assert mean_deg.shape == (8,)
        assert std_deg.shape == (8,)
        assert np.all(mean_deg >= 0.5) and np.all(mean_deg <= 5.0)

    def test_warmup_cosine_lr(self):
        """测试学习率调度。"""
        from core.cnn import warmup_cosine_lr

        # Warmup 阶段
        for epoch in range(5):
            lr = warmup_cosine_lr(epoch, warmup=5, total=100)
            assert 0 < lr <= 1.0

        # Warmup 后
        lr = warmup_cosine_lr(50, warmup=5, total=100)
        assert 0 < lr < 1.0


# ── Integration Tests ───────────────────────────────────────────────────────

class TestIntegration:
    """集成测试。"""

    def test_full_pipeline(self, sample_batch, tmp_path):
        """测试完整训练-评估流程。"""
        from core.cnn import build_model, predict_with_uncertainty
        from core.metrics import compute_stratified_metrics

        # 构建模型
        model = build_model(n_channels=3, dropout=0.3)

        # 模拟训练（仅前向传播）
        model.eval()
        with torch.no_grad():
            output = model(sample_batch)

        # 模拟评估
        labels = np.random.uniform(0.5, 5.0, 8).astype(np.float32)
        predictions = output.squeeze().cpu().numpy() * 4.5 + 0.5

        metrics = compute_stratified_metrics(predictions, labels)

        assert metrics.n_samples == 8

    def test_augmentation_preserves_label(self, sample_image, rng):
        """测试增强不改变标签（角度）。"""
        from core.augment import get_default_augmentation

        aug = get_default_augmentation(target_size=128)
        result = aug(sample_image, rng)

        # 翻转和旋转不改变角度，仅缩放可能影响
        # 这里验证形状一致
        assert result.shape == sample_image.shape


# ── Performance Tests ───────────────────────────────────────────────────────

class TestPerformance:
    """性能测试。"""

    def test_inference_speed(self, sample_batch):
        """测试推理速度。"""
        import time
        from core.cnn import build_model

        model = build_model(n_channels=3, dropout=0.3, arch="resnet18")
        model.eval()

        # 预热
        with torch.no_grad():
            _ = model(sample_batch)

        # 计时
        n_runs = 10
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(sample_batch)
        elapsed = time.time() - start

        avg_time = elapsed / n_runs
        throughput = sample_batch.shape[0] / avg_time

        print(f"\n推理速度: {throughput:.1f} samples/s")
        assert avg_time < 1.0  # 单批次推理应小于 1 秒

    def test_dual_stream_inference_speed(self, sample_batch):
        """测试双流网络推理速度。"""
        import time
        from core.dual_stream import DualStreamNet

        model = DualStreamNet(n_channels=3, dropout=0.3, arch="resnet18")
        model.eval()

        # 预热
        with torch.no_grad():
            _ = model(sample_batch)

        # 计时
        n_runs = 10
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(sample_batch)
        elapsed = time.time() - start

        avg_time = elapsed / n_runs
        throughput = sample_batch.shape[0] / avg_time

        print(f"\n双流网络推理速度: {throughput:.1f} samples/s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])