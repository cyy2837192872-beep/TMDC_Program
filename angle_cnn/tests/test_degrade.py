"""Tests for core.degrade — AFM degradation functions."""

import numpy as np
import pytest


class TestAffineDistortion:
    def test_identity(self):
        from angle_cnn.core.degrade import apply_affine_distortion
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_affine_distortion(img, 0, 0, 1.0, 1.0)
        np.testing.assert_array_almost_equal(out, img)

    def test_shape_preserved(self):
        from angle_cnn.core.degrade import apply_affine_distortion
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_affine_distortion(img, 0.05, 0.03, 1.1, 0.9)
        assert out.shape == (64, 64)


class TestGaussianNoise:
    def test_shape_preserved(self):
        from angle_cnn.core.degrade import apply_isotropic_gaussian_noise
        rng = np.random.default_rng(42)
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_isotropic_gaussian_noise(img, 0.1, rng)
        assert out.shape == (64, 64)

    def test_zero_noise(self):
        from angle_cnn.core.degrade import apply_isotropic_gaussian_noise
        rng = np.random.default_rng(42)
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_isotropic_gaussian_noise(img, 0.0, rng)
        np.testing.assert_array_equal(out, img)


class TestBackgroundTilt:
    def test_shape_preserved(self):
        from angle_cnn.core.degrade import apply_background_tilt
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_background_tilt(img, 0.1, 1.0, 0.5)
        assert out.shape == (64, 64)

    def test_zero_amplitude(self):
        from angle_cnn.core.degrade import apply_background_tilt
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_background_tilt(img, 0.0, 1.0, 0.5)
        np.testing.assert_array_almost_equal(out, img)


class TestGaussianBlur:
    def test_shape_preserved(self):
        from angle_cnn.core.degrade import apply_gaussian_blur
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_gaussian_blur(img, sigma=2.0)
        assert out.shape == (64, 64)

    def test_zero_sigma(self):
        from angle_cnn.core.degrade import apply_gaussian_blur
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_gaussian_blur(img, sigma=0.0)
        np.testing.assert_array_equal(out, img)


class TestTipConvolution:
    def test_zero_radius(self):
        from angle_cnn.core.degrade import apply_tip_convolution
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_tip_convolution(img, 0.0, 1.0)
        np.testing.assert_array_equal(out, img)

    def test_shape_preserved(self):
        from angle_cnn.core.degrade import apply_tip_convolution
        img = np.random.randn(64, 64).astype(np.float32)
        out = apply_tip_convolution(img, 7.0, 0.5)
        assert out.shape == (64, 64)
