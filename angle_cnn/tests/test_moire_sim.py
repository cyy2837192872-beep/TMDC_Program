"""Tests for core.moire_sim — moiré image synthesis."""

import numpy as np
import pytest

from core.physics import FIXED_FOV_NM


class TestSynthesize:
    def test_single_channel_shape(self):
        from core.moire_sim import synthesize_reconstructed_moire
        img, fov = synthesize_reconstructed_moire(2.0, FIXED_FOV_NM, n=128)
        assert img.shape == (128, 128)
        assert fov > 0

    def test_single_channel_deterministic(self):
        from core.moire_sim import synthesize_reconstructed_moire
        img1, fov1 = synthesize_reconstructed_moire(2.0, FIXED_FOV_NM, n=64)
        img2, fov2 = synthesize_reconstructed_moire(2.0, FIXED_FOV_NM, n=64)
        np.testing.assert_array_equal(img1, img2)
        assert fov1 == fov2

    def test_single_channel_different_angles(self):
        from core.moire_sim import synthesize_reconstructed_moire
        img_small, _ = synthesize_reconstructed_moire(0.5, FIXED_FOV_NM, n=64)
        img_large, _ = synthesize_reconstructed_moire(5.0, FIXED_FOV_NM, n=64)
        assert not np.allclose(img_small, img_large)

    def test_multichannel_keys(self):
        from core.moire_sim import synthesize_multichannel_moire
        result, fov = synthesize_multichannel_moire(2.0, FIXED_FOV_NM, n=64)
        assert "height" in result
        assert "phase" in result
        assert "amplitude" in result
        for key in result:
            assert result[key].shape == (64, 64)

    def test_multichannel_subset(self):
        from core.moire_sim import synthesize_multichannel_moire
        result, _ = synthesize_multichannel_moire(2.0, FIXED_FOV_NM, n=64, channels=("height", "phase"))
        assert "height" in result
        assert "phase" in result
        assert "amplitude" not in result

    def test_fov_consistency(self):
        from core.moire_sim import synthesize_reconstructed_moire
        _, fov = synthesize_reconstructed_moire(2.0, FIXED_FOV_NM, n=512)
        assert abs(fov - FIXED_FOV_NM) < 1e-6
