"""Regression checks for physical scale and report provenance."""

import numpy as np
import pytest

from angle_cnn.core.afm_preprocess import TRAIN_CROP_FOV_NM, prepare_afm_channels
from angle_cnn.core.fusion import fuse_angles
from angle_cnn.inference_real_data import load_preprocessed_npz


def test_afm_crop_preserves_training_scale_and_fft_resolution():
    n = 512
    height = np.tile(np.arange(n, dtype=np.float32), (n, 1))
    image, fft_height = prepare_afm_channels({"height": height}, TRAIN_CROP_FOV_NM * 4)
    assert image.shape == (3, 128, 128)
    assert fft_height.shape == (512, 512)
    np.testing.assert_allclose(image[0], fft_height[192:320, 192:320])
    np.testing.assert_allclose(image[1], image[0])


def test_afm_rejects_undersampled_or_unknown_fov():
    image = {"height": np.ones((256, 256), dtype=np.float32)}
    with pytest.raises(ValueError, match="resolution"):
        prepare_afm_channels(image, TRAIN_CROP_FOV_NM * 4)
    with pytest.raises(ValueError, match="scan_size_nm"):
        prepare_afm_channels(image, 0)


def test_legacy_npz_cannot_be_used_as_full_resolution_fft(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, image=np.zeros((3, 128, 128)), metadata="{}")
    with pytest.raises(ValueError, match="height_fft"):
        load_preprocessed_npz(str(path))


def test_fusion_is_shared_and_nan_fft_falls_back_to_cnn():
    fused, weight = fuse_angles(np.array([2.0, np.nan]), np.array([1.0, 1.0]),
                                np.array([0.5, 0.5]), 0.5)
    np.testing.assert_allclose(fused, [1.5, 1.0])
    np.testing.assert_allclose(weight, [0.5, 0.0])
