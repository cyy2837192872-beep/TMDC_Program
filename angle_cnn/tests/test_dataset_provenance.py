"""Physical FOV and metric-schema regressions."""

import csv
import json

import numpy as np
import pytest

from angle_cnn.core.io_utils import read_dataset_config
from angle_cnn.core.physics import FIXED_FOV_NM
from angle_cnn.dataset_generator import generate_sample, generate_dataset
from angle_cnn.generate_thesis_figures import gen_table_robustness
from angle_cnn import robustness_sweep


def test_generated_sample_fov_describes_saved_crop():
    image, crop_fov = generate_sample(
        2.0, np.random.default_rng(7), img_size=128,
        fixed_fov_nm=FIXED_FOV_NM, n_channels=1, n_sim=256,
    )
    assert image.shape == (128, 128)
    assert crop_fov == pytest.approx(FIXED_FOV_NM / 2)


def test_worker_count_does_not_change_generated_dataset():
    args = dict(n_total=4, seed=19, img_size=64, n_sim=128,
                n_channels=1, no_progress=True)
    sequential = generate_dataset(parallel_workers=0, **args)
    parallel = generate_dataset(parallel_workers=2, **args)
    for a, b in zip(sequential, parallel):
        np.testing.assert_array_equal(a, b)


def test_dataset_config_is_readable_without_pickle(tmp_path):
    path = tmp_path / "dataset.npz"
    np.savez(path, config=json.dumps({"format_version": 2, "fov_semantics": "cnn_crop_nm"}))
    with np.load(path, allow_pickle=False) as data:
        assert read_dataset_config(data)["fov_semantics"] == "cnn_crop_nm"


def test_legacy_pickle_config_is_not_trusted(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, config={"n_sim": 512})
    with np.load(path, allow_pickle=False) as data:
        assert read_dataset_config(data) is None


def test_robustness_csv_distinguishes_mae_medae_and_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(robustness_sweep, "OUT_DIR", str(tmp_path))
    row = {
        "levels": [0.1], "train_range": (0, 0.2), "n_trials": 3,
        "fft_mae": [0.3], "fft128_mae": [float("nan")], "cnn_mae": [0.2], "cnn_std": [0.05],
        "fft_medae": [0.1], "fft128_medae": [float("nan")], "cnn_medae": [0.15],
        "fft_fail_rate_pct": [33.3], "fft128_fail_rate_pct": [100.0], "cnn_fail_rate_pct": [0.0],
    }
    robustness_sweep.save_csv({"noise": row})
    with (tmp_path / "robustness_sweep.csv").open(newline="") as f:
        saved = next(csv.DictReader(f))
    assert saved["fft512_mae_deg"] == "0.3000"
    assert saved["fft512_medae_deg"] == "0.1000"
    assert saved["fft512_fail_rate_pct"] == "33.3"
    assert saved["fft_period_prior"] == "none"


def test_robustness_table_marks_all_failed_fft_unavailable():
    tex = gen_table_robustness([{
        "dim": "noise", "level": "0.2", "fft512_mae_deg": "nan",
        "cnn_mae_deg": "0.15", "fft512_fail_rate_pct": "100", "is_ood": "True",
    }])
    assert "100.0\\%" in tex
    assert "---" in tex
    assert "nan" not in tex.lower()
