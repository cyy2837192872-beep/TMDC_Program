"""Regressions for the failures found during project acceptance."""
import argparse
import csv
import sys
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from angle_cnn import inference_real_data as inference
from angle_cnn.preprocess_afm_data import load_ibw_file
from angle_cnn.train_cnn import train_one_epoch


@pytest.mark.parametrize("channel_axis", [0, 2])
def test_ibw_axis_notes_units_and_channel_names(tmp_path, monkeypatch, channel_axis):
    raw = np.stack([np.full((32, 32), v) for v in [1, 2, 3]], axis=channel_axis)
    package = types.ModuleType("igor2")
    binary = types.ModuleType("igor2.binarywave")
    package.binarywave = binary
    binary.load = lambda path: {"wave": {"wData": raw, "note": b"ScanSize:3e-7\rScanRate:2\rChannelName:Phase;Height;Amplitude"}}
    monkeypatch.setitem(sys.modules, "igor2", package)
    monkeypatch.setitem(sys.modules, "igor2.binarywave", binary)
    path = tmp_path / "scan.ibw"
    path.touch()
    channels, meta = load_ibw_file(str(path))
    assert channels["height"].shape == (32, 32)
    assert np.all(channels["height"] == 2)
    assert meta.scan_size_nm == pytest.approx(300)
    assert meta.scan_rate_hz == 2


def test_training_rejects_empty_batches_and_counts_processed_samples():
    model = torch.nn.Linear(1, 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.SGD(model.parameters(), lr=0)
    ds = TensorDataset(torch.zeros(5, 1), torch.tensor([1., 1., 1., 1., 99.]))
    with pytest.raises(ValueError, match="no batches"):
        train_one_epoch(model, DataLoader(ds, batch_size=8, drop_last=True), opt, "cpu")
    loss = train_one_epoch(model, DataLoader(ds, batch_size=2, drop_last=True), opt, "cpu", huber_delta=1)
    assert loss == pytest.approx(0.5)


def setup_inference(tmp_path, monkeypatch):
    args = argparse.Namespace(seed=42, mc_samples=3, model=__file__, output=str(tmp_path),
                              fov_nm=None, raw=False, no_preview=True, tau=0.5)
    monkeypatch.setattr(inference, "parse_args", lambda: args)
    monkeypatch.setattr(inference, "collect_input_files", lambda args: [("scan.npz", "scan")])
    monkeypatch.setattr(inference, "load_cnn_model", lambda *a: (None, False, 3, {}))
    monkeypatch.setattr(inference, "load_preprocessed_npz", lambda *a: (
        np.zeros((3, 128, 128)), np.zeros((512, 512)), {"scan_size_nm": 300}))
    monkeypatch.setattr(inference, "run_cnn_inference", lambda *a: (float(torch.rand(())), float(torch.rand(())), None))
    monkeypatch.setattr(inference, "run_fft_inference", lambda *a: 2.)
    return args


def test_inference_seed_repeats_outputs(tmp_path, monkeypatch):
    setup_inference(tmp_path, monkeypatch)
    assert inference.main() == 0
    first = (tmp_path / "real_data_results.csv").read_text()
    assert inference.main() == 0
    assert (tmp_path / "real_data_results.csv").read_text() == first


def test_inference_failure_returns_nonzero_and_keeps_error_rows(tmp_path, monkeypatch):
    setup_inference(tmp_path, monkeypatch)
    def invalid(*a):
        raise ValueError("invalid FOV")
    monkeypatch.setattr(inference, "load_preprocessed_npz", invalid)
    assert inference.main() == 1
    with (tmp_path / "real_data_results.csv").open() as f:
        assert next(csv.DictReader(f))["cnn_angle_deg"] == "ERROR"


def test_graded_fft_uses_no_label_prior_and_repeats_scenes(monkeypatch):
    from angle_cnn import graded_eval as graded
    monkeypatch.setattr(graded, "TEST_THETAS", [2.])
    monkeypatch.setattr(graded, "N_TRIALS", 2)
    monkeypatch.setattr(graded, "DEGRADATION_LEVELS", [("test", 0, 0, 0, 0, 0, 0, 0, 0)])
    images = []
    def scene(*args, **kwargs):
        im = args[-1].random((16, 16)).astype(np.float32)
        images.append(im.copy())
        # Sentinel oracle values must never be passed to the FFT implementation.
        return im, im, 300., object(), im, 75., object()
    monkeypatch.setattr(graded, "generate_test_image", scene)
    monkeypatch.setattr(graded, "extract_angle_fft_blind", lambda image, fov: np.nan)
    monkeypatch.setattr(graded, "cnn_predict_single", lambda *a, **kw: 2.)
    one = graded.run_graded_eval(None, "cpu")
    graded.run_graded_eval(None, "cpu")
    assert one["test"]["fft_fail"] == [100.]
    np.testing.assert_array_equal(images[:2], images[2:])
