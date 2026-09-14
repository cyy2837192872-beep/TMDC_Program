"""Exercise the actual training CLI across process boundaries with a small test network."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


RUNNER = '''
import torch
from angle_cnn import train_cnn as train
torch.set_num_threads(1)
def tiny(n_channels, **kw):
    return torch.nn.Sequential(torch.nn.Conv2d(n_channels, 4, 3), torch.nn.BatchNorm2d(4),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Dropout(.3), torch.nn.Linear(4,1))
train.build_model = tiny
train.get_device = lambda: torch.device('cpu')
train.main()
'''


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("workers,extras", [(0, []), (1, []), (0, ["--swa", "--ema-decay", "0.9", "--fft-channel"])])
def test_interrupted_training_matches_uninterrupted(tmp_path, workers, extras):
    rng = np.random.default_rng(7)
    arrays = {}
    for split, n in [("train", 8), ("val", 3), ("test", 3)]:
        arrays[f"images_{split}"] = rng.random((n, 3, 16, 16), dtype=np.float32)
        arrays[f"labels_{split}"] = rng.uniform(.5, 5, n).astype(np.float32)
        arrays[f"fovs_{split}"] = np.full(n, 78.4, np.float32)
    np.savez(tmp_path / "moire_dataset.npz", **arrays)
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo), "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
    base = ["--data-dir", str(tmp_path), "--epochs", "3", "--batch-size", "4", "--warmup-epochs", "0",
            "--num-workers", str(workers), "--no-compile", "--no-amp", "--no-progress", "--mc-samples", "0", *extras]
    def run(output, extra):
        p = subprocess.run([sys.executable, "-c", RUNNER, *base, "--output-dir", str(output), *extra],
                           env=env, cwd=repo, capture_output=True, text=True, timeout=90)
        assert p.returncode == 0, p.stdout + p.stderr
    full, partial = tmp_path / "full", tmp_path / "partial"
    run(full, ["--stop-after-epoch", "3"])
    run(partial, ["--stop-after-epoch", "1"])
    run(partial, ["--resume", str(partial / "last_training.pt"), "--stop-after-epoch", "3"])
    a = torch.load(full / "last_training.pt", weights_only=False)
    b = torch.load(partial / "last_training.pt", weights_only=False)
    for key in ["model_state_dict", "optimizer", "scheduler", "rng", "history", "log_rows",
                "ema", "swa", "swa_scheduler", "best_checkpoint", "patience_cnt"]:
        assert_equal(a[key], b[key])
    assert (full / "train_log.csv").read_text() == (partial / "train_log.csv").read_text()
    if workers == 0 and not extras:
        arrays['labels_train'][0] += 0.01
        np.savez(tmp_path / 'moire_dataset.npz', **arrays)
        p = subprocess.run([sys.executable, '-c', RUNNER, *base, '--output-dir', str(partial),
                            '--resume', str(partial / 'last_training.pt')],
                           env=env, cwd=repo, capture_output=True, text=True, timeout=90)
        assert p.returncode != 0
        assert 'Resume dataset differs' in p.stderr
