"""Tests for core/metrics.py — stratified evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from angle_cnn.core.metrics import (
    StratifiedMetrics,
    compute_stratified_metrics,
    compute_calibration_metrics,
)


class TestComputeStratifiedMetrics:
    """Tests for compute_stratified_metrics — the backbone of all evaluation."""

    def test_perfect_predictions(self):
        preds = np.array([0.5, 1.0, 2.0, 3.0, 4.5], dtype=np.float32)
        labels = np.array([0.5, 1.0, 2.0, 3.0, 4.5], dtype=np.float32)
        m = compute_stratified_metrics(preds, labels)
        assert m.overall_mae == 0.0
        assert m.max_error == 0.0
        assert m.median_error == 0.0
        assert m.failure_rate == 0.0
        assert m.n_valid == 5

    def test_constant_offset(self):
        preds = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        labels = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        m = compute_stratified_metrics(preds, labels)
        assert m.overall_mae == pytest.approx(0.5)
        assert m.n_valid == 4

    def test_with_nan_predictions(self):
        preds = np.array([0.5, np.nan, 2.0, np.nan, 4.0], dtype=np.float32)
        labels = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        m = compute_stratified_metrics(preds, labels)
        assert m.n_valid == 3
        assert m.failure_rate == pytest.approx(40.0)
        assert m.overall_mae == 0.0  # The 3 valid predictions are exact

    def test_all_nan(self):
        preds = np.full(5, np.nan, dtype=np.float32)
        labels = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        m = compute_stratified_metrics(preds, labels)
        assert m.n_valid == 0
        assert m.failure_rate == 100.0
        assert m.overall_mae == 0.0

    def test_percentile_errors(self):
        rng = np.random.default_rng(42)
        labels = np.linspace(0.5, 5.0, 1000)
        preds = labels + rng.normal(0, 0.1, size=1000).astype(np.float32)
        errors = np.abs(preds - labels)
        m = compute_stratified_metrics(preds, labels)
        assert m.percentile_errors[50] == pytest.approx(np.median(errors), rel=0.05)
        assert m.percentile_errors[90] == pytest.approx(np.percentile(errors, 90), rel=0.05)
        assert m.percentile_errors[95] == pytest.approx(np.percentile(errors, 95), rel=0.05)
        assert m.percentile_errors[99] == pytest.approx(np.percentile(errors, 99), rel=0.05)

    def test_small_vs_large_angle(self):
        labels = np.array([0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 4.5, 4.8], dtype=np.float32)
        preds = labels.copy()
        preds[0:3] += 0.2  # small-angle errors
        preds[-2:] += 0.1  # large-angle errors
        m = compute_stratified_metrics(preds, labels,
                                        small_angle_threshold=1.5,
                                        large_angle_threshold=3.5)
        assert m.small_angle_mae > 0  # small angle has errors
        assert m.large_angle_mae > 0  # large angle has errors
        # small angle errors were larger
        assert m.small_angle_mae > m.large_angle_mae

    def test_angle_bin_mae(self):
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        preds = labels.copy()
        m = compute_stratified_metrics(preds, labels, n_angle_bins=5)
        assert len(m.angle_bin_mae) == 5
        for center, mae in m.angle_bin_mae.items():
            assert mae == 0.0  # perfect predictions

    def test_empty_input(self):
        preds = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.float32)
        m = compute_stratified_metrics(preds, labels)
        assert m.n_samples == 0
        assert m.n_valid == 0
        assert m.failure_rate == 100.0

    def test_single_sample(self):
        preds = np.array([2.5], dtype=np.float32)
        labels = np.array([2.5], dtype=np.float32)
        m = compute_stratified_metrics(preds, labels)
        assert m.overall_mae == 0.0
        assert m.n_valid == 1

    def test_custom_percentiles(self):
        labels = np.linspace(0.5, 5.0, 100)
        preds = labels + 0.05
        errors = np.abs(preds - labels)
        m = compute_stratified_metrics(preds, labels, percentiles=(80, 99))
        assert 80 in m.percentile_errors
        assert 99 in m.percentile_errors
        assert 50 not in m.percentile_errors
        assert m.percentile_errors[80] == pytest.approx(np.percentile(errors, 80), rel=0.05)

    def test_to_dict(self):
        preds = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m = compute_stratified_metrics(preds, labels)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "overall_mae" in d
        assert "p95" in d
        assert "small_angle_mae" in d

    def test_summary_string(self):
        preds = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m = compute_stratified_metrics(preds, labels)
        s = m.summary()
        assert "整体 MAE" in s
        assert "最大误差" in s


class TestComputeCalibrationMetrics:
    """Tests for compute_calibration_metrics (MC Dropout calibration)."""

    def test_perfect_calibration(self):
        preds = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        labels = preds.copy()
        uncs = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        cal = compute_calibration_metrics(preds, labels, uncs)
        # No error → all errors are within any σ
        assert cal["within_1sigma_pct"] == 100.0
        assert cal["within_2sigma_pct"] == 100.0

    def test_with_nan(self):
        preds = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        uncs = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        cal = compute_calibration_metrics(preds, labels, uncs)
        # NaN is filtered out, only 2 valid
        assert cal["within_1sigma_pct"] >= 0

    def test_output_keys(self):
        preds = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        labels = preds.copy()
        uncs = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cal = compute_calibration_metrics(preds, labels, uncs)
        expected_keys = {
            "within_1sigma_pct", "within_2sigma_pct", "within_3sigma_pct",
            "error_uncertainty_correlation", "calibration_mse",
            "expected_1sigma", "expected_2sigma",
        }
        assert expected_keys.issubset(cal.keys())
