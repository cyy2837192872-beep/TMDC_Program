"""Tests for core.physics — physical formula correctness."""

import math

import numpy as np
import pytest


class TestMoirePeriod:
    def test_known_value(self):
        from core.physics import moire_period
        # Manual calculation: L = 0.316 * sqrt(3) / (4 * sin(0.5°))
        theta = 1.0
        expected = 0.316 * math.sqrt(3) / (4 * math.sin(math.radians(theta / 2.0)))
        assert abs(moire_period(theta) - expected) < 1e-10

    def test_zero_angle_raises(self):
        from core.physics import moire_period
        with pytest.raises(ValueError):
            moire_period(0.0)

    def test_period_decreases_with_angle(self):
        from core.physics import moire_period
        assert moire_period(0.5) > moire_period(5.0)


class TestThetaFromPeriod:
    def test_roundtrip(self):
        from core.physics import moire_period, theta_from_period
        for theta in [0.5, 1.0, 2.0, 3.0, 5.0]:
            L = moire_period(theta)
            theta_back = theta_from_period(L)
            assert abs(theta_back - theta) < 1e-6

    def test_small_angle(self):
        from core.physics import theta_from_period
        # Very large period -> very small angle
        theta = theta_from_period(1000.0)
        assert 0 < theta < 0.1


class TestAngleUncertainty:
    def test_decreases_with_fov(self):
        from core.physics import angle_uncertainty
        assert angle_uncertainty(100.0) > angle_uncertainty(400.0)

    def test_positive(self):
        from core.physics import angle_uncertainty
        assert angle_uncertainty(100.0) > 0

    def test_known_value(self):
        from core.physics import angle_uncertainty
        # δθ = a√3 / (2·fov) in radians, then convert to degrees
        fov = 313.6  # FIXED_FOV_NM
        expected_rad = 0.316 * math.sqrt(3) / (2.0 * fov)
        expected_deg = math.degrees(expected_rad)
        assert abs(angle_uncertainty(fov) - expected_deg) < 1e-10


class TestFixedFov:
    def test_positive(self):
        from core.physics import FIXED_FOV_NM
        assert FIXED_FOV_NM > 0

    def test_matches_formula(self):
        from core.config import THETA_MIN
        from core.physics import moire_period, FIXED_FOV_NM
        expected = 10.0 * moire_period(THETA_MIN)
        assert abs(FIXED_FOV_NM - expected) < 1e-10


class TestPixelsPerMoirePeriod:
    def test_matches_n_L_over_fov(self):
        from core.physics import FIXED_FOV_NM, moire_period, pixels_per_moire_period

        n = 512
        theta = 1.0
        L = moire_period(theta)
        ppp = pixels_per_moire_period(n, theta, FIXED_FOV_NM)
        assert abs(ppp - n * L / FIXED_FOV_NM) < 1e-6

    def test_r_peak_identity(self):
        """extract_angle_fft uses r_peak = n_img / ppp = fov / L."""
        from core.physics import FIXED_FOV_NM, moire_period, pixels_per_moire_period

        n = 512
        theta = 2.0
        L = moire_period(theta)
        ppp = pixels_per_moire_period(n, theta, FIXED_FOV_NM)
        r_peak = n / ppp
        assert abs(r_peak - FIXED_FOV_NM / L) < 1e-6
