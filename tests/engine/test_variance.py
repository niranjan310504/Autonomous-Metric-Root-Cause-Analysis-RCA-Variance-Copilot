"""Tests for deterministic variance calculations."""

import pytest

from engine.variance import relative_variance


def test_relative_variance() -> None:
    assert relative_variance(120.0, 100.0) == pytest.approx(0.2)


def test_relative_variance_rejects_zero_comparison() -> None:
    with pytest.raises(ValueError, match="must not be zero"):
        relative_variance(1.0, 0.0)
