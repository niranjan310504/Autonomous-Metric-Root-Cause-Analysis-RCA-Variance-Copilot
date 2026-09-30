"""Tests for deterministic synthetic data generation."""

import pytest

from data.synthetic import generate_observations


def test_generate_observations_validates_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        generate_observations("revenue", [1.0], [{}, {}])
