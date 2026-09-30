"""Deterministic variance calculations."""


def relative_variance(current_value: float, comparison_value: float) -> float:
    """Return the relative change from comparison to current value."""

    if comparison_value == 0:
        raise ValueError("comparison_value must not be zero")
    return (current_value - comparison_value) / comparison_value
